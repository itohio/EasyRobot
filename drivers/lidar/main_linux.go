//go:build linux || windows

package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/itohio/dndm"
	"github.com/itohio/dndm/endpoint/direct"
	"github.com/itohio/dndm/x/bus"

	pbdevices "github.com/itohio/EasyRobot/types/devices"
	devio "github.com/itohio/EasyRobot/x/devices"
	"github.com/itohio/EasyRobot/x/devices/lidar/ld06"
	"github.com/itohio/EasyRobot/x/devices/lidar/xwpftb"
	mat "github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"

	"image"
	"image/color"
	"math"

	"github.com/itohio/EasyRobot/x/marshaller/gocv"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	cv "gocv.io/x/gocv"
)

var (
	serialPort   = flag.String("serial", "", "Serial port device (e.g., /dev/ttyUSB0 or COM3)")
	lidarType    = flag.String("type", "xwpftb", "LiDAR type: xwpftb or ld06")
	targetPts    = flag.Int("target-points", 0, "Target points per rotation (0 = auto-calibrate)")
	display      = flag.Bool("display", false, "Display LiDAR scan visualization")
	windowWidth  = flag.Int("window-width", 0, "Display window width in pixels (0 = use image width)")
	windowHeight = flag.Int("window-height", 0, "Display window height in pixels (0 = use image height)")
	imageWidth   = flag.Int("image-width", 2400, "Image width in pixels (default allows 12m range at 10mm/pixel)")
	imageHeight  = flag.Int("image-height", 2400, "Image height in pixels (default allows 12m range at 10mm/pixel)")
	scaleFactor  = flag.Float64("scale", 10.0, "Scaling factor in mm per pixel (default 10 = 1cm per pixel)")
)

func main() {
	flag.Parse()

	if *serialPort == "" {
		fmt.Fprintf(os.Stderr, "Serial port required (--serial)\n")
		flag.Usage()
		os.Exit(1)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Setup DNDM router (direct endpoint for local use)
	router, err := setupRouter(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to setup router: %v\n", err)
		os.Exit(1)
	}
	defer router.Close()

	// Create LiDAR reading producer
	intentPath := "LIDARReading@lidar.scan"
	producer, err := bus.NewProducer[*pbdevices.LIDARReading](ctx, router, intentPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create producer: %v\n", err)
		os.Exit(1)
	}
	defer producer.Close()

	fmt.Printf("LiDAR driver initialized with intent: %s\n", intentPath)

	// Setup serial
	ser, err := devio.NewSerial(*serialPort)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to open serial port: %v\n", err)
		os.Exit(1)
	}

	// Create LiDAR device (no motor control on Linux/Windows)
	var lidar lidarDevice
	switch *lidarType {
	case "xwpftb":
		dev := xwpftb.New(ctx, ser, nil, *targetPts, 2048)
		if err := dev.Configure(true); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to configure XWPFTB: %v\n", err)
			os.Exit(1)
		}
		lidar = dev
	case "ld06":
		dev := ld06.New(ctx, ser, nil, *targetPts, 3600)
		if err := dev.Configure(true); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to configure LD06: %v\n", err)
			os.Exit(1)
		}
		lidar = dev
	default:
		fmt.Fprintf(os.Stderr, "Unknown LiDAR type: %s\n", *lidarType)
		os.Exit(1)
	}
	defer lidar.Close()

	// Validate display parameters
	if *display {
		if *imageWidth <= 0 || *imageHeight <= 0 {
			fmt.Fprintf(os.Stderr, "Image dimensions must be positive\n")
			os.Exit(1)
		}
		if *scaleFactor <= 0 {
			fmt.Fprintf(os.Stderr, "Scale factor must be positive\n")
			os.Exit(1)
		}
		if *windowWidth < 0 || *windowHeight < 0 {
			fmt.Fprintf(os.Stderr, "Window dimensions must be non-negative\n")
			os.Exit(1)
		}
	}

	// Setup display if requested
	var displayStream types.FrameStream
	var frameChan chan types.Frame
	if *display {
		frameChan = make(chan types.Frame, 1)
		displayStream = types.NewFrameStream(frameChan, func() {
			close(frameChan)
		})

		// Determine window size (use image size if not specified)
		winW := *windowWidth
		winH := *windowHeight
		if winW == 0 {
			winW = *imageWidth
		}
		if winH == 0 {
			winH = *imageHeight
		}

		// Create gocv marshaller with display
		marshaller := gocv.NewMarshaller(
			gocv.WithDisplay(ctx),
			gocv.WithTitle("LiDAR Scan Visualization (Press ESC to exit)"),
			gocv.WithWindowSize(winW, winH),
		)

		// Start marshaller in background
		go func() {
			if err := marshaller.Marshal(nil, displayStream); err != nil {
				fmt.Fprintf(os.Stderr, "Display marshaller error: %v\n", err)
			}
		}()
	}

	// Preallocate image tensor for visualization
	var displayTensor types.Tensor
	var displayMat *cv.Mat
	if *display {
		var err error
		displayTensor, err = tensorgocv.NewImage(*imageHeight, *imageWidth, 3)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to create display tensor: %v\n", err)
			os.Exit(1)
		}
		defer displayTensor.Release()

		accessor, ok := displayTensor.(tensorgocv.Accessor)
		if !ok {
			fmt.Fprintf(os.Stderr, "Display tensor does not implement Accessor\n")
			os.Exit(1)
		}

		displayMat, err = accessor.MatRef()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to get Mat reference: %v\n", err)
			os.Exit(1)
		}
	}

	// Register callback to publish readings and optionally display
	lidar.OnRead(func(m matTypes.Matrix) {
		reading := matrixToLIDARReading(m)
		if err := producer.Send(ctx, reading); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to send reading: %v\n", err)
		}

		// Display visualization if enabled
		if *display && displayTensor != nil && displayMat != nil && frameChan != nil {
			drawLIDARReading(displayMat, reading, *imageWidth, *imageHeight, *scaleFactor)
			select {
			case frameChan <- types.Frame{
				Tensors:   []types.Tensor{displayTensor},
				Index:     0,
				Timestamp: time.Now().UnixNano(),
			}:
			default:
				// Channel full, skip this frame
			}
		}
	})

	// Keep running
	<-ctx.Done()

	// Cleanup display stream
	if displayStream.C != nil {
		displayStream.Close()
	}
}

type lidarDevice interface {
	Configure(init bool) error
	OnRead(fn func(matTypes.Matrix))
	Close()
}

func setupRouter(ctx context.Context) (*dndm.Router, error) {
	directEP := direct.New(10)
	err := directEP.Init(ctx, nil,
		func(intent dndm.Intent, ep dndm.Endpoint) error { return nil },
		func(interest dndm.Interest, ep dndm.Endpoint) error { return nil },
	)
	if err != nil {
		return nil, err
	}
	router, err := dndm.New(
		dndm.WithContext(ctx),
		dndm.WithQueueSize(10),
		dndm.WithEndpoint(directEP),
	)
	if err != nil {
		return nil, err
	}
	return router, nil
}

func matrixToLIDARReading(m matTypes.Matrix) *pbdevices.LIDARReading {
	if m == nil || m.Cols() == 0 {
		return &pbdevices.LIDARReading{
			Valid: false,
		}
	}

	view := m.View().(mat.Matrix)
	distances := make([]float32, m.Cols())
	angles := make([]float32, m.Cols())

	for i := 0; i < m.Cols(); i++ {
		distances[i] = view[0][i]
		angles[i] = view[1][i]
	}

	return &pbdevices.LIDARReading{
		DistancesMm: distances,
		AnglesDeg:   angles,
		Timestamp:   time.Now().UnixNano(),
		Valid:       true,
	}
}

// drawLIDARReading draws a LiDAR reading visualization on a preallocated Mat.
// The Mat is cleared and then redrawn with the current reading.
// imgWidth and imgHeight specify the image dimensions in pixels.
// scaleFactor specifies the scaling in mm per pixel (e.g., 10.0 = 1cm per pixel).
func drawLIDARReading(mat *cv.Mat, reading *pbdevices.LIDARReading, imgWidth, imgHeight int, scaleFactor float64) {
	centerX := imgWidth / 2
	centerY := imgHeight / 2
	maxRange := float64(math.Min(float64(imgWidth), float64(imgHeight))/2) * scaleFactor // Maximum range in mm

	// Clear the image (black background)
	mat.SetTo(cv.NewScalar(0, 0, 0, 0))

	// Draw LiDAR points
	if reading.Valid && len(reading.DistancesMm) > 0 {
		for i := 0; i < len(reading.DistancesMm) && i < len(reading.AnglesDeg); i++ {
			dist := reading.DistancesMm[i]
			angle := reading.AnglesDeg[i]

			distFloat := float64(dist)
			if distFloat <= 0 || distFloat > maxRange {
				continue
			}

			// Convert polar to Cartesian
			// dist is in mm, scaleFactor is mm per pixel
			angleRad := float64(angle) * math.Pi / 180.0
			pixelsFromCenter := distFloat / scaleFactor
			x := centerX + int(float64(pixelsFromCenter)*math.Sin(angleRad))
			y := centerY - int(float64(pixelsFromCenter)*math.Cos(angleRad))

			// Draw point (white)
			cv.Circle(mat, image.Pt(x, y), 2, color.RGBA{255, 255, 255, 0}, -1)
		}

		// Draw center point
		cv.Circle(mat, image.Pt(centerX, centerY), 5, color.RGBA{0, 255, 0, 0}, -1)

		// Draw range circles at 1m, 2m, 3m, etc. intervals
		for r := 1000.0; r <= maxRange; r += 1000.0 {
			radius := int(r / scaleFactor)
			if radius > 0 && radius < centerX && radius < centerY {
				cv.Circle(mat, image.Pt(centerX, centerY), radius, color.RGBA{64, 64, 64, 0}, 1)
			}
		}
	}
}
