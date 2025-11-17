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
	mat "github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"

	"image"
	"image/color"
	"math"

	"github.com/itohio/EasyRobot/x/marshaller/types"
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

	cfg := &Config{
		SerialPort:   *serialPort,
		LidarType:    *lidarType,
		TargetPts:    *targetPts,
		Display:      *display,
		WindowWidth:  *windowWidth,
		WindowHeight: *windowHeight,
		ImageWidth:   *imageWidth,
		ImageHeight:  *imageHeight,
		ScaleFactor:  *scaleFactor,
	}

	if err := cfg.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "Configuration error: %v\n", err)
		flag.Usage()
		os.Exit(1)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := run(ctx, cfg); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, cfg *Config) error {
	router, err := setupRouter(ctx)
	if err != nil {
		return fmt.Errorf("setup router: %w", err)
	}
	defer router.Close()

	producer, err := setupProducer(ctx, router)
	if err != nil {
		return fmt.Errorf("setup producer: %w", err)
	}
	defer producer.Close()

	ser, err := devio.NewSerial(cfg.SerialPort)
	if err != nil {
		return fmt.Errorf("open serial port: %w", err)
	}

	factory := NewDeviceFactory()
	lidar, err := factory.CreateDevice(ctx, cfg, ser)
	if err != nil {
		return fmt.Errorf("create device: %w", err)
	}
	defer lidar.Close()

	displaySetup, err := SetupDisplay(ctx, cfg)
	if err != nil {
		return fmt.Errorf("setup display: %w", err)
	}
	defer displaySetup.Close()

	setupCallback(ctx, lidar, producer, displaySetup)

	<-ctx.Done()
	return nil
}

func setupProducer(ctx context.Context, router *dndm.Router) (*bus.Producer[*pbdevices.LIDARReading], error) {
	intentPath := "LIDARReading@lidar.scan"
	producer, err := bus.NewProducer[*pbdevices.LIDARReading](ctx, router, intentPath)
	if err != nil {
		return nil, err
	}
	fmt.Printf("LiDAR driver initialized with intent: %s\n", intentPath)
	return producer, nil
}

func setupCallback(ctx context.Context, lidar lidarDevice, producer *bus.Producer[*pbdevices.LIDARReading], display *DisplaySetup) {
	lidar.OnRead(func(m matTypes.Matrix) {
		reading := matrixToLIDARReading(m)
		if err := producer.Send(ctx, reading); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to send reading: %v\n", err)
		}

		if display != nil && display.FrameChan != nil {
			drawLIDARReading(display.Mat, reading, display.ImageWidth, display.ImageHeight, display.ScaleFactor)
			select {
			case display.FrameChan <- types.Frame{
				Tensors:   []types.Tensor{display.Tensor},
				Index:     0,
				Timestamp: time.Now().UnixNano(),
			}:
			default:
				// Channel full, skip this frame
			}
		}
	})
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
