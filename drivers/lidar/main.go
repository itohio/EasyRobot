//go:build !rp2040 && cgo

package main

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/itohio/dndm"
	"github.com/itohio/dndm/endpoint/direct"

	"github.com/itohio/EasyRobot/cmd/display/destination"
	pbdevices "github.com/itohio/EasyRobot/types/devices"
	devio "github.com/itohio/EasyRobot/x/devices"
	"github.com/itohio/EasyRobot/x/devices/lidar/ld06"
	"github.com/itohio/EasyRobot/x/devices/lidar/xwpftb"
	mat "github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"

	"image"
	"image/color"
	"math"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	cv "gocv.io/x/gocv"
)

var (
	serialPort = flag.String("serial", "", "Serial port device (e.g., /dev/ttyUSB0 or COM3)")
	lidarType  = flag.String("type", "xwpftb", "LiDAR type: xwpftb or ld06")
	targetPts  = flag.Int("target-points", 0, "Target points per rotation (0 = auto-calibrate)")
	baudRate   = flag.Int("baud", 0, "Serial port baud rate (0 = auto-detect based on LiDAR type: xwpftb=115200, ld06=230400)")

	// Display options (reusing cmd/display/destination flags)
	imageWidth  = flag.Int("image-width", 2400, "Image width in pixels (default allows 12m range at 10mm/pixel)")
	imageHeight = flag.Int("image-height", 2400, "Image height in pixels (default allows 12m range at 10mm/pixel)")
	scaleFactor = flag.Float64("scale", 10.0, "Scaling factor in mm per pixel (default 10 = 1cm per pixel)")

	// DNDM routing options
	disableRouting = flag.Bool("disable-routing", false, "Disable DNDM routing (no message bus)")
	queueSize      = flag.Int("queue-size", 10, "DNDM router queue size")
)

func main() {
	// Register destination flags (display and intent)
	destination.RegisterAllFlags()
	destination.RegisterIntentFlags()

	flag.Parse()

	if *serialPort == "" {
		slog.Error("Serial port required", "flag", "--serial")
		flag.Usage()
		os.Exit(1)
	}

	slog.Info("LiDAR driver starting",
		"serial", *serialPort,
		"lidar_type", *lidarType,
		"target_points", *targetPts,
		"disable_routing", *disableRouting,
	)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	slog.Info("Context initialized with signal handling")

	// Setup DNDM router (optional)
	var router *dndm.Router
	var intentDest destination.Destination
	if !*disableRouting {
		slog.Info("Setting up DNDM router", "queue_size", *queueSize)
		var err error
		router, err = setupRouter(ctx, *queueSize)
		if err != nil {
			slog.Error("Failed to setup router", "err", err)
			os.Exit(1)
		}
		defer func() {
			slog.Info("Closing DNDM router")
			router.Close()
		}()

		slog.Info("DNDM router created successfully")

		// Create intent destination from flags
		intentDest = destination.NewIntentFromFlags(router)
		if intentDest != nil {
			slog.Info("Starting intent destination")
			if err := intentDest.Start(ctx); err != nil {
				slog.Error("Failed to start intent destination", "err", err)
				os.Exit(1)
			}
			defer func() {
				slog.Info("Closing intent destination")
				intentDest.Close()
			}()
			slog.Info("LiDAR driver initialized with DNDM intent publishing")
		} else {
			slog.Info("LiDAR driver initialized (no intent routes specified via --intent flags)")
		}
	} else {
		slog.Info("LiDAR driver initialized (routing disabled via --disable-routing)")
	}

	// Determine baud rate (auto-detect based on LiDAR type if not specified)
	actualBaudRate := *baudRate
	if actualBaudRate == 0 {
		switch *lidarType {
		case "xwpftb":
			actualBaudRate = 250000
			slog.Info("Auto-detected baud rate for XWPFTB", "baud", actualBaudRate)
		case "ld06":
			actualBaudRate = 230400
			slog.Info("Auto-detected baud rate for LD06", "baud", actualBaudRate)
		default:
			actualBaudRate = 115200 // Default fallback
			slog.Warn("Unknown LiDAR type, using default baud rate", "baud", actualBaudRate, "lidar_type", *lidarType)
		}
	} else {
		slog.Info("Using specified baud rate", "baud", actualBaudRate)
	}

	// Setup serial with baud rate
	slog.Info("Opening serial port", "port", *serialPort, "baud", actualBaudRate)
	config := devio.DefaultSerialConfig()
	config.BaudRate = actualBaudRate
	ser, err := devio.NewSerialWithConfig(*serialPort, config)
	if err != nil {
		slog.Error("Failed to open serial port", "port", *serialPort, "baud", actualBaudRate, "err", err)
		os.Exit(1)
	}
	defer func() {
		slog.Info("Closing serial port", "port", *serialPort)
		ser.Close()
	}()
	slog.Info("Serial port opened successfully", "port", *serialPort, "baud", actualBaudRate)

	// Create LiDAR device (no motor control on Linux/Windows)
	slog.Info("Creating LiDAR device", "type", *lidarType, "target_points", *targetPts)
	var lidar lidarDevice
	switch *lidarType {
	case "xwpftb":
		slog.Info("Initializing XWPFTB LiDAR", "target_points", *targetPts, "max_points", 3600)
		dev := xwpftb.New(ctx, ser, nil, *targetPts, 3600)
		slog.Info("Configuring XWPFTB LiDAR")
		if err := dev.Configure(true); err != nil {
			slog.Error("Failed to configure XWPFTB", "err", err)
			os.Exit(1)
		}
		lidar = dev
		slog.Info("XWPFTB LiDAR configured successfully")
	case "ld06":
		slog.Info("Initializing LD06 LiDAR", "target_points", *targetPts, "max_points", 3600)
		dev := ld06.New(ctx, ser, nil, *targetPts, 3600)
		slog.Info("Configuring LD06 LiDAR")
		if err := dev.Configure(true); err != nil {
			slog.Error("Failed to configure LD06", "err", err)
			os.Exit(1)
		}
		lidar = dev
		slog.Info("LD06 LiDAR configured successfully")
	default:
		slog.Error("Unknown LiDAR type", "type", *lidarType, "supported", []string{"xwpftb", "ld06"})
		os.Exit(1)
	}
	defer func() {
		slog.Info("Closing LiDAR device")
		lidar.Close()
	}()

	// Validate display parameters
	if *imageWidth <= 0 || *imageHeight <= 0 {
		slog.Error("Image dimensions must be positive", "width", *imageWidth, "height", *imageHeight)
		os.Exit(1)
	}
	if *scaleFactor <= 0 {
		slog.Error("Scale factor must be positive", "scale", *scaleFactor)
		os.Exit(1)
	}

	slog.Info("Display configuration",
		"image_width", *imageWidth,
		"image_height", *imageHeight,
		"scale_factor", *scaleFactor,
	)

	// Setup display using gocv marshaller pattern
	displayCfg := &Config{
		Display:      true,
		ImageWidth:   *imageWidth,
		ImageHeight:  *imageHeight,
		ScaleFactor:  *scaleFactor,
		WindowWidth:  0, // Use image size
		WindowHeight: 0,
	}
	displaySetup, err := SetupDisplay(ctx, displayCfg, cancel)
	if err != nil {
		slog.Error("Failed to setup display", "err", err)
		os.Exit(1)
	}
	if displaySetup == nil {
		slog.Error("Display setup returned nil")
		os.Exit(1)
	}
	defer func() {
		slog.Info("Closing display setup")
		displaySetup.Close()
	}()
	slog.Info("Display setup completed successfully")

	// Register callback to publish readings and display
	readingCount := 0
	lastLogTime := time.Now()
	readingsSinceLastLog := 0
	lidar.OnRead(func(m matTypes.Matrix) {
		readingCount++
		readingsSinceLastLog++
		now := time.Now()

		// Log reading statistics every 5 seconds
		if now.Sub(lastLogTime) >= 5*time.Second {
			elapsed := now.Sub(lastLogTime).Seconds()
			rate := float64(readingsSinceLastLog) / elapsed
			slog.Info("LiDAR reading statistics",
				"total_readings", readingCount,
				"points", m.Cols(),
				"readings_in_period", readingsSinceLastLog,
				"rate_per_sec", rate,
			)
			lastLogTime = now
			readingsSinceLastLog = 0
		}

		reading := matrixToLIDARReading(m)
		if !reading.Valid {
			slog.Warn("Received invalid LiDAR reading", "points", len(reading.DistancesMm))
			return
		}

		// Publish to intent destination if available
		if intentDest != nil {
			// Use type assertion to access SetLIDARReading method
			// This is safe because we know intentDest is created from NewIntentFromFlags
			if intentDestImpl, ok := intentDest.(*destination.IntentDestination); ok {
				if err := intentDestImpl.SetLIDARReading(reading); err != nil {
					slog.Error("Failed to send reading to intent destination", "err", err, "points", len(reading.DistancesMm))
				}
			}
		}

		// Draw visualization and send to display stream
		drawLIDARReading(displaySetup.Mat, reading, displaySetup.ImageWidth, displaySetup.ImageHeight, displaySetup.ScaleFactor)
		frame := types.Frame{
			Tensors:   []types.Tensor{displaySetup.Tensor},
			Index:     readingCount,
			Timestamp: time.Now().UnixNano(),
		}
		select {
		case displaySetup.FrameChan <- frame:
			// Frame sent successfully
		case <-ctx.Done():
			// Context cancelled, stop processing
			return
		default:
			// Channel full, skip this frame to avoid blocking
			slog.Warn("Display frame channel full, skipping frame", "frame_index", readingCount)
		}
	})

	slog.Info("LiDAR driver fully initialized and ready",
		"lidar_type", *lidarType,
		"serial_port", *serialPort,
		"baud_rate", actualBaudRate,
		"routing_enabled", !*disableRouting,
		"display_enabled", true,
	)

	// Keep running
	slog.Info("Entering main loop, waiting for LiDAR readings...")
	<-ctx.Done()
	slog.Info("Context cancelled, shutting down", "reason", ctx.Err())
}

type lidarDevice interface {
	Configure(init bool) error
	OnRead(fn func(matTypes.Matrix))
	Close()
}

func setupRouter(ctx context.Context, queueSize int) (*dndm.Router, error) {
	slog.Debug("Creating direct endpoint", "queue_size", queueSize)
	directEP := direct.New(queueSize)
	slog.Debug("Creating DNDM router", "queue_size", queueSize)
	router, err := dndm.New(
		dndm.WithContext(ctx),
		dndm.WithQueueSize(queueSize),
		dndm.WithEndpoint(directEP),
	)
	if err != nil {
		slog.Error("Failed to create DNDM router", "err", err, "queue_size", queueSize)
		return nil, err
	}
	slog.Debug("DNDM router created successfully", "queue_size", queueSize)
	return router, nil
}

func matrixToLIDARReading(m matTypes.Matrix) *pbdevices.LIDARReading {
	if m == nil || m.Cols() == 0 {
		slog.Debug("Empty matrix received, returning invalid reading")
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

	reading := &pbdevices.LIDARReading{
		DistancesMm: distances,
		AnglesDeg:   angles,
		Timestamp:   time.Now().UnixNano(),
		Valid:       true,
	}

	slog.Debug("Converted matrix to LIDAR reading", "points", len(distances))
	return reading
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
