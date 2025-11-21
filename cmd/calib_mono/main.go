package main

import (
	"context"
	"flag"
	"fmt"
	"image"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"

	"github.com/itohio/EasyRobot/cmd/display/destination"
	"github.com/itohio/EasyRobot/cmd/display/source"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	cv "gocv.io/x/gocv"
)

func main() {
	// Register source and destination flags
	source.RegisterAllFlags()
	destination.RegisterAllFlags()

	// Calibration-specific flags
	gridStr := flag.String("grid", "9,7", "Calibration grid shape (width,height)")
	samples := flag.Int("samples", 30, "Number of calibration samples to collect")
	output := flag.String("output", "camera.json", "Output file path")
	format := flag.String("format", "json", "Output format: json, yaml, gocv")
	testMode := flag.Bool("test", false, "Test mode - load existing calibration file")
	help := flag.Bool("help", false, "Show help message")

	flag.Parse()

	if *help {
		flag.PrintDefaults()
		return
	}

	// Validate monocular calibration (exactly one camera if camera source is used)
	// Check source flags to ensure only one camera is specified
	cameraIDs := source.GetCameraIDs()
	if len(cameraIDs) > 1 {
		fmt.Fprintf(os.Stderr, "Error: monocular calibration requires exactly one camera\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	// Parse grid size
	gridWidth, gridHeight, err := parseGrid(*gridStr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing grid: %v\n", err)
		os.Exit(1)
	}

	// Create source from flags
	src, err := source.NewFromFlags()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		flag.PrintDefaults()
		os.Exit(1)
	}

	// Create destinations from flags
	dests := destination.NewAllDestinations()

	// Setup context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-sigChan
		cancel()
	}()

	// Start source
	if err := src.Start(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "Error starting source: %v\n", err)
		os.Exit(1)
	}
	defer src.Close()

	// Start destinations
	for _, dest := range dests {
		if err := dest.Start(ctx); err != nil {
			fmt.Fprintf(os.Stderr, "Error starting destination: %v\n", err)
			os.Exit(1)
		}
		defer dest.Close()
	}

	// Process frames
	if *testMode {
		if err := testCalibration(ctx, src, dests, *output); err != nil {
			fmt.Fprintf(os.Stderr, "Error testing calibration: %v\n", err)
			os.Exit(1)
		}
	} else {
		processor := NewCalibrationProcessor(image.Point{X: gridWidth, Y: gridHeight}, *samples)
		defer processor.Close()
		if err := calibrateCamera(ctx, src, processor, dests, *output, *format); err != nil {
			fmt.Fprintf(os.Stderr, "Error calibrating camera: %v\n", err)
			os.Exit(1)
		}
	}
}

func parseGrid(gridStr string) (width, height int, err error) {
	parts := strings.Split(gridStr, ",")
	if len(parts) != 2 {
		return 0, 0, fmt.Errorf("invalid grid format: %s (expected width,height)", gridStr)
	}
	width, err = strconv.Atoi(strings.TrimSpace(parts[0]))
	if err != nil {
		return 0, 0, fmt.Errorf("invalid width: %w", err)
	}
	height, err = strconv.Atoi(strings.TrimSpace(parts[1]))
	if err != nil {
		return 0, 0, fmt.Errorf("invalid height: %w", err)
	}
	return width, height, nil
}

func calibrateCamera(ctx context.Context, src source.Source, processor *CalibrationProcessor, dests []destination.Destination, outputPath, format string) error {
	fmt.Printf("Starting camera calibration...\n")
	fmt.Printf("Grid size: %dx%d\n", processor.gridSize.X, processor.gridSize.Y)
	fmt.Printf("Target samples: %d\n", processor.targetSamples)

	if err := processCalibrationLoop(ctx, src, processor, dests); err != nil {
		return err
	}

	calibration, err := processor.Calibrate()
	if err != nil {
		return fmt.Errorf("calibration failed: %w", err)
	}
	defer calibration.Close()

	if err := saveCalibration(calibration, outputPath, format); err != nil {
		return fmt.Errorf("failed to save calibration: %w", err)
	}

	fmt.Printf("Calibration saved to %s\n", outputPath)
	fmt.Printf("Reprojection error: %.4f\n", calibration.ReprojectionError)

	return nil
}

func testCalibration(ctx context.Context, src source.Source, dests []destination.Destination, calibPath string) error {
	// Load calibration file
	calibration, err := loadCalibration(calibPath)
	if err != nil {
		return fmt.Errorf("failed to load calibration: %w", err)
	}
	defer calibration.Close()

	fmt.Printf("Loaded calibration from %s\n", calibPath)
	fmt.Printf("Grid size: %dx%d\n", calibration.GridShape.X, calibration.GridShape.Y)
	fmt.Printf("Image size: %dx%d\n", calibration.ImageSize.X, calibration.ImageSize.Y)
	fmt.Printf("Samples used: %d\n", calibration.NumSamples)
	fmt.Printf("Reprojection error: %.4f\n", calibration.ReprojectionError)

	// Process frames and undistort
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Read frame from source
		frame, err := src.ReadFrame()
		if err != nil {
			if err == source.ErrSourceExhausted {
				return nil
			}
			fmt.Fprintf(os.Stderr, "Warning: failed to read frame: %v\n", err)
			continue
		}

		if len(frame.Tensors) == 0 {
			continue
		}

		// Convert tensor to Mat - note: original tensor will be released by destinations with WithRelease()
		mat, err := tensorToMat(frame.Tensors[0])
		if err != nil {
			// If tensorToMat fails, release original tensor if possible
			if releaser, ok := frame.Tensors[0].(types.Releaser); ok {
				releaser.Release()
			}
			continue
		}
		defer mat.Close()

		// Undistort
		undistorted := cv.NewMat()
		defer undistorted.Close()
		emptyMat := cv.NewMat()
		defer emptyMat.Close()
		if err := cv.Undistort(mat, &undistorted, calibration.CameraMatrix, calibration.DistortionCoeffs, emptyMat); err != nil {
			// Error in undistort - mat already closed via defer, original tensor released by destinations
			continue
		}

		// Convert back to tensor
		undistortedTensor, err := matToTensor(undistorted)
		if err != nil {
			// Error in matToTensor - mats already closed via defer, original tensor released by destinations
			continue
		}

		// Create display frame
		// undistortedTensor will be released by destinations that use WithRelease()
		displayFrame := types.Frame{
			Tensors:  []types.Tensor{undistortedTensor},
			Metadata: frame.Metadata,
		}

		// Send to all destinations - destinations with WithRelease() will release undistortedTensor after consumption
		for _, dest := range dests {
			if err := dest.AddFrame(displayFrame); err != nil {
				// If destination fails and doesn't use WithRelease(), we need to release
				// But since display destination now uses WithRelease(), this is handled
				// Note: If no destinations use WithRelease(), undistortedTensor leaks here
				_ = err // Ignore destination errors
			}
		}
		// Note: undistortedTensor release is handled by destinations with WithRelease()
		// Original frame.Tensors[0] release is handled by source if it uses WithRelease() on unmarshaller
		// (But unmarshallers should NOT use WithRelease() - they are producers)
	}
}

// tensorToMat converts a tensor to gocv.Mat using the destination package helper
func tensorToMat(tensor types.Tensor) (cv.Mat, error) {
	return destination.TensorToMat(tensor)
}

// matToTensor converts a gocv.Mat to a tensor
func matToTensor(mat cv.Mat) (types.Tensor, error) {
	return tensorgocv.FromMat(mat, tensorgocv.WithAdoptedMat())
}
