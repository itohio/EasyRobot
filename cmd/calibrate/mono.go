package main

import (
	"context"
	"fmt"
	"image"
	"os"
	"os/signal"
	"syscall"

	"github.com/itohio/EasyRobot/cmd/display/destination"
	"github.com/itohio/EasyRobot/cmd/display/source"
	"github.com/spf13/cobra"
)

func newCamMonoCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "cam-mono",
		Short: "Monocular camera calibration",
		Long:  "Calibrate a single camera using a checkerboard pattern. Collects calibration samples and saves camera parameters.",
		RunE:  runCamMono,
	}

	// Add source and destination flags
	source.AddSourceFlags(cmd)
	destination.AddDestinationFlags(cmd)

	// Calibration-specific flags
	cmd.Flags().String("grid", "9,7", "Calibration grid shape (width,height)")
	cmd.Flags().Int("samples", 30, "Number of calibration samples to collect")
	cmd.Flags().String("output", "camera.json", "Output file path")
	cmd.Flags().String("format", "json", "Output format: json, yaml, gocv")
	cmd.Flags().Bool("test", false, "Test mode - load existing calibration file")

	return cmd
}

func runCamMono(cmd *cobra.Command, args []string) error {
	// Validate monocular calibration (exactly one camera if camera source is used)
	cameraIDs := source.GetCameraIDs(cmd)
	if len(cameraIDs) > 1 {
		return fmt.Errorf("monocular calibration requires exactly one camera")
	}

	// Read calibration-specific flags
	gridStr, _ := cmd.Flags().GetString("grid")
	samples, _ := cmd.Flags().GetInt("samples")
	output, _ := cmd.Flags().GetString("output")
	format, _ := cmd.Flags().GetString("format")
	testMode, _ := cmd.Flags().GetBool("test")

	// Parse grid size
	gridWidth, gridHeight, err := parseGrid(gridStr)
	if err != nil {
		return fmt.Errorf("error parsing grid: %w", err)
	}

	// Create source from cobra command
	src, err := source.NewFromCobra(cmd)
	if err != nil {
		return fmt.Errorf("error creating source: %w", err)
	}

	// Create destinations from cobra command
	dests := destination.NewFromCobra(cmd)

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
		return fmt.Errorf("error starting source: %w", err)
	}
	defer src.Close()

	// Start destinations
	for _, dest := range dests {
		if err := dest.Start(ctx); err != nil {
			return fmt.Errorf("error starting destination: %w", err)
		}
		defer dest.Close()
	}

	// Process frames
	if testMode {
		if err := testMonoCalibration(ctx, src, dests, output); err != nil {
			return fmt.Errorf("error testing calibration: %w", err)
		}
	} else {
		processor := NewMonoCalibrationProcessor(image.Point{X: gridWidth, Y: gridHeight}, samples)
		defer processor.Close()
		if err := calibrateMonoCamera(ctx, src, processor, dests, output, format); err != nil {
			return fmt.Errorf("error calibrating camera: %w", err)
		}
	}

	return nil
}

