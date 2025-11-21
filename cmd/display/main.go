package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/itohio/EasyRobot/cmd/display/destination"
	"github.com/itohio/EasyRobot/cmd/display/source"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
)

func main() {
	// Register flags
	source.RegisterAllFlags()
	destination.RegisterAllFlags()
	// Register DNDM flags if needed (currently optional)
	source.RegisterInterestFlags()
	destination.RegisterIntentFlags()
	help := flag.Bool("help", false, "Show help message")

	// Parse flags
	flag.Parse()

	if *help {
		flag.PrintDefaults()
		return
	}

	// Handle list-cameras flag - list cameras and exit (don't start streaming)
	if source.IsListCamerasFlagSet() {
		if err := source.ListCameras(); err != nil {
			fmt.Fprintf(os.Stderr, "Error listing cameras: %v\n", err)
			os.Exit(1)
		}
		return // Exit after listing cameras
	}

	// Create source from flags
	slog.Info("Creating source from flags")
	src, err := source.NewFromFlags()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		flag.PrintDefaults()
		os.Exit(1)
	}
	slog.Info("Source created successfully")

	// Create destinations from flags
	slog.Info("Creating destinations from flags")
	dests := destination.NewAllDestinations()
	slog.Info("Destinations created", "count", len(dests))

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
	slog.Info("Starting source")
	if err := src.Start(ctx); err != nil {
		slog.Error("Error starting source", "err", err)
		fmt.Fprintf(os.Stderr, "Error starting source: %v\n", err)
		os.Exit(1)
	}
	slog.Info("Source started successfully")
	defer func() {
		slog.Info("Closing source")
		src.Close()
	}()

	// Start destinations
	slog.Info("Starting destinations", "count", len(dests))
	for i, dest := range dests {
		slog.Info("Starting destination", "index", i)
		// If this is a display destination, set the cancel function so window close cancels parent context
		if cancelSetter, ok := dest.(destination.CancelSetter); ok {
			cancelSetter.SetCancelFunc(cancel)
		}
		// Each destination gets the same parent context - when main context is cancelled,
		// all destinations will be notified. The display destination can cancel the parent
		// context when the main window is closed.
		if err := dest.Start(ctx); err != nil {
			slog.Error("Error starting destination", "index", i, "err", err)
			fmt.Fprintf(os.Stderr, "Error starting destination: %v\n", err)
			os.Exit(1)
		}
		slog.Info("Destination started", "index", i)
		defer func(idx int, d destination.Destination) {
			slog.Info("Closing destination", "index", idx)
			d.Close()
		}(i, dest)
	}
	slog.Info("All destinations started successfully")

	// Process frames
	frameCount := 0
	slog.Info("Starting frame processing loop")
	for {
		// Check context cancellation
		select {
		case <-ctx.Done():
			slog.Info("Frame processing stopped", "reason", "context cancelled", "frames_processed", frameCount)
			return
		default:
		}

		// Read frame from source
		slog.Debug("Reading frame from source")
		frame, err := src.ReadFrame()
		if err != nil {
			if err == source.ErrSourceExhausted {
				slog.Info("Source exhausted", "frames_processed", frameCount)
				// Source exhausted, exit gracefully
				return
			}
			slog.Warn("Error reading frame", "err", err, "frames_processed", frameCount)
			fmt.Fprintf(os.Stderr, "Error reading frame: %v\n", err)
			// Continue processing if error is recoverable
			continue
		}

		frameCount++
		slog.Debug("Frame read successfully",
			"frame_index", frame.Index,
			"frame_count", frameCount,
			"timestamp", frame.Timestamp,
			"tensors", len(frame.Tensors),
		)

		// Use smart tensors with reference counting for fan-out to multiple destinations
		// Wrap each tensor with reference counting and create views for additional destinations
		numDests := len(dests)
		if numDests > 1 && len(frame.Tensors) > 0 {
			// Wrap all tensors with reference counting
			smartTensors := make([]*tensor.SmartTensor, len(frame.Tensors))
			for i, t := range frame.Tensors {
				smartTensors[i] = tensor.WithRefcount(t)
			}

			// Create views for each destination (N-1 views needed)
			viewFrames := make([]types.Frame, numDests-1)
			for i := 0; i < numDests-1; i++ {
				viewTensors := make([]types.Tensor, len(smartTensors))
				for j, st := range smartTensors {
					viewTensors[j] = st.View()
				}
				viewFrames[i] = types.Frame{
					Index:     frame.Index,
					Timestamp: frame.Timestamp,
					Metadata:  frame.Metadata,
					Tensors:   viewTensors,
				}
			}

			// Create frame with smart tensors for first destination
			smartFrame := types.Frame{
				Index:     frame.Index,
				Timestamp: frame.Timestamp,
				Metadata:  frame.Metadata,
				Tensors:   make([]types.Tensor, len(smartTensors)),
			}
			for i, st := range smartTensors {
				smartFrame.Tensors[i] = st
			}

			// Send to first destination with smart tensors
			if len(dests) > 0 {
				slog.Debug("Sending frame to destination", "destination_index", 0, "frame_index", frame.Index)
				if err := dests[0].AddFrame(smartFrame); err != nil {
					slog.Warn("Error adding frame to destination", "destination_index", 0, "frame_index", frame.Index, "err", err)
					fmt.Fprintf(os.Stderr, "Error adding frame to destination: %v\n", err)
				} else {
					slog.Debug("Frame sent to destination successfully", "destination_index", 0, "frame_index", frame.Index)
				}
			}

			// Send views to remaining destinations
			for i := 1; i < numDests; i++ {
				slog.Debug("Sending frame to destination", "destination_index", i, "frame_index", frame.Index)
				if err := dests[i].AddFrame(viewFrames[i-1]); err != nil {
					slog.Warn("Error adding frame to destination", "destination_index", i, "frame_index", frame.Index, "err", err)
					fmt.Fprintf(os.Stderr, "Error adding frame to destination: %v\n", err)
				} else {
					slog.Debug("Frame sent to destination successfully", "destination_index", i, "frame_index", frame.Index)
				}
			}
		} else {
			// Single destination or no tensors - send frame as-is
			for i, dest := range dests {
				slog.Debug("Sending frame to destination", "destination_index", i, "frame_index", frame.Index)
				if err := dest.AddFrame(frame); err != nil {
					slog.Warn("Error adding frame to destination", "destination_index", i, "frame_index", frame.Index, "err", err)
					fmt.Fprintf(os.Stderr, "Error adding frame to destination: %v\n", err)
				} else {
					slog.Debug("Frame sent to destination successfully", "destination_index", i, "frame_index", frame.Index)
				}
			}

			// Note: Tensor release is now handled by destinations that use WithRelease() option
			// For single destination with WithRelease(), the destination will release after consumption
			// For multiple destinations, smart tensors handle refcounting automatically
			// No manual Release() needed here
		}

		if frameCount%100 == 0 {
			slog.Info("Frame processing progress", "frames_processed", frameCount)
		}

		// Check if display was closed (ESC key pressed)
		select {
		case <-ctx.Done():
			slog.Info("Frame processing stopped", "reason", "context cancelled", "frames_processed", frameCount)
			return
		default:
		}
	}
}
