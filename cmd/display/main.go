package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/itohio/EasyRobot/cmd/display/destination"
	"github.com/itohio/EasyRobot/cmd/display/source"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/spf13/cobra"
)

func main() {
	rootCmd := &cobra.Command{
		Use:   "display",
		Short: "Display frames from various sources",
		Long:  "Display frames from cameras, videos, images, or generated test patterns. Supports multiple output destinations including display windows and video files.",
		RunE:  runDisplay,
	}

	// Add source flags
	source.AddSourceFlags(rootCmd)
	source.AddInterestFlags(rootCmd)

	// Add destination flags
	destination.AddDestinationFlags(rootCmd)
	destination.AddIntentFlags(rootCmd)

	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func runDisplay(cmd *cobra.Command, args []string) error {
	// Handle list-cameras flag - list cameras and exit (don't start streaming)
	if source.IsListCamerasFlagSet(cmd) {
		if err := source.ListCameras(); err != nil {
			return fmt.Errorf("error listing cameras: %w", err)
		}
		return nil // Exit after listing cameras
	}

	// Create source from cobra command
	slog.Info("Creating source from cobra command")
	src, err := source.NewFromCobra(cmd)
	if err != nil {
		return fmt.Errorf("error creating source: %w", err)
	}
	slog.Info("Source created successfully")

	// Create destinations from cobra command
	slog.Info("Creating destinations from cobra command")
	dests := destination.NewFromCobra(cmd)
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
		return fmt.Errorf("error starting source: %w", err)
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
			return fmt.Errorf("error starting destination: %w", err)
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
			return nil
		default:
		}

		// Read frame from source
		slog.Debug("Reading frame from source")
		frame, err := src.ReadFrame()
		if err != nil {
			if err == source.ErrSourceExhausted {
				slog.Info("Source exhausted", "frames_processed", frameCount)
				// Source exhausted, exit gracefully
				return nil
			}
			slog.Warn("Error reading frame", "err", err, "frames_processed", frameCount)
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
				} else {
					slog.Debug("Frame sent to destination successfully", "destination_index", 0, "frame_index", frame.Index)
				}
			}

			// Send views to remaining destinations
			for i := 1; i < numDests; i++ {
				slog.Debug("Sending frame to destination", "destination_index", i, "frame_index", frame.Index)
				if err := dests[i].AddFrame(viewFrames[i-1]); err != nil {
					slog.Warn("Error adding frame to destination", "destination_index", i, "frame_index", frame.Index, "err", err)
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
			return nil
		default:
		}
	}
}
