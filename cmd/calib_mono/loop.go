package main

import (
	"context"
	"fmt"
	"os"

	"github.com/itohio/EasyRobot/cmd/display/destination"
	"github.com/itohio/EasyRobot/cmd/display/source"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	cv "gocv.io/x/gocv"
)

// FrameProcessor processes frames for calibration.
type FrameProcessor interface {
	ProcessFrame(mat cv.Mat) (bool, error)
	DrawCorners(mat cv.Mat, found bool)
	NumSamples() int
	TargetSamples() int
}

// processCalibrationLoop runs the calibration loop.
func processCalibrationLoop(ctx context.Context, src source.Source, processor FrameProcessor, dests []destination.Destination) error {
	for {
		if err := checkContext(ctx); err != nil {
			return err
		}

		frame, cont := readFrameOrContinue(src)
		if !cont {
			break
		}
		if len(frame.Tensors) == 0 {
			continue
		}

		mat, err := tensorToMat(frame.Tensors[0])
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: failed to convert tensor: %v\n", err)
			// Note: Original tensor will be released by destinations with WithRelease()
			continue
		}
		defer mat.Close()

		found, err := processor.ProcessFrame(mat)
		if err != nil {
			// mat already closed via defer, original tensor released by destinations
			fmt.Fprintf(os.Stderr, "Warning: failed to process frame: %v\n", err)
			continue
		}

		if err := sendVisualization(ctx, mat, processor, found, frame.Metadata, dests); err != nil {
			// mat already closed via defer
			return err
		}
		// mat already closed via defer

		if processor.NumSamples() >= processor.TargetSamples() {
			fmt.Printf("\nCollected %d samples, computing calibration...\n", processor.NumSamples())
			break
		}

		if processor.NumSamples()%5 == 0 {
			fmt.Printf("Collected %d/%d samples\n", processor.NumSamples(), processor.TargetSamples())
		}
	}
	return nil
}

func checkContext(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
		return nil
	}
}

func readFrameOrContinue(src source.Source) (types.Frame, bool) {
	frame, err := src.ReadFrame()
	if err != nil {
		if err == source.ErrSourceExhausted {
			return types.Frame{}, false
		}
		fmt.Fprintf(os.Stderr, "Warning: failed to read frame: %v\n", err)
		return types.Frame{}, true // Continue
	}
	return frame, true
}

func sendVisualization(ctx context.Context, mat cv.Mat, processor FrameProcessor, found bool, metadata map[string]any, dests []destination.Destination) error {
	visMat := mat.Clone()
	defer visMat.Close()
	processor.DrawCorners(visMat, found)

	visTensor, err := matToTensor(visMat)
	if err != nil {
		// visMat already closed via defer
		return fmt.Errorf("failed to convert mat to tensor: %w", err)
	}

	displayFrame := types.Frame{
		Tensors:  []types.Tensor{visTensor},
		Metadata: metadata,
	}

	// Send to destinations - destinations with WithRelease() will release visTensor after consumption
	for _, dest := range dests {
		if err := dest.AddFrame(displayFrame); err != nil {
			select {
			case <-ctx.Done():
				// visMat already closed via defer, visTensor released by destinations
				return ctx.Err()
			default:
				// Continue processing
			}
		}
	}
	// visMat already closed via defer
	// visTensor release is handled by destinations with WithRelease()
	return nil
}

