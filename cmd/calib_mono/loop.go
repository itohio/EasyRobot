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
			continue
		}

		found, err := processor.ProcessFrame(mat)
		if err != nil {
			mat.Close()
			fmt.Fprintf(os.Stderr, "Warning: failed to process frame: %v\n", err)
			continue
		}

		if err := sendVisualization(ctx, mat, processor, found, frame.Metadata, dests); err != nil {
			mat.Close()
			return err
		}

		mat.Close()

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
	processor.DrawCorners(visMat, found)

	visTensor, err := matToTensor(visMat)
	if err != nil {
		visMat.Close()
		return fmt.Errorf("failed to convert mat to tensor: %w", err)
	}

	displayFrame := types.Frame{
		Tensors:  []types.Tensor{visTensor},
		Metadata: metadata,
	}

	for _, dest := range dests {
		if err := dest.AddFrame(displayFrame); err != nil {
			select {
			case <-ctx.Done():
				visMat.Close()
				return ctx.Err()
			default:
				// Continue processing
			}
		}
	}

	visMat.Close()
	return nil
}

