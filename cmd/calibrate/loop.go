package main

import (
	"context"
	"fmt"
	"os"

	"github.com/itohio/EasyRobot/cmd/display/destination"
	"github.com/itohio/EasyRobot/cmd/display/source"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
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
			// Release tensor if possible
			if releaser, ok := frame.Tensors[0].(types.Releaser); ok {
				releaser.Release()
			}
			continue
		}

		found, err := processor.ProcessFrame(mat)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: failed to process frame: %v\n", err)
			mat.Close()
			// Release tensor if possible
			if releaser, ok := frame.Tensors[0].(types.Releaser); ok {
				releaser.Release()
			}
			continue
		}

		if err := sendVisualization(ctx, mat, processor, found, frame.Metadata, dests); err != nil {
			mat.Close()
			// Release tensor if possible
			if releaser, ok := frame.Tensors[0].(types.Releaser); ok {
				releaser.Release()
			}
			return err
		}

		mat.Close()
		// Release tensor if possible
		if releaser, ok := frame.Tensors[0].(types.Releaser); ok {
			releaser.Release()
		}

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
	if len(dests) == 0 {
		return nil
	}

	visMat := mat.Clone()
	processor.DrawCorners(visMat, found)

	visTensor, err := matToTensor(visMat)
	if err != nil {
		visMat.Close()
		return fmt.Errorf("failed to convert mat to tensor: %w", err)
	}

	// Use SharedTensor for multiple destinations to avoid cloning overhead
	numDests := len(dests)
	if numDests > 1 {
		// Wrap tensor with reference counting
		sharedTensor := tensor.WithRefcount(visTensor)

		// Create frame with shared tensor for first destination
		sharedFrame := types.Frame{
			Tensors:  []types.Tensor{sharedTensor},
			Metadata: metadata,
		}

		// Create view frames for remaining destinations
		viewFrames := make([]types.Frame, numDests-1)
		for i := range numDests - 1 {
			viewTensors := []types.Tensor{sharedTensor.View()}
			viewFrames[i] = types.Frame{
				Tensors:  viewTensors,
				Metadata: metadata,
			}
		}

		// Send to first destination with shared tensor
		if err := dests[0].AddFrame(sharedFrame); err != nil {
			select {
			case <-ctx.Done():
				visMat.Close()
				return ctx.Err()
			default:
				// Continue processing
			}
		}

		// Send views to remaining destinations
		for i := 1; i < numDests; i++ {
			if err := dests[i].AddFrame(viewFrames[i-1]); err != nil {
				select {
				case <-ctx.Done():
					visMat.Close()
					return ctx.Err()
				default:
					// Continue processing
				}
			}
		}
	} else {
		// Single destination - send tensor as-is
		displayFrame := types.Frame{
			Tensors:  []types.Tensor{visTensor},
			Metadata: metadata,
		}

		if err := dests[0].AddFrame(displayFrame); err != nil {
			visMat.Close()
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
				// Continue processing
			}
		}
	}

	// Mat is adopted by tensor, so we don't close it here
	// Tensor lifecycle is managed by destinations with WithRelease() option

	return nil
}

