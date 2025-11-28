package main

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/tensor"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/spf13/cobra"
)

func newMixCmd() *cobra.Command {
	var (
		inputs   []string
		output   string
		offsets  []string
		dataType string
	)

	cmd := &cobra.Command{
		Use:   "mix",
		Short: "Mix multiple audio inputs into a single output",
		Long:  "Mix multiple audio files or devices, applying time offsets and summing the signals. All inputs must have the same number of channels and sample rate.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runMix(inputs, output, offsets, dataType)
		},
	}

	cmd.Flags().StringSliceVarP(&inputs, "input", "i", nil, "Input WAV file or device (can be specified multiple times)")
	cmd.Flags().StringVarP(&output, "output", "o", "", "Output WAV file path (required)")
	cmd.Flags().StringSliceVar(&offsets, "offset", nil, "Time offset in seconds (e.g., '1.5' or '0:1.5' for input 0). Can be specified multiple times")
	cmd.Flags().StringVar(&dataType, "data-type", "INT16", "Output data type (INT16, INT32, FP32, UINT8, default: INT16)")

	cmd.MarkFlagRequired("output")
	cmd.MarkFlagRequired("input")

	return cmd
}

func runMix(inputs []string, output string, offsetStrs []string, dataType string) error {
	if len(inputs) == 0 {
		return fmt.Errorf("at least one input is required")
	}

	// Parse data type
	dt, err := parseDataType(dataType)
	if err != nil {
		return err
	}

	// Parse offsets
	offsets, err := parseOffsets(offsetStrs, len(inputs))
	if err != nil {
		return fmt.Errorf("invalid offsets: %w", err)
	}

	ctx := context.Background()

	// Create unmarshallers for all inputs
	unmarshallers := make([]*audio.Unmarshaller, len(inputs))
	for i, input := range inputs {
		unmarshallers[i] = audio.NewUnmarshaller(audio.WithWAVPath(input))
	}

	// Validate all inputs have same channels and sample rate
	// Read first frame from each to get metadata
	sampleRate := 0
	channels := 0
	for i, unm := range unmarshallers {
		var stream types.FrameStream
		if err := unm.Unmarshal(nil, &stream); err != nil {
			return fmt.Errorf("failed to read input %d (%s): %w", i, inputs[i], err)
		}

		// Read first frame to get metadata
		frame, ok := <-stream.C
		if !ok {
			stream.Close()
			return fmt.Errorf("input %d (%s) has no frames", i, inputs[i])
		}

		if len(frame.Tensors) == 0 {
			stream.Close()
			return fmt.Errorf("input %d (%s) has no tensors in first frame", i, inputs[i])
		}

		// Get metadata from unmarshaller
		unmSampleRate := unm.SampleRate()
		unmChannels := unm.Channels()

		if unmSampleRate == 0 || unmChannels == 0 {
			// Try to infer from tensor shape
			t := frame.Tensors[0]
			shape := t.Shape()
			if len(shape) >= 2 {
				unmChannels = shape[1]
			}
			// Sample rate might not be available from tensor alone
			if unmSampleRate == 0 {
				unmSampleRate = 48000 // default assumption
			}
		}

		if i == 0 {
			sampleRate = unmSampleRate
			channels = unmChannels
		} else {
			if unmSampleRate != sampleRate {
				stream.Close()
				return fmt.Errorf("input %d (%s) has sample rate %d, expected %d", i, inputs[i], unmSampleRate, sampleRate)
			}
			if unmChannels != channels {
				stream.Close()
				return fmt.Errorf("input %d (%s) has %d channels, expected %d", i, inputs[i], unmChannels, channels)
			}
		}

		// Close stream - we'll recreate it in the mixing function
		stream.Close()
		if frame.Tensors[0] != nil {
			frame.Tensors[0].Release()
		}
	}

	// Create output marshaller
	marshaller := audio.NewMarshaller(
		audio.WithWAVPath(output),
		audio.WithSampleRate(sampleRate),
		audio.WithChannels(channels),
		types.WithDestinationType(dt),
	)

	// Perform mixing
	if err := mixAudio(unmarshallers, marshaller, offsets, sampleRate, channels, dt, ctx); err != nil {
		return fmt.Errorf("failed to mix audio: %w", err)
	}

	fmt.Printf("Mixed audio: %s\n", output)
	fmt.Printf("  Inputs: %d\n", len(inputs))
	fmt.Printf("  Sample rate: %d Hz\n", sampleRate)
	fmt.Printf("  Channels: %d\n", channels)
	fmt.Printf("  Data type: %s\n", dataType)

	return nil
}

// parseOffsets parses offset strings in two formats:
// - "X" (duration in seconds) - appends to existing offsets
// - "N:X" (N is input index, X is duration) - sets specific offset
func parseOffsets(offsetStrs []string, numInputs int) ([]time.Duration, error) {
	offsets := make([]time.Duration, numInputs)

	for _, offsetStr := range offsetStrs {
		offsetStr = strings.TrimSpace(offsetStr)
		if offsetStr == "" {
			continue
		}

		// Check if format is "N:X"
		parts := strings.SplitN(offsetStr, ":", 2)
		if len(parts) == 2 {
			// Format: "N:X"
			index, err := strconv.Atoi(strings.TrimSpace(parts[0]))
			if err != nil {
				return nil, fmt.Errorf("invalid offset index '%s': %w", parts[0], err)
			}
			if index < 0 || index >= numInputs {
				return nil, fmt.Errorf("offset index %d out of range [0, %d)", index, numInputs)
			}

			duration, err := time.ParseDuration(strings.TrimSpace(parts[1]) + "s")
			if err != nil {
				return nil, fmt.Errorf("invalid offset duration '%s': %w", parts[1], err)
			}

			offsets[index] = duration
		} else {
			// Format: "X" - append to existing offsets
			duration, err := time.ParseDuration(offsetStr + "s")
			if err != nil {
				return nil, fmt.Errorf("invalid offset duration '%s': %w", offsetStr, err)
			}

			// Find first zero offset and set it
			found := false
			for i := range offsets {
				if offsets[i] == 0 {
					offsets[i] = duration
					found = true
					break
				}
			}
			if !found {
				// All offsets are set, append to last one
				offsets = append(offsets, duration)
			}
		}
	}

	return offsets, nil
}

// inputSource tracks an input source and its state
type inputSource struct {
	unmarshaller   *audio.Unmarshaller
	stream         types.FrameStream
	offsetSamples  int
	currentSamples int
	depleted       bool
	rmsAccumulator *rmsAccumulator
}

// rmsAccumulator accumulates RMS statistics
type rmsAccumulator struct {
	sumSquares float64
	count      int64
}

func (r *rmsAccumulator) add(samples []float32) {
	for _, s := range samples {
		r.sumSquares += float64(s * s)
		r.count++
	}
}

func (r *rmsAccumulator) calculate() dsp.RMSStatistics {
	if r.count == 0 {
		return dsp.RMSStatistics{
			MinRMSDb: math.Inf(-1),
			MaxRMSDb: math.Inf(-1),
			AvgRMSDb: math.Inf(-1),
		}
	}
	avgRMS := math.Inf(-1)
	if r.sumSquares > 0 {
		avgRMS = 20.0 * math.Log10(math.Sqrt(r.sumSquares/float64(r.count)))
	}
	return dsp.RMSStatistics{
		MinRMSDb: avgRMS,
		MaxRMSDb: avgRMS,
		AvgRMSDb: avgRMS,
	}
}

// mixAudio is the centralized function that reads fixed chunks from unmarshallers,
// applies offsets, marks depleted sources, and writes to marshaller.
// All work is done without additional allocations by preallocating buffer tensors.
func mixAudio(
	unmarshallers []*audio.Unmarshaller,
	marshaller *audio.Marshaller,
	offsets []time.Duration,
	sampleRate int,
	channels int,
	dataType types.DataType,
	ctx context.Context,
) error {
	// Convert offsets from time.Duration to samples
	offsetSamples := make([]int, len(offsets))
	for i, offset := range offsets {
		offsetSamples[i] = int(float64(offset.Seconds()) * float64(sampleRate))
	}

	// Initialize input sources
	sources := make([]*inputSource, len(unmarshallers))
	for i, unm := range unmarshallers {
		var stream types.FrameStream
		if err := unm.Unmarshal(nil, &stream); err != nil {
			return fmt.Errorf("failed to create stream for input %d: %w", i, err)
		}

		sources[i] = &inputSource{
			unmarshaller:   unm,
			stream:         stream,
			offsetSamples:  offsetSamples[i],
			rmsAccumulator: &rmsAccumulator{},
		}
	}
	defer func() {
		for _, src := range sources {
			if src.stream.C != nil {
				src.stream.Close()
			}
		}
	}()

	// Pre-allocate buffers (fixedBufferSize samples per channel)
	const fixedBufferSize = 8192
	totalBufferSize := fixedBufferSize * channels

	// Buffer for reading from each source (float32)
	sourceBuffers := make([][]float32, len(sources))
	for i := range sourceBuffers {
		sourceBuffers[i] = make([]float32, totalBufferSize)
	}

	// Buffer for mixed output (float32)
	mixedBuffer := make([]float32, totalBufferSize)

	// Buffer for output RMS statistics
	outputRMSAccumulator := &rmsAccumulator{}

	// Track frame buffers for each source
	frameBuffers := make([][]float32, len(sources))
	for i := range frameBuffers {
		frameBuffers[i] = make([]float32, 0, totalBufferSize)
	}

	// Main mixing loop
	outputFrameIndex := 0
	anyActive := true

	for anyActive {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		// Clear mixed buffer
		for i := range mixedBuffer {
			mixedBuffer[i] = 0.0
		}

		anyActive = false
		samplesToMix := fixedBufferSize
		hasActiveSources := false

		// Read from each source and apply offset
		for i, src := range sources {
			if src.depleted {
				// Source is depleted, it won't contribute (already padded with zeros in mixedBuffer)
				continue
			}

			hasActiveSources = true

			// Handle offset: pad with zeros if offset is positive
			// When currentSamples < offsetSamples, source hasn't started yet
			if src.currentSamples < src.offsetSamples {
				// Source hasn't started yet due to offset - it contributes zeros (already in mixedBuffer)
				// Advance currentSamples to track progress
				skipSamples := src.offsetSamples - src.currentSamples
				if skipSamples > fixedBufferSize {
					skipSamples = fixedBufferSize
				}
				src.currentSamples += skipSamples
				// Source contributes zeros (mixedBuffer already initialized to zeros)
				// Mark as active so we continue writing output
				anyActive = true
				continue
			}

			// Read data from source
			readSamples := 0

			// Check if we have buffered data
			if len(frameBuffers[i]) > 0 {
				copySize := fixedBufferSize
				if copySize > len(frameBuffers[i]) {
					copySize = len(frameBuffers[i])
				}
				copy(sourceBuffers[i][:copySize*channels], frameBuffers[i][:copySize*channels])
				readSamples = copySize
				frameBuffers[i] = frameBuffers[i][copySize*channels:]
			}

			// Read more frames if needed
			for readSamples < fixedBufferSize {
				select {
				case frame, ok := <-src.stream.C:
					if !ok {
						// Source depleted
						src.depleted = true
						break
					}

					if len(frame.Tensors) == 0 {
						if frame.Tensors != nil {
							for _, t := range frame.Tensors {
								if t != nil {
									t.Release()
								}
							}
						}
						continue
					}

					// Extract float32 data from tensor
					t := frame.Tensors[0]
					tensorData := t.Data()
					convertedData := tensortypes.CloneTensorDataTo(tensortypes.FP32, tensorData)
					if convertedData == nil {
						if t != nil {
							t.Release()
						}
						continue
					}
					floatData, ok := convertedData.([]float32)
					if !ok {
						if t != nil {
							t.Release()
						}
						continue
					}

					// Accumulate RMS statistics for this input
					src.rmsAccumulator.add(floatData)

					// Append to frame buffer
					remaining := fixedBufferSize - readSamples
					if remaining*channels <= len(floatData) {
						// Use what we need
						copy(sourceBuffers[i][readSamples*channels:(readSamples+remaining)*channels], floatData[:remaining*channels])
						readSamples += remaining
						// Store remainder in frame buffer
						frameBuffers[i] = append(frameBuffers[i], floatData[remaining*channels:]...)
					} else {
						// Use all data
						copySize := len(floatData) / channels
						if readSamples+copySize > fixedBufferSize {
							copySize = fixedBufferSize - readSamples
						}
						copy(sourceBuffers[i][readSamples*channels:(readSamples+copySize)*channels], floatData[:copySize*channels])
						readSamples += copySize
						if copySize*channels < len(floatData) {
							frameBuffers[i] = append(frameBuffers[i], floatData[copySize*channels:]...)
						}
					}

					if t != nil {
						t.Release()
					}
				default:
					// No more data available right now
					break
				}
			}

			if readSamples > 0 {
				anyActive = true
				src.currentSamples += readSamples

				// Mix into output buffer
				for j := 0; j < readSamples; j++ {
					for ch := 0; ch < channels; ch++ {
						idx := j*channels + ch
						mixedBuffer[idx] += sourceBuffers[i][idx]
					}
				}

				if readSamples < samplesToMix {
					samplesToMix = readSamples
				}
			} else if !src.depleted {
				anyActive = true
			}
		}

		// Continue if we have active sources (not all depleted) or if any source contributed data
		if !hasActiveSources && !anyActive {
			break
		}

		// Clamp mixed buffer to [-1, 1]
		for i := range mixedBuffer[:samplesToMix*channels] {
			if mixedBuffer[i] > 1.0 {
				mixedBuffer[i] = 1.0
			} else if mixedBuffer[i] < -1.0 {
				mixedBuffer[i] = -1.0
			}
		}

		// Accumulate output RMS statistics
		outputRMSAccumulator.add(mixedBuffer[:samplesToMix*channels])

		// Create tensor from mixed buffer
		shape := tensor.NewShape(samplesToMix, channels)
		t := tensor.FromFloat32(shape, mixedBuffer[:samplesToMix*channels])
		if t.Empty() {
			return fmt.Errorf("failed to create output tensor")
		}

		// Write to marshaller
		if err := marshaller.Marshal(nil, t, types.WithContext(ctx)); err != nil {
			return fmt.Errorf("failed to write to marshaller: %w", err)
		}

		outputFrameIndex++
	}

	// Print RMS statistics
	fmt.Println("\nRMS Statistics:")
	for i, src := range sources {
		rmsStats := src.rmsAccumulator.calculate()
		fmt.Printf("  Input %d: AVG RMS: %.2f dB\n", i, rmsStats.AvgRMSDb)
	}
	outputRMSStats := outputRMSAccumulator.calculate()
	fmt.Printf("  Output: AVG RMS: %.2f dB\n", outputRMSStats.AvgRMSDb)

	return nil
}
