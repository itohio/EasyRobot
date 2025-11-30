package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
	mathtensor "github.com/itohio/EasyRobot/x/math/tensor"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
)

func newRecvChirpCmd() *cobra.Command {
	var (
		input          string
		deviceID       string
		output         string
		sampleRate     int
		channels       int
		refFrequency   float64
		refDuration    time.Duration
		delay          time.Duration
		chirpDuration  time.Duration
		chirpFreqStart float64
		chirpFreqEnd   float64
		chirpType      string
		padding        time.Duration
		channel        int
		refAmplitude   float64
		chirpAmplitude float64
	)

	cmd := &cobra.Command{
		Use:   "recv-chirp",
		Short: "Receive and process chirp signal with reference detection",
		Long:  "Record or read audio, detect reference signal using Goertzel, and perform chirp correlation.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runRecvChirp(input, deviceID, output, sampleRate, channels, refFrequency, refDuration, delay,
				chirpDuration, chirpFreqStart, chirpFreqEnd, chirpType, padding, channel,
				refAmplitude, chirpAmplitude)
		},
	}

	cmd.Flags().StringVarP(&input, "input", "i", "", "Input WAV file (alternative to device capture)")
	cmd.Flags().StringVar(&deviceID, "device-id", "", "Capture device ID (optional, uses default if not specified)")
	cmd.Flags().StringVarP(&output, "output", "o", "", "Output WAV file path (required)")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 48000, "Sample rate (Hz)")
	cmd.Flags().IntVarP(&channels, "channels", "c", 1, "Number of channels")
	cmd.Flags().Float64Var(&refFrequency, "ref-frequency", 4400, "Reference signal frequency in Hz (default: 4400)")
	cmd.Flags().DurationVar(&refDuration, "ref-duration", 250*time.Millisecond, "Reference signal duration (e.g., 100ms)")
	cmd.Flags().DurationVar(&delay, "delay", time.Second, "Delay between ref signal end and chirp signal start (e.g., 50ms)")
	cmd.Flags().DurationVar(&chirpDuration, "chirp-duration", 5*time.Second, "Chirp signal duration (e.g., 5s)")
	cmd.Flags().Float64Var(&chirpFreqStart, "chirp-freq-start", 20, "Chirp start frequency in Hz (default: 20)")
	cmd.Flags().Float64Var(&chirpFreqEnd, "chirp-freq-end", 22000, "Chirp end frequency in Hz (default: 22000)")
	cmd.Flags().StringVar(&chirpType, "chirp-type", "pink-log", "Chirp type: linear, log, or pink-log (default: pink-log)")
	cmd.Flags().DurationVar(&padding, "padding", time.Second, "Padding after chirp signal (e.g., 100ms)")
	cmd.Flags().IntVar(&channel, "channel", 0, "Channel index to process (contains both reference and chirp signals, default: 0)")
	cmd.Flags().Float64Var(&refAmplitude, "ref-amplitude", 0, "Expected reference signal amplitude in dB (0 = auto-detect from noise floor, for threshold calculation)")
	cmd.Flags().Float64Var(&chirpAmplitude, "chirp-amplitude", -6, "Expected chirp signal amplitude in dB (for correlation, default: -6)")

	cmd.MarkFlagRequired("output")

	return cmd
}

func runRecvChirp(input, deviceID, output string, sampleRate, channels int, refFrequency float64, refDuration, delay time.Duration,
	chirpDuration time.Duration, chirpFreqStart, chirpFreqEnd float64, chirpType string, padding time.Duration,
	channel int, refAmplitude, chirpAmplitude float64) error {
	// Validate channel index
	if channel < 0 || channel >= channels {
		return fmt.Errorf("channel %d is out of range [0, %d)", channel, channels)
	}

	// Convert amplitudes from dB to linear
	var refAmpLinear float32
	if refAmplitude != 0 {
		refAmpLinear = dbToAmplitude(refAmplitude)
	}

	chirpAmpLinear := dbToAmplitude(chirpAmplitude)

	// Calculate required samples
	refSamples := int(float64(sampleRate) * refDuration.Seconds())
	delaySamples := int(float64(sampleRate) * delay.Seconds())
	chirpSamples := int(float64(sampleRate) * chirpDuration.Seconds())
	paddingSamples := int(float64(sampleRate) * padding.Seconds())
	requiredSamples := refSamples + delaySamples + chirpSamples + paddingSamples

	if requiredSamples <= 0 {
		return fmt.Errorf("total required duration yields no samples")
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigCh
		fmt.Println("\nStopping...")
		cancel()
	}()

	// Create unmarshaller (uses device/file native type)
	var unmarshaller types.Unmarshaller
	if input != "" {
		fmt.Printf("Reading audio from WAV file: %s\n", input)
		unmarshaller = audio.NewUnmarshaller(audio.WithWAVPath(input))
	} else {
		fmt.Printf("Recording from audio device...\n")
		opts := []types.Option{
			audio.WithSampleRate(sampleRate),
			audio.WithChannels(channels),
		}
		if deviceID != "" {
			opts = append(opts, audio.WithCaptureDevice(deviceID))
		}
		unmarshaller = audio.NewUnmarshaller(opts...)
	}

	// Collect audio tensors
	var collectedTensors []types.Tensor
	var stream types.FrameStream
	var tensor types.Tensor

	if err := unmarshaller.Unmarshal(nil, &stream); err == nil {
		defer stream.Close()
		fmt.Println("Streaming audio...")
		for {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case frame, ok := <-stream.C:
				if !ok {
					goto processCollected
				}
				if len(frame.Tensors) == 0 {
					continue
				}
				// Clone tensor to keep it
				for _, t := range frame.Tensors {
					if !t.Empty() {
						cloned := t.Clone()
						if !cloned.Empty() {
							collectedTensors = append(collectedTensors, cloned)
						}
					}
				}
			}
		}
	}

	if err := unmarshaller.Unmarshal(nil, &tensor); err != nil {
		return fmt.Errorf("failed to read from unmarshaller: %w", err)
	}
	defer tensor.Release()
	collectedTensors = []types.Tensor{tensor}

processCollected:
	if len(collectedTensors) == 0 {
		return fmt.Errorf("no audio data collected")
	}

	// Concatenate all tensors into a single tensor
	concatenated, err := concatenateTensors(collectedTensors)
	if err != nil {
		return fmt.Errorf("failed to concatenate tensors: %w", err)
	}
	defer concatenated.Release()

	// Convert to float32 for processing
	shape := concatenated.Shape()
	if len(shape) != 2 {
		return fmt.Errorf("expected tensor shape [frames, channels], got %v", shape)
	}
	totalFrames := shape[0]
	actualChannels := shape[1]

	if actualChannels != channels {
		return fmt.Errorf("expected %d channels, got %d", channels, actualChannels)
	}

	// Extract channel (contains both reference and chirp signals)
	channelData, err := extractChannel(concatenated, channel)
	if err != nil {
		return fmt.Errorf("failed to extract channel: %w", err)
	}

	// Detect reference signal
	refStart, refEnd, err := detectRefSignal(channelData, float32(refFrequency), float32(sampleRate), refSamples, refAmpLinear)
	if err != nil {
		return fmt.Errorf("failed to detect reference signal: %w", err)
	}

	fmt.Printf("Detected reference signal: samples %d-%d (%.3f-%.3f s)\n",
		refStart, refEnd, float64(refStart)/float64(sampleRate), float64(refEnd)/float64(sampleRate))

	// Verify we have enough data
	chirpStart := refEnd + delaySamples
	chirpEnd := chirpStart + chirpSamples
	requiredEnd := chirpEnd + paddingSamples

	if requiredEnd > totalFrames {
		return fmt.Errorf("insufficient audio data: need %d frames, have %d", requiredEnd, totalFrames)
	}

	// Extract chirp region from the same channel (cut everything before ref signal start)
	chirpRegion := channelData[chirpStart:chirpEnd]

	// Generate expected chirp for correlation
	correlationLen := len(chirpRegion) + len(chirpRegion) - 1
	correlationResult := make([]float32, correlationLen)

	switch strings.ToLower(chirpType) {
	case "linear":
		chirpGen := dsp.NewChirpGenerator(float32(sampleRate), float32(chirpFreqStart), float32(chirpFreqEnd), chirpAmpLinear)
		chirpGen.Correlate(correlationResult, chirpRegion)
	case "log", "logarithmic":
		chirpGen := dsp.NewLogarithmicChirpGenerator(float32(sampleRate), float32(chirpFreqStart), float32(chirpFreqEnd), chirpAmpLinear)
		chirpGen.Correlate(correlationResult, chirpRegion)
	case "pink-log", "pink":
		// PinkLogarithmicChirpGenerator doesn't have Correlate, use LogarithmicChirpGenerator as approximation
		chirpGen := dsp.NewLogarithmicChirpGenerator(float32(sampleRate), float32(chirpFreqStart), float32(chirpFreqEnd), chirpAmpLinear)
		fmt.Printf("Warning: pink-log chirp correlation uses logarithmic generator\n")
		chirpGen.Correlate(correlationResult, chirpRegion)
	default:
		return fmt.Errorf("unsupported chirp type: %s", chirpType)
	}

	// Create output tensor with correlation result
	outputShape := mathtensor.NewShape(correlationLen, 1)
	outputTensor := mathtensor.New(tensortypes.DataType(types.FP32), outputShape)
	outputData := outputTensor.Data().([]float32)
	copy(outputData, correlationResult)

	// Write to WAV file
	file, err := os.Create(output)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer file.Close()

	m := audio.NewMarshaller(
		audio.WithWAVPath(output),
		audio.WithSampleRate(sampleRate),
		audio.WithChannels(1),
		types.WithDestinationType(types.FP32),
	)

	if err := m.Marshal(file, outputTensor); err != nil {
		return fmt.Errorf("failed to write WAV file: %w", err)
	}

	fmt.Printf("Correlation result written to: %s\n", output)
	fmt.Printf("  Correlation length: %d samples (%.3f s)\n", correlationLen, float64(correlationLen)/float64(sampleRate))
	fmt.Printf("  Chirp region: samples %d-%d\n", chirpStart, chirpEnd)

	return nil
}

// detectRefSignal finds the reference signal using Goertzel algorithm
func detectRefSignal(signal []float32, refFreq, sampleRate float32, refSamples int, expectedAmp float32) (start, end int, err error) {
	if len(signal) < refSamples {
		return 0, 0, fmt.Errorf("signal too short: need at least %d samples, got %d", refSamples, len(signal))
	}

	// Use a sliding window to find the reference signal
	windowSize := refSamples
	stepSize := windowSize / 4 // Overlap windows for better detection
	if stepSize < 1 {
		stepSize = 1
	}

	var maxAmp float32
	var bestStart int
	threshold := expectedAmp * 0.3 // 30% of expected amplitude as threshold
	if expectedAmp <= 0 {
		// If no expected amplitude, use adaptive threshold
		// Calculate noise floor from first part of signal
		noiseSamples := windowSize
		if noiseSamples > len(signal)/4 {
			noiseSamples = len(signal) / 4
		}
		noiseAmps := make([]float32, 0, noiseSamples/windowSize)
		for i := 0; i < noiseSamples-windowSize; i += stepSize {
			window := vec.NewFrom(signal[i : i+windowSize]...)
			meas := dsp.Goertzel(window, refFreq, sampleRate)
			noiseAmps = append(noiseAmps, meas.Amplitude)
		}
		// Calculate median as noise floor
		if len(noiseAmps) > 0 {
			threshold = noiseAmps[len(noiseAmps)/2] * 3.0 // 3x noise floor
		} else {
			threshold = 0.01 // Default threshold
		}
	}

	amplitudes := make([]float32, 0)
	for i := 0; i <= len(signal)-windowSize; i += stepSize {
		window := vec.NewFrom(signal[i : i+windowSize]...)
		meas := dsp.Goertzel(window, refFreq, sampleRate)
		amplitudes = append(amplitudes, meas.Amplitude)

		if meas.Amplitude > maxAmp {
			maxAmp = meas.Amplitude
			bestStart = i
		}
	}

	if maxAmp < threshold {
		return 0, 0, fmt.Errorf("reference signal not found: max amplitude %.6f below threshold %.6f", maxAmp, threshold)
	}

	// Refine start position by finding peak in the region
	searchStart := bestStart - stepSize
	if searchStart < 0 {
		searchStart = 0
	}
	searchEnd := bestStart + stepSize + windowSize
	if searchEnd > len(signal) {
		searchEnd = len(signal)
	}

	// Fine-grained search around bestStart
	refinedStart := bestStart
	maxAmpRefined := maxAmp
	for i := searchStart; i <= searchEnd-windowSize && i+windowSize <= len(signal); i++ {
		window := vec.NewFrom(signal[i : i+windowSize]...)
		meas := dsp.Goertzel(window, refFreq, sampleRate)
		if meas.Amplitude > maxAmpRefined {
			maxAmpRefined = meas.Amplitude
			refinedStart = i
		}
	}

	refinedEnd := refinedStart + refSamples
	if refinedEnd > len(signal) {
		refinedEnd = len(signal)
	}

	return refinedStart, refinedEnd, nil
}

// extractChannel extracts a single channel from a multi-channel tensor
func extractChannel(t types.Tensor, channelIndex int) ([]float32, error) {
	shape := t.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("expected tensor shape [frames, channels], got %v", shape)
	}
	frames := shape[0]
	channels := shape[1]

	if channelIndex < 0 || channelIndex >= channels {
		return nil, fmt.Errorf("channel index %d out of range [0, %d)", channelIndex, channels)
	}

	// Access data and convert to float32 if needed
	data := t.Data()
	var floatData []float32
	if t.DataType() == types.FP32 {
		var ok bool
		floatData, ok = data.([]float32)
		if !ok {
			return nil, fmt.Errorf("tensor data type is FP32 but data is not []float32")
		}
	} else {
		// Convert to FP32
		convertedData := tensortypes.CloneTensorDataTo(tensortypes.FP32, data)
		if convertedData == nil {
			return nil, fmt.Errorf("failed to convert tensor data to float32")
		}
		var ok bool
		floatData, ok = convertedData.([]float32)
		if !ok {
			return nil, fmt.Errorf("converted data is not []float32")
		}
	}

	// Extract channel
	result := make([]float32, frames)
	for i := 0; i < frames; i++ {
		result[i] = floatData[i*channels+channelIndex]
	}

	return result, nil
}

// concatenateTensors concatenates multiple tensors along the frame dimension
func concatenateTensors(tensors []types.Tensor) (types.Tensor, error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("no tensors to concatenate")
	}
	if len(tensors) == 1 {
		return tensors[0], nil
	}

	// Validate all tensors have compatible shapes
	firstShape := tensors[0].Shape()
	if len(firstShape) != 2 {
		return nil, fmt.Errorf("tensor shape must be [frames, channels]")
	}
	channels := firstShape[1]
	dtype := tensors[0].DataType()

	totalFrames := firstShape[0]
	for i := 1; i < len(tensors); i++ {
		shape := tensors[i].Shape()
		if len(shape) != 2 {
			return nil, fmt.Errorf("tensor %d: shape must be [frames, channels]", i)
		}
		if shape[1] != channels {
			return nil, fmt.Errorf("tensor %d: channel count mismatch (%d != %d)", i, shape[1], channels)
		}
		if tensors[i].DataType() != dtype {
			return nil, fmt.Errorf("tensor %d: data type mismatch", i)
		}
		totalFrames += shape[0]
	}

	// Create concatenated tensor
	concatenatedShape := mathtensor.NewShape(totalFrames, channels)
	concatenated := mathtensor.New(tensortypes.DataType(dtype), concatenatedShape)

	// Convert all to float32 for concatenation
	allData := make([][]float32, len(tensors))
	for i, t := range tensors {
		data := t.Data()
		var floatData []float32
		if t.DataType() == types.FP32 {
			var ok bool
			floatData, ok = data.([]float32)
			if !ok {
				return nil, fmt.Errorf("tensor %d: data type is FP32 but data is not []float32", i)
			}
		} else {
			// Convert to FP32
			converted := tensortypes.CloneTensorDataTo(tensortypes.FP32, data)
			if converted == nil {
				return nil, fmt.Errorf("failed to convert tensor %d to float32", i)
			}
			var ok bool
			floatData, ok = converted.([]float32)
			if !ok {
				return nil, fmt.Errorf("tensor %d: converted data is not []float32", i)
			}
		}
		allData[i] = floatData
	}

	// Copy data
	outputData := concatenated.Data().([]float32)
	offset := 0
	for _, data := range allData {
		copy(outputData[offset:], data)
		offset += len(data)
	}

	return concatenated, nil
}
