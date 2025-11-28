package main

import (
	"context"
	encodingHex "encoding/hex"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/marshaller/audio"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/dsss"
	"github.com/itohio/EasyRobot/x/math/tensor"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/spf13/cobra"
)

// runGenCodeOnly generates and displays codes without creating a signal
func runGenCodeOnly(genCode string, spreadingFactor int) error {
	// Parse "NxM" or "N" format
	numCodes := 0
	codeLength := 16 // default

	if strings.Contains(genCode, "x") {
		parts := strings.Split(genCode, "x")
		if len(parts) != 2 {
			return fmt.Errorf("invalid --gen-code format: %s (expected \"NxM\" or \"N\")", genCode)
		}
		var err1, err2 error
		numCodes, err1 = strconv.Atoi(strings.TrimSpace(parts[0]))
		codeLength, err2 = strconv.Atoi(strings.TrimSpace(parts[1]))
		if err1 != nil || err2 != nil {
			return fmt.Errorf("invalid --gen-code format: %s (N and M must be integers)", genCode)
		}
	} else {
		var err1 error
		numCodes, err1 = strconv.Atoi(strings.TrimSpace(genCode))
		if err1 != nil {
			return fmt.Errorf("invalid --gen-code format: %s (expected \"NxM\" or \"N\")", genCode)
		}
	}

	if numCodes <= 0 {
		return fmt.Errorf("number of codes must be positive, got %d", numCodes)
	}
	if codeLength <= 0 {
		return fmt.Errorf("code length must be positive, got %d", codeLength)
	}

	// Generate codes as [][]int8 (bipolar: -1, +1)
	dsssCodes := generateOrthogonalCodes(numCodes, codeLength)

	// Display codes
	fmt.Printf("Generated %d codes of %d chips each:\n", len(dsssCodes), codeLength)
	fmt.Printf("Modulation order: %d", len(dsssCodes))
	if len(dsssCodes) == 1 {
		fmt.Print(" (Binary DSSS)")
	} else {
		fmt.Printf(" (%d-ary DSSS, %d bits per symbol)", len(dsssCodes), int(math.Log2(float64(len(dsssCodes)))))
	}
	fmt.Println()
	fmt.Println("Use these codes with --code flags:")
	// Output all codes on a single line for easy copying
	codeFlags := make([]string, len(dsssCodes))
	for i, code := range dsssCodes {
		// Convert bipolar code to hex representation
		codeBytes := make([]byte, (len(code)+7)/8)
		for j := 0; j < len(code); j++ {
			byteIdx := j / 8
			bitIdx := j % 8
			if code[j] == 1 {
				codeBytes[byteIdx] |= 1 << bitIdx
			}
		}
		hexStr := fmt.Sprintf("%X", codeBytes)
		codeFlags[i] = fmt.Sprintf("--code %s", hexStr)
	}
	fmt.Printf("  %s\n", strings.Join(codeFlags, " "))

	return nil
}

// generateOrthogonalCodes generates Walsh codes using Hadamard matrix construction.
// This ensures all codes are orthogonal to each other.
func generateOrthogonalCodes(numCodes, length int) [][]int8 {
	if numCodes <= 0 {
		numCodes = 1
	}
	// Ensure length is a power of 2 for Walsh codes
	actualLength := length
	if length <= 0 {
		actualLength = 1
	}
	// Find next power of 2 >= length
	nextPowerOf2 := 1
	for nextPowerOf2 < actualLength {
		nextPowerOf2 <<= 1
	}
	actualLength = nextPowerOf2

	// Generate Hadamard matrix (Walsh codes)
	hadamard := make([][]int8, actualLength)
	for i := range hadamard {
		hadamard[i] = make([]int8, actualLength)
	}

	// Initialize first element
	hadamard[0][0] = 1

	// Build Hadamard matrix recursively: H(2n) = [H(n) H(n); H(n) -H(n)]
	for n := 1; n < actualLength; n <<= 1 {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				// Copy to top-right, top-left, bottom-left
				hadamard[i][j+n] = hadamard[i][j]
				hadamard[i+n][j] = hadamard[i][j]
				// Negate for bottom-right
				hadamard[i+n][j+n] = -hadamard[i][j]
			}
		}
	}

	// Extract requested number of codes, truncate to requested length
	codes := make([][]int8, numCodes)
	for i := 0; i < numCodes && i < actualLength; i++ {
		codes[i] = make([]int8, length)
		copy(codes[i], hadamard[i][:length])
	}

	// If we need more codes than the Hadamard matrix provides, repeat with different patterns
	// For now, we'll just use what we have (Walsh codes are limited to power-of-2 lengths)
	return codes
}

func newGenDSSSCmd() *cobra.Command {
	var (
		output          string
		deviceID        string
		amplitude       string
		frequency       float64
		data            string
		hex             string
		sampleRate      int
		symbolRate      float64
		spreadingFactor int
		chips           int
		codes           []string
		genCode         string
		modulationOrder int
		dataType        string
		pad             time.Duration
		padType         string
		padAmplitude    string
		noise           string
		noiseType       string
	)

	cmd := &cobra.Command{
		Use:   "gen-dsss",
		Short: "Generate DSSS modulated signal",
		Long:  "Generate a Direct Sequence Spread Spectrum (DSSS) modulated signal from data. Supports protocol wrapping, padding, and noise.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runGenDSSS(output, deviceID, amplitude, frequency, data, hex, sampleRate, symbolRate, spreadingFactor, chips, codes, genCode, modulationOrder, dataType, pad, padType, padAmplitude, noise, noiseType)
		},
	}

	cmd.Flags().StringVarP(&output, "output", "o", "", "Output WAV file path (or use --device-id for audio device)")
	cmd.Flags().StringVar(&deviceID, "device-id", "", "Playback device ID (alternative to --output)")
	cmd.Flags().StringVarP(&amplitude, "amplitude", "a", "-6", "Amplitude in dB (default: -6dB)")
	cmd.Flags().Float64VarP(&frequency, "frequency", "f", 10000, "Carrier frequency in Hz (default: 10000)")
	cmd.Flags().StringVar(&data, "data", "Hello", "Data as string (default: \"Hello\")")
	cmd.Flags().StringVar(&hex, "hex", "", "Data as hex string (alternative to -data)")
	cmd.Flags().IntVarP(&sampleRate, "sample-rate", "r", 96000, "Sample rate (Hz)")
	cmd.Flags().Float64Var(&symbolRate, "symbol-rate", 500, "Symbol rate in baud (default: 500)")
	cmd.Flags().IntVarP(&modulationOrder, "modulation-order", "m", 4, "Modulation order (default: 4)")
	cmd.Flags().IntVar(&spreadingFactor, "spreading-factor", 31, "Spreading factor (chips per symbol, default: 31, ignored if --gen-code is used)")
	cmd.Flags().IntVar(&chips, "chips", 16, "Code length in chips (default: 16, required when using --code, inferred from hex string if not specified)")
	cmd.Flags().StringSliceVarP(&codes, "code", "c", []string{}, "Spreading code as hex string (can be specified multiple times or comma-separated, e.g., --code A5,F3 or -c A5 -c F3 for 2-ary DSSS)")
	cmd.Flags().StringVar(&genCode, "gen-code", "", "Generate code table: \"NxM\" for N codes of M chips, or \"N\" for N codes of 16 chips (e.g., \"4x8\" or \"4\")")
	cmd.Flags().StringVar(&dataType, "data-type", "INT16", "Data type (INT16, INT32, FP32, UINT8, default: INT16)")
	cmd.Flags().DurationVar(&pad, "pad", 1*time.Second, "Padding duration (adds padding at beginning and end, e.g., 1s, 100ms)")
	cmd.Flags().StringVar(&padType, "pad-type", "zero", "Padding type: \"zero\" or \"noise\" (default: zero)")
	cmd.Flags().StringVar(&padAmplitude, "pad-amplitude", "-20", "Padding noise amplitude in dB (only used if pad-type=noise, default: -20dB)")
	cmd.Flags().StringVar(&noise, "noise", "-120", "Noise amplitude in dB to add on top of signal (default: -120, meaning no noise)")
	cmd.Flags().StringVar(&noiseType, "noise-type", "white", "Noise type: \"white\", \"pink\", or \"brown\" (default: white)")

	return cmd
}

func runGenDSSS(output, deviceID, amplitude string, frequency float64, data, hex string, sampleRate int, symbolRate float64, spreadingFactor, chips int, codes []string, genCode string, modulationOrder int, dataType string, pad time.Duration, padType, padAmplitude, noise, noiseType string) error {
	// If only --gen-code is provided, just generate and display codes
	if genCode != "" && len(codes) == 0 && output == "" && deviceID == "" {
		return runGenCodeOnly(genCode, spreadingFactor)
	}

	// Validate output (required if generating signal)
	if output == "" && deviceID == "" {
		return fmt.Errorf("either --output or --device-id must be specified")
	}

	// Parse data
	dataBytes, err := parseData(data, hex)
	if err != nil {
		return err
	}

	// Work directly with user data - no protocol wrapping

	// Parse amplitude (dB)
	ampDb, err := strconv.ParseFloat(amplitude, 64)
	if err != nil {
		return fmt.Errorf("invalid amplitude: %w", err)
	}
	ampLinear := dbToAmplitude(ampDb)

	// Parse data type
	dt, err := parseDataType(dataType)
	if err != nil {
		return err
	}

	// Parse spreading codes - either from --code flags or --gen-code
	var dsssCodes [][]int8

	// Default to modulation-order codes if nothing is provided
	if genCode == "" && len(codes) == 0 {
		codeLength := chips
		if codeLength == 0 {
			codeLength = 16 // Default key size is 16 bits
		}
		genCode = fmt.Sprintf("%dx%d", modulationOrder, codeLength)
		fmt.Printf("  Using default codes: %d codes of %d chips each\n", modulationOrder, codeLength)
	}

	if genCode != "" {
		// Generate codes using --gen-code
		if len(codes) > 0 {
			return fmt.Errorf("--code and --gen-code cannot be used together")
		}

		// Parse "NxM" or "N" format
		numCodes := 0
		codeLength := chips
		if codeLength == 0 {
			codeLength = 16 // Default key size is 16 bits
		}

		if strings.Contains(genCode, "x") {
			parts := strings.Split(genCode, "x")
			if len(parts) != 2 {
				return fmt.Errorf("invalid --gen-code format: %s (expected \"NxM\" or \"N\")", genCode)
			}
			var err1, err2 error
			numCodes, err1 = strconv.Atoi(strings.TrimSpace(parts[0]))
			codeLength, err2 = strconv.Atoi(strings.TrimSpace(parts[1]))
			if err1 != nil || err2 != nil {
				return fmt.Errorf("invalid --gen-code format: %s (N and M must be integers)", genCode)
			}
		} else {
			var err1 error
			numCodes, err1 = strconv.Atoi(strings.TrimSpace(genCode))
			if err1 != nil {
				return fmt.Errorf("invalid --gen-code format: %s (expected \"NxM\" or \"N\")", genCode)
			}
			// Use modulation order if not specified in genCode
			if numCodes == 0 {
				numCodes = modulationOrder
			}
		}

		// Ensure numCodes matches modulation order
		if numCodes != modulationOrder {
			fmt.Printf("  Warning: generated codes count (%d) doesn't match modulation order (%d), using modulation order\n", numCodes, modulationOrder)
			numCodes = modulationOrder
		}

		if numCodes <= 0 {
			return fmt.Errorf("number of codes must be positive, got %d", numCodes)
		}
		if codeLength <= 0 {
			return fmt.Errorf("code length must be positive, got %d", codeLength)
		}

		// Validation will be done by modulator constructor

		// Generate codes as [][]int8 (bipolar: -1, +1)
		dsssCodes = generateOrthogonalCodes(numCodes, codeLength)

		// Update spreading factor from generated codes
		spreadingFactor = len(dsssCodes[0])
	} else {
		// Parse codes from hex strings
		if len(codes) == 0 {
			// This should not happen due to default above, but keep as safety check
			return fmt.Errorf("either --code or --gen-code is required")
		}

		// Ensure number of codes matches modulation order
		if len(codes) != modulationOrder {
			return fmt.Errorf("number of codes (%d) must match modulation order (%d)", len(codes), modulationOrder)
		}

		// Determine code length: use --chips if provided, otherwise infer from hex string
		codeLength := chips
		if codeLength == 0 {
			// Infer code length from first hex string (all codes should have same length)
			// Hex string length: 2 hex chars = 1 byte = 8 bits
			// Default: use all bits from decoded hex bytes
			hexBytes, err := encodingHex.DecodeString(codes[0])
			if err != nil {
				return fmt.Errorf("invalid hex code: %s: %w", codes[0], err)
			}
			codeLength = len(hexBytes) * 8
			if codeLength == 0 {
				codeLength = 16 // Default key size is 16 bits
			}
		}

		if codeLength <= 0 {
			return fmt.Errorf("code length (--chips) must be positive, got %d", codeLength)
		}

		// Parse all hex codes to [][]int8
		dsssCodes = make([][]int8, len(codes))
		for i, hexStr := range codes {
			hexBytes, err := encodingHex.DecodeString(hexStr)
			if err != nil {
				return fmt.Errorf("invalid hex code %d: %s: %w", i, hexStr, err)
			}
			// Convert bytes to bipolar int8 codes (-1, +1)
			code := make([]int8, codeLength)
			for j := 0; j < codeLength && j < len(hexBytes)*8; j++ {
				byteIdx := j / 8
				bitIdx := j % 8
				if (hexBytes[byteIdx]>>bitIdx)&1 == 1 {
					code[j] = 1
				} else {
					code[j] = -1
				}
			}
			dsssCodes[i] = code
		}

		// Update spreading factor to match actual code length
		spreadingFactor = codeLength
		// Validation will be done by modulator constructor
	}

	// Calculate bits per symbol from modulation order
	bitsPerSymbol := int(math.Log2(float64(len(dsssCodes))))
	if bitsPerSymbol == 0 {
		bitsPerSymbol = 1
	}

	// Create DSSS config
	config := dsss.Config{
		BitsPerSymbol:   bitsPerSymbol,
		ModulationOrder: len(dsssCodes),
		Codes:           dsssCodes,
		SpreadingFactor: spreadingFactor,
		SampleRate:      float32(sampleRate),
		SymbolRate:      float32(symbolRate),
		CarrierFreq:     float32(frequency),
		PulseShape:      "raised_cosine",
		Rolloff:         0.35,
		PulseFilterSpan: 4,
		PreambleLength:  0,
		Amplitude:       ampLinear,
	}

	// Create DSSS modulator
	dsssModulator, err := dsss.NewModulator(config)
	if err != nil {
		return fmt.Errorf("failed to create DSSS modulator: %w", err)
	}

	// Convert data to bits buffer
	dataBits, err := buffer.NewBits(bitsPerSymbol)
	if err != nil {
		return fmt.Errorf("failed to create bits buffer: %w", err)
	}
	dataBits.Write(dataBytes)

	// Estimate length needed
	expectedSize := dsssModulator.Process(nil, dataBits)
	if expectedSize == 0 {
		return fmt.Errorf("DSSS modulator expects 0 samples (check modulator configuration or data)")
	}

	// Generate signal
	modulatedVec := vec.New(expectedSize)
	modulatedBuf := buffer.New(modulatedVec)
	dataBits.Reset() // Reset for actual processing
	samplesWritten := dsssModulator.Process(modulatedBuf, dataBits)

	if samplesWritten == 0 {
		return fmt.Errorf("DSSS modulator wrote 0 samples (check modulator configuration)")
	}

	modData := []float32(modulatedVec)[:samplesWritten]

	// Calculate RMS statistics
	if len(modData) > 0 {
		signalVec := vec.NewFrom(modData...)
		rmsStats := dsp.CalculateRMSStatistics(signalVec)
		fmt.Printf("  RMS Statistics:\n")
		fmt.Printf("    MIN RMS: %.2f dB\n", rmsStats.MinRMSDb)
		fmt.Printf("    MAX RMS: %.2f dB\n", rmsStats.MaxRMSDb)
		fmt.Printf("    AVG RMS: %.2f dB\n", rmsStats.AvgRMSDb)
	}

	// Apply padding if requested
	if pad > 0 {
		padSamples := int(float64(sampleRate) * pad.Seconds())
		modData = applyPadding(vec.NewFrom(modData...), padSamples, padType, padAmplitude, float32(sampleRate))
	}

	// Apply noise if requested (if noise > -120 dB)
	noiseDb, err := strconv.ParseFloat(noise, 64)
	if err != nil {
		return fmt.Errorf("invalid noise amplitude: %w", err)
	}
	if noiseDb > -120 {
		noiseAmp := dbToAmplitude(noiseDb)
		modData = applyNoise(vec.NewFrom(modData...), noiseAmp, noiseType, float32(sampleRate))
	}

	numSamples := len(modData)

	// Create tensor [frames, 1]
	shape := tensor.NewShape(numSamples, 1)
	t := tensor.New(tensortypes.DataType(dt), shape)

	// Fill tensor and calculate statistics
	var tensorMax, tensorMin float32
	var tensorSumSquares float64
	switch dt {
	case types.INT16:
		data := t.Data().([]int16)
		for i := range numSamples {
			val := modData[i] * 32767.0
			if val > 32767.0 {
				val = 32767.0
			} else if val < -32768.0 {
				val = -32768.0
			}
			data[i] = int16(val)
			// Calculate stats on normalized value
			normalized := float32(data[i]) / 32768.0
			if normalized > tensorMax {
				tensorMax = normalized
			}
			if normalized < tensorMin {
				tensorMin = normalized
			}
			tensorSumSquares += float64(normalized * normalized)
		}
	case types.INT32:
		data := t.Data().([]int32)
		// Verify tensor data length matches expected
		expectedLen := numSamples * 1 // shape [numSamples, 1]
		if len(data) != expectedLen {
			return fmt.Errorf("tensor data length mismatch: expected %d, got %d", expectedLen, len(data))
		}
		for i := range numSamples {
			val := modData[i] * 2147483647.0
			if val > 2147483647.0 {
				val = 2147483647.0
			} else if val < -2147483648.0 {
				val = -2147483648.0
			}
			// Write to tensor in row-major order: [frame, channel] = data[frame*numChannels+channel]
			// For shape [numSamples, 1]: [i, 0] = data[i*1+0] = data[i]
			data[i*1+0] = int32(val)
			// Calculate stats on normalized value
			normalized := float32(data[i*1+0]) / 2147483648.0
			if normalized > tensorMax {
				tensorMax = normalized
			}
			if normalized < tensorMin {
				tensorMin = normalized
			}
			tensorSumSquares += float64(normalized * normalized)
		}
	case types.FP32:
		data := t.Data().([]float32)
		// Verify tensor data length matches expected
		expectedLen := numSamples * 1 // shape [numSamples, 1]
		if len(data) != expectedLen {
			return fmt.Errorf("tensor data length mismatch: expected %d, got %d", expectedLen, len(data))
		}
		// Write to tensor in row-major order: [frame, channel] = data[frame*numChannels+channel]
		// For shape [numSamples, 1]: [i, 0] = data[i*1+0] = data[i]
		for i := range numSamples {
			data[i*1+0] = modData[i]
		}
		// Calculate stats
		for _, v := range data {
			if v > tensorMax {
				tensorMax = v
			}
			if v < tensorMin {
				tensorMin = v
			}
			tensorSumSquares += float64(v * v)
		}
	case types.UINT8:
		data := t.Data().([]uint8)
		// Verify tensor data length matches expected
		expectedLen := numSamples * 1 // shape [numSamples, 1]
		if len(data) != expectedLen {
			return fmt.Errorf("tensor data length mismatch: expected %d, got %d", expectedLen, len(data))
		}
		for i := range numSamples {
			val := (modData[i] + 1.0) * 127.5
			if val > 255.0 {
				val = 255.0
			} else if val < 0.0 {
				val = 0.0
			}
			// Write to tensor in row-major order: [frame, channel] = data[frame*numChannels+channel]
			// For shape [numSamples, 1]: [i, 0] = data[i*1+0] = data[i]
			data[i*1+0] = uint8(val)
			// Calculate stats on normalized value
			normalized := (float32(data[i*1+0]) - 128.0) / 128.0
			if normalized > tensorMax {
				tensorMax = normalized
			}
			if normalized < tensorMin {
				tensorMin = normalized
			}
			tensorSumSquares += float64(normalized * normalized)
		}
	default:
		return fmt.Errorf("unsupported data type: %v", dt)
	}

	// Write to WAV file or device
	if output != "" {
		// Create marshaller based on output config
		ctx := context.Background()
		m := audio.NewMarshaller(
			audio.WithWAVPath(output),
			audio.WithSampleRate(sampleRate),
			audio.WithChannels(1),
			types.WithDestinationType(dt),
		)

		if err := m.Marshal(nil, t, types.WithContext(ctx)); err != nil {
			return fmt.Errorf("failed to write: %w", err)
		}

		fmt.Printf("Generated DSSS signal: %s\n", output)
	} else {
		// TODO: Implement device playback
		return fmt.Errorf("device playback not yet implemented (use --output to save to WAV file)")
	}

	fmt.Printf("  Sample rate: %d Hz\n", sampleRate)
	fmt.Printf("  Data type: %s\n", dataType)
	fmt.Printf("  Amplitude: %.2f dB\n", ampDb)
	fmt.Printf("  Carrier frequency: %.0f Hz\n", frequency)
	fmt.Printf("  Symbol rate: %.0f baud\n", symbolRate)
	fmt.Printf("  Spreading factor: %d\n", spreadingFactor)
	if genCode != "" {
		fmt.Printf("  Generated codes: %s (%d codes of %d chips, non-deterministic)\n", genCode, len(dsssCodes), spreadingFactor)
		// Print all generated codes as hex so user can use them for decoding
		codeHexStrs := make([]string, len(dsssCodes))
		for i := range dsssCodes {
			// Convert bipolar code to hex representation
			codeBytes := make([]byte, (len(dsssCodes[i])+7)/8)
			for j := 0; j < len(dsssCodes[i]); j++ {
				byteIdx := j / 8
				bitIdx := j % 8
				if dsssCodes[i][j] == 1 {
					codeBytes[byteIdx] |= 1 << bitIdx
				}
			}
			codeHexStrs[i] = fmt.Sprintf("%X", codeBytes)
		}
		fmt.Printf("  Generated codes (hex, use these for --code flags):\n")
		// Output all codes on a single line for easy copying
		codeFlags := make([]string, len(codeHexStrs))
		for i, hexStr := range codeHexStrs {
			codeFlags[i] = fmt.Sprintf("--code %s", hexStr)
		}
		fmt.Printf("    %s\n", strings.Join(codeFlags, " "))
	} else {
		fmt.Printf("  Codes: %d code(s): %s\n", len(codes), strings.Join(codes, ", "))
	}
	fmt.Printf("  Modulation order: %d", len(dsssCodes))
	if len(dsssCodes) == 1 {
		fmt.Print(" (Binary DSSS)")
	} else {
		fmt.Printf(" (%d-ary DSSS, %d bits per symbol)", len(dsssCodes), int(math.Log2(float64(len(dsssCodes)))))
	}
	fmt.Println()

	// Print signal statistics if available
	if len(modData) > 0 {
		var signalMin, signalMax float32
		var signalSumSquares float64
		for _, v := range modData {
			if v < signalMin {
				signalMin = v
			}
			if v > signalMax {
				signalMax = v
			}
			signalSumSquares += float64(v * v)
		}
		signalRMS := float32(math.Sqrt(signalSumSquares / float64(len(modData))))
		fmt.Printf("  Signal statistics:\n")
		fmt.Printf("    Samples: %d\n", len(modData))
		fmt.Printf("    MIN amplitude: %.6f\n", signalMin)
		fmt.Printf("    MAX amplitude: %.6f\n", signalMax)
		fmt.Printf("    RMS amplitude: %.6f\n", signalRMS)
	}

	// Print tensor statistics if available
	if numSamples > 0 && tensorSumSquares > 0 {
		tensorRMS := float32(math.Sqrt(tensorSumSquares / float64(numSamples)))
		fmt.Printf("  Tensor statistics:\n")
		fmt.Printf("    Shape: %v\n", shape)
		fmt.Printf("    MIN amplitude: %.6f\n", tensorMin)
		fmt.Printf("    MAX amplitude: %.6f\n", tensorMax)
		fmt.Printf("    RMS amplitude: %.6f\n", tensorRMS)
	}
	fmt.Printf("  Data: %d bytes\n", len(dataBytes))
	if pad > 0 {
		fmt.Printf("  Padding: %v (%s)\n", pad, padType)
	}
	if noiseDb > -120 {
		fmt.Printf("  Noise: %.2f dB (%s)\n", noiseDb, noiseType)
	}

	return nil
}
