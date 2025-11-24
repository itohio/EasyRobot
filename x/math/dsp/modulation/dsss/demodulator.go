package dsss

import (
	"fmt"
	"math"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// Demodulator implements DSSS demodulation from scratch following SPEC.md
// Uses raised cosine matched filter and coherent correlation
type Demodulator struct {
	// Parameters
	sampleRate      float32
	carrierFreq     float32
	symbolRate      float32
	spreadingFactor int // L: chips per symbol
	modulationOrder int // M: 2 for binary, 4 for 4-ary, etc.

	// Derived parameters
	samplesPerChip   int // N_sc
	samplesPerSymbol int // L * N_sc
	chipDuration     float32
	bitsPerSymbol    int // k = log2(M)

	// Pulse shaping (matched filter)
	rolloffFactor float32
	pulseShape    []float32 // Matched filter (same as raised cosine for now)

	// Spreading codes
	code      Code
	codeTable CodeTable

	// State
	bitBuffer        buffer.Bits[byte]
	sampleCounter    int64 // Global sample counter for phase tracking
	bytesExtracted   int   // Number of complete bytes extracted
	symbolOffset     int64 // Offset for symbol timing (for synchronization)
	symbolsProcessed int64 // Number of symbols processed (for buffer advancement)

	// Buffers
	sampleBuffer []float32 // Ring buffer for incoming samples
	bufferSize   int
	bufferIndex  int
	totalSamples int64
}

// NewDemodulator creates a new DSSS demodulator following SPEC.md
// Parameters:
//   - sampleRate: sampling frequency in Hz
//   - carrierFreq: carrier frequency in Hz (0 for baseband)
//   - symbolRate: symbol rate in symbols/sec
//   - rolloffFactor: raised cosine roll-off factor (must match modulator)
//   - codes: spreading codes (must match modulator)
func NewDemodulator(sampleRate, carrierFreq, symbolRate, rolloffFactor float32, codes ...Code) *Demodulator {
	if len(codes) == 0 {
		panic("DSSS demodulator: at least one code must be provided")
	}

	// Validate codes (same as modulator)
	firstCodeLength := codes[0].length
	if firstCodeLength == 0 {
		panic("DSSS demodulator: codes must not be empty")
	}

	for i, code := range codes {
		if code.length == 0 {
			panic(fmt.Sprintf("DSSS demodulator: code %d must not be empty", i))
		}
		if code.length != firstCodeLength {
			panic(fmt.Sprintf("DSSS demodulator: code %d has length %d, expected %d", i, code.length, firstCodeLength))
		}
	}

	// Determine modulation order
	numCodes := len(codes)
	var modulationOrder int
	if numCodes == 1 {
		modulationOrder = 2 // Binary DSSS
	} else {
		modulationOrder = numCodes
		if (modulationOrder & (modulationOrder - 1)) != 0 {
			panic(fmt.Sprintf("DSSS demodulator: M-ary requires M to be a power of 2, got %d codes", modulationOrder))
		}
	}

	spreadingFactor := firstCodeLength
	chipRate := symbolRate * float32(spreadingFactor)
	samplesPerChip := int(sampleRate / chipRate)
	if samplesPerChip <= 0 {
		panic(fmt.Sprintf("DSSS demodulator: samplesPerChip must be positive, got %d", samplesPerChip))
	}

	chipDuration := 1.0 / chipRate
	bitsPerSymbol := int(math.Log2(float64(modulationOrder)))
	samplesPerSymbol := spreadingFactor * samplesPerChip

	// Pre-compute matched filter (raised cosine pulse)
	pulseShape := computeRaisedCosinePulse(samplesPerChip, chipDuration, sampleRate, rolloffFactor)

	// Buffer size: enough for multiple symbols
	bufferSize := samplesPerSymbol * 8

	demod := &Demodulator{
		sampleRate:       sampleRate,
		carrierFreq:      carrierFreq,
		symbolRate:       symbolRate,
		spreadingFactor:  spreadingFactor,
		modulationOrder:  modulationOrder,
		samplesPerChip:   samplesPerChip,
		samplesPerSymbol: samplesPerSymbol,
		chipDuration:     chipDuration,
		bitsPerSymbol:    bitsPerSymbol,
		rolloffFactor:    rolloffFactor,
		bitBuffer:        buffer.NewBits[byte](bitsPerSymbol),
		sampleCounter:    0,
		pulseShape:       pulseShape,
		sampleBuffer:     make([]float32, bufferSize),
		bufferSize:       bufferSize,
		symbolOffset:     0,
	}

	// Set code or codeTable
	if numCodes == 1 {
		demod.code = codes[0]
	} else {
		demod.codeTable = make(CodeTable, modulationOrder)
		copy(demod.codeTable, codes)
	}

	return demod
}

// Reset resets the demodulator state
func (d *Demodulator) Reset() {
	d.bitBuffer.Clear()
	d.sampleCounter = 0
	d.bytesExtracted = 0
	d.bufferIndex = 0
	d.totalSamples = 0
	d.symbolOffset = 0
	d.symbolsProcessed = 0
}

// Decode decodes amplitude vector into bytes
// Implements modulation.Demodulator interface
func (d *Demodulator) Decode(data []byte, amplitudes vecTypes.Vector) int {
	if data == nil {
		// Calculate expected length
		maxSymbols := d.bufferSize / d.samplesPerSymbol
		return (maxSymbols * d.bitsPerSymbol) / 8
	}

	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)

	// Add samples to buffer
	for _, sample := range ampData {
		d.sampleBuffer[d.bufferIndex] = sample
		d.bufferIndex = (d.bufferIndex + 1) % d.bufferSize
		d.totalSamples++
	}

	bytesDecoded := 0

	// Process symbols
	for bytesDecoded < len(data) {
		// Check if we have enough samples for a symbol
		// Need at least samplesPerSymbol samples in buffer
		availableSamples := d.totalSamples - d.symbolsProcessed*int64(d.samplesPerSymbol)
		if availableSamples < int64(d.samplesPerSymbol) {
			break
		}

		// Extract symbol window from buffer
		// Calculate which symbol we're processing (relative to start of received data)
		symbolStartSample := d.symbolsProcessed * int64(d.samplesPerSymbol)
		window := d.extractSymbolWindowAt(symbolStartSample)

		// Calculate absolute sample index for carrier phase calculation
		// This is the absolute index from when we started receiving
		absoluteSampleIdx := symbolStartSample

		// Correlate with codes (Component 5: Chip Correlator)
		correlations := d.correlateWithCodes(window, absoluteSampleIdx)

		// Decode symbol (Component 6: Symbol Decoder)
		symbol := d.decodeSymbol(correlations)

		// Debug: log first few symbols
		if d.symbolsProcessed < 8 {
			// Check window samples
			nonZeroInWindow := 0
			for _, s := range window {
				if s != 0 {
					nonZeroInWindow++
				}
			}
			// Note: can't use t.Logf here, but we can check correlations
		}

		if symbol >= 0 {
			// Write symbol to BitBuffer
			d.bitBuffer.Write(byte(symbol))

			// Convert bits to bytes (Component 7: Symbol Demapper)
			// Only extract bytes when we have complete bytes (multiples of 8 bits)
			bitsWritten := d.bitBuffer.GetWrittenBits()
			completeBytes := (bitsWritten / 8) * 8
			if completeBytes > d.bytesExtracted*8 {
				allBytes := d.bitBuffer.ToBytes()
				availableBytes := completeBytes / 8
				if availableBytes > d.bytesExtracted && bytesDecoded < len(data) {
					// Copy new complete bytes to output
					newBytes := allBytes[d.bytesExtracted:availableBytes]
					remaining := data[bytesDecoded:]
					copied := copy(remaining, newBytes)
					bytesDecoded += copied
					d.bytesExtracted += copied

					// Compact buffer after extracting bytes
					if d.bytesExtracted*8 >= completeBytes && completeBytes > 0 {
						d.bitBuffer.SetPosition(completeBytes)
						d.bitBuffer.Compact()
						d.bytesExtracted = 0
					}
				}
			}
		}

		// Advance to next symbol
		d.symbolsProcessed++
	}

	return bytesDecoded
}

// extractSymbolWindowAt extracts one symbol period from the buffer at a specific sample offset
func (d *Demodulator) extractSymbolWindowAt(symbolStartSample int64) []float32 {
	window := make([]float32, d.samplesPerSymbol)

	// Calculate buffer index for the start of this symbol
	// bufferIndex points to the next write position (where we'll write next)
	// totalSamples is the total number of samples we've received
	// symbolStartSample is the sample index relative to the start of received data

	// Check if symbol is in buffer
	if symbolStartSample < 0 || symbolStartSample+int64(d.samplesPerSymbol) > d.totalSamples {
		// Symbol is not fully in buffer
		return window
	}

	// Calculate position in ring buffer
	// If buffer hasn't wrapped: samples start at index 0
	// If buffer has wrapped: oldest sample is at bufferIndex
	var symbolStartInBuffer int
	if d.totalSamples <= int64(d.bufferSize) {
		// Buffer hasn't wrapped yet - samples are stored starting at 0
		symbolStartInBuffer = int(symbolStartSample)
	} else {
		// Buffer has wrapped - calculate offset from bufferIndex
		oldestSampleInBuffer := d.totalSamples - int64(d.bufferSize)
		offsetFromOldest := symbolStartSample - oldestSampleInBuffer
		if offsetFromOldest < 0 {
			// Symbol is too old, not in buffer
			return window
		}
		symbolStartInBuffer = (d.bufferIndex + int(offsetFromOldest)) % d.bufferSize
	}

	for i := range d.samplesPerSymbol {
		idx := (symbolStartInBuffer + i) % d.bufferSize
		window[i] = d.sampleBuffer[idx]
	}

	return window
}

// correlateWithCodes computes correlation with all codes (Component 5)
// Returns correlation values for each code
func (d *Demodulator) correlateWithCodes(samples []float32, symbolStartSample int64) []float32 {
	correlations := make([]float32, d.modulationOrder)

	// For each code, compute correlation
	for codeIdx := range d.modulationOrder {
		var code Code
		if d.modulationOrder == 2 {
			code = d.code
		} else {
			code = d.codeTable[codeIdx]
		}

		// Coherent correlation with carrier demodulation
		corr := d.computeCorrelation(samples, code, symbolStartSample)
		correlations[codeIdx] = corr
	}

	return correlations
}

// computeCorrelation computes correlation between received samples and code
// Following SPEC.md Component 5, Section B (Coherent Correlation)
func (d *Demodulator) computeCorrelation(samples []float32, code Code, symbolStartSample int64) float32 {
	var sum float32

	// Calculate starting sample index for this symbol
	startSampleIdx := symbolStartSample

	// For each chip
	for chipIdx := 0; chipIdx < d.spreadingFactor; chipIdx++ {
		chipStart := chipIdx * d.samplesPerChip
		chipValue := code.GetBipolar(chipIdx)

		// Integrate over chip samples
		for s := 0; s < d.samplesPerChip; s++ {
			sampleIdx := chipStart + s
			if sampleIdx >= len(samples) {
				break
			}

			sample := samples[sampleIdx]

			// Carrier demodulation (if carrier present)
			if d.carrierFreq > 0 {
				// Calculate carrier phase for this sample
				globalSampleIdx := startSampleIdx + int64(sampleIdx)
				phase := 2.0 * math32.Pi * d.carrierFreq * float32(globalSampleIdx) / d.sampleRate
				carrier := math32.Cos(phase)
				sample = sample * carrier
			}

			// Apply matched filter (pulse shape)
			pulseWeight := d.pulseShape[s]

			// Correlate: sample × pulse × code
			sum += sample * pulseWeight * chipValue
		}
	}

	// Normalize by number of samples
	return sum / float32(d.samplesPerSymbol)
}

// decodeSymbol decodes symbol from correlation values (Component 6)
// Returns symbol index or -1 if invalid
func (d *Demodulator) decodeSymbol(correlations []float32) int {
	if d.modulationOrder == 2 {
		// Binary DSSS: Maximum Likelihood decision
		// Symbol 0: negative correlation, Symbol 1: positive correlation
		corr := correlations[0]
		if corr < 0 {
			return 0
		}
		return 1
	} else {
		// M-ary DSSS: Maximum Likelihood (arg max |ρ_i|)
		maxIdx := 0
		maxAbs := math32.Abs(correlations[0])
		for i := 1; i < len(correlations); i++ {
			abs := math32.Abs(correlations[i])
			if abs > maxAbs {
				maxAbs = abs
				maxIdx = i
			}
		}
		return maxIdx
	}
}

// advanceBuffer is no longer needed - we track symbolsProcessed instead
