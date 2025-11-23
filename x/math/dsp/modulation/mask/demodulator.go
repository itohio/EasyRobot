package mask

import (
	"math"

	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/iq"
	"github.com/itohio/EasyRobot/x/math/filter"
	agcFilter "github.com/itohio/EasyRobot/x/math/filter/agc"
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// Demodulator handles Amplitude Shift Keying demodulation for digital communication.
// Supports various ASK constellations (2-ASK, 4-ASK, 8-ASK, etc.).
type Demodulator struct {
	sampleRate       float32
	carrierFreq      float32
	symbolRate       float32
	modulationOrder  int
	samplesPerSymbol int
	lowPassFilter    filter.Processor[float32]

	// Phase and timing recovery
	costasLoop     *modulation.CostasLoop
	timingRecovery *modulation.TimingRecovery

	// Signal detection and AGC
	signalDetector *modulation.SignalDetector
	agc            *agcFilter.Filter

	// State
	iqDemodulator  *iq.Decoder
	currentPhase   float32
	bitBuffer      buffer.Bits[byte]
	bytesExtracted int

	// Streaming state
	sampleBuffer []float32

	// Pre-computed values
	omega float32

	// Lock state
	locked        bool
	wasLocked     bool // Track previous lock state to detect lock transition
	startDecoding bool // Track if we've started decoding (to skip padding)
}

// NewDemodulator creates a new ASK demodulator.
// sampleRate: sampling frequency in Hz
// carrierFreq: carrier frequency in Hz
// symbolRate: symbol rate in baud
// modulationOrder: number of amplitude levels (2, 4, 8, 16, etc.)
func NewDemodulator(sampleRate, carrierFreq, symbolRate float32, modulationOrder int) *Demodulator {
	samplesPerSymbol := int(sampleRate / symbolRate)

	// Initialize Costas loop for phase recovery
	costasLoop := modulation.NewCostasLoop(2, 0.1, 0.01)

	// Initialize timing recovery
	timingRecovery := modulation.NewTimingRecovery(float32(samplesPerSymbol), 0.5)

	// Initialize signal detector
	// For higher order ASK, use lower threshold since signal energy is distributed across more levels
	detectorWindow := samplesPerSymbol * 4
	energyThreshold := float32(modulation.MinSignalEnergy * 10)
	if modulationOrder == 8 {
		// Lower threshold for 8-ASK since amplitude levels are closer together
		energyThreshold = float32(modulation.MinSignalEnergy * 5)
	} else if modulationOrder >= 16 {
		// Even lower threshold for 16-ASK since amplitude levels are very close together
		energyThreshold = float32(modulation.MinSignalEnergy * 3)
	}
	signalDetector := modulation.NewSignalDetector(detectorWindow, energyThreshold)
	// Lower minimum SNR requirement for higher order ASK
	if modulationOrder >= 16 {
		signalDetector.SetMinSNR(2.0) // Lower SNR requirement for 16-ASK
	} else if modulationOrder >= 8 {
		signalDetector.SetMinSNR(3.0) // Lower SNR requirement for 8-ASK
	}

	// Initialize AGC
	agc := agcFilter.New(0.5, 0.01)
	agc.SetSampleRate(sampleRate)

	return &Demodulator{
		sampleRate:       sampleRate,
		carrierFreq:      carrierFreq,
		symbolRate:       symbolRate,
		modulationOrder:  modulationOrder,
		samplesPerSymbol: samplesPerSymbol,
		omega:            2 * math.Pi * carrierFreq / sampleRate,
		costasLoop:       costasLoop,
		timingRecovery:   timingRecovery,
		signalDetector:   signalDetector,
		agc:              agc,
		iqDemodulator:    iq.NewDecoder(sampleRate, carrierFreq),
		bitBuffer:        buffer.NewBits[byte](int(math.Log2(float64(modulationOrder)))), // bits per symbol
		bytesExtracted:   0,
		sampleBuffer:     make([]float32, 0, samplesPerSymbol*2),
	}
}

// SetFilter sets the filter for baseband filtering.
// SetFilter sets the low-pass filter for demodulation
//
// Deprecated: This method is not part of the Demodulator interface.
func (mask *Demodulator) SetFilter(f filter.Processor[float32]) {
	mask.lowPassFilter = f
	if mask.iqDemodulator != nil {
		mask.iqDemodulator.SetFilter(f)
	}
}

// Reset resets the demodulator state.
func (mask *Demodulator) Reset() {
	mask.locked = false
	mask.wasLocked = false
	mask.startDecoding = false
	mask.bitBuffer.Clear()
	mask.bytesExtracted = 0
	mask.sampleBuffer = mask.sampleBuffer[:0]
	mask.currentPhase = 0
	if mask.costasLoop != nil {
		mask.costasLoop.Reset()
	}
	if mask.timingRecovery != nil {
		mask.timingRecovery.Reset()
	}
	if mask.signalDetector != nil {
		mask.signalDetector.Reset()
	}
	if mask.agc != nil {
		mask.agc.Reset()
	}
	if mask.iqDemodulator != nil {
		mask.iqDemodulator.Reset()
	}
	if mask.lowPassFilter != nil {
		mask.lowPassFilter.Reset()
	}
}

// Decode decodes amplitude vector into bytes.
// Implements modulation.Demodulator interface.
// Returns the number of bytes written.
// Returns expected data length if data is nil.
func (mask *Demodulator) Decode(data []byte, amplitudes vecTypes.Vector) int {
	if data == nil {
		// Calculate expected length
		bitsPerSymbol := int(math.Log2(float64(mask.modulationOrder)))
		symbolsAvailable := len(mask.sampleBuffer) / mask.samplesPerSymbol
		bitsAvailable := symbolsAvailable * bitsPerSymbol
		return bitsAvailable / 8
	}

	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)

	bytesDecoded, _ := mask.ProcessChunk(ampData, data)
	return bytesDecoded
}

// ProcessChunk processes a chunk of samples for streaming decoding.
//
// Deprecated: This method is not part of the Demodulator interface.
// Use Decode instead.
func (mask *Demodulator) ProcessChunk(samples []float32, output []byte) (bytesDecoded int, hasMore bool) {
	// Apply AGC and signal detection
	for i := range samples {
		if mask.agc != nil {
			samples[i] = mask.agc.Process(samples[i])
		}
		if mask.signalDetector != nil {
			mask.signalDetector.Update(samples[i])
			mask.locked = mask.signalDetector.IsLocked()
		}
	}

	// Check if we just locked (transition from unlocked to locked)
	justLocked := !mask.wasLocked && mask.locked
	if justLocked {
		// When we first lock, start decoding
		// Don't clear the buffer - the signal detector has already filtered out most padding
		// by requiring consistent signal energy. The remaining samples should be mostly signal.
		mask.startDecoding = true
	}
	mask.wasLocked = mask.locked

	// Always add samples to buffer (even if not locked yet)
	// This allows signal detector to accumulate samples and lock
	// We'll only decode symbols when locked
	mask.sampleBuffer = append(mask.sampleBuffer, samples...)

	// Only decode symbols if locked and we've started decoding (to skip padding)
	if !mask.locked || !mask.startDecoding {
		return 0, len(mask.sampleBuffer) >= mask.samplesPerSymbol
	}

	bytesDecoded = 0

	// Process complete symbols
	for len(mask.sampleBuffer) >= mask.samplesPerSymbol {
		// Extract one symbol
		symbolSamples := mask.sampleBuffer[:mask.samplesPerSymbol]
		mask.sampleBuffer = mask.sampleBuffer[mask.samplesPerSymbol:]

		// Demodulate symbol to get amplitude
		amplitude := mask.demodulateSymbol(symbolSamples)

		// Convert amplitude back to symbol and write to bit buffer
		symbol := mask.quantizeAmplitude(amplitude)
		mask.bitBuffer.Write(byte(symbol))

		// Extract bytes when we have enough
		allBytes := mask.bitBuffer.ToBytes()
		if len(allBytes) > mask.bytesExtracted {
			// Copy new bytes to output
			newBytes := allBytes[mask.bytesExtracted:]
			remaining := output[bytesDecoded:]
			copied := copy(remaining, newBytes)
			bytesDecoded += copied
			mask.bytesExtracted += copied

			// If we extracted all complete bytes, compact the buffer
			bitsWritten := mask.bitBuffer.GetWrittenBits()
			completeBytes := (bitsWritten >> 3) * 8
			if mask.bytesExtracted*8 >= completeBytes && completeBytes > 0 {
				// Move remaining bits to beginning
				mask.bitBuffer.SetPosition(completeBytes)
				mask.bitBuffer.Compact()
				mask.bytesExtracted = 0
			}
		}
	}

	return bytesDecoded, len(mask.sampleBuffer) > 0
}

// demodulateSymbol extracts amplitude from symbol samples.
func (mask *Demodulator) demodulateSymbol(samples []float32) float32 {
	// For ASK, we can use direct envelope detection (rectification + averaging)
	// This is simpler and more robust than IQ demodulation for amplitude-only modulation
	var envelopeAcc float32

	for _, sample := range samples {
		// Rectify (absolute value) to get envelope
		// For ASK, the signal is amplitude-modulated, so the envelope is the absolute value
		envelope := float32(math.Abs(float64(sample)))
		envelopeAcc += envelope
	}

	// Average envelope over symbol duration
	return envelopeAcc / float32(len(samples))
}

// amplitudeToBits converts amplitude value back to bits for M-ASK.
func (mask *Demodulator) amplitudeToBits(amplitude float32) []byte {
	// Quantize amplitude to nearest constellation point
	symbol := mask.quantizeAmplitude(amplitude)

	// Convert symbol to bits
	bitsPerSymbol := int(math.Log2(float64(mask.modulationOrder)))
	bits := make([]byte, bitsPerSymbol)

	for i := bitsPerSymbol - 1; i >= 0; i-- {
		bits[i] = byte(symbol & 1)
		symbol >>= 1
	}

	return bits
}

// quantizeAmplitude maps continuous amplitude value to nearest constellation point.
func (mask *Demodulator) quantizeAmplitude(amplitude float32) int {
	if mask.modulationOrder == 2 {
		// 2-ASK (OOK): threshold at half of max expected amplitude
		// This is a simplified threshold - in practice, AGC should normalize
		if amplitude < 0.3 {
			return 0
		}
		return 1
	}

	// Multi-level ASK: map to nearest level
	// Normalize amplitude to [0, 1] range (assuming AGC has normalized)
	normalized := amplitude
	if normalized < 0 {
		normalized = 0
	}
	if normalized > 1 {
		normalized = 1
	}

	// Map to one of modulationOrder levels
	// Divide [0, 1] into modulationOrder equal bins: [0, 1/modOrder), [1/modOrder, 2/modOrder), ..., [(modOrder-1)/modOrder, 1]
	// For normalized=0.75 and modOrder=4: int(0.75 * 4) = 3 (maps to bin 3, which is level 3)
	level := int(normalized * float32(mask.modulationOrder))
	if level < 0 {
		level = 0
	}
	if level >= mask.modulationOrder {
		level = mask.modulationOrder - 1
	}

	return level
}

// IsLocked returns true if demodulator is locked onto signal.
// IsLocked returns true if demodulator is locked onto signal
//
// Deprecated: This method is not part of the Demodulator interface.
func (mask *Demodulator) IsLocked() bool {
	return mask.locked && mask.signalDetector != nil && mask.signalDetector.IsLocked()
}
