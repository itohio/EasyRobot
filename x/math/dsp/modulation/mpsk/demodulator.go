package mpsk

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

// Demodulator handles Phase Shift Keying demodulation
// Improved with phase recovery, timing recovery, signal detection, and AGC
type Demodulator struct {
	sampleRate      float32
	carrierFreq     float32
	symbolRate      float32
	modulationOrder int
	lowPassFilter   filter.Processor[float32]

	// Phase and timing recovery
	costasLoop     *modulation.CostasLoop
	timingRecovery *modulation.TimingRecovery

	// Signal detection and AGC
	signalDetector *modulation.SignalDetector
	agc            *agcFilter.Filter

	// Pre-computed values
	samplesPerSymbol int
	omega            float32

	// State
	iqDecoder *iq.Decoder
	phaseAcc  float32

	// Streaming state
	sampleBuffer   []float32            // Buffer for partial symbols
	bitBuffer      buffer.Bits[byte]    // Buffer for symbols
	bytesExtracted int

	// Lock state
	locked        bool
	wasLocked     bool // Track previous lock state to detect lock transition
	startDecoding bool // Track if we've started decoding (to skip padding)
}

// NewDemodulator creates a new PSK demodulator
func NewDemodulator(sampleRate, carrierFreq, symbolRate float32, modulationOrder int) *Demodulator {
	samplesPerSymbol := int(sampleRate / symbolRate)

	// Initialize Costas loop with reasonable gains
	alpha := float32(0.1) // Proportional gain
	beta := float32(0.01) // Integral gain
	costasLoop := modulation.NewCostasLoop(modulationOrder, alpha, beta)

	// Initialize timing recovery
	timingRecovery := modulation.NewTimingRecovery(float32(samplesPerSymbol), 0.5)

	// Initialize signal detector (window size = 2 symbols worth of samples)
	detectorWindow := samplesPerSymbol * 2
	signalDetector := modulation.NewSignalDetector(detectorWindow, modulation.MinSignalEnergy*10)

	// Initialize AGC
	agc := agcFilter.New(0.5, 0.01) // 0.01 is AGC time constant
	agc.SetSampleRate(sampleRate)

	decoder := &Demodulator{
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
		iqDecoder:        iq.NewDecoder(sampleRate, carrierFreq),
		sampleBuffer:     make([]float32, 0, samplesPerSymbol*2),
		bitBuffer:        buffer.NewBits[byte](int(math.Log2(float64(modulationOrder)))), // bits per symbol
		bytesExtracted:   0,
	}

	return decoder
}

// SetFilter sets the filter for baseband filtering
// SetFilter sets the low-pass filter for demodulation
//
// Deprecated: This method is not part of the Demodulator interface.
func (psk *Demodulator) SetFilter(f filter.Processor[float32]) {
	psk.lowPassFilter = f
	if psk.iqDecoder != nil {
		psk.iqDecoder.SetFilter(f)
	}
}

// Reset resets the demodulator state
func (psk *Demodulator) Reset() {
	psk.phaseAcc = 0
	psk.locked = false
	psk.wasLocked = false
	psk.startDecoding = false
	psk.sampleBuffer = psk.sampleBuffer[:0]
	psk.bitBuffer.Clear()
	psk.bytesExtracted = 0
	if psk.costasLoop != nil {
		psk.costasLoop.Reset()
	}
	if psk.timingRecovery != nil {
		psk.timingRecovery.Reset()
	}
	if psk.signalDetector != nil {
		psk.signalDetector.Reset()
	}
	if psk.agc != nil {
		psk.agc.Reset()
	}
	if psk.iqDecoder != nil {
		psk.iqDecoder.Reset()
	}
	if psk.lowPassFilter != nil {
		psk.lowPassFilter.Reset()
	}
}

// IsLocked returns true if demodulator is locked onto signal
//
// Deprecated: This method is not part of the Demodulator interface.
func (psk *Demodulator) IsLocked() bool {
	return psk.locked && psk.signalDetector != nil && psk.signalDetector.IsLocked()
}

// Demodulate demodulates PSK signal into symbols
//
// Deprecated: This method is not part of the Demodulator interface.
func (psk *Demodulator) Demodulate(signal vecTypes.Vector) vec.Vector {
	// First, IQ demodulate
	I, Q := psk.iqDecoder.Demodulate(signal)
	iData := []float32(I)
	qData := []float32(Q)

	// Apply phase recovery using Costas loop
	// Process samples to lock phase
	for i := 0; i < len(iData) && i < 1000; i++ { // Lock on first 1000 samples
		psk.costasLoop.Update(iData[i], qData[i])
	}

	// Apply phase correction
	phaseCorr := psk.costasLoop.GetPhase()
	cosPhase := float32(math.Cos(float64(phaseCorr)))
	sinPhase := float32(math.Sin(float64(phaseCorr)))

	// Rotate I/Q by phase correction
	for i := range iData {
		iRot := iData[i]*cosPhase - qData[i]*sinPhase
		qRot := iData[i]*sinPhase + qData[i]*cosPhase
		iData[i] = iRot
		qData[i] = qRot
	}

	// Sample at symbol rate with timing recovery
	numSymbols := len(iData) / psk.samplesPerSymbol
	symbols := vec.New(numSymbols)
	symbolData := []float32(symbols)

	for i := 0; i < numSymbols; i++ {
		sampleIdx := i * psk.samplesPerSymbol
		if sampleIdx >= len(iData) {
			break
		}

		// Symbol decision based on modulation order
		if psk.modulationOrder == 2 {
			// BPSK
			if iData[sampleIdx] > 0 {
				symbolData[i] = 1.0
			} else {
				symbolData[i] = 0.0
			}
		} else if psk.modulationOrder == 4 {
			// QPSK - map to 0,1,2,3
			iBit := 0
			qBit := 0
			if iData[sampleIdx] > 0 {
				iBit = 1
			}
			if qData[sampleIdx] > 0 {
				qBit = 1
			}
			symbolData[i] = float32(iBit*2 + qBit)
		} else {
			// Higher order PSK - use phase angle
			phase := float32(math.Atan2(float64(qData[sampleIdx]), float64(iData[sampleIdx])))
			if phase < 0 {
				phase += 2 * math.Pi
			}
			symbolData[i] = phase / (2 * math.Pi / float32(psk.modulationOrder))
		}
	}

	return symbols
}

// ProcessChunk processes a chunk of samples for streaming decoding
// Returns number of bytes decoded and whether more data is expected
//
// Deprecated: This method is not part of the Demodulator interface.
// Use Decode instead.
func (psk *Demodulator) ProcessChunk(samples []float32, output []byte) (bytesDecoded int, hasMore bool) {
	// Apply AGC and signal detection
	for i := range samples {
		// AGC
		if psk.agc != nil {
			samples[i] = psk.agc.Process(samples[i])
		}

		// Signal detection
		if psk.signalDetector != nil {
			psk.signalDetector.Update(samples[i])
			psk.locked = psk.signalDetector.IsLocked()
		}
	}

	// Check if we just locked (transition from unlocked to locked)
	justLocked := !psk.wasLocked && psk.locked
	if justLocked {
		// When we first lock, clear the sample buffer to discard padding samples
		// Keep only the last few samples for continuity (enough for signal detector to work)
		keepSamples := psk.samplesPerSymbol * 2 // Keep 2 symbols worth for better continuity
		if len(psk.sampleBuffer) > keepSamples {
			psk.sampleBuffer = psk.sampleBuffer[len(psk.sampleBuffer)-keepSamples:]
		} else {
			// If buffer is small, keep it all (might be all signal, no padding)
		}
		psk.startDecoding = true
	}
	psk.wasLocked = psk.locked

	// Always add samples to buffer (even if not locked yet)
	// This allows signal detector to accumulate samples and lock
	// We'll only decode symbols when locked
	psk.sampleBuffer = append(psk.sampleBuffer, samples...)

	// Only decode symbols if locked and we've started decoding (to skip padding)
	if !psk.locked || !psk.startDecoding {
		return 0, len(psk.sampleBuffer) >= psk.samplesPerSymbol
	}

	// Process complete symbols
	bytesDecoded = 0

	for len(psk.sampleBuffer) >= psk.samplesPerSymbol {
		// Extract one symbol
		symbolSamples := psk.sampleBuffer[:psk.samplesPerSymbol]
		psk.sampleBuffer = psk.sampleBuffer[psk.samplesPerSymbol:]

		// IQ demodulate symbol
		var iAcc, qAcc float32
		for _, sample := range symbolSamples {
			iVal, qVal := psk.iqDecoder.ProcessSample(sample)
			iAcc += iVal
			qAcc += qVal
		}
		iAcc /= float32(len(symbolSamples))
		qAcc /= float32(len(symbolSamples))

		// Apply filter if set
		if psk.lowPassFilter != nil {
			iAcc = psk.lowPassFilter.Process(iAcc)
			qAcc = psk.lowPassFilter.Process(qAcc)
		}

		// Update Costas loop
		psk.costasLoop.Update(iAcc, qAcc)

		// Apply phase correction
		phaseCorr := psk.costasLoop.GetPhase()
		cosPhase := float32(math.Cos(float64(phaseCorr)))
		sinPhase := float32(math.Sin(float64(phaseCorr)))
		iRot := iAcc*cosPhase - qAcc*sinPhase
		qRot := iAcc*sinPhase + qAcc*cosPhase

		// Symbol decision
		var symbol int
		if psk.modulationOrder == 2 {
			if iRot > 0 {
				symbol = 1
			} else {
				symbol = 0
			}
		} else if psk.modulationOrder == 4 {
			iBit := 0
			qBit := 0
			if iRot > 0 {
				iBit = 1
			}
			if qRot > 0 {
				qBit = 1
			}
			symbol = iBit*2 + qBit
		} else {
			phase := float32(math.Atan2(float64(qRot), float64(iRot)))
			if phase < 0 {
				phase += 2 * math.Pi
			}
			symbol = int(phase / (2 * math.Pi / float32(psk.modulationOrder)))
		}

		// Write symbol to bit buffer
		psk.bitBuffer.Write(byte(symbol))

		// Extract bytes when we have enough
		allBytes := psk.bitBuffer.ToBytes()
		if len(allBytes) > psk.bytesExtracted {
			// Copy new bytes to output
			newBytes := allBytes[psk.bytesExtracted:]
			remaining := output[bytesDecoded:]
			copied := copy(remaining, newBytes)
			bytesDecoded += copied
			psk.bytesExtracted += copied

			// If we extracted all complete bytes, compact the buffer
			bitsWritten := psk.bitBuffer.GetWrittenBits()
			completeBytes := (bitsWritten >> 3) * 8
			if psk.bytesExtracted*8 >= completeBytes && completeBytes > 0 {
				// Move remaining bits to beginning
				psk.bitBuffer.SetPosition(completeBytes)
				psk.bitBuffer.Compact()
				psk.bytesExtracted = 0
			}
		}
	}

	return bytesDecoded, len(psk.sampleBuffer) > 0
}

// Decode decodes amplitude vector into bytes
// Implements modulation.Demodulator interface
// Returns the number of bytes written
// Returns expected data length if data is nil
func (psk *Demodulator) Decode(data []byte, amplitudes vecTypes.Vector) int {
	if data == nil {
		// Calculate expected length
		bitsPerSymbol := int(math.Log2(float64(psk.modulationOrder)))
		symbolsAvailable := len(psk.sampleBuffer) / psk.samplesPerSymbol
		bitsAvailable := symbolsAvailable * bitsPerSymbol
		return bitsAvailable / 8
	}

	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)

	bytesDecoded, _ := psk.ProcessChunk(ampData, data)
	return bytesDecoded
}
