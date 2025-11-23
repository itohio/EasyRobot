package qam

import (
	"math"

	"github.com/itohio/EasyRobot/x/math/filter"
	agcFilter "github.com/itohio/EasyRobot/x/math/filter/agc"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation/iq"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// Decoder handles 16-QAM demodulation (4 bits per symbol).
// Implements modulation.Demodulator interface.
// Improved with phase recovery, timing recovery, signal detection, and AGC.
type Decoder struct {
	sampleRate       float32
	carrierFreq      float32
	symbolRate       float32
	samplesPerSymbol int
	lowPassFilter    filter.Processor[float32]

	// Phase and timing recovery
	costasLoop     *modulation.CostasLoop
	timingRecovery *modulation.TimingRecovery

	// Signal detection and AGC
	signalDetector *modulation.SignalDetector
	agc            *agcFilter.Filter

	// State
	iqDecoder    *iq.Decoder
	currentPhase float32
	bitBuffer    []byte
	endOfPacket  func([]byte) bool

	// Streaming state
	sampleBuffer []float32

	// Pre-computed values
	omega float32

	// Lock state
	locked bool
}

// NewDecoder creates a new 16-QAM decoder
func NewDecoder(sampleRate, carrierFreq, symbolRate float32) *Decoder {
	samplesPerSymbol := int(sampleRate / symbolRate)

	// Initialize Costas loop for QPSK-like phase recovery
	costasLoop := modulation.NewCostasLoop(4, 0.1, 0.01)

	// Initialize timing recovery
	timingRecovery := modulation.NewTimingRecovery(float32(samplesPerSymbol), 0.5)

	// Initialize signal detector
	// For 16-QAM, use similar threshold to mask's 16-ASK approach
	detectorWindow := samplesPerSymbol * 4
	// Lower threshold for 16-QAM similar to 16-ASK (levels are close together)
	energyThreshold := float32(modulation.MinSignalEnergy * 5)
	signalDetector := modulation.NewSignalDetector(detectorWindow, energyThreshold)
	// Set minimum SNR lower to allow detection at 12dB SNR
	// The decoder should be able to decode at 12dB SNR, so detector should lock at lower SNR
	signalDetector.SetMinSNR(8.0)

	// Initialize AGC
	agc := agcFilter.New(0.5, 0.01) // 0.01 is AGC time constant
	agc.SetSampleRate(sampleRate)

	return &Decoder{
		sampleRate:       sampleRate,
		carrierFreq:      carrierFreq,
		symbolRate:       symbolRate,
		samplesPerSymbol: samplesPerSymbol,
		omega:            2 * math.Pi * carrierFreq / sampleRate,
		costasLoop:       costasLoop,
		timingRecovery:   timingRecovery,
		signalDetector:   signalDetector,
		agc:              agc,
		iqDecoder:        iq.NewDecoder(sampleRate, carrierFreq),
		bitBuffer:        make([]byte, 0, 1024),
		sampleBuffer:     make([]float32, 0, samplesPerSymbol*2),
	}
}

// SetFilter sets the filter for baseband filtering
func (qam *Decoder) SetFilter(f filter.Processor[float32]) {
	qam.lowPassFilter = f
	if qam.iqDecoder != nil {
		qam.iqDecoder.SetFilter(f)
	}
}

// Reset resets the decoder state
func (qam *Decoder) Reset() {
	qam.locked = false
	qam.bitBuffer = qam.bitBuffer[:0]
	qam.sampleBuffer = qam.sampleBuffer[:0]
	qam.currentPhase = 0
	if qam.costasLoop != nil {
		qam.costasLoop.Reset()
	}
	if qam.timingRecovery != nil {
		qam.timingRecovery.Reset()
	}
	if qam.signalDetector != nil {
		qam.signalDetector.Reset()
	}
	if qam.agc != nil {
		qam.agc.Reset()
	}
	if qam.iqDecoder != nil {
		qam.iqDecoder.Reset()
	}
	if qam.lowPassFilter != nil {
		qam.lowPassFilter.Reset()
	}
}

// IsLocked returns true if decoder is locked onto signal
func (qam *Decoder) IsLocked() bool {
	return qam.locked && qam.signalDetector != nil && qam.signalDetector.IsLocked()
}

// OnIsEndOfPacket sets a callback to check if packet is complete
func (qam *Decoder) OnIsEndOfPacket(callback func([]byte) bool) {
	qam.endOfPacket = callback
}

// ProcessChunk processes a chunk of samples for streaming decoding
func (qam *Decoder) ProcessChunk(samples []float32, output []byte) (bytesDecoded int, hasMore bool) {
	// Apply AGC and signal detection
	for i := range samples {
		if qam.agc != nil {
			samples[i] = qam.agc.Process(samples[i])
		}
		if qam.signalDetector != nil {
			qam.signalDetector.Update(samples[i])
			qam.locked = qam.signalDetector.IsLocked()
		}
	}

	// Add samples to buffer (always buffer, even if not locked yet)
	// Signal detector needs time to accumulate samples and lock
	qam.sampleBuffer = append(qam.sampleBuffer, samples...)

	// If not locked yet, don't decode but continue buffering
	if !qam.locked {
		return 0, len(qam.sampleBuffer) >= qam.samplesPerSymbol
	}

	bytesDecoded = 0

	// Process complete symbols
	for len(qam.sampleBuffer) >= qam.samplesPerSymbol {
		// Extract one symbol
		symbolSamples := qam.sampleBuffer[:qam.samplesPerSymbol]
		qam.sampleBuffer = qam.sampleBuffer[qam.samplesPerSymbol:]

		// Demodulate symbol to get I/Q values
		iVal, qVal := qam.demodulateSymbol(symbolSamples)

		// Convert I/Q back to 4 bits
		bits := qam.iqToBits(iVal, qVal)
		qam.bitBuffer = append(qam.bitBuffer, bits...)

		// Convert bits to bytes when we have enough
		for len(qam.bitBuffer) >= 8 && bytesDecoded < len(output) {
			byteVal := byte(0)
			for i := 0; i < 8; i++ {
				byteVal |= qam.bitBuffer[i] << uint(7-i)
			}
			output[bytesDecoded] = byteVal
			bytesDecoded++

			// Check if packet is complete
			if qam.endOfPacket != nil && qam.endOfPacket(output[:bytesDecoded]) {
				return bytesDecoded, false // Packet complete
			}
			qam.bitBuffer = qam.bitBuffer[8:]
		}
	}

	return bytesDecoded, len(qam.sampleBuffer) > 0
}

// Decode decodes amplitude vector into bytes.
// Implements modulation.Demodulator interface.
// Returns the number of bytes written.
// Returns expected data length if data is nil.
func (qam *Decoder) Decode(data []byte, amplitudes vecTypes.Vector) int {
	if data == nil {
		// Calculate expected length
		bitsPerSymbol := 4 // 16-QAM: 4 bits per symbol
		symbolsAvailable := len(qam.sampleBuffer) / qam.samplesPerSymbol
		bitsAvailable := symbolsAvailable * bitsPerSymbol
		return bitsAvailable / 8
	}

	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)

	bytesDecoded, _ := qam.ProcessChunk(ampData, data)
	return bytesDecoded
}

// demodulateSymbol extracts I/Q values from symbol samples
func (qam *Decoder) demodulateSymbol(samples []float32) (float32, float32) {
	// Use IQ decoder to get I/Q
	var iAcc, qAcc float32

	for _, sample := range samples {
		iVal, qVal := qam.iqDecoder.ProcessSample(sample)
		iAcc += iVal
		qAcc += qVal
	}

	// Normalize
	iAcc /= float32(len(samples))
	qAcc /= float32(len(samples))

	// Update Costas loop for phase recovery
	qam.costasLoop.Update(iAcc, qAcc)

	// Apply phase correction
	phaseCorr := qam.costasLoop.GetPhase()
	cosPhase := float32(math.Cos(float64(phaseCorr)))
	sinPhase := float32(math.Sin(float64(phaseCorr)))

	iRot := iAcc*cosPhase - qAcc*sinPhase
	qRot := iAcc*sinPhase + qAcc*cosPhase

	return iRot, qRot
}

// iqToBits converts I/Q values back to 4 bits for 16-QAM
func (qam *Decoder) iqToBits(iVal, qVal float32) []byte {
	// Quantize I/Q values back to constellation points
	iBits := qam.quantizeValue(iVal)
	qBits := qam.quantizeValue(qVal)

	// Return as [i1 i0 q1 q0]
	return []byte{
		byte((iBits >> 1) & 1), // i1
		byte(iBits & 1),        // i0
		byte((qBits >> 1) & 1), // q1
		byte(qBits & 1),        // q0
	}
}

// quantizeValue maps continuous value to nearest constellation point (0,1,2,3).
// After AGC normalization, constellation points are at approximately -1, -1/3, 1/3, 1
// (assuming amplitude=1.0 and scaling by amplitude/3.0).
// Thresholds are set to midpoints between constellation points.
func (qam *Decoder) quantizeValue(val float32) int {
	// Constellation points (after scaling): -1, -1/3, 1/3, 1
	// Thresholds at midpoints: -2/3, 0, 2/3
	if val < -2.0/3.0 {
		return 0 // Maps to -1 (bit pattern 00)
	} else if val < 0 {
		return 1 // Maps to -1/3 (bit pattern 01)
	} else if val < 2.0/3.0 {
		return 2 // Maps to 1/3 (bit pattern 10)
	} else {
		return 3 // Maps to 1 (bit pattern 11)
	}
}

