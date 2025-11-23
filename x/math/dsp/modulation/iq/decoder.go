package iq

import (
	"math"

	"github.com/itohio/EasyRobot/x/math/filter"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// Decoder handles IQ demodulation for digital communication
// Optimized for real-time embedded systems (no allocations in hot paths)
type Decoder struct {
	sampleRate    float32
	carrierFreq   float32
	lowPassFilter filter.Processor[float32]

	// Pre-computed values for efficiency
	omega float32

	// Phase accumulator for continuous phase tracking
	phaseAcc float32
}

// NewDecoder creates a new IQ decoder
func NewDecoder(sampleRate, carrierFreq float32) *Decoder {
	return &Decoder{
		sampleRate:  sampleRate,
		carrierFreq: carrierFreq,
		omega:       2 * math.Pi * carrierFreq / sampleRate,
	}
}

// SetFilter sets the filter for baseband filtering
func (iq *Decoder) SetFilter(f filter.Processor[float32]) {
	iq.lowPassFilter = f
}

// Reset resets the decoder state
func (iq *Decoder) Reset() {
	iq.phaseAcc = 0
	if iq.lowPassFilter != nil {
		iq.lowPassFilter.Reset()
	}
}

// Demodulate demodulates a modulated signal into I/Q components
// Returns I and Q components (allocates output vectors, but no allocations in processing loop)
func (iq *Decoder) Demodulate(signal vecTypes.Vector) (I, Q vec.Vector) {
	signalData := []float32(signal.(vec.Vector))
	length := len(signalData)

	I = vec.New(length)
	Q = vec.New(length)

	iData := []float32(I)
	qData := []float32(Q)

	// Process samples with pre-computed omega
	for i := 0; i < length; i++ {
		phase := iq.omega * float32(i)
		carrierI := float32(math.Cos(float64(phase)))
		carrierQ := float32(math.Sin(float64(phase)))
		iData[i] = signalData[i] * carrierI
		qData[i] = -signalData[i] * carrierQ
	}

	// Apply low-pass filtering if filter is set
	if iq.lowPassFilter != nil {
		for i := 0; i < length; i++ {
			iData[i] = iq.lowPassFilter.Process(iData[i])
			qData[i] = iq.lowPassFilter.Process(qData[i])
		}
	}

	return I, Q
}

// ProcessSample processes a single sample and updates I/Q (for streaming)
// Returns I and Q values (no allocations)
func (iq *Decoder) ProcessSample(sample float32) (iVal, qVal float32) {
	carrierI := float32(math.Cos(float64(iq.phaseAcc)))
	carrierQ := float32(math.Sin(float64(iq.phaseAcc)))

	iVal = sample * carrierI
	qVal = -sample * carrierQ

	iq.phaseAcc += iq.omega
	if iq.phaseAcc > 2*math.Pi {
		iq.phaseAcc -= 2 * math.Pi
	}

	if iq.lowPassFilter != nil {
		iVal = iq.lowPassFilter.Process(iVal)
		qVal = iq.lowPassFilter.Process(qVal)
	}

	return iVal, qVal
}

