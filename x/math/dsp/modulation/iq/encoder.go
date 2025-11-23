package iq

import (
	"math"

	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// Encoder handles IQ modulation for digital communication
type Encoder struct {
	sampleRate  float32
	carrierFreq float32
}

// NewEncoder creates a new IQ encoder
func NewEncoder(sampleRate, carrierFreq float32) *Encoder {
	return &Encoder{
		sampleRate:  sampleRate,
		carrierFreq: carrierFreq,
	}
}

// Modulate modulates baseband I/Q signals onto carrier frequency
// I and Q are the in-phase and quadrature components
// Returns the modulated signal
func (iq *Encoder) Modulate(I, Q vecTypes.Vector) vec.Vector {
	iVec := I.(vec.Vector)
	qVec := Q.(vec.Vector)
	iData := []float32(iVec)
	qData := []float32(qVec)

	if len(iData) != len(qData) {
		panic("IQEncoder.Modulate: I and Q vectors must have same length")
	}

	length := len(iData)
	result := vec.New(length)
	resultData := []float32(result)

	omega := 2 * math.Pi * iq.carrierFreq / iq.sampleRate

	for i := 0; i < length; i++ {
		carrierI := float32(math.Cos(float64(omega * float32(i))))
		carrierQ := float32(math.Sin(float64(omega * float32(i))))
		resultData[i] = iData[i]*carrierI - qData[i]*carrierQ
	}

	return result
}

// GenerateIQSignal generates an IQ modulated signal from I and Q vectors
func (iq *Encoder) GenerateIQSignal(I, Q vecTypes.Vector) vec.Vector {
	return iq.Modulate(I, Q)
}

