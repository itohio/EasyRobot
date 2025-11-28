package tc

import (
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// LTCModulator modulates SMPTE timecode using LTC (Linear Timecode) bi-phase mark encoding.
// This is the standard method for transmitting timecode as an audio signal.
type LTCModulator struct {
	sampleRate float32
	frameRate  FrameRate
}

// NewLTCModulator creates a new LTC modulator for SMPTE timecode.
func NewLTCModulator(sampleRate float32, frameRate FrameRate) *LTCModulator {
	return &LTCModulator{
		sampleRate: sampleRate,
		frameRate:  frameRate,
	}
}

// Encode encodes a SMPTE timecode to LTC audio samples using bi-phase mark encoding.
// Returns the number of samples written (one frame period).
// Returns expected samples length if amplitudes is nil.
func (ltc *LTCModulator) Encode(amplitudes vecTypes.Vector, timecode *SMPTETimeCode) int {
	// Marshal timecode to LTC frame (80 bits = 10 bytes)
	frameBytes, err := MarshalLTC(timecode)
	if err != nil {
		return 0
	}

	// Encode using bi-phase mark
	samples := EncodeBiPhase(frameBytes, ltc.sampleRate, ltc.frameRate)

	if amplitudes == nil {
		return len(samples)
	}

	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)
	if len(ampData) < len(samples) {
		return 0
	}

	copy(ampData, samples)
	return len(samples)
}

// EncodeMultiple encodes multiple timecodes sequentially.
func (ltc *LTCModulator) EncodeMultiple(amplitudes vecTypes.Vector, timecodes []*SMPTETimeCode) int {
	if amplitudes == nil {
		// Calculate expected length
		totalSamples := 0
		for range timecodes {
			samples := ltc.Encode(nil, &SMPTETimeCode{FrameRate: ltc.frameRate})
			totalSamples += samples
		}
		return totalSamples
	}

	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)
	totalSamples := 0
	offset := 0

	for _, tc := range timecodes {
		samplesNeeded := ltc.Encode(nil, tc)
		if offset+samplesNeeded > len(ampData) {
			break
		}

		frameSamplesVec := vec.New(samplesNeeded)
		samplesWritten := ltc.Encode(frameSamplesVec, tc)
		offset += samplesWritten
		totalSamples += samplesWritten
	}

	return totalSamples
}

// Reset resets the modulator state (stateless, no-op).
func (ltc *LTCModulator) Reset() {
	// LTC modulator is stateless
}
