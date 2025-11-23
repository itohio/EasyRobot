package tc

import (
	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
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

// Modulator modulates SMPTE timecode using a generic modulation scheme.
// This allows using BPSK, QPSK, DSSS, etc. on top of timecode data.
type Modulator struct {
	modulator modulation.Modulator
	frameRate FrameRate
}

// NewModulator creates a new timecode modulator using a generic modulation scheme.
// The underlying modulator (e.g., BPSK, QPSK) is provided for composability.
func NewModulator(modulator modulation.Modulator, frameRate FrameRate) *Modulator {
	if modulator == nil {
		panic("modulator must be provided - use mpsk.NewModulator or dsss.NewModulator")
	}

	return &Modulator{
		modulator: modulator,
		frameRate: frameRate,
	}
}

// Encode encodes a SMPTE timecode to amplitude samples using the underlying modulator.
// Returns the number of samples written.
// Returns expected samples length if amplitudes is nil.
func (tcm *Modulator) Encode(amplitudes vecTypes.Vector, timecode *SMPTETimeCode) int {
	// Marshal timecode to LTC frame (80 bits = 10 bytes)
	frameBytes, err := MarshalLTC(timecode)
	if err != nil {
		return 0
	}

	// Use underlying modulator to encode bytes to samples
	return tcm.modulator.Encode(amplitudes, frameBytes)
}

// EncodeMultiple encodes multiple timecodes sequentially.
func (tcm *Modulator) EncodeMultiple(amplitudes vecTypes.Vector, timecodes []*SMPTETimeCode) int {
	if amplitudes == nil {
		// Calculate expected length
		totalSamples := 0
		for _, tc := range timecodes {
			frameBytes, err := MarshalLTC(tc)
			if err != nil {
				continue
			}
			samples := tcm.modulator.Encode(nil, frameBytes)
			totalSamples += samples
		}
		return totalSamples
	}

	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)
	totalSamples := 0
	offset := 0

	for _, timecode := range timecodes {
		frameBytes, err := MarshalLTC(timecode)
		if err != nil {
			continue
		}

		// Calculate samples needed for this frame
		samplesNeeded := tcm.modulator.Encode(nil, frameBytes)
		if offset+samplesNeeded > len(ampData) {
			break // Not enough space
		}

		// Create slice for this frame
		frameSamples := vec.New(samplesNeeded)
		frameSamplesData := []float32(frameSamples)
		samplesWritten := tcm.modulator.Encode(frameSamples, frameBytes)

		// Copy to output
		copy(ampData[offset:], frameSamplesData[:samplesWritten])
		offset += samplesWritten
		totalSamples += samplesWritten
	}

	return totalSamples
}

// Reset resets the modulator state.
func (tcm *Modulator) Reset() {
	tcm.modulator.Reset()
}

// FrameRate returns the frame rate being used.
func (tcm *Modulator) FrameRate() FrameRate {
	return tcm.frameRate
}
