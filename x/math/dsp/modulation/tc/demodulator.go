package tc

import (
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// LTCDemodulator demodulates LTC (Linear Timecode) audio samples to SMPTE timecode.
// Uses bi-phase mark decoding.
type LTCDemodulator struct {
	sampleRate float32
	frameRate  FrameRate
	byteBuffer []byte
}

// NewLTCDemodulator creates a new LTC demodulator for SMPTE timecode.
func NewLTCDemodulator(sampleRate float32, frameRate FrameRate) *LTCDemodulator {
	return &LTCDemodulator{
		sampleRate: sampleRate,
		frameRate:  frameRate,
		byteBuffer: make([]byte, 0, 20), // Buffer for partial frames
	}
}

// Decode decodes LTC audio samples into SMPTE timecode structures.
// Returns the number of timecodes decoded.
// Returns expected count if timecodes is nil.
func (ltcd *LTCDemodulator) Decode(timecodes []*SMPTETimeCode, amplitudes vecTypes.Vector) int {
	if timecodes == nil {
		// Calculate expected count
		// One frame = 80 bits = bi-phase encoded samples
		framePeriod := 1.0 / float32(ltcd.frameRate)
		samplesPerFrame := int(ltcd.sampleRate * framePeriod)
		if amplitudes == nil {
			return 0
		}
		ampVec := amplitudes.(vec.Vector)
		return len(ampVec) / samplesPerFrame
	}

	if amplitudes == nil {
		return 0
	}

	// Convert Vector to []float32 for bi-phase decoding
	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)

	if len(ampData) == 0 {
		return 0
	}

	// Decode bi-phase mark to bits
	frameBytes, err := DecodeBiPhase(ampData, ltcd.sampleRate, ltcd.frameRate)
	if err != nil {
		return 0
	}

	// Unmarshal LTC frame to timecode
	tc, err := UnmarshalLTC(frameBytes)
	if err != nil {
		return 0
	}

	if len(timecodes) > 0 {
		*timecodes[0] = *tc
		return 1
	}

	return 0
}

// DecodeStream processes samples in a streaming fashion.
func (ltcd *LTCDemodulator) DecodeStream(timecodes []*SMPTETimeCode, amplitudes vecTypes.Vector) int {
	// For now, use same logic as Decode
	// In a full implementation, this would handle partial frames and buffering
	return ltcd.Decode(timecodes, amplitudes)
}

// Reset resets the demodulator state.
func (ltcd *LTCDemodulator) Reset() {
	ltcd.byteBuffer = ltcd.byteBuffer[:0]
}
