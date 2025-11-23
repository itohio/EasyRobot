package tc

import (
	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
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

// Demodulator demodulates samples into SMPTE timecode using a generic demodulation scheme.
type Demodulator struct {
	demodulator modulation.Demodulator
	frameRate   FrameRate
	byteBuffer  []byte
}

// NewDemodulator creates a new timecode demodulator using a generic demodulation scheme.
func NewDemodulator(demodulator modulation.Demodulator, frameRate FrameRate) *Demodulator {
	if demodulator == nil {
		panic("demodulator must be provided - use mpsk.NewDemodulator or dsss.NewDemodulator")
	}

	return &Demodulator{
		demodulator: demodulator,
		frameRate:   frameRate,
		byteBuffer:  make([]byte, 0, 20), // Buffer for partial frames
	}
}

// Decode decodes amplitude samples into SMPTE timecode structures.
// Returns the number of timecodes decoded.
// Returns expected count if timecodes is nil.
func (tcd *Demodulator) Decode(timecodes []*SMPTETimeCode, amplitudes vecTypes.Vector) int {
	if timecodes == nil {
		// Calculate expected count
		expectedBytes := tcd.demodulator.Decode(nil, amplitudes)
		if expectedBytes == 0 {
			return 0
		}
		// Each timecode frame is 10 bytes (80 bits)
		return expectedBytes / 10
	}

	if amplitudes == nil {
		return 0
	}

	// Demodulate samples to bytes
	expectedBytes := tcd.demodulator.Decode(nil, amplitudes)
	if expectedBytes == 0 {
		return 0
	}

	// Ensure buffer is large enough
	if len(tcd.byteBuffer) < expectedBytes {
		tcd.byteBuffer = make([]byte, expectedBytes)
	}

	// Demodulate to bytes
	bytesDecoded := tcd.demodulator.Decode(tcd.byteBuffer, amplitudes)
	if bytesDecoded == 0 {
		return 0
	}

	// Parse frames from byte buffer
	framesDecoded := 0
	offset := 0

	for offset+10 <= bytesDecoded && framesDecoded < len(timecodes) {
		frameData := tcd.byteBuffer[offset : offset+10]

		// Try to unmarshal frame
		tc, err := UnmarshalLTC(frameData)
		if err != nil {
			// Invalid frame - skip one byte and try again (sliding window)
			offset++
			continue
		}

		// Valid frame - copy to output
		*timecodes[framesDecoded] = *tc
		framesDecoded++
		offset += 10
	}

	// Remove processed bytes from buffer
	if offset > 0 {
		remaining := bytesDecoded - offset
		if remaining > 0 {
			copy(tcd.byteBuffer, tcd.byteBuffer[offset:bytesDecoded])
			tcd.byteBuffer = tcd.byteBuffer[:remaining]
		} else {
			tcd.byteBuffer = tcd.byteBuffer[:0]
		}
	}

	return framesDecoded
}

// DecodeStream processes samples in a streaming fashion.
func (tcd *Demodulator) DecodeStream(timecodes []*SMPTETimeCode, amplitudes vecTypes.Vector) int {
	if timecodes == nil {
		// Estimate based on current buffer + new samples
		expectedBytes := tcd.demodulator.Decode(nil, amplitudes)
		totalBytes := len(tcd.byteBuffer) + expectedBytes
		return totalBytes / 10
	}

	if amplitudes == nil {
		return 0
	}

	// Demodulate new samples to bytes
	expectedBytes := tcd.demodulator.Decode(nil, amplitudes)
	if expectedBytes == 0 {
		// Check if we have complete frames in buffer
		return tcd.parseFramesFromBuffer(timecodes)
	}

	// Ensure buffer is large enough
	newSize := len(tcd.byteBuffer) + expectedBytes
	if cap(tcd.byteBuffer) < newSize {
		newBuffer := make([]byte, len(tcd.byteBuffer), newSize*2)
		copy(newBuffer, tcd.byteBuffer)
		tcd.byteBuffer = newBuffer
	}

	// Demodulate to bytes, append to buffer
	oldLen := len(tcd.byteBuffer)
	tcd.byteBuffer = tcd.byteBuffer[:newSize]
	bytesDecoded := tcd.demodulator.Decode(tcd.byteBuffer[oldLen:], amplitudes)
	tcd.byteBuffer = tcd.byteBuffer[:oldLen+bytesDecoded]

	// Parse complete frames from buffer
	return tcd.parseFramesFromBuffer(timecodes)
}

// parseFramesFromBuffer parses complete frames from the byte buffer.
func (tcd *Demodulator) parseFramesFromBuffer(timecodes []*SMPTETimeCode) int {
	framesDecoded := 0
	offset := 0

	for offset+10 <= len(tcd.byteBuffer) && framesDecoded < len(timecodes) {
		frameData := tcd.byteBuffer[offset : offset+10]

		// Try to unmarshal frame
		tc, err := UnmarshalLTC(frameData)
		if err != nil {
			// Invalid frame - skip one byte and try again (sliding window for frame sync)
			offset++
			continue
		}

		// Valid frame - copy to output
		*timecodes[framesDecoded] = *tc
		framesDecoded++
		offset += 10
	}

	// Remove processed bytes from buffer
	if offset > 0 {
		remaining := len(tcd.byteBuffer) - offset
		if remaining > 0 {
			copy(tcd.byteBuffer, tcd.byteBuffer[offset:])
			tcd.byteBuffer = tcd.byteBuffer[:remaining]
		} else {
			tcd.byteBuffer = tcd.byteBuffer[:0]
		}
	}

	return framesDecoded
}

// Reset resets the demodulator state.
func (tcd *Demodulator) Reset() {
	tcd.demodulator.Reset()
	tcd.byteBuffer = tcd.byteBuffer[:0]
}

// FrameRate returns the frame rate being used.
func (tcd *Demodulator) FrameRate() FrameRate {
	return tcd.frameRate
}

// SetEndOfPacketCallback sets a callback to detect end of packet.
func (tcd *Demodulator) SetEndOfPacketCallback(callback func([]byte) bool) {
	tcd.demodulator.OnIsEndOfPacket(callback)
}
