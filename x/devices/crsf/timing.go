package crsf

import "encoding/binary"

const timingSubCmd byte = 0x10

// TimingCorrection represents CRSF Shot timing data (Type 0x3A, sub 0x10).
// Allows FC to synchronize PID loop to the exact moment channels arrive.
type TimingCorrection struct {
	Dest   Address
	Origin Address
	// UpdateInterval is the expected microseconds between channel frames.
	UpdateInterval int32
	// Offset is the time offset in microseconds to apply to the PID loop.
	Offset int32
}

// DecodeTimingCorrection parses a timing correction extended frame.
func DecodeTimingCorrection(f *Frame) (*TimingCorrection, error) {
	if f.Type != TypeTimingCorrect {
		return nil, ErrTypeMismatch
	}
	if len(f.Payload) < 11 { // dest(1) + origin(1) + sub(1) + interval(4) + offset(4)
		return nil, ErrFrameTooShort
	}
	p := f.Payload
	if p[2] != timingSubCmd {
		return nil, ErrTypeMismatch
	}
	return &TimingCorrection{
		Dest:           Address(p[0]),
		Origin:         Address(p[1]),
		UpdateInterval: int32(binary.LittleEndian.Uint32(p[3:7])),
		Offset:         int32(binary.LittleEndian.Uint32(p[7:11])),
	}, nil
}

// EncodeTimingCorrection creates a timing correction frame.
func EncodeTimingCorrection(tc *TimingCorrection) *Frame {
	p := make([]byte, 11)
	p[0] = byte(tc.Dest)
	p[1] = byte(tc.Origin)
	p[2] = timingSubCmd
	binary.LittleEndian.PutUint32(p[3:7], uint32(tc.UpdateInterval))
	binary.LittleEndian.PutUint32(p[7:11], uint32(tc.Offset))
	return &Frame{Type: TypeTimingCorrect, Payload: p}
}
