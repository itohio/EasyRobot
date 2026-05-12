package crsf

import (
	"encoding/binary"
	"errors"
)

// ParamType identifies the data type of a CRSF parameter.
type ParamType uint8

const (
	ParamUint8      ParamType = 0
	ParamInt8       ParamType = 1
	ParamUint16     ParamType = 2
	ParamInt16      ParamType = 3
	ParamUint32     ParamType = 8
	ParamInt32      ParamType = 9
	ParamFloat      ParamType = 10
	ParamTextSelect ParamType = 11
	ParamString     ParamType = 12
	ParamFolder     ParamType = 13
	ParamCommand    ParamType = 14
	ParamOutOfRange ParamType = 127
)

// ParamAccess flags for parameter read/write access.
type ParamAccess uint8

const (
	ParamHidden   ParamAccess = 0
	ParamReadOnly ParamAccess = 1
	ParamReadWrite ParamAccess = 3
)

var (
	ErrParamChunkIncomplete = errors.New("crsf: parameter chunk incomplete")
	ErrParamInvalid         = errors.New("crsf: invalid parameter data")
)

// ParamEntry represents a decoded CRSF parameter (Type 0x2B).
type ParamEntry struct {
	Dest   Address
	Origin Address
	Index  uint8
	Chunks uint8 // total number of chunks
	Chunk  uint8 // chunk index being delivered (0-based)

	// Fields below are populated from chunk 0
	Parent uint8
	Type   ParamType
	Name   string

	// Raw value payload (interpretation depends on Type)
	Value []byte
}

// ParamRead is a request to read a parameter entry (Type 0x2C).
type ParamRead struct {
	Dest   Address
	Origin Address
	Index  uint8
	Chunk  uint8
}

// ParamWrite is a request to write a parameter value (Type 0x2D).
type ParamWrite struct {
	Dest   Address
	Origin Address
	Index  uint8
	Value  []byte
}

// EncodeParamRead creates a parameter read request frame.
func EncodeParamRead(pr *ParamRead) *Frame {
	p := []byte{byte(pr.Dest), byte(pr.Origin), pr.Index, pr.Chunk}
	return &Frame{Type: TypeParamRead, Payload: p}
}

// DecodeParamRead parses a parameter read request frame.
func DecodeParamRead(f *Frame) (*ParamRead, error) {
	if f.Type != TypeParamRead {
		return nil, ErrTypeMismatch
	}
	if len(f.Payload) < 4 {
		return nil, ErrFrameTooShort
	}
	return &ParamRead{
		Dest:   Address(f.Payload[0]),
		Origin: Address(f.Payload[1]),
		Index:  f.Payload[2],
		Chunk:  f.Payload[3],
	}, nil
}

// EncodeParamWrite creates a parameter write request frame.
func EncodeParamWrite(pw *ParamWrite) *Frame {
	p := make([]byte, 3+len(pw.Value))
	p[0] = byte(pw.Dest)
	p[1] = byte(pw.Origin)
	p[2] = pw.Index
	copy(p[3:], pw.Value)
	return &Frame{Type: TypeParamWrite, Payload: p}
}

// DecodeParamWrite parses a parameter write request frame.
func DecodeParamWrite(f *Frame) (*ParamWrite, error) {
	if f.Type != TypeParamWrite {
		return nil, ErrTypeMismatch
	}
	if len(f.Payload) < 4 {
		return nil, ErrFrameTooShort
	}
	return &ParamWrite{
		Dest:   Address(f.Payload[0]),
		Origin: Address(f.Payload[1]),
		Index:  f.Payload[2],
		Value:  f.Payload[3:],
	}, nil
}

// DecodeParamEntry parses a parameter entry frame (Type 0x2B).
// Chunk 0 includes the header (parent, type, name); subsequent chunks
// carry only the remaining value bytes.
func DecodeParamEntry(f *Frame) (*ParamEntry, error) {
	if f.Type != TypeParamEntry {
		return nil, ErrTypeMismatch
	}
	if len(f.Payload) < 5 {
		return nil, ErrFrameTooShort
	}
	pe := &ParamEntry{
		Dest:   Address(f.Payload[0]),
		Origin: Address(f.Payload[1]),
		Index:  f.Payload[2],
		Chunks: f.Payload[3],
		Chunk:  f.Payload[4],
	}

	data := f.Payload[5:]
	if pe.Chunk == 0 {
		if len(data) < 3 {
			return nil, ErrParamInvalid
		}
		pe.Parent = data[0]
		pe.Type = ParamType(data[1] & 0x7F)
		nameEnd := 2
		for nameEnd < len(data) && data[nameEnd] != 0 {
			nameEnd++
		}
		pe.Name = string(data[2:nameEnd])
		if nameEnd+1 < len(data) {
			pe.Value = data[nameEnd+1:]
		}
	} else {
		pe.Value = data
	}
	return pe, nil
}

// EncodeParamEntry creates a parameter entry response frame.
func EncodeParamEntry(pe *ParamEntry) *Frame {
	var payload []byte
	if pe.Chunk == 0 {
		nameBytes := len(pe.Name) + 1 // null-terminated
		headerSize := 5 + 2 + nameBytes
		payload = make([]byte, headerSize+len(pe.Value))
		payload[0] = byte(pe.Dest)
		payload[1] = byte(pe.Origin)
		payload[2] = pe.Index
		payload[3] = pe.Chunks
		payload[4] = pe.Chunk
		payload[5] = pe.Parent
		payload[6] = byte(pe.Type)
		copy(payload[7:], pe.Name)
		// null terminator at payload[7+len(pe.Name)] is already zero
		copy(payload[7+nameBytes:], pe.Value)
	} else {
		payload = make([]byte, 5+len(pe.Value))
		payload[0] = byte(pe.Dest)
		payload[1] = byte(pe.Origin)
		payload[2] = pe.Index
		payload[3] = pe.Chunks
		payload[4] = pe.Chunk
		copy(payload[5:], pe.Value)
	}
	return &Frame{Type: TypeParamEntry, Payload: payload}
}

// ParamValueUint8 extracts a uint8 value from a parameter entry's Value field.
func ParamValueUint8(pe *ParamEntry) (value, min, max, def uint8, err error) {
	if len(pe.Value) < 4 {
		return 0, 0, 0, 0, ErrParamInvalid
	}
	return pe.Value[0], pe.Value[1], pe.Value[2], pe.Value[3], nil
}

// ParamValueUint16 extracts a uint16 value from a parameter entry's Value field.
func ParamValueUint16(pe *ParamEntry) (value, min, max, def uint16, err error) {
	if len(pe.Value) < 8 {
		return 0, 0, 0, 0, ErrParamInvalid
	}
	return binary.LittleEndian.Uint16(pe.Value[0:2]),
		binary.LittleEndian.Uint16(pe.Value[2:4]),
		binary.LittleEndian.Uint16(pe.Value[4:6]),
		binary.LittleEndian.Uint16(pe.Value[6:8]),
		nil
}

// ParamValueTextSelect extracts a text-select value (current index + semicolon-separated options).
func ParamValueTextSelect(pe *ParamEntry) (current uint8, options string, err error) {
	if len(pe.Value) < 2 {
		return 0, "", ErrParamInvalid
	}
	current = pe.Value[0]
	// Options are null-terminated string following the value byte
	optEnd := 1
	for optEnd < len(pe.Value) && pe.Value[optEnd] != 0 {
		optEnd++
	}
	return current, string(pe.Value[1:optEnd]), nil
}
