package tfl

import (
	"encoding/binary"
	"math"
)

// A minimal, bounds-checked FlexBuffers reader for custom operator options:
// the root must be a map; its numeric and boolean values (inline or
// indirect, 1–8 bytes wide) are returned as float64, other values are
// skipped. Malformed data yields ErrFormat, never a panic.

// FlexBuffers value types (flexbuffers.h).
const (
	flexInt           = 1
	flexUInt          = 2
	flexFloat         = 3
	flexIndirectInt   = 6
	flexIndirectUInt  = 7
	flexIndirectFloat = 8
	flexMap           = 9
	flexBool          = 26
)

type flexReader struct {
	b  []byte
	ok bool
}

func (r *flexReader) in(pos, w int) bool {
	if pos < 0 || w <= 0 || pos+w > len(r.b) {
		r.ok = false
	}
	return r.ok
}

func (r *flexReader) uint(pos, w int) uint64 {
	if !r.in(pos, w) {
		return 0
	}
	switch w {
	case 1:
		return uint64(r.b[pos])
	case 2:
		return uint64(binary.LittleEndian.Uint16(r.b[pos:]))
	case 4:
		return uint64(binary.LittleEndian.Uint32(r.b[pos:]))
	case 8:
		return binary.LittleEndian.Uint64(r.b[pos:])
	}
	r.ok = false
	return 0
}

func (r *flexReader) int(pos, w int) int64 {
	u := r.uint(pos, w)
	sh := 64 - 8*uint(w)
	return int64(u<<sh) >> sh
}

func (r *flexReader) float(pos, w int) float64 {
	switch w {
	case 4:
		return float64(math.Float32frombits(uint32(r.uint(pos, 4))))
	case 8:
		return math.Float64frombits(r.uint(pos, 8))
	}
	r.ok = false
	return 0
}

// indirect follows the offset stored at pos (w bytes wide, back from pos).
func (r *flexReader) indirect(pos, w int) int {
	off := r.uint(pos, w)
	if off > uint64(pos) {
		r.ok = false
		return 0
	}
	return pos - int(off)
}

// cstring reads the zero-terminated key at pos.
func (r *flexReader) cstring(pos int) string {
	if !r.in(pos, 1) {
		return ""
	}
	for end := pos; end < len(r.b); end++ {
		if r.b[end] == 0 {
			return string(r.b[pos:end])
		}
	}
	r.ok = false
	return ""
}

// flexNumbers returns the numeric and boolean entries of the FlexBuffers
// map b (booleans as 0 or 1).
func flexNumbers(b []byte) (map[string]float64, error) {
	r := flexReader{b: b, ok: len(b) >= 3}
	if !r.ok {
		return nil, ErrFormat
	}
	rw := int(b[len(b)-1])
	rt := b[len(b)-2]
	root := len(b) - 2 - rw
	if rt>>2 != flexMap {
		return nil, ErrFormat
	}
	p := r.indirect(root, rw)
	w := 1 << (rt & 3)
	n := int(r.uint(p-w, w))
	kw := int(r.uint(p-2*w, w))
	keys := r.indirect(p-3*w, w)
	if !r.ok || n < 0 || n > len(b) || !r.in(p, n*w+n) || !r.in(keys, n*kw) {
		return nil, ErrFormat
	}
	out := make(map[string]float64, n)
	for k := range n {
		key := r.cstring(r.indirect(keys+k*kw, kw))
		if !r.ok {
			return nil, ErrFormat
		}
		pos, t := p+k*w, b[p+n*w+k]
		cw := 1 << (t & 3)
		var v float64
		switch t >> 2 {
		case flexInt:
			v = float64(r.int(pos, w))
		case flexUInt, flexBool:
			v = float64(r.uint(pos, w))
		case flexFloat:
			v = r.float(pos, w)
		case flexIndirectInt:
			v = float64(r.int(r.indirect(pos, w), cw))
		case flexIndirectUInt:
			v = float64(r.uint(r.indirect(pos, w), cw))
		case flexIndirectFloat:
			v = r.float(r.indirect(pos, w), cw)
		default:
			continue
		}
		if !r.ok {
			return nil, ErrFormat
		}
		out[key] = v
	}
	return out, nil
}
