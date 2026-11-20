package kgraph

import (
	"encoding/binary"
	"hash/crc32"
	"math"
	"unsafe"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// File is a parsed kgraph file. Graph's parameters, the segments' kmodels
// and the extras lie inside the file's bytes, which must stay alive and
// unchanged while the File is used.
type File struct {
	// Graph is the compiled graph. The layers of KPU segments carry no
	// parameters: run it with qnet.NewRunner and the backend of NewBackend.
	Graph    qnet.Graph
	Segments []Segment
	extras   []Extra
}

// Extra returns the block tagged tag, or nil.
func (f *File) Extra(tag string) []byte {
	for _, e := range f.extras {
		if e.Tag == tag {
			return e.Data
		}
	}
	return nil
}

// ReadFloat32s decodes little-endian float32 values (an Extra written with
// Float32s) into a new slice; ok is false when b is not a whole number of
// values.
func ReadFloat32s(b []byte) (v []float32, ok bool) {
	if len(b)%4 != 0 {
		return nil, false
	}
	v = make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v, true
}

// littleEndian reports the byte order the program runs with: the int32
// tables are used in place, so the file's order must be the machine's.
func littleEndian() bool {
	x := uint16(1)
	return *(*byte)(unsafe.Pointer(&x)) == 1
}

// reader bounds-checks every access into the file.
type reader struct{ d []byte }

// at returns n bytes at off, a multiple of align; ok is false outside the
// file.
func (r reader) at(off, n uint64, align uint64) ([]byte, bool) {
	if off%align != 0 || off > uint64(len(r.d)) || n > uint64(len(r.d))-off {
		return nil, false
	}
	return r.d[off : off+n : off+n], true
}

func (r reader) u32(off int) uint32 { return binary.LittleEndian.Uint32(r.d[off:]) }

// Parse reads a kgraph file in place: data must lie at a 128-byte-aligned
// address (ErrAlign) on a little-endian machine, and may be longer than the
// file. It checks the checksum and every table against the file's bounds;
// it allocates the layer structs and copies nothing else. Shapes and
// parameters are validated when the graph is planned and run
// (qnet.NewRunner).
func Parse(data []byte) (*File, error) {
	if len(data) < headerSize || string(data[:4]) != magic {
		return nil, ErrFormat
	}
	le := binary.LittleEndian
	size := uint64(le.Uint32(data[8:]))
	if size < headerSize || size > uint64(len(data)) {
		return nil, ErrFormat
	}
	data = data[:size:size]
	if crc32.ChecksumIEEE(data[16:]) != le.Uint32(data[12:]) {
		return nil, ErrChecksum
	}
	if le.Uint32(data[4:]) != version {
		return nil, ErrVersion
	}
	if uintptr(unsafe.Pointer(unsafe.SliceData(data)))%Align != 0 {
		return nil, ErrAlign
	}
	if !littleEndian() {
		return nil, ErrByteOrder
	}
	r := reader{data}
	table := func(k int, record uint64) ([]byte, int, bool) {
		n := uint64(r.u32(16 + 8*k))
		b, ok := r.at(uint64(r.u32(20+8*k)), n*record, 4)
		return b, int(n), ok
	}
	lt, nLayers, ok1 := table(0, layerSize)
	st, nSegs, ok2 := table(1, segSize)
	ot, nOuts, ok3 := table(2, 4)
	et, nExtras, ok4 := table(3, extraSize)
	if !ok1 || !ok2 || !ok3 || !ok4 {
		return nil, ErrFormat
	}

	f := &File{Graph: qnet.Graph{Layers: make([]qnet.Layer, nLayers)}}
	for i := range f.Graph.Layers {
		rec := lt[i*layerSize:][:layerSize]
		u := func(off int) uint64 { return uint64(le.Uint32(rec[off:])) }
		l := &f.Graph.Layers[i]
		l.Kind, l.Padding, l.Round, l.TiesUp = qnet.Kind(rec[0]), qi8.Padding(rec[1]), qi8.Rounding(rec[2]), rec[3]&1 != 0
		if !supported(l.Kind) {
			return nil, ErrFormat
		}
		l.In = qi8.Shape{H: int(u(4)), W: int(u(8)), C: int(u(12))}
		l.Out = qi8.Shape{H: int(u(16)), W: int(u(20)), C: int(u(24))}
		for _, s := range [...]qi8.Shape{l.In, l.Out} {
			// Sizes far beyond any tensor would overflow the planner's sums.
			if max(s.H, s.W, s.C) > maxDim || uint64(s.H)*uint64(s.W)*uint64(s.C) > maxLen {
				return nil, ErrFormat
			}
		}
		l.InQ = qnet.Quant{Scale: math.Float32frombits(uint32(u(28))), ZP: int8(rec[36])}
		l.OutQ = qnet.Quant{Scale: math.Float32frombits(uint32(u(32))), ZP: int8(rec[37])}
		l.ActMin, l.ActMax = int8(rec[38]), int8(rec[39])
		l.KH, l.KW, l.StrideH, l.StrideW = int(rec[40]), int(rec[41]), int(rec[42]), int(rec[43])
		l.WZP, l.PadValue, l.Axis = int8(rec[44]), int8(rec[45]), int(rec[46])
		for k := range l.Pads {
			l.Pads[k] = int(le.Uint16(rec[48+2*k:]))
		}
		if n := uint64(rec[47]); n > 0 {
			b, ok := r.at(u(60), 4*n, 4)
			if !ok {
				return nil, ErrFormat
			}
			l.From = make([]int, n)
			for k := range l.From {
				l.From[k] = int(int32(le.Uint32(b[4*k:])))
			}
		}
		if n := u(68); n > 0 {
			b, ok := r.at(u(64), n, 1)
			if !ok {
				return nil, ErrFormat
			}
			l.Weights = unsafe.Slice((*int8)(unsafe.Pointer(unsafe.SliceData(b))), len(b))
		}
		if n := u(76); n > 0 {
			b, ok := r.at(u(72), 4*n, 4)
			if !ok {
				return nil, ErrFormat
			}
			l.Bias = unsafe.Slice((*int32)(unsafe.Pointer(unsafe.SliceData(b))), n)
		}
		if n := u(84); n > 0 {
			b, ok := r.at(u(80), 8*n, 4)
			if !ok {
				return nil, ErrFormat
			}
			l.Requant = unsafe.Slice((*qi8.Requant)(unsafe.Pointer(unsafe.SliceData(b))), n)
		}
		switch off := u(88); {
		case off == 0:
		case l.Kind == qnet.Logistic:
			b, ok := r.at(off, 256, 1)
			if !ok {
				return nil, ErrFormat
			}
			l.LogisticParams = (*qi8.LogisticParams)(unsafe.Pointer(unsafe.SliceData(b)))
		case l.Kind == qnet.Concat:
			offs, ok := r.at(off, 4*uint64(len(l.From)), 4)
			if !ok || len(l.From) == 0 {
				return nil, ErrFormat
			}
			l.Rescale = make([]*[256]int8, len(l.From))
			for k := range l.Rescale {
				mo := uint64(le.Uint32(offs[4*k:]))
				if mo == 0 {
					continue
				}
				b, ok := r.at(mo, 256, 1)
				if !ok {
					return nil, ErrFormat
				}
				l.Rescale[k] = (*[256]int8)(unsafe.Pointer(unsafe.SliceData(b)))
			}
		default:
			return nil, ErrFormat
		}
	}

	f.Segments = make([]Segment, nSegs)
	next := 0
	for k := range f.Segments {
		rec := st[k*segSize:]
		first, count := uint64(le.Uint32(rec)), uint64(le.Uint32(rec[4:]))
		b, ok := r.at(uint64(le.Uint32(rec[8:])), uint64(le.Uint32(rec[12:])), Align)
		if !ok || len(b) == 0 || count == 0 || first < uint64(next) || first+count > uint64(nLayers) {
			return nil, ErrFormat
		}
		f.Segments[k] = Segment{First: int(first), Count: int(count), Kmodel: b}
		next = int(first + count)
	}
	if nOuts > 0 {
		f.Graph.Outputs = make([]int, nOuts)
		for k := range f.Graph.Outputs {
			v := le.Uint32(ot[4*k:])
			if uint64(v) >= uint64(nLayers) {
				return nil, ErrFormat
			}
			f.Graph.Outputs[k] = int(v)
		}
	}
	f.extras = make([]Extra, nExtras)
	for k := range f.extras {
		rec := et[k*extraSize:]
		b, ok := r.at(uint64(le.Uint32(rec[8:])), uint64(le.Uint32(rec[12:])), tableAlign)
		if !ok {
			return nil, ErrFormat
		}
		tag := rec[:tagSize]
		for len(tag) > 0 && tag[len(tag)-1] == 0 {
			tag = tag[:len(tag)-1]
		}
		f.extras[k] = Extra{Tag: string(tag), Data: b}
	}
	return f, nil
}
