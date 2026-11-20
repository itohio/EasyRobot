// Package kgraph is the one-file form of a qnet graph compiled for the
// K210's KPU: the graph's layers, the kmodel of every KPU segment, the
// parameters of the layers the CPU runs, and tagged extras (a detector's
// anchors). A host writes it (Encode, or compile.Backend.File); a board
// holds it once in memory and runs it in place (Parse, NewBackend, then
// qnet.NewRunner on File.Graph): nothing in it is copied or compiled on the
// device.
//
// # Layout (little-endian)
//
//	header, 64 bytes
//	   0 "KGRF"   4 version (1)   8 file size   12 CRC-32 (IEEE) of bytes 16…
//	  16 layers:   count, table offset
//	  24 segments: count, table offset
//	  32 outputs:  count, table offset (u32 layer indices; 0 = the last layer)
//	  40 extras:   count, table offset
//	layer record, 96 bytes
//	   0 kind, padding, rounding, flags (1 = TiesUp)          (u8 each)
//	   4 In H, W, C   16 Out H, W, C                           (u32 each)
//	  28 InQ scale  32 OutQ scale                              (float32)
//	  36 InQ zero point, OutQ zero point, ActMin, ActMax       (i8 each)
//	  40 KH, KW, StrideH, StrideW                              (u8 each)
//	  44 WZP, PadValue (i8), Axis, number of From entries      (u8)
//	  48 Pads                                                  (6 × u16)
//	  60 From offset (i32 entries; none = the previous layer)
//	  64 weights offset, length (bytes)
//	  72 bias offset, count (i32)   80 requant offset, count (Mult, Shift i32)
//	  88 table offset: Logistic's 256-byte table; Concat's u32 offsets, one
//	     per input, of 256-byte rescale maps (0 = copy); 0 = none
//	segment record, 16 bytes: first layer, layer count, kmodel offset, length
//	extra record, 16 bytes:   tag (8 bytes, zero padded), offset, length
//
// Kmodels lie at multiples of 128 bytes, everything else at multiples of
// 16, so with the file at a 128-byte-aligned address the KPU reads the
// weights where they are and the CPU tables are usable as int32 and float32
// in place. The layers of a KPU segment carry no parameters: their kmodel
// holds them.
package kgraph

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"math"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
)

// Align is the alignment of the kmodels inside the file, and of the address
// the file must be held at (the KPU fetches weights from 128-byte-aligned
// addresses).
const Align = 128

const (
	magic      = "KGRF"
	version    = 1
	headerSize = 64
	layerSize  = 96
	segSize    = 16
	extraSize  = 16
	tableAlign = 16
	tagSize    = 8
	maxDim     = 1 << 20 // a tensor side or channel count
	maxLen     = 1 << 30 // a tensor's bytes
)

var (
	// ErrFormat reports a file that is not a kgraph file or whose tables
	// are inconsistent, and segments or extras Encode cannot write.
	ErrFormat = errors.New("kgraph: malformed file")
	// ErrVersion reports a file of another format version.
	ErrVersion = errors.New("kgraph: unsupported version")
	// ErrChecksum reports a file whose contents do not match its checksum.
	ErrChecksum = errors.New("kgraph: checksum mismatch")
	// ErrAlign reports a file held at an address that is not a multiple of
	// Align.
	ErrAlign = errors.New("kgraph: the file must be held at a 128-byte-aligned address")
	// ErrByteOrder reports a big-endian machine: the file's int32 tables
	// are used in place.
	ErrByteOrder = errors.New("kgraph: the file is little-endian, the machine is not")
	// ErrUnsupported reports a layer kind the format does not hold.
	ErrUnsupported = errors.New("kgraph: layer kind not supported by the format")
)

// Segment is a run of graph layers the KPU executes as one kmodel.
type Segment struct {
	First, Count int
	// Kmodel is the segment's kmodel v3 file: one input (the first layer's
	// In as CHW uint8) and one main-memory output (the last layer's Out).
	// In a parsed File it lies inside the file.
	Kmodel []byte
}

// Extra is a tagged block of bytes the application interprets (Tag is at
// most 8 bytes).
type Extra struct {
	Tag  string
	Data []byte
}

// supported reports whether the layer record holds everything kind k needs.
func supported(k qnet.Kind) bool {
	switch k {
	case qnet.Conv2D, qnet.DepthwiseConv2D, qnet.MaxPool2D, qnet.AvgPool2D, qnet.ReLU, qnet.ReLU6,
		qnet.Reshape, qnet.Logistic, qnet.Concat, qnet.Pad:
		return true
	}
	return false
}

// Encode writes g, compiled into segs, as one file. g must be valid with
// its parameters bound; segs are ordered, disjoint runs of its layers. The
// layers inside segments are written without parameters.
func Encode(g *qnet.Graph, segs []Segment, extras ...Extra) ([]byte, error) {
	if err := g.Validate(); err != nil {
		return nil, err
	}
	onKPU := make([]bool, len(g.Layers))
	next := 0
	for k, s := range segs {
		if s.First < next || s.Count <= 0 || s.First+s.Count > len(g.Layers) || len(s.Kmodel) == 0 {
			return nil, fmt.Errorf("%w: segment %d (layers %d+%d of %d, %d kmodel bytes)", ErrFormat, k, s.First, s.Count, len(g.Layers), len(s.Kmodel))
		}
		for i := s.First; i < s.First+s.Count; i++ {
			onKPU[i] = true
		}
		next = s.First + s.Count
	}
	for _, e := range extras {
		if len(e.Tag) == 0 || len(e.Tag) > tagSize {
			return nil, fmt.Errorf("%w: extra tag %q (1 to %d bytes)", ErrFormat, e.Tag, tagSize)
		}
	}
	outs := g.Outputs
	layersOff := headerSize
	segsOff := layersOff + layerSize*len(g.Layers)
	outsOff := segsOff + segSize*len(segs)
	extrasOff := outsOff + 4*len(outs)
	w := writer{buf: make([]byte, extrasOff+extraSize*len(extras))}
	le := binary.LittleEndian

	for i := range g.Layers {
		l := &g.Layers[i]
		if !supported(l.Kind) {
			return nil, fmt.Errorf("%w: layer %d (%v)", ErrUnsupported, i, l.Kind)
		}
		for _, v := range [...]int{l.In.H, l.In.W, l.In.C, l.Out.H, l.Out.W, l.Out.C} {
			if v < 0 || int64(v) > math.MaxUint32 {
				return nil, fmt.Errorf("%w: layer %d shape", ErrFormat, i)
			}
		}
		for _, v := range [...]int{l.KH, l.KW, l.StrideH, l.StrideW, l.Axis, len(l.From)} {
			if v < 0 || v > math.MaxUint8 {
				return nil, fmt.Errorf("%w: layer %d geometry", ErrFormat, i)
			}
		}
		r := make([]byte, layerSize)
		r[0], r[1], r[2] = byte(l.Kind), byte(l.Padding), byte(l.Round)
		if l.TiesUp {
			r[3] = 1
		}
		for k, v := range [...]int{l.In.H, l.In.W, l.In.C, l.Out.H, l.Out.W, l.Out.C} {
			le.PutUint32(r[4+4*k:], uint32(v))
		}
		le.PutUint32(r[28:], math.Float32bits(l.InQ.Scale))
		le.PutUint32(r[32:], math.Float32bits(l.OutQ.Scale))
		r[36], r[37], r[38], r[39] = byte(l.InQ.ZP), byte(l.OutQ.ZP), byte(l.ActMin), byte(l.ActMax)
		r[40], r[41], r[42], r[43] = byte(l.KH), byte(l.KW), byte(l.StrideH), byte(l.StrideW)
		r[44], r[45], r[46], r[47] = byte(l.WZP), byte(l.PadValue), byte(l.Axis), byte(len(l.From))
		for k, v := range l.Pads {
			if v < 0 || v > math.MaxUint16 {
				return nil, fmt.Errorf("%w: layer %d pads", ErrFormat, i)
			}
			le.PutUint16(r[48+2*k:], uint16(v))
		}
		if len(l.From) > 0 {
			b := make([]byte, 4*len(l.From))
			for k, v := range l.From {
				le.PutUint32(b[4*k:], uint32(int32(v)))
			}
			le.PutUint32(r[60:], w.add(b, tableAlign))
		}
		if !onKPU[i] {
			if len(l.Weights) > 0 {
				b := make([]byte, len(l.Weights))
				for k, v := range l.Weights {
					b[k] = byte(v)
				}
				le.PutUint32(r[64:], w.add(b, tableAlign))
				le.PutUint32(r[68:], uint32(len(b)))
			}
			if len(l.Bias) > 0 {
				b := make([]byte, 4*len(l.Bias))
				for k, v := range l.Bias {
					le.PutUint32(b[4*k:], uint32(v))
				}
				le.PutUint32(r[72:], w.add(b, tableAlign))
				le.PutUint32(r[76:], uint32(len(l.Bias)))
			}
			if len(l.Requant) > 0 {
				b := make([]byte, 8*len(l.Requant))
				for k, v := range l.Requant {
					le.PutUint32(b[8*k:], uint32(v.Mult))
					le.PutUint32(b[8*k+4:], uint32(v.Shift))
				}
				le.PutUint32(r[80:], w.add(b, tableAlign))
				le.PutUint32(r[84:], uint32(len(l.Requant)))
			}
		}
		switch {
		case l.Kind == qnet.Logistic:
			t := l.LogisticParams.Table
			b := make([]byte, len(t))
			for k, v := range t {
				b[k] = byte(v)
			}
			le.PutUint32(r[88:], w.add(b, tableAlign))
		case l.Kind == qnet.Concat && l.Rescale != nil:
			offs := make([]byte, 4*len(l.Rescale))
			for k, m := range l.Rescale {
				if m == nil {
					continue
				}
				b := make([]byte, len(m))
				for j, v := range m {
					b[j] = byte(v)
				}
				le.PutUint32(offs[4*k:], w.add(b, tableAlign))
			}
			le.PutUint32(r[88:], w.add(offs, tableAlign))
		}
		copy(w.buf[layersOff+layerSize*i:], r)
	}
	for k, s := range segs {
		off := w.add(s.Kmodel, Align)
		r := w.buf[segsOff+segSize*k:]
		le.PutUint32(r[0:], uint32(s.First))
		le.PutUint32(r[4:], uint32(s.Count))
		le.PutUint32(r[8:], off)
		le.PutUint32(r[12:], uint32(len(s.Kmodel)))
	}
	for k, v := range outs {
		le.PutUint32(w.buf[outsOff+4*k:], uint32(v))
	}
	for k, e := range extras {
		off := w.add(e.Data, tableAlign)
		r := w.buf[extrasOff+extraSize*k:]
		copy(r[:tagSize], e.Tag)
		le.PutUint32(r[8:], off)
		le.PutUint32(r[12:], uint32(len(e.Data)))
	}
	if int64(len(w.buf)) > math.MaxUint32 {
		return nil, fmt.Errorf("%w: %d bytes", ErrFormat, len(w.buf))
	}
	h := w.buf
	copy(h, magic)
	le.PutUint32(h[4:], version)
	le.PutUint32(h[8:], uint32(len(h)))
	for k, v := range [...]int{len(g.Layers), layersOff, len(segs), segsOff, len(outs), outsOff, len(extras), extrasOff} {
		le.PutUint32(h[16+4*k:], uint32(v))
	}
	le.PutUint32(h[12:], crc32.ChecksumIEEE(h[16:]))
	return h, nil
}

// writer appends aligned blocks to the file being built.
type writer struct{ buf []byte }

// add appends b at the next multiple of align and returns its offset.
func (w *writer) add(b []byte, align int) uint32 {
	for len(w.buf)%align != 0 {
		w.buf = append(w.buf, 0)
	}
	off := len(w.buf)
	w.buf = append(w.buf, b...)
	return uint32(off)
}

// Float32s returns v as little-endian bytes, for an Extra.
func Float32s(v []float32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return b
}
