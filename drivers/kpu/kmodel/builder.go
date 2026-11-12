package kmodel

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Builder assembles a kmodel v3 file in memory, laid out as nncase rc5's
// ncc writes it (Generator.GenerateBin, K210Conv2d.GenerateBin):
//
//	header, outputs, layer table, then each layer body in order
//	conv body: flags, main-memory output, 4 offsets, then inside the body
//	           the layer argument (8-aligned), weights (128-aligned),
//	           BN table (128-aligned), activation table (256-aligned)
//	every body padded to a multiple of 8 (the padding counts in its size)
//
// Alignment is to absolute file offsets; padding is zero. A compiler adds
// layers in execution order and calls Build.
type Builder struct {
	flags, maxStart, mainMem uint32
	outputs                  []Output
	layers                   []builtLayer
}

type builtLayer struct {
	typ  LayerType
	body []uint32 // fixed fields; for a conv: flags, main-memory output
	conv *convBlobs
}

type convBlobs struct {
	args             Args
	weights, bn, act []byte
}

// NewBuilder starts a model. eightBit sets the header's KPU 8-bit mode
// flag (8-bit weights); maxStart is the header's max start address field
// (ncc writes its KPU allocator's high-water mark; the runtime ignores it).
func NewBuilder(eightBit bool, maxStart uint32) *Builder {
	b := &Builder{maxStart: maxStart}
	if eightBit {
		b.flags = 1
	}
	return b
}

// SetMainMem sets the main-memory arena size the layers address.
func (b *Builder) SetMainMem(n uint32) { b.mainMem = n }

// AddOutput declares Size bytes at Address in main memory as an output.
func (b *Builder) AddOutput(o Output) { b.outputs = append(b.outputs, o) }

// AddConv adds a K210 conv whose output stays in KPU memory: its layer
// argument (the weight, BN and activation addresses are patched at run
// time and may be zero) and its tables, encoded as the KPU reads them (see
// EncodeBN, EncodeAct).
func (b *Builder) AddConv(a Args, weights, bn, act []byte) {
	b.addConv(0, 0, a, weights, bn, act)
}

// AddConvToMain adds a K210 conv whose output is also read back to main
// memory at addr (KLF_MAIN_MEM_OUT), for a CPU layer or a model output.
func (b *Builder) AddConvToMain(addr uint32, a Args, weights, bn, act []byte) {
	b.addConv(FlagMainMemOut, addr, a, weights, bn, act)
}

func (b *Builder) addConv(flags, mainMemOut uint32, a Args, weights, bn, act []byte) {
	b.layers = append(b.layers, builtLayer{typ: K210Conv, body: []uint32{flags, mainMemOut},
		conv: &convBlobs{a, weights, bn, act}})
}

func (b *Builder) addQuant(t LayerType, q Quant) {
	b.layers = append(b.layers, builtLayer{typ: t, body: []uint32{q.Flags, q.In, q.Out, q.Count,
		math.Float32bits(q.Scale), math.Float32bits(q.Bias)}})
}

// AddQuantize adds a Quantize layer (float32 → uint8).
func (b *Builder) AddQuantize(q Quant) { b.addQuant(Quantize, q) }

// AddDequantize adds a Dequantize layer (uint8 → float32).
func (b *Builder) AddDequantize(q Quant) { b.addQuant(Dequantize, q) }

// AddGAP adds a GlobalAveragePool2D layer.
func (b *Builder) AddGAP(g GAP) {
	b.layers = append(b.layers, builtLayer{typ: GlobalAveragePool2D, body: []uint32{g.Flags, g.In, g.Out, g.KernelSize, g.Channels}})
}

func (b *Builder) addPadding(t LayerType, p Padding) {
	b.layers = append(b.layers, builtLayer{typ: t, body: []uint32{p.Flags, p.In, p.Out, p.Channels}})
}

// AddAddPadding adds a K210AddPadding layer (main memory → a 4×4 map in KPU
// memory per channel).
func (b *Builder) AddAddPadding(p Padding) { b.addPadding(K210AddPadding, p) }

// AddRemovePadding adds a K210RemovePadding layer (keeps byte 16·c of each
// channel).
func (b *Builder) AddRemovePadding(p Padding) { b.addPadding(K210RemovePadding, p) }

// AddUpload adds a K210Upload layer.
func (b *Builder) AddUpload(u Upload) {
	b.layers = append(b.layers, builtLayer{typ: K210Upload, body: []uint32{u.Flags, u.In, u.Out, u.Width, u.Height, u.Channels}})
}

// Build serialises the model and checks it (Parse, then Check), so a
// returned file is one drivers/kpu can run.
func (b *Builder) Build() ([]byte, error) {
	out := binary.LittleEndian.AppendUint32(nil, 3)
	for _, v := range []uint32{b.flags, 0, uint32(len(b.layers)), b.maxStart, b.mainMem, uint32(len(b.outputs))} {
		out = binary.LittleEndian.AppendUint32(out, v)
	}
	for _, o := range b.outputs {
		out = binary.LittleEndian.AppendUint32(out, o.Address)
		out = binary.LittleEndian.AppendUint32(out, o.Size)
	}
	table := len(out)
	out = append(out, make([]byte, 8*len(b.layers))...)
	for i, l := range b.layers {
		start := len(out)
		for _, v := range l.body {
			out = binary.LittleEndian.AppendUint32(out, v)
		}
		if c := l.conv; c != nil {
			fix := len(out)
			out = append(out, make([]byte, 16)...)
			var offs [4]uint32
			out, offs[0] = align(out, 8)
			for _, w := range c.args {
				out = binary.LittleEndian.AppendUint64(out, w)
			}
			out, offs[1] = align(out, 128)
			out = append(out, c.weights...)
			out, offs[2] = align(out, 128)
			out = append(out, c.bn...)
			out, offs[3] = align(out, 256)
			out = append(out, c.act...)
			for j, o := range offs {
				binary.LittleEndian.PutUint32(out[fix+4*j:], o)
			}
		}
		out, _ = align(out, 8)
		binary.LittleEndian.PutUint32(out[table+8*i:], uint32(l.typ))
		binary.LittleEndian.PutUint32(out[table+8*i+4:], uint32(len(out)-start))
	}
	m, err := Parse(out)
	if err != nil {
		return nil, err
	}
	if err := m.Check(); err != nil {
		return nil, fmt.Errorf("kmodel: built model fails Check: %w", err)
	}
	return out, nil
}

// align pads b with zeros to a multiple of n and returns the new length.
func align(b []byte, n int) ([]byte, uint32) {
	for len(b)%n != 0 {
		b = append(b, 0)
	}
	return b, uint32(len(b))
}

// EncodeBN encodes per-channel batch-norm entries (kpu_batchnorm_argument_t:
// norm_mul bits 0–23, norm_add 24–55, norm_shift 56–59), 8 bytes each.
// Values outside the fields' ranges are an error.
func EncodeBN(bn []BN) ([]byte, error) {
	out := make([]byte, 0, 8*len(bn))
	for i, e := range bn {
		if e.Mul < -1<<23 || e.Mul >= 1<<23 || e.Add < math.MinInt32 || e.Add > math.MaxInt32 || e.Shift > 15 {
			return nil, fmt.Errorf("%w: BN entry %d out of range: %+v", ErrFormat, i, e)
		}
		v := uint64(e.Mul)&0xff_ffff | (uint64(e.Add)&0xffff_ffff)<<24 | uint64(e.Shift)<<56
		out = binary.LittleEndian.AppendUint64(out, v)
	}
	return out, nil
}

// EncodeAct encodes the 16-segment activation table (activate_para_t:
// shift_number bits 0–7, y_mul 8–23, x_start 24–59; then 16 result_bias
// bytes), 144 bytes. Values outside the fields' ranges are an error.
func EncodeAct(act [16]ActSegment) ([]byte, error) {
	out := make([]byte, 0, actTableSize)
	for i, s := range act {
		if s.StartX < -1<<35 || s.StartX >= 1<<35 || s.Mul < -1<<15 || s.Mul >= 1<<15 || s.Shift > 255 || s.Add < 0 || s.Add > 255 {
			return nil, fmt.Errorf("%w: activation segment %d out of range: %+v", ErrFormat, i, s)
		}
		v := uint64(s.Shift) | (uint64(s.Mul)&0xffff)<<8 | (uint64(s.StartX)&0xf_ffff_ffff)<<24
		out = binary.LittleEndian.AppendUint64(out, v)
	}
	for _, s := range act {
		out = append(out, byte(s.Add))
	}
	return out, nil
}
