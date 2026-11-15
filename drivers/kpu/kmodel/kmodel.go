// Package kmodel parses Kendryte kmodel v3 files (nncase v0.1.0-rc5) and
// implements the parts of their runtime that run on the CPU: the KPU memory
// layout and the CPU layers. It has no hardware access, so it builds and is
// tested on any host; package drivers/kpu runs a parsed model on the K210.
//
// The format and every layer here follow the kmodel v3 runtime in
// kendryte-standalone-sdk V0.5.6 (lib/drivers/kpu.c, kpu.h):
//
//	header   7 × u32: version (3), flags (bit 0: KPU 8-bit mode), arch (0),
//	         layer count, max start address, main memory size, output count
//	outputs  output count × {address, size u32} in main memory
//	layers   layer count × {type, body size u32}
//	bodies   the layer bodies, back to back, in layer order
//
// A K210 conv body points at a 12 × u64 KPU layer argument, the weights, the
// batch-norm table and the activation table elsewhere in the file. All
// fields are little-endian.
package kmodel

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/bits"
)

// LayerType is a kmodel v3 layer type (kpu_model_layer_type_t).
type LayerType uint32

// Layer types this package decodes. The rest of the v3 set exists in the
// format but is rejected by Model.Check.
const (
	GlobalAveragePool2D LayerType = 5
	Quantize            LayerType = 11
	Dequantize          LayerType = 12
	K210Conv            LayerType = 10240
	K210AddPadding      LayerType = 10241
	K210RemovePadding   LayerType = 10242
	K210Upload          LayerType = 10243
)

func (t LayerType) String() string {
	switch t {
	case GlobalAveragePool2D:
		return "GlobalAveragePool2D"
	case Quantize:
		return "Quantize"
	case Dequantize:
		return "Dequantize"
	case K210Conv:
		return "K210Conv"
	case K210AddPadding:
		return "K210AddPadding"
	case K210RemovePadding:
		return "K210RemovePadding"
	case K210Upload:
		return "K210Upload"
	}
	return fmt.Sprintf("LayerType(%d)", uint32(t))
}

// FlagMainMemOut (KLF_MAIN_MEM_OUT) marks a conv whose output goes to main
// memory instead of staying in KPU memory.
const FlagMainMemOut = 1

const (
	headerSize   = 7 * 4
	argsSize     = 12 * 8
	actTableSize = 18 * 8 // 16 segments + 2 words of result_bias
	// KPU register limits (image_size, image_channel_num).
	maxWidth, maxHeight, maxChannels = 1024, 512, 1024
	// RAMSize is the KPU's AI SRAM: 2 MiB, addressed in 64-byte lines.
	RAMSize = 2 << 20
)

var (
	ErrFormat      = errors.New("kmodel: malformed file")
	ErrVersion     = errors.New("kmodel: not a kmodel v3 for the K210 (version 3, arch 0)")
	ErrUnsupported = errors.New("kmodel: layer type not supported")
)

// MaxInputRows is the most input rows a KPU layer runs on silicon: the
// field holds 512, but a 257- or 258-row input never raises the done
// interrupt (observed on the K210).
const MaxInputRows = 256

// Output is one model output: Size bytes at Address in main memory.
type Output struct{ Address, Size uint32 }

// Layer is one layer: its type and its body bytes (a sub-slice of the file).
type Layer struct {
	Type LayerType
	Body []byte
}

// Model is a parsed kmodel. Data is the whole file; Layer bodies and conv
// arguments point into it.
type Model struct {
	Data         []byte
	Flags        uint32
	MainMemUsage uint32
	Outputs      []Output
	Layers       []Layer
}

// Parse checks the header and splits the layer table. It does not copy
// data. Per-layer bounds are checked by Check.
func Parse(data []byte) (*Model, error) {
	if len(data) < headerSize {
		return nil, ErrFormat
	}
	u := func(i int) uint32 { return binary.LittleEndian.Uint32(data[4*i:]) }
	if u(0) != 3 || u(2) != 0 {
		return nil, ErrVersion
	}
	nLayers, nOut := uint64(u(3)), uint64(u(6))
	p := uint64(headerSize)
	end := p + 8*nOut + 8*nLayers
	if end > uint64(len(data)) {
		return nil, ErrFormat
	}
	m := &Model{
		Data:         data,
		Flags:        u(1),
		MainMemUsage: u(5),
		Outputs:      make([]Output, nOut),
		Layers:       make([]Layer, nLayers),
	}
	for i := range m.Outputs {
		o := Output{binary.LittleEndian.Uint32(data[p:]), binary.LittleEndian.Uint32(data[p+4:])}
		if uint64(o.Address)+uint64(o.Size) > uint64(m.MainMemUsage) {
			return nil, ErrFormat
		}
		m.Outputs[i] = o
		p += 8
	}
	body := end
	for i := range m.Layers {
		t := LayerType(binary.LittleEndian.Uint32(data[p:]))
		n := uint64(binary.LittleEndian.Uint32(data[p+4:]))
		if body+n > uint64(len(data)) {
			return nil, ErrFormat
		}
		m.Layers[i] = Layer{Type: t, Body: data[body : body+n : body+n]}
		body += n
		p += 8
	}
	return m, nil
}

// EightBitMode reports the KPU 8-bit mode bit of the header flags.
func (m *Model) EightBitMode() bool { return m.Flags&1 != 0 }

// Check verifies that every output lies in main memory, that every layer is
// a supported type with a long enough body, and that everything it
// addresses lies inside the file, main memory (MainMemUsage bytes) or KPU
// memory. After Check, running the model cannot index out of range.
func (m *Model) Check() error {
	if len(m.Layers) == 0 || m.Layers[0].Type != K210Conv {
		return fmt.Errorf("%w: the first layer must be a K210 conv", ErrFormat)
	}
	// Parse checks the outputs too, but a compiler may assemble a Model
	// without it.
	for i, o := range m.Outputs {
		if uint64(o.Address)+uint64(o.Size) > uint64(m.MainMemUsage) {
			return fmt.Errorf("%w: output %d past main memory", ErrFormat, i)
		}
	}
	for i, l := range m.Layers {
		if err := m.checkLayer(l); err != nil {
			return fmt.Errorf("layer %d (%v): %w", i, l.Type, err)
		}
	}
	return nil
}

func (m *Model) checkLayer(l Layer) error {
	main := uint64(m.MainMemUsage)
	in := func(addr uint32, n uint64) error {
		if uint64(addr)+n > main {
			return ErrFormat
		}
		return nil
	}
	switch l.Type {
	case K210Conv:
		c, err := l.Conv()
		if err != nil {
			return err
		}
		if uint64(c.LayerOffset)+argsSize > uint64(len(m.Data)) {
			return ErrFormat
		}
		a := m.Args(c)
		if a[0]>>3&1 != 0 && a[4]>>4&0xf != 0 {
			// The silicon computes a wrong output (observed on the K210).
			return fmt.Errorf("%w: pool_type %d on a depthwise conv (the KPU does not pool in depthwise mode)", ErrUnsupported, a[4]>>4&0xf)
		}
		if err := m.checkTables(c, &a); err != nil {
			return err
		}
		w, h, ch := a.InShape()
		if h > MaxInputRows {
			// Each Run would time out.
			return fmt.Errorf("%w: a %d-row input (the KPU runs at most %d)", ErrUnsupported, h, MaxInputRows)
		}
		if err := a.InLayout().check(a.SrcAddr(), w, h, ch); err != nil {
			return err
		}
		w, h, ch = a.OutShape()
		if err := a.OutLayout().check(a.DstAddr(), w, h, ch); err != nil {
			return err
		}
		// dma_parameter's byte counts follow the output shape (as SetOutShape
		// writes them). Above 65,536 pixels per
		// channel the KPU ignores channel_byte_num (send_data_out must be
		// off): SetOutShape saturates it and ncc truncates it to 16 bits, so
		// both are accepted (K9a-2).
		chb, total, px := a[11]>>16&0xffff, a[11]>>32, uint64(w*h)
		chbOK := chb == min(px, maxChannelBytes)-1 || (px > maxChannelBytes && chb == (px-1)&0xffff)
		if total != px*uint64(ch)-1 || !chbOK || (a[11]&1 != 0 && px > maxChannelBytes) {
			return fmt.Errorf("%w: dma_parameter byte counts %d/%d do not match the output %d×%d×%d", ErrFormat, chb+1, total+1, w, h, ch)
		}
		if c.Flags&FlagMainMemOut != 0 {
			return in(c.MainMemOut, uint64(w*h*ch))
		}
	case Dequantize, Quantize:
		q, err := l.Quant()
		if err != nil {
			return err
		}
		if !finite(q.Scale) || q.Scale == 0 || !finite(q.Bias) {
			return ErrFormat
		}
		if err := in(q.In, uint64(q.Count)*inSize(l.Type)); err != nil {
			return err
		}
		return in(q.Out, uint64(q.Count)*outSize(l.Type))
	case GlobalAveragePool2D:
		g, err := l.GAP()
		if err != nil {
			return err
		}
		if g.KernelSize == 0 {
			return ErrFormat
		}
		n, ok := mul(4*uint64(g.Channels), uint64(g.KernelSize))
		if !ok {
			return ErrFormat
		}
		if err := in(g.In, n); err != nil {
			return err
		}
		return in(g.Out, 4*uint64(g.Channels))
	case K210AddPadding:
		p, err := l.Padding()
		if err != nil {
			return err
		}
		if p.Channels > maxChannels {
			return ErrFormat
		}
		if err := in(p.In, uint64(p.Channels)); err != nil {
			return err
		}
		// The bytes go into a 4×4 map per channel, which the next conv reads
		// whole: check that map, not just the bytes written.
		return padLayout.check(p.Out, 4, 4, int(p.Channels))
	case K210RemovePadding:
		p, err := l.Padding()
		if err != nil {
			return err
		}
		if p.Channels > 0 {
			if err := in(p.In, 16*uint64(p.Channels-1)+1); err != nil {
				return err
			}
		}
		return in(p.Out, uint64(p.Channels))
	case K210Upload:
		u, err := l.Upload()
		if err != nil {
			return err
		}
		// The layout check first: it bounds the shape by the KPU limits, so
		// the product below cannot wrap. Values above
		// the int range on 32-bit hosts convert to negative and fail it too.
		if err := UploadLayout(int(u.Width), int(u.Height)).check(u.Out, int(u.Width), int(u.Height), int(u.Channels)); err != nil {
			return err
		}
		return in(u.In, uint64(u.Width)*uint64(u.Height)*uint64(u.Channels))
	default:
		return ErrUnsupported
	}
	return nil
}

// checkTables checks that conv c's weight, batch-norm and activation
// tables lie inside the file over their whole size, since the KPU fetches
// them by DMA. The weight extent is the size the
// layer shape implies (8- or 16-bit weights by the header's mode). It is
// not para_size × (load_time + 1): nncase's last load is partial (MobileNet
// layer 26: 3 loads of 30,720 B for 65,536 B of weights, the file ending
// 24,280 B before 3 × 30,720 would), and the board runs give correct results.
func (m *Model) checkTables(c Conv, a *Args) error {
	_, _, ic := a.InShape()
	_, _, oc := a.OutShape()
	k := uint64(1)
	if a[4]&7 == 1 {
		k = 3
	}
	nw := uint64(ic) * k * k
	if a[0]>>3&1 == 0 { // not depthwise
		nw *= uint64(oc)
	}
	if !m.EightBitMode() {
		nw *= 2
	}
	size := uint64(len(m.Data))
	if uint64(c.WeightsOffset)+nw > size ||
		uint64(c.BNOffset)+8*uint64(oc) > size ||
		uint64(c.ActOffset)+actTableSize > size {
		return ErrFormat
	}
	return nil
}

func finite(f float32) bool { return !math.IsNaN(float64(f)) && !math.IsInf(float64(f), 0) }

// mul returns a·b and whether it did not overflow.
func mul(a, b uint64) (uint64, bool) {
	hi, lo := bits.Mul64(a, b)
	return lo, hi == 0
}

func inSize(t LayerType) uint64 {
	if t == Quantize {
		return 4
	}
	return 1
}

func outSize(t LayerType) uint64 {
	if t == Dequantize {
		return 4
	}
	return 1
}

// u32 reads little-endian word i of b; callers check len(b) first.
func u32(b []byte, i int) uint32 { return binary.LittleEndian.Uint32(b[4*i:]) }

// Conv is a K210 conv body (kpu_model_conv_layer_argument_t). The offsets
// are from the start of the file.
type Conv struct {
	Flags, MainMemOut                               uint32
	LayerOffset, WeightsOffset, BNOffset, ActOffset uint32
}

// Conv decodes a K210Conv body.
func (l Layer) Conv() (Conv, error) {
	if len(l.Body) < 24 {
		return Conv{}, ErrFormat
	}
	b := l.Body
	return Conv{u32(b, 0), u32(b, 1), u32(b, 2), u32(b, 3), u32(b, 4), u32(b, 5)}, nil
}

// Quant is a Quantize or Dequantize body: Count values from In to Out in
// main memory; float = q·Scale + Bias.
type Quant struct {
	Flags, In, Out, Count uint32
	Scale, Bias           float32
}

// Quant decodes a Quantize or Dequantize body.
func (l Layer) Quant() (Quant, error) {
	if len(l.Body) < 24 {
		return Quant{}, ErrFormat
	}
	b := l.Body
	return Quant{u32(b, 0), u32(b, 1), u32(b, 2), u32(b, 3), f32(u32(b, 4)), f32(u32(b, 5))}, nil
}

// GAP is a GlobalAveragePool2D body: Channels float averages over
// KernelSize floats each, In and Out in main memory.
type GAP struct{ Flags, In, Out, KernelSize, Channels uint32 }

// GAP decodes a GlobalAveragePool2D body.
func (l Layer) GAP() (GAP, error) {
	b := l.Body
	if len(b) < 20 {
		return GAP{}, ErrFormat
	}
	return GAP{u32(b, 0), u32(b, 1), u32(b, 2), u32(b, 3), u32(b, 4)}, nil
}

// Padding is a K210AddPadding body (In in main memory, Out a KPU memory
// line address) or a K210RemovePadding body (both in main memory).
type Padding struct{ Flags, In, Out, Channels uint32 }

// Padding decodes a K210AddPadding or K210RemovePadding body.
func (l Layer) Padding() (Padding, error) {
	b := l.Body
	if len(b) < 16 {
		return Padding{}, ErrFormat
	}
	return Padding{u32(b, 0), u32(b, 1), u32(b, 2), u32(b, 3)}, nil
}

// Upload is a K210Upload body: a CHW uint8 tensor at In in main memory to
// KPU memory line address Out.
type Upload struct{ Flags, In, Out, Width, Height, Channels uint32 }

// Upload decodes a K210Upload body.
func (l Layer) Upload() (Upload, error) {
	b := l.Body
	if len(b) < 24 {
		return Upload{}, ErrFormat
	}
	return Upload{u32(b, 0), u32(b, 1), u32(b, 2), u32(b, 3), u32(b, 4), u32(b, 5)}, nil
}
