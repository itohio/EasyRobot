package kmodel

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"
)

// Every bound of Parse/Check, one case each: a valid model with exactly one
// field changed must be rejected, and the boundary value (the region ending
// exactly at the limit) must still be accepted. Review R-K210-K8b K1, K2, K6.

const (
	fxW, fxH, fxC, fxOC = 16, 16, 3, 4
	fxN                 = fxW * fxH * fxOC // conv output bytes
	fxMain              = 5144
)

// fixture is a model with one layer of every supported type, laid out as
// ncc does. Its fields locate each body and blob so that tests can change
// exactly one value.
type fixture struct {
	data                      []byte
	body                      []int // offset of each layer body in data
	args, wOff, bnOff, actOff int
	tableEnd                  int // len(data): the activation table ends here
	upW, upH, upC, padC, rmC  uint32
	upOut, padOut             uint32
	weightBytes, bnBytes      int
}

// Main memory: conv out [0,1024), Dequantize → floats [1024,5120), GAP →
// [5120,5136), Quantize → [5136,5140), AddPadding reads [5136,5140),
// Upload reads [0,1024), RemovePadding reads [0,49) → [5140,5144).
func newFixture() *fixture {
	f := &fixture{upW: fxW, upH: fxH, upC: fxOC, padC: fxOC, rmC: fxOC, upOut: 2000, padOut: 1000}
	bodies := [][]byte{
		le32(FlagMainMemOut, 0, 0, 0, 0, 0), // conv; offsets set below
		le32(0, 0, 1024, fxN, math.Float32bits(0.5), math.Float32bits(-1)),
		le32(0, 1024, 5120, fxW*fxH, fxOC),
		le32(0, 5120, 5136, fxOC, math.Float32bits(0.1), 0),
		le32(0, 5136, f.padOut, f.padC),
		le32(0, 0, f.upOut, f.upW, f.upH, f.upC),
		le32(0, 0, 5140, f.rmC),
	}
	types := []LayerType{K210Conv, Dequantize, GlobalAveragePool2D, Quantize, K210AddPadding, K210Upload, K210RemovePadding}
	head := 28 + 8 + 8*len(bodies)
	p := head
	for _, b := range bodies {
		f.body = append(f.body, p)
		p += len(b)
	}
	f.args = (p + 127) &^ 127
	f.wOff = f.args + 128
	f.weightBytes = fxC * fxOC // 1×1 conv, 8-bit
	f.bnOff = f.wOff + 128
	f.bnBytes = 8 * fxOC
	f.actOff = f.bnOff + 128
	f.tableEnd = f.actOff + 18*8
	data := make([]byte, f.tableEnd)
	copy(data, le32(3, 1, 0, uint32(len(bodies)), 0, fxMain, 1))
	copy(data[28:], le32(1024, 4*fxN))
	for i, b := range bodies {
		copy(data[36+8*i:], le32(uint32(types[i]), uint32(len(b))))
		copy(data[f.body[i]:], b)
	}
	copy(data[f.body[0]+8:], le32(uint32(f.args), uint32(f.wOff), uint32(f.bnOff), uint32(f.actOff)))
	a := convArgs(100, 200, fxW, fxH, fxC, fxOC, UploadLayout(fxW, fxH), UploadLayout(fxW, fxH))
	a[5] |= uint64(f.weightBytes) << 15 // para_size, load_time 0
	for i, v := range a {
		binary.LittleEndian.PutUint64(data[f.args+8*i:], v)
	}
	f.data = data
	return f
}

func (f *fixture) setBody(layer, word int, v uint32) {
	binary.LittleEndian.PutUint32(f.data[f.body[layer]+4*word:], v)
}

func (f *fixture) setArg(word int, mask, v uint64) {
	o := f.args + 8*word
	x := binary.LittleEndian.Uint64(f.data[o:])
	binary.LittleEndian.PutUint64(f.data[o:], x&^mask|v&mask)
}

func (f *fixture) check(t *testing.T) error {
	t.Helper()
	m, err := Parse(f.data)
	if err != nil {
		return err
	}
	return m.Check()
}

func TestFixtureValid(t *testing.T) {
	if err := newFixture().check(t); err != nil {
		t.Fatalf("fixture rejected: %v", err)
	}
}

func TestCheckRejects(t *testing.T) {
	const (
		conv, deq, gap, quant, pad, up, rm = 0, 1, 2, 3, 4, 5, 6
		nan                                = 0x7fc00000
		inf                                = 0x7f800000
	)
	bodyLen := func(layer int, n uint32) func(f *fixture) {
		return func(f *fixture) { binary.LittleEndian.PutUint32(f.data[36+8*layer+4:], n) }
	}
	cases := []struct {
		name string
		mod  func(f *fixture)
		want error
	}{
		{"first layer not a conv", func(f *fixture) { binary.LittleEndian.PutUint32(f.data[36:], uint32(Dequantize)) }, ErrFormat},
		{"unsupported layer type", func(f *fixture) { binary.LittleEndian.PutUint32(f.data[36+8*rm:], 15) }, ErrUnsupported},

		{"conv body short", bodyLen(conv, 20), ErrFormat},
		{"conv args past the file", func(f *fixture) { f.setBody(conv, 2, uint32(len(f.data)-argsSize+1)) }, ErrFormat},
		{"weights start past the file", func(f *fixture) { f.setBody(conv, 3, uint32(len(f.data))) }, ErrFormat},
		{"weights extent past the file (K2)", func(f *fixture) { f.setBody(conv, 3, uint32(len(f.data)-f.weightBytes+1)) }, ErrFormat},
		{"weights extent past the file at 16-bit weights", func(f *fixture) {
			binary.LittleEndian.PutUint32(f.data[4:], 0) // not 8-bit mode: 2 bytes per weight
			f.setBody(conv, 3, uint32(len(f.data)-2*f.weightBytes+1))
		}, ErrFormat},
		{"3×3 weights extent past the file", func(f *fixture) {
			f.setArg(4, 7, 1) // kernel 3×3: 9× the weights
			f.setBody(conv, 3, uint32(len(f.data)-9*f.weightBytes+1))
		}, ErrFormat},
		{"conv weights sized ic·oc (12 B), 4 B before the end", func(f *fixture) {
			// Depthwise needs ic·k·k = 3 bytes, not ic·oc: a start 4 bytes
			// before the end is fine for depthwise only, so with the flag
			// cleared (regular conv, 12 bytes) it must be rejected.
			f.setBody(conv, 3, uint32(len(f.data)-4))
		}, ErrFormat},
		{"BN extent past the file (K2)", func(f *fixture) { f.setBody(conv, 4, uint32(len(f.data)-f.bnBytes+1)) }, ErrFormat},
		{"activation table past the file (K2)", func(f *fixture) { f.setBody(conv, 5, uint32(len(f.data)-18*8+1)) }, ErrFormat},
		{"conv input past KPU memory", func(f *fixture) { f.setArg(1, 0x7fff, 0x7fff) }, ErrFormat},
		{"conv output past KPU memory", func(f *fixture) { f.setArg(1, 0x7fff<<32, 0x7fff<<32) }, ErrFormat},
		{"conv input group 3", func(f *fixture) { f.setArg(7, 7<<28, 3<<28) }, ErrFormat},
		{"conv output group 0", func(f *fixture) { f.setArg(8, 7<<20, 0) }, ErrFormat},
		{"conv input wider than its slot", func(f *fixture) { f.setArg(7, 7<<28, 4<<28); f.setArg(3, 0x3ff, 31) }, ErrFormat},
		{"conv main-memory output past main memory", func(f *fixture) { f.setBody(conv, 1, fxMain-fxN+1) }, ErrFormat},
		{"pool on a depthwise conv (the KPU does not pool in depthwise mode)", func(f *fixture) {
			f.setArg(0, 1<<3, 1<<3) // depthwise
			f.setArg(4, 0xf<<4, 5<<4)
		}, ErrUnsupported},

		{"dequantize body short", bodyLen(deq, 20), ErrFormat},
		{"dequantize input past main memory", func(f *fixture) { f.setBody(deq, 1, fxMain-fxN+1) }, ErrFormat},
		{"dequantize output past main memory", func(f *fixture) { f.setBody(deq, 2, fxMain-4*fxN+1) }, ErrFormat},
		{"dequantize scale 0 (K6)", func(f *fixture) { f.setBody(deq, 4, 0) }, ErrFormat},
		{"dequantize scale NaN (K6)", func(f *fixture) { f.setBody(deq, 4, nan) }, ErrFormat},
		{"dequantize bias Inf (K6)", func(f *fixture) { f.setBody(deq, 5, inf) }, ErrFormat},
		{"quantize scale 0 (K6)", func(f *fixture) { f.setBody(quant, 4, 0) }, ErrFormat},
		{"quantize scale Inf (K6)", func(f *fixture) { f.setBody(quant, 4, inf) }, ErrFormat},
		{"quantize bias NaN (K6)", func(f *fixture) { f.setBody(quant, 5, nan) }, ErrFormat},
		{"quantize input past main memory", func(f *fixture) { f.setBody(quant, 1, fxMain-4*fxOC+1) }, ErrFormat},
		{"quantize output past main memory", func(f *fixture) { f.setBody(quant, 2, fxMain-fxOC+1) }, ErrFormat},

		{"GAP body short", bodyLen(gap, 16), ErrFormat},
		{"GAP kernel 0", func(f *fixture) { f.setBody(gap, 3, 0) }, ErrFormat},
		{"GAP input past main memory", func(f *fixture) { f.setBody(gap, 1, fxMain-4*fxN+1) }, ErrFormat},
		{"GAP output past main memory", func(f *fixture) { f.setBody(gap, 2, fxMain-4*fxOC+1) }, ErrFormat},
		{"GAP input size overflows uint64", func(f *fixture) { f.setBody(gap, 3, 1<<31); f.setBody(gap, 4, 1<<31) }, ErrFormat},

		{"add-padding body short", bodyLen(pad, 12), ErrFormat},
		{"add-padding input past main memory", func(f *fixture) { f.setBody(pad, 1, fxMain-fxOC+1) }, ErrFormat},
		{"add-padding output past KPU memory", func(f *fixture) { f.setBody(pad, 2, RAMSize/64-4+1) }, ErrFormat},
		{"add-padding channels above 1024", func(f *fixture) { f.setBody(pad, 3, 1025); f.setBody(pad, 1, 0) }, ErrFormat},

		{"remove-padding input past main memory", func(f *fixture) { f.setBody(rm, 1, fxMain-16*(fxOC-1)) }, ErrFormat},
		{"remove-padding output past main memory", func(f *fixture) { f.setBody(rm, 2, fxMain-fxOC+1) }, ErrFormat},

		{"upload body short", bodyLen(up, 20), ErrFormat},
		{"upload input past main memory", func(f *fixture) { f.setBody(up, 1, fxMain-fxN+1) }, ErrFormat},
		{"upload output past KPU memory", func(f *fixture) { f.setBody(up, 2, RAMSize/64-16+1) }, ErrFormat},
		{"upload shape overflows (K1: 16×2³⁰×2³⁰)", func(f *fixture) { f.setBody(up, 4, 1<<30); f.setBody(up, 5, 1<<30) }, ErrFormat},
		{"upload width above 1024", func(f *fixture) { f.setBody(up, 3, 1025); f.setBody(up, 4, 1); f.setBody(up, 5, 1) }, ErrFormat},
		{"upload height above 512", func(f *fixture) { f.setBody(up, 3, 1); f.setBody(up, 4, 513); f.setBody(up, 5, 1) }, ErrFormat},
		{"upload channels above 1024", func(f *fixture) { f.setBody(up, 3, 1); f.setBody(up, 4, 1); f.setBody(up, 5, 1025) }, ErrFormat},
		{"upload width 0", func(f *fixture) { f.setBody(up, 3, 0) }, ErrFormat},
		{"upload height 0", func(f *fixture) { f.setBody(up, 4, 0) }, ErrFormat},
		{"upload channels 0", func(f *fixture) { f.setBody(up, 5, 0) }, ErrFormat},
		{"upload last byte exactly at RAMSize", func(f *fixture) {
			// 1×1×1 in a 16-byte slot: the last byte sits at addr·64.
			f.setBody(up, 3, 1)
			f.setBody(up, 4, 1)
			f.setBody(up, 5, 1)
			f.setBody(up, 2, RAMSize/64)
		}, ErrFormat},
	}
	for _, c := range cases {
		f := newFixture()
		c.mod(f)
		if err := f.check(t); !errors.Is(err, c.want) {
			t.Errorf("%s: err %v, want %v", c.name, err, c.want)
		}
	}
}

// TestCheckBoundariesAccepted moves each region to end exactly at its
// limit: the check must be "past the end", not "at the end".
func TestCheckBoundariesAccepted(t *testing.T) {
	const conv, deq, quant, pad, up, rm = 0, 1, 3, 4, 5, 6
	cases := []struct {
		name string
		mod  func(f *fixture)
	}{
		{"conv args end at the file end", func(f *fixture) {
			f.data = append(f.data, f.data[f.args:f.args+argsSize]...)
			f.setBody(conv, 2, uint32(len(f.data)-argsSize))
		}},
		{"weights end at the file end", func(f *fixture) { f.setBody(conv, 3, uint32(len(f.data)-f.weightBytes)) }},
		{"BN ends at the file end", func(f *fixture) { f.setBody(conv, 4, uint32(len(f.data)-f.bnBytes)) }},
		{"activation table ends at the file end", func(f *fixture) { f.setBody(conv, 5, uint32(len(f.data)-actTableSize)) }},
		{"depthwise weights end at the file end", func(f *fixture) {
			f.setArg(0, 1<<3, 1<<3) // depthwise: ic·k·k bytes
			f.setArg(2, 0x3ff<<32, uint64(fxC-1)<<32)
			f.setBody(conv, 3, uint32(len(f.data)-fxC))
			f.setBody(conv, 0, 0) // output stays in KPU memory (3 channels now)
		}},
		{"conv output ends at KPU memory end", func(f *fixture) {
			// 16×16×4 in groups of 4: 16 lines.
			f.setArg(1, 0x7fff<<32, uint64(RAMSize/64-16)<<32)
		}},
		{"conv main-memory output ends at main memory end", func(f *fixture) { f.setBody(conv, 1, fxMain-fxN) }},
		{"dequantize output ends at main memory end", func(f *fixture) { f.setBody(deq, 2, fxMain-4*fxN) }},
		{"quantize output ends at main memory end", func(f *fixture) { f.setBody(quant, 2, fxMain-fxOC) }},
		{"add-padding output ends at KPU memory end", func(f *fixture) { f.setBody(pad, 2, RAMSize/64-4) }},
		{"upload output ends at KPU memory end", func(f *fixture) { f.setBody(up, 2, RAMSize/64-16) }},
		{"upload at the KPU limits 1024×512×1", func(f *fixture) {
			f.setBody(up, 3, 1024)
			f.setBody(up, 4, 512)
			f.setBody(up, 5, 1)
			binary.LittleEndian.PutUint32(f.data[20:], 1024*512) // main memory large enough
			f.setBody(up, 2, 0)
		}},
		{"remove-padding input ends at main memory end", func(f *fixture) { f.setBody(rm, 1, fxMain-16*(fxOC-1)-1) }},
		{"remove-padding of 0 channels is a no-op", func(f *fixture) { f.setBody(rm, 3, 0); f.setBody(rm, 1, fxMain) }},
		{"upload last byte just below RAMSize", func(f *fixture) {
			f.setBody(up, 3, 1)
			f.setBody(up, 4, 1)
			f.setBody(up, 5, 1)
			f.setBody(up, 2, RAMSize/64-1)
		}},
		{"upload with 1024 channels", func(f *fixture) {
			f.setBody(up, 3, 1)
			f.setBody(up, 4, 1)
			f.setBody(up, 5, 1024)
			f.setBody(up, 2, 0)
		}},
		{"add-padding with 1024 channels", func(f *fixture) {
			f.setBody(pad, 3, 1024)
			f.setBody(pad, 1, 0)
			f.setBody(pad, 2, 0)
		}},
	}
	for _, c := range cases {
		f := newFixture()
		c.mod(f)
		if err := f.check(t); err != nil {
			t.Errorf("%s: rejected: %v", c.name, err)
		}
	}
}

// FuzzCheckedModelRuns pins Check's promise: a model that passes Check can
// be run without indexing out of range. It runs every layer the way
// kpu.Runner does (convs as an input upload and an output read-back in
// their layouts, the KPU itself skipped).
func FuzzCheckedModelRuns(f *testing.F) {
	f.Add(newFixture().data)
	f.Add(tinyModel(f))
	k1 := newFixture() // review R-K210-K8b K1: 16×2³⁰×2³⁰ upload
	k1.setBody(5, 4, 1<<30)
	k1.setBody(5, 5, 1<<30)
	f.Add(k1.data)
	f.Fuzz(func(t *testing.T, data []byte) {
		m, err := Parse(data)
		if err != nil || m.Check() != nil || m.MainMemUsage > 1<<24 {
			return
		}
		main := make([]byte, m.MainMemUsage)
		ram := make([]byte, RAMSize)
		var lut [256]uint32
		for _, l := range m.Layers {
			switch l.Type {
			case K210Conv:
				c, _ := l.Conv()
				a := m.Args(c)
				w, h, ch := a.InShape()
				if err := a.InLayout().UploadTo(ram, a.SrcAddr(), w, h, ch, make([]byte, w*h*ch)); err != nil {
					t.Fatalf("checked conv input: %v", err)
				}
				w, h, ch = a.OutShape()
				if c.Flags&FlagMainMemOut != 0 {
					if err := a.OutLayout().DownloadFrom(ram, a.DstAddr(), w, h, ch, main[c.MainMemOut:]); err != nil {
						t.Fatalf("checked conv output: %v", err)
					}
				}
			case Dequantize:
				q, _ := l.Quant()
				q.Dequantize(main, &lut)
			case Quantize:
				q, _ := l.Quant()
				q.Quantize(main)
			case GlobalAveragePool2D:
				g, _ := l.GAP()
				g.Run(main)
			case K210AddPadding:
				p, _ := l.Padding()
				p.AddPadding(main, ram)
			case K210RemovePadding:
				p, _ := l.Padding()
				p.RemovePadding(main)
			case K210Upload:
				u, _ := l.Upload()
				if err := u.Run(main, ram); err != nil {
					t.Fatalf("checked upload: %v", err)
				}
			}
		}
	})
}

// TestCheckOutputs: a Model assembled without Parse (as a compiler may)
// must have its outputs checked too (review R-K210-K8b K4).
func TestCheckOutputs(t *testing.T) {
	m, err := Parse(newFixture().data)
	if err != nil {
		t.Fatal(err)
	}
	m.Outputs = []Output{{Address: fxMain - 16, Size: 16}}
	if err := m.Check(); err != nil {
		t.Errorf("output ending at main memory end rejected: %v", err)
	}
	m.Outputs = []Output{{Address: fxMain - 16, Size: 17}}
	if err := m.Check(); !errors.Is(err, ErrFormat) {
		t.Errorf("output past main memory: err %v, want ErrFormat", err)
	}
	m.Outputs = []Output{{Address: 1 << 31, Size: 1 << 31}}
	if err := m.Check(); !errors.Is(err, ErrFormat) {
		t.Errorf("output address+size overflowing uint32: err %v, want ErrFormat", err)
	}
}
