package kmodel

import (
	"encoding/binary"
	"fmt"
)

// BN is one output channel's batch-norm entry (kpu_batchnorm_argument_t):
// y = (acc·Mul >> Shift) + Add.
type BN struct {
	Mul, Add int64
	Shift    uint
}

// ActSegment is one of the 16 activation segments (activate_para_t plus its
// result_bias byte): for StartX ≤ y below the next segment's start,
// out = clamp(carry_shift((y − StartX)·Mul, Shift) + Add, 0, 255).
type ActSegment struct {
	StartX, Mul, Add int64
	Shift            uint
}

// ConvParams is a K210 conv layer decoded from its layer argument and the
// weight, batch-norm and activation blobs it points at.
type ConvParams struct {
	Args       Args
	Depthwise  bool
	Kernel     int // 1 or 3
	Pool       int // pool_type
	PadValue   int64
	ArgX, ArgW int64
	ShrX, ShrW uint
	ArgAdd     int64
	Weights    []byte // 8-bit weights, see docs/architecture/k210-kpu.md
	BN         []BN
	Act        [16]ActSegment
}

func signed(v uint64, bits uint) int64 { return int64(v<<(64-bits)) >> (64 - bits) }

// ConvParams decodes conv c (8-bit weights only).
func (m *Model) ConvParams(c Conv) (ConvParams, error) {
	if uint64(c.LayerOffset)+argsSize > uint64(len(m.Data)) {
		return ConvParams{}, ErrFormat
	}
	return m.ParamsAt(m.Args(c), c.WeightsOffset, c.BNOffset, c.ActOffset)
}

// ParamsAt decodes layer argument a with its weights, batch-norm and
// activation tables at the given file offsets (8-bit weights only).
func (m *Model) ParamsAt(a Args, weights, bn, act uint32) (ConvParams, error) {
	p := ConvParams{
		Args:      a,
		Depthwise: a[0]>>3&1 != 0,
		Kernel:    1,
		Pool:      int(a[4] >> 4 & 0xf),
		PadValue:  int64(a[4] >> 24 & 0xff),
		ShrW:      uint(a[9] & 0xf),
		ShrX:      uint(a[9] >> 4 & 0xf),
		ArgW:      signed(a[9]>>8&0xff_ffff, 24),
		ArgX:      signed(a[9]>>32&0xff_ffff, 24),
		ArgAdd:    signed(a[10]&0xff_ffff_ffff, 40),
	}
	if a[4]&7 == 1 {
		p.Kernel = 3
	}
	_, _, ic := a.InShape()
	_, _, oc := a.OutShape()
	nw := ic * p.Kernel * p.Kernel
	if !p.Depthwise {
		nw *= oc
	}
	d := m.Data
	if uint64(weights)+uint64(nw) > uint64(len(d)) ||
		uint64(bn)+8*uint64(oc) > uint64(len(d)) ||
		uint64(act)+18*8 > uint64(len(d)) {
		return ConvParams{}, ErrFormat
	}
	p.Weights = d[weights : weights+uint32(nw)]
	p.BN = make([]BN, oc)
	for i := range p.BN {
		v := binary.LittleEndian.Uint64(d[bn+uint32(8*i):])
		p.BN[i] = BN{Mul: signed(v&0xff_ffff, 24), Add: signed(v>>24&0xffff_ffff, 32), Shift: uint(v >> 56 & 0xf)}
	}
	for i := range p.Act {
		v := binary.LittleEndian.Uint64(d[act+uint32(8*i):])
		p.Act[i] = ActSegment{StartX: signed(v>>24&0xf_ffff_ffff, 36), Mul: signed(v>>8&0xffff, 16), Shift: uint(v & 0xff)}
		p.Act[i].Add = int64(d[act+16*8+uint32(i)])
	}
	return p, nil
}

func carryShift(v int64, shift uint) int64 {
	if shift == 0 {
		return v
	}
	v >>= shift - 1
	if v&1 != 0 {
		if v < 0 {
			return v>>1 - 1
		}
		return v>>1 + 1
	}
	return v >> 1
}

// Emulate computes the layer in integer arithmetic, from in (CHW uint8,
// the layer's input shape) into out (CHW uint8, its output shape). It is a
// port of nncase v0.1.0-rc5's K210Conv2d.Forward (the arithmetic its
// simulator uses), for checking the KPU layer by layer; the silicon agrees
// with it to about 1 LSB (ER-K210-K8 evidence).
func (p *ConvParams) Emulate(in, out []byte) error {
	return p.EmulateInto(in, out, nil)
}

// EmulateInto is Emulate with a caller-provided scratch buffer of at least
// W·H·OC bytes (the stride-1 output before pooling); with a shorter one it
// allocates.
func (p *ConvParams) EmulateInto(in, out, scratch []byte) error {
	w, h, ic := p.Args.InShape()
	ow, oh, oc := p.Args.OutShape()
	if len(in) < w*h*ic || len(out) < ow*oh*oc || (p.Depthwise && ic != oc) {
		return ErrFormat
	}
	pix := scratch // after conv, BN and activation, before pooling
	if len(pix) < w*h*oc {
		pix = make([]byte, w*h*oc)
	}
	k := p.Kernel
	for o := range oc {
		var ws []byte
		if p.Depthwise {
			ws = p.Weights[o*k*k : (o+1)*k*k]
		} else {
			ws = p.Weights[o*ic*k*k : (o+1)*ic*k*k]
		}
		var sumW int64
		for _, v := range ws {
			sumW += int64(v)
		}
		add := p.ArgAdd * int64(ic)
		if p.Depthwise {
			add = p.ArgAdd
		}
		zw := p.ArgW * sumW >> p.ShrW
		for y := range h {
			for x := range w {
				var acc, sumX int64
				chans := ic
				if p.Depthwise {
					chans = 1
				}
				for c := range chans {
					src := c
					if p.Depthwise {
						src = o
					}
					for wy := range k {
						for wx := range k {
							iy, ix := y+wy-k/2, x+wx-k/2
							xv := p.PadValue
							if iy >= 0 && iy < h && ix >= 0 && ix < w {
								xv = int64(in[(src*h+iy)*w+ix])
							}
							sumX += xv
							acc += int64(ws[(c*k+wy)*k+wx]) * xv
						}
					}
				}
				acc += p.ArgX*sumX>>p.ShrX + zw + add
				bn := p.BN[o]
				v := acc*bn.Mul>>bn.Shift + bn.Add
				seg := 0
				for i := range p.Act {
					if p.Act[i].StartX <= v {
						seg = i
					}
				}
				s := p.Act[seg]
				r := carryShift((v-s.StartX)*s.Mul, s.Shift) + s.Add
				pix[(o*h+y)*w+x] = byte(max(0, min(255, r)))
			}
		}
	}
	return pool(p.Pool, pix, w, h, oc, out, ow, oh)
}

// pool applies pool_type t (nncase's emulator: stride = kernel, no padding;
// type 6, the right-top sampler, from nncase v0.2's K210 kernels).
func pool(t int, src []byte, w, h, c int, dst []byte, ow, oh int) error {
	if t == 0 {
		if ow != w || oh != h {
			return ErrFormat
		}
		copy(dst, src[:w*h*c])
		return nil
	}
	k := 2
	switch t {
	case 1, 2, 5, 6:
	case 3, 4:
		k = 4
	default:
		return fmt.Errorf("%w: pool type %d", ErrUnsupported, t)
	}
	for ch := range c {
		for oy := range oh {
			for ox := range ow {
				var mx, sum, n int
				first, rightTop := -1, 0
				for ky := range k {
					for kx := range k {
						iy, ix := oy*k+ky, ox*k+kx
						if iy >= h || ix >= w {
							continue
						}
						v := int(src[(ch*h+iy)*w+ix])
						if first < 0 {
							first = v
						}
						if ky == 0 && kx == 1 {
							rightTop = v
						}
						mx = max(mx, v)
						sum += v
						n++
					}
				}
				if n == 0 {
					return ErrFormat // output larger than the pooled input
				}
				var v int
				switch t {
				case 1, 3:
					v = mx
				case 2, 4:
					v = sum / n
				case 5:
					v = first
				case 6: // right-top: offset (row 0, column 1); 0 off the image (nncase v0.2)
					v = rightTop
				}
				dst[(ch*oh+oy)*ow+ox] = byte(v)
			}
		}
	}
	return nil
}
