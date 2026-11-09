//go:build esp32s3 && !qi8generic

package qi8

// ESP32-S3 PIE backend for the prepared layers (see impl_generic.go for the
// seam contract). Conv2D and FullyConnected run their dot products with PIE
// (pie_esp32s3.c, ACCX accumulation); DepthwiseConv2D accumulates 16
// channels per lane group in QACC; pools and activations use the generic
// inner loops.
//
// Exactness. The generic reference computes acc = bias + Σ (x − zpIn)·w in
// wrapping int32. This backend computes Σ x·w over a gathered patch in which
// padded cells hold zpIn, and adds biasEff = bias − zpIn·Σw (precomputed at
// Prepare over the full kernel window). Every term is an exact integer and
// int32 addition wraps modulo 2^32, so the result is the same int32 bit for
// bit; requantisation then uses the shared requantize. SPEC.md allows this
// zero-point folding for accelerated paths.
//
// Scratch layout (from the first 16-byte-aligned byte of Scratch), with
// K = taps·inC (conv) or In (FC), K32 = K rounded up to 32, R = output
// channels or FC outputs:
//
//	wpack   R·K32 bytes   weight rows, zero-padded to K32
//	slack   16 bytes      zero; the pipelined loop reads one block past a row
//	patch   K32 bytes     gathered input, zero tail
//	slack   16 bytes      zero
//	biasEff R int32       bias − zpIn·Σw
//	acc     R int32       per-row dot products

// #include <stdint.h>
// void qi8_pie_enable(void);
// void qi8_pie_dot_rows(const int8_t *patch, const int8_t *w, int k, int rows, int32_t *out);
// void qi8_pie_dw_px(const int8_t *const *tap, int taps, int direct, const int8_t *zprow, int8_t *stage, const int8_t *w, int ch, int c16, const int32_t *biasEff, const void *rq, int32_t zp, int32_t lo, int32_t hi, int8_t *dst, uint8_t *tmp);
import "C"

import "unsafe"

// backendName is reported by Backend.
const backendName = "esp32s3-pie"

func init() { C.qi8_pie_enable() }

func roundUp32(n int) int { return (n + 31) &^ 31 }

// pieLayout is the scratch layout for one weighted layer.
type pieLayout struct {
	k, k32, rows int
	wpack, patch []int8
	biasEff, acc []int32
}

func pieScratchLen(k, rows int) int {
	k32 := roundUp32(k)
	return 15 + rows*k32 + 16 + k32 + 16 + 8*rows
}

func newPieLayout(scratch []int8, k, rows int) pieLayout {
	off := int((16 - uintptr(unsafe.Pointer(&scratch[0]))&15) & 15)
	s := scratch[off:]
	k32 := roundUp32(k)
	wEnd := rows * k32
	pStart := wEnd + 16
	bStart := pStart + k32 + 16
	return pieLayout{
		k: k, k32: k32, rows: rows,
		wpack:   s[:wEnd+16],
		patch:   s[pStart : pStart+k32+16],
		biasEff: unsafe.Slice((*int32)(unsafe.Pointer(&s[bStart])), rows),
		acc:     unsafe.Slice((*int32)(unsafe.Pointer(&s[bStart+4*rows])), rows),
	}
}

// pack copies the weight rows (k each) zero-padded to k32, zeroes the slack
// and patch, and precomputes biasEff.
//
// Rows are packed from last to first with memmove semantics, so w may alias
// the start of the aligned wpack region (w == wpack[:rows*k]): packing in
// place lets a caller keep only one copy of large weights in SRAM. The
// original w contents are then gone; Run never reads w.
func (l *pieLayout) pack(w []int8, bias []int32, inZP int32) {
	for r := l.rows - 1; r >= 0; r-- {
		dst := l.wpack[r*l.k32 : (r+1)*l.k32]
		copy(dst[:l.k], w[r*l.k:(r+1)*l.k])
		fill(dst[l.k:], 0)
	}
	fill(l.wpack[l.rows*l.k32:], 0)
	fill(l.patch, 0)
	for r := 0; r < l.rows; r++ {
		var sum int32
		for _, v := range l.wpack[r*l.k32 : r*l.k32+l.k] {
			sum += int32(v)
		}
		var b int32
		if len(bias) != 0 {
			b = bias[r]
		}
		l.biasEff[r] = b - inZP*sum
	}
}

func (l *pieLayout) dotRows() {
	C.qi8_pie_dot_rows(
		(*C.int8_t)(unsafe.Pointer(&l.patch[0])),
		(*C.int8_t)(unsafe.Pointer(&l.wpack[0])),
		C.int(l.k32), C.int(l.rows),
		(*C.int32_t)(unsafe.Pointer(&l.acc[0])))
}

func conv2DScratchLen(pl *convPlan) int { return pieScratchLen(pl.wPerOut, pl.outC) }

func conv2DPack(scratch, w []int8, bias []int32, _ []Requant, pl *convPlan) {
	l := newPieLayout(scratch, pl.wPerOut, pl.outC)
	l.pack(w, bias, pl.inZP)
}

func conv2DRun(dst, src, _, scratch []int8, _ []int32, rq []Requant, pl *convPlan) {
	l := newPieLayout(scratch, pl.wPerOut, pl.outC)
	g := &pl.geometry
	inC := g.ch
	rowLen := g.kw * inC
	zp := int8(pl.inZP)
	zpOut, lo, hi := pl.outZP, pl.lo, pl.hi
	patch := l.patch
	di := 0
	for oy := range g.outH {
		iy0 := oy*g.strH - g.padT
		for ox := range g.outW {
			ix, kx0, kx1 := span(ox, g.strW, g.padL, g.kw, g.inW)
			pi := 0
			for ky := 0; ky < g.kh; ky++ {
				y := iy0 + ky
				if y < 0 || y >= g.inH {
					fill(patch[pi:pi+rowLen], zp)
					pi += rowLen
					continue
				}
				left := kx0 * inC
				n := (kx1 - kx0) * inC
				fill(patch[pi:pi+left], zp)
				xo := (y*g.inW + ix + kx0) * inC
				copy(patch[pi+left:pi+left+n], src[xo:xo+n])
				fill(patch[pi+left+n:pi+rowLen], zp)
				pi += rowLen
			}
			l.dotRows()
			for oc := range pl.outC {
				dst[di] = requantize(l.acc[oc]+l.biasEff[oc], rq[oc], zpOut, lo, hi)
				di++
			}
		}
	}
}

func fill(s []int8, v int8) {
	for i := range s {
		s[i] = v
	}
}

// Depthwise scratch layout (aligned), with T = taps, C = channels and
// C16 = C rounded up to 16:
//
//	wpack   T·C16 bytes   w[t][c], channel tail zero
//	zprow   C16 bytes     zpIn (padded taps in the direct path)
//	stage   T·16 bytes    staged 16-channel group (unaligned or partial input)
//	tmp     64 bytes      QACC spill
//	tap     T pointers    per-pixel tap addresses
//	biasEff C int32       bias − zpIn·Σ_t w[t][c]
type pieDWLayout struct {
	taps, c, c16 int
	wpack        []int8
	zprow        []int8
	stage, tmp   []int8
	tap          []uintptr
	biasEff      []int32
}

func roundUp16(n int) int { return (n + 15) &^ 15 }

func depthwiseConv2DScratchLen(pl *convPlan) int {
	c16 := roundUp16(pl.ch)
	return 15 + pl.taps*c16 + c16 + 16*pl.taps + 64 + 4*pl.taps + 4*pl.ch
}

func newPieDWLayout(scratch []int8, pl *convPlan) pieDWLayout {
	off := int((16 - uintptr(unsafe.Pointer(&scratch[0]))&15) & 15)
	s := scratch[off:]
	taps, c := pl.taps, pl.ch
	c16 := roundUp16(c)
	w := taps * c16
	z := w + c16
	st := z + 16*taps
	tp := st + 64
	b := tp + 4*taps
	return pieDWLayout{
		taps: taps, c: c, c16: c16,
		wpack:   s[:w],
		zprow:   s[w:z],
		stage:   s[z:st],
		tmp:     s[st:tp],
		tap:     unsafe.Slice((*uintptr)(unsafe.Pointer(&s[tp])), taps),
		biasEff: unsafe.Slice((*int32)(unsafe.Pointer(&s[b])), c),
	}
}

// depthwiseConv2DPack repacks w[t][c] to stride C16 (from the last tap down,
// so w may alias the start of the aligned scratch), fills the zero-point row
// and precomputes biasEff.
func depthwiseConv2DPack(scratch, w []int8, bias []int32, _ []Requant, pl *convPlan) {
	l := newPieDWLayout(scratch, pl)
	for t := l.taps - 1; t >= 0; t-- {
		row := l.wpack[t*l.c16 : (t+1)*l.c16]
		copy(row[:l.c], w[t*l.c:(t+1)*l.c])
		fill(row[l.c:], 0)
	}
	fill(l.zprow, int8(pl.inZP))
	fill(l.stage, 0)
	for ch := 0; ch < l.c; ch++ {
		var sum int32
		for t := 0; t < l.taps; t++ {
			sum += int32(l.wpack[t*l.c16+ch])
		}
		var b int32
		if len(bias) != 0 {
			b = bias[ch]
		}
		l.biasEff[ch] = b - pl.inZP*sum
	}
}

// depthwiseConv2DRun loads 16-channel groups straight from src when C is a
// multiple of 16 and src is 16-byte aligned (pixel rows are then aligned),
// and stages them otherwise.
func depthwiseConv2DRun(dst, src, _, scratch []int8, _ []int32, rq []Requant, pl *convPlan) {
	l := newPieDWLayout(scratch, pl)
	g := &pl.geometry
	c := l.c
	base := uintptr(unsafe.Pointer(&src[0]))
	direct := c%16 == 0 && base&15 == 0
	var pad uintptr // padded taps: zprow (direct) or NULL (staged)
	var directFlag C.int
	if direct {
		pad = uintptr(unsafe.Pointer(&l.zprow[0]))
		directFlag = 1
	}
	di := 0
	for oy := range g.outH {
		iy0 := oy*g.strH - g.padT
		for ox := range g.outW {
			ix0 := ox*g.strW - g.padL
			t := 0
			for ky := 0; ky < g.kh; ky++ {
				y := iy0 + ky
				for kx := 0; kx < g.kw; kx++ {
					x := ix0 + kx
					if y < 0 || y >= g.inH || x < 0 || x >= g.inW {
						l.tap[t] = pad
					} else {
						l.tap[t] = base + uintptr((y*g.inW+x)*c)
					}
					t++
				}
			}
			C.qi8_pie_dw_px(
				(**C.int8_t)(unsafe.Pointer(&l.tap[0])), C.int(l.taps), directFlag,
				(*C.int8_t)(unsafe.Pointer(&l.zprow[0])),
				(*C.int8_t)(unsafe.Pointer(&l.stage[0])),
				(*C.int8_t)(unsafe.Pointer(&l.wpack[0])), C.int(c), C.int(l.c16),
				(*C.int32_t)(unsafe.Pointer(&l.biasEff[0])), unsafe.Pointer(&rq[0]),
				C.int32_t(pl.outZP), C.int32_t(pl.lo), C.int32_t(pl.hi),
				(*C.int8_t)(unsafe.Pointer(&dst[di])),
				(*C.uint8_t)(unsafe.Pointer(&l.tmp[0])))
			di += c
		}
	}
}

func fullyConnectedScratchLen(pl *fcPlan) int { return pieScratchLen(pl.in, pl.out) }

func fullyConnectedPack(scratch, w []int8, bias []int32, _ []Requant, pl *fcPlan) {
	l := newPieLayout(scratch, pl.in, pl.out)
	l.pack(w, bias, pl.inZP)
}

func fullyConnectedRun(dst, src, _, scratch []int8, _ []int32, rq []Requant, pl *fcPlan) {
	l := newPieLayout(scratch, pl.in, pl.out)
	copy(l.patch, src[:pl.in])
	l.dotRows()
	for o := range pl.out {
		dst[o] = requantize(l.acc[o]+l.biasEff[o], rq[o], pl.outZP, pl.lo, pl.hi)
	}
}

func maxPool2DRun(dst, src []int8, pl *poolPlan) { maxPool2DGeneric(dst, src, pl) }

func avgPool2DRun(dst, src []int8, pl *poolPlan) { avgPool2DGeneric(dst, src, pl) }

func reluRun(dst, src []int8, zp int8) { reluGeneric(dst, src, zp) }

func relu6Run(dst, src []int8, zp, q6 int8) { relu6Generic(dst, src, zp, q6) }

// addRun stays generic: the same 32-bit loop compiled from C by clang was
// not faster on the chip, and PIE has no element-wise 32-bit multiply.
func addRun(dst, a, b []int8, p *AddParams) { addGeneric(dst, a, b, p) }
