//go:build qi8pieemu && !qi8generic

package qi8

// Test-only emulation of the PIE kernels (build tag qi8pieemu): pure-Go
// models of qi8_pie_dot_rows and qi8_pie_dw_px, so that impl_esp32s3.go's
// layouts, packing and loops run under `go test -tags qi8pieemu` on the
// host. The models compute the kernels' documented results and panic when
// a precondition the assembly relies on is violated (16-byte alignment, the
// 16 readable bytes after each row and the patch, K a multiple of 32, tap
// addresses inside the input or zprow). It is never part of a firmware
// build: the backend is still chosen at compile time.

import "unsafe"

func emuAligned(s []int8, what string) {
	if uintptr(unsafe.Pointer(&s[0]))&15 != 0 {
		panic("qi8 pie emu: " + what + " not 16-byte aligned")
	}
}

func pieDotRows(patch, w []int8, k32, rows int, acc []int32) {
	if k32%32 != 0 || k32 <= 0 || rows <= 0 {
		panic("qi8 pie emu: bad dot_rows shape")
	}
	emuAligned(patch, "patch")
	emuAligned(w, "weights")
	// The pipelined loop loads one 16-byte block past every row and the patch.
	if len(patch) < k32+16 || len(w) < rows*k32+16 || len(acc) < rows {
		panic("qi8 pie emu: dot_rows buffers lack the 16-byte read-ahead slack")
	}
	for r := range rows {
		var s int64 // ACCX is 40 bits; the kernel returns its low 32
		row := w[r*k32 : (r+1)*k32]
		for i, x := range patch[:k32] {
			s += int64(x) * int64(row[i])
		}
		acc[r] = int32(s)
	}
}

func pieDotRowsRQ(l *pieLayout, rq []Requant, zp, lo, hi int64, round Rounding, dst []int8) {
	pieDotRows(l.patch, l.wpack, l.k32, l.rows, l.acc)
	for r := range l.rows {
		dst[r] = requantizeMode(l.acc[r]+l.biasEff[r], rq[r], zp, lo, hi, round)
	}
}

func pieDWPx(l *pieDWLayout, src []int8, direct bool, rq []Requant, pl *convPlan, dst []int8) {
	emuAligned(l.wpack, "depthwise weights")
	emuAligned(l.zprow, "zprow")
	base := uintptr(unsafe.Pointer(&src[0]))
	zpAddr := uintptr(unsafe.Pointer(&l.zprow[0]))
	if direct && (l.c%16 != 0 || base&15 != 0) {
		panic("qi8 pie emu: direct depthwise needs C%16 == 0 and an aligned input")
	}
	for ch := range l.c {
		acc := l.biasEff[ch]
		for t, a := range l.tap[:l.taps] {
			var x int8
			switch {
			case a == 0 && !direct, a == zpAddr && direct:
				x = l.zprow[ch]
			case a == 0 || a == zpAddr:
				panic("qi8 pie emu: padded tap encoded for the wrong path")
			default:
				off := int(a - base)
				if a < base || off%l.c != 0 || off+l.c > len(src) {
					panic("qi8 pie emu: tap address outside the input")
				}
				if direct && a&15 != 0 {
					panic("qi8 pie emu: direct tap not 16-byte aligned")
				}
				x = src[off+ch]
			}
			acc += int32(x) * int32(l.wpack[t*l.c16+ch])
		}
		dst[ch] = requantizeMode(acc, rq[ch], pl.outZP, pl.lo, pl.hi, pl.round)
	}
}
