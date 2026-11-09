//go:build esp32s3 && !qi8generic && !qi8pieemu

package qi8

// The chip side of the PIE backend: thin wrappers over the C kernels in
// pie_esp32s3.c (impl_esp32s3.go holds the layouts and loops).

// #include <stdint.h>
// void qi8_pie_enable(void);
// void qi8_pie_dot_rows(const int8_t *patch, const int8_t *w, int k, int rows, int32_t *out);
// void qi8_pie_dw_px(const int8_t *const *tap, int taps, int direct, const int8_t *zprow, int8_t *stage, const int8_t *w, int ch, int c16, const int32_t *biasEff, const void *rq, int32_t zp, int32_t lo, int32_t hi, int8_t *dst, uint8_t *tmp);
import "C"

import "unsafe"

func init() { C.qi8_pie_enable() }

// pieDotRows computes acc[r] = Σ_{i<k32} patch[i]·w[r·k32+i] (wrapping int32)
// for r < rows; patch and w are 16-byte aligned and followed by 16 readable
// bytes.
func pieDotRows(patch, w []int8, k32, rows int, acc []int32) {
	C.qi8_pie_dot_rows(
		(*C.int8_t)(unsafe.Pointer(&patch[0])),
		(*C.int8_t)(unsafe.Pointer(&w[0])),
		C.int(k32), C.int(rows),
		(*C.int32_t)(unsafe.Pointer(&acc[0])))
}

// pieDWPx computes one depthwise output pixel into dst (C bytes) from the
// tap addresses in l.tap (see qi8_pie_dw_px). src is the layer input; the C
// kernel reads it through the tap addresses.
func pieDWPx(l *pieDWLayout, _ []int8, direct bool, rq []Requant, pl *convPlan, dst []int8) {
	var directFlag C.int
	if direct {
		directFlag = 1
	}
	C.qi8_pie_dw_px(
		(**C.int8_t)(unsafe.Pointer(&l.tap[0])), C.int(l.taps), directFlag,
		(*C.int8_t)(unsafe.Pointer(&l.zprow[0])),
		(*C.int8_t)(unsafe.Pointer(&l.stage[0])),
		(*C.int8_t)(unsafe.Pointer(&l.wpack[0])), C.int(l.c), C.int(l.c16),
		(*C.int32_t)(unsafe.Pointer(&l.biasEff[0])), unsafe.Pointer(&rq[0]),
		C.int32_t(pl.outZP), C.int32_t(pl.lo), C.int32_t(pl.hi),
		(*C.int8_t)(unsafe.Pointer(&dst[0])),
		(*C.uint8_t)(unsafe.Pointer(&l.tmp[0])))
}
