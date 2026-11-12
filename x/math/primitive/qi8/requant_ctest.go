//go:build qi8ctest && cgo

package qi8

// Host test harness, tag qi8ctest: compiles the chip kernels' C
// requantisation (requant_c.h, included by pie_esp32s3.c) for the host with
// UndefinedBehaviorSanitizer, so requant_ctest_test.go can compare it with
// the Go reference. Not part of any build without the tag.

// #cgo CFLAGS: -O2 -fsanitize=undefined
// #cgo LDFLAGS: -fsanitize=undefined
// #include "requant_c.h"
// const char *__ubsan_default_options(void) { return "halt_on_error=1:print_stacktrace=1"; }
// static int8_t qi8_ctest_requantize_mode(int32_t acc, int32_t mult, int32_t shift, int32_t zp, int32_t lo, int32_t hi, int round) {
//     qi8_requant_t r = {mult, shift};
//     return qi8_requantize_mode(acc, &r, zp, lo, hi, round);
// }
import "C"

// cRequantizeMode is requantizeMode computed by the C code.
func cRequantizeMode(acc int32, r Requant, zp, lo, hi int64, mode Rounding) int8 {
	return int8(C.qi8_ctest_requantize_mode(C.int32_t(acc), C.int32_t(r.Mult), C.int32_t(r.Shift),
		C.int32_t(zp), C.int32_t(lo), C.int32_t(hi), C.int(mode)))
}
