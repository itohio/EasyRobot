//go:build esp32s3 && !qi8generic

// ESP32-S3 PIE (SIMD) inner loops for the qi8 prepared layers.
// See impl_esp32s3.go for the data layout and the exactness argument.

#include <stdint.h>

// qi8_pie_enable sets CPENABLE bit 3 (the PIE coprocessor). TinyGo enables
// only the FPU (bit 0); any ee.* instruction traps until this runs.
void qi8_pie_enable(void) {
    uint32_t v;
    __asm__ volatile ("rsr.cpenable %0" : "=r"(v));
    v |= 0x08;
    __asm__ volatile ("wsr.cpenable %0\n rsync" :: "r"(v));
}

// qi8_pie_dot_rows computes out[r] = sum_{i<k} patch[i] * w[r*k + i] for
// r < rows (wrapping int32, the low 32 bits of the 40-bit ACCX).
//
// Requirements: patch and w 16-byte aligned; k a positive multiple of 32;
// 16 readable bytes after patch[k-1] and after w[rows*k-1] (the software
// pipeline loads one block ahead; the extra block is never multiplied).
void qi8_pie_dot_rows(const int8_t *patch, const int8_t *w, int k, int rows, int32_t *out) {
    for (int r = 0; r < rows; r++) {
        const int8_t *a = patch;
        int n = k >> 5;
        int32_t acc;
        __asm__ volatile (
            "ee.zero.accx\n"
            "ee.vld.128.ip q0, %1, 16\n"
            "ee.vld.128.ip q1, %2, 16\n"
            "loopnez %3, 1f\n"
            "ee.vld.128.ip q3, %2, 16\n"
            "ee.vmulas.s8.accx.ld.ip q2, %1, 16, q0, q1\n"
            "ee.vld.128.ip q1, %2, 16\n"
            "ee.vmulas.s8.accx.ld.ip q0, %1, 16, q2, q3\n"
            "1:\n"
            "rur.accx_0 %0\n"
            : "=r"(acc), "+r"(a), "+r"(w), "+r"(n)
            :
            : "memory");
        w -= 16; // the pipeline read one block past this row
        out[r] = acc;
    }
}
