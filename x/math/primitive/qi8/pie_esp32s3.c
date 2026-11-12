//go:build esp32s3 && !qi8generic && !qi8pieemu

// ESP32-S3 PIE (SIMD) inner loops for the qi8 prepared layers.
// See impl_esp32s3.go for the data layout and the exactness argument.
//
// Undeclared registers (assumption; recheck on every TinyGo/clang upgrade).
// The asm below uses the zero-overhead loop registers LBEG/LEND/LCOUNT
// (loopnez), the PIE vector registers q0–q3 and the accumulators ACCX and
// QACC without naming them as clobbers: TinyGo 0.43's clang rejects those
// register names in clobber lists. This is correct only while no
// compiler-generated code in these functions uses hardware loops or PIE.
// Checked 2026-11-12 for qi8_dot_row (inlined into qi8_pie_dot_rows_rq)
// and qi8_pie_dw_px: `tinygo clang --target=xtensa -mcpu=esp32s3 -S` at
// -Oz, -O1, -O2, -O3 and -Os emits exactly the asm's own 3 loop
// instructions and 23 ee.* instructions (the enclosing C loops are plain
// branch loops),
// and while no interrupt handler uses them: TinyGo's interrupt entry saves
// neither the loop registers nor PIE state, so an ISR that used either would
// corrupt a running kernel. Moving the loops into a separate .S file (as
// ESP-DSP does) would remove the first assumption, not the second.

#include <stdint.h>

#include "requant_c.h"

// qi8_pie_enable sets CPENABLE bit 3 (the PIE coprocessor). TinyGo enables
// only the FPU (bit 0); any ee.* instruction traps until this runs.
void qi8_pie_enable(void) {
    uint32_t v;
    __asm__ volatile ("rsr.cpenable %0" : "=r"(v));
    v |= 0x08;
    __asm__ volatile ("wsr.cpenable %0\n rsync" :: "r"(v));
}

// qi8_dot_row returns sum_{i<k} patch[i] * w[i] for one row (wrapping int32,
// the low 32 bits of the 40-bit ACCX) and advances *w to the next row.
//
// Requirements: patch and w 16-byte aligned; k a positive multiple of 32;
// 16 readable bytes after patch[k-1] and after the row (the software
// pipeline loads one block ahead; the extra block is never multiplied).
static inline int32_t qi8_dot_row(const int8_t *patch, const int8_t **wp, int k) {
    const int8_t *a = patch;
    const int8_t *w = *wp;
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
    *wp = w - 16; // the pipeline read one block past this row
    return acc;
}

// qi8_pie_dot_rows_rq computes each row's dot product (qi8_dot_row) and
// then the layer's requantisation: dst[r] = requantize(dot_r + biasEff[r],
// rq[r]) with output zero point zp, clamp [lo, hi] and rounding mode round
// (as qi8_requantize_mode). It saves the per-output round trip to Go.
void qi8_pie_dot_rows_rq(const int8_t *patch, const int8_t *w, int k, int rows, const int32_t *biasEff,
                         const qi8_requant_t *rq, int32_t zp, int32_t lo, int32_t hi, int round, int8_t *dst) {
    for (int r = 0; r < rows; r++) {
        int32_t acc = qi8_dot_row(patch, &w, k);
        dst[r] = qi8_requantize_mode((int32_t)((uint32_t)acc + (uint32_t)biasEff[r]), &rq[r], zp, lo, hi, round);
    }
}

// qi8_pie_dw_px computes one output pixel of a depthwise convolution:
//
//	dst[c] = requantize(biasEff[c] + sum_t in[t][c] * w[t*c16 + c], rq[c], ...)
//
// for c < ch (c16 = ch rounded up to 16; weight lanes past ch are zero, w is
// 16-byte aligned). tap[t] points at channel 0 of tap t's input pixel, or is
// NULL for a padded cell (which reads as zp, matching the zp·Σw folded into
// biasEff).
//
// direct != 0 (ch a multiple of 16, every non-NULL tap 16-byte aligned):
// 16-channel groups are loaded straight from the input; the caller passes
// zprow (c16 bytes of the input zero point, aligned) for padded taps.
// Otherwise each group is staged: up to 16 bytes per tap are copied (padded
// taps copy zprow) into stage (16·taps aligned bytes) and loaded from there.
// zp, lo and hi are the output zero point and clamp; round selects the
// rounding mode (0 RoundExact, 1 RoundTFLite).
//
// Products accumulate per lane in QACC (16 lanes of 20 bits), which saturates
// at ±(2^19−1): taps are summed in chunks of at most 31 (31·128·128 = 507 904
// < 2^19) and chunks are added in wrapping int32, so the sum is exact. tmp is
// 64 aligned bytes. QACC_L holds lanes 0–7 and QACC_H lanes 8–15, each as 8
// little-endian 20-bit two's-complement fields (lane i at bits 20i..20i+19),
// stored with ee.st.qacc_{l,h}.l.128 (bytes 0–15) and .h.32 (bytes 16–19).
void qi8_pie_dw_px(const int8_t *const *tap, int taps, int direct, const int8_t *zprow, int8_t *stage,
                   const int8_t *w, int ch, int c16,
                   const int32_t *biasEff, const qi8_requant_t *rq, int32_t zp, int32_t lo, int32_t hi, int round,
                   int8_t *dst, uint8_t *tmp) {
    uint32_t a[16]; // wrapping int32 arithmetic, as in the Go reference
    for (int g = 0; g < c16; g += 16) {
        int width = ch - g < 16 ? ch - g : 16;
        for (int i = 0; i < 16; i++) {
            a[i] = 0;
        }
        for (int t0 = 0; t0 < taps; t0 += 31) {
            int n = taps - t0;
            if (n > 31) {
                n = 31;
            }
            const int8_t *pw = w + t0 * c16 + g;
            uint8_t *o = tmp;
            if (direct) {
                const int8_t *const *pp = tap + t0;
                const int8_t *pt;
                __asm__ volatile (
                    "ee.zero.qacc\n"
                    "loopnez %[n], 1f\n"
                    "l32i %[pt], %[pp], 0\n"
                    "addi %[pp], %[pp], 4\n"
                    "add %[pt], %[pt], %[g]\n"
                    "ee.vld.128.ip q0, %[pt], 0\n"
                    "ee.vld.128.xp q1, %[pw], %[c16]\n"
                    "ee.vmulas.s8.qacc q0, q1\n"
                    "1:\n"
                    "ee.st.qacc_l.l.128.ip %[o], 16\n"
                    "ee.st.qacc_l.h.32.ip %[o], 16\n"
                    "ee.st.qacc_h.l.128.ip %[o], 16\n"
                    "ee.st.qacc_h.h.32.ip %[o], 16\n"
                    : [pp] "+r"(pp), [pw] "+r"(pw), [pt] "=&r"(pt), [o] "+r"(o)
                    : [n] "r"(n), [g] "r"(g), [c16] "r"(c16)
                    : "memory");
            } else {
                for (int t = 0; t < n; t++) {
                    int8_t *s = stage + 16 * t;
                    const int8_t *in = tap[t0 + t];
                    if (in == 0) {
                        for (int i = 0; i < 16; i++) {
                            s[i] = zprow[i];
                        }
                    } else {
                        for (int i = 0; i < width; i++) {
                            s[i] = in[g + i];
                        }
                    }
                }
                const int8_t *ps = stage;
                __asm__ volatile (
                    "ee.zero.qacc\n"
                    "loopnez %[n], 2f\n"
                    "ee.vld.128.ip q0, %[ps], 16\n"
                    "ee.vld.128.xp q1, %[pw], %[c16]\n"
                    "ee.vmulas.s8.qacc q0, q1\n"
                    "2:\n"
                    "ee.st.qacc_l.l.128.ip %[o], 16\n"
                    "ee.st.qacc_l.h.32.ip %[o], 16\n"
                    "ee.st.qacc_h.l.128.ip %[o], 16\n"
                    "ee.st.qacc_h.h.32.ip %[o], 16\n"
                    : [ps] "+r"(ps), [pw] "+r"(pw), [o] "+r"(o)
                    : [n] "r"(n), [c16] "r"(c16)
                    : "memory");
            }
            for (int h = 0; h < 2; h++) {
                const uint8_t *b = tmp + 32 * h;
                for (int i = 0; i < 8; i++) {
                    int bit = 20 * i;
                    const uint8_t *p = b + (bit >> 3);
                    uint32_t v = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
                    v = (v >> (bit & 7)) & 0xFFFFF;
                    a[h * 8 + i] += (uint32_t)((int32_t)(v ^ 0x80000) - 0x80000);
                }
            }
        }
        for (int i = 0; i < width; i++) {
            int32_t acc = (int32_t)(a[i] + (uint32_t)biasEff[g + i]);
            dst[g + i] = qi8_requantize_mode(acc, &rq[g + i], zp, lo, hi, round);
        }
    }
}
