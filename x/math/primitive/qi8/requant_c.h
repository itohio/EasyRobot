// Requantisation for the C kernels (pie_esp32s3.c), in plain C so the host
// test harness (requant_ctest.go, tag qi8ctest) compiles the same code and
// checks it against the Go reference (quant.go, tflite.go).

#ifndef QI8_REQUANT_C_H
#define QI8_REQUANT_C_H

#include <stdint.h>

// qi8_requant_t matches qi8.Requant{Mult, Shift int32}.
typedef struct {
    int32_t mult, shift;
} qi8_requant_t;

// qi8_mulhi returns the high 32 bits of a·b (and bit 31 of the low word in
// *lo31) from four 16×16-bit products, which on the chip is faster than the
// toolchain's 64-bit multiply (measured).
static inline uint32_t qi8_mulhi(uint32_t a, uint32_t b, uint32_t *lo31) {
    uint32_t al = a & 0xFFFF, ah = a >> 16, bl = b & 0xFFFF, bh = b >> 16;
    uint32_t lh = al * bh, hl = ah * bl;
    uint32_t mid = ((al * bl) >> 16) + (lh & 0xFFFF) + (hl & 0xFFFF);
    *lo31 = (mid >> 15) & 1;
    return ah * bh + (lh >> 16) + (hl >> 16) + (mid >> 16);
}

// qi8_add_zp_clamp is clamp(v + zp, lo, hi) as Go computes it in int64,
// without forming v + zp: v may be any int32, and the sum can overflow int32
// (undefined in C). lo − zp and hi − zp cannot overflow: lo, hi and zp are
// int8-range values.
static inline int8_t qi8_add_zp_clamp(int32_t v, int32_t zp, int32_t lo, int32_t hi) {
    if (v < lo - zp) {
        return (int8_t)lo;
    }
    if (v > hi - zp) {
        return (int8_t)hi;
    }
    return (int8_t)(v + zp);
}

// qi8_requantize is requantize from quant.go: one exact rounding (half away
// from zero) of acc·mult / 2^(31−shift), plus zp, clamped to [lo, hi]. For
// shift < 0 (total shift ≥ 32) it uses the same 32-bit form as Requant.Apply:
// only the high word of |acc|·mult and, at shift −1, the low word's top bit
// matter (|result| < 2^30, so adding zp and clamping stays in 32 bits).
static inline int8_t qi8_requantize(int32_t acc, const qi8_requant_t *r, int32_t zp, int32_t lo, int32_t hi) {
    if (r->shift < 0) {
        uint32_t a = acc < 0 ? 0u - (uint32_t)acc : (uint32_t)acc;
        uint32_t lo31;
        uint32_t h = qi8_mulhi(a, (uint32_t)r->mult, &lo31);
        uint32_t k = (uint32_t)(-1 - r->shift);
        uint32_t m = k == 0 ? h + lo31 : (h + (1u << (k - 1))) >> k;
        int32_t v = (acc < 0 ? -(int32_t)m : (int32_t)m) + zp;
        if (v < lo) {
            v = lo;
        }
        if (v > hi) {
            v = hi;
        }
        return (int8_t)v;
    }
    int64_t p = (int64_t)acc * (int64_t)r->mult;
    int s = 31 - r->shift;
    int64_t half = (int64_t)1 << (s - 1);
    int64_t v = p >= 0 ? (p + half) >> s : -((-p + half) >> s);
    v += zp;
    if (v < lo) {
        v = lo;
    }
    if (v > hi) {
        v = hi;
    }
    return (int8_t)v;
}

// qi8_apply_tflite is Requant.ApplyTFLite (tflite.go): TFLite's
// MultiplyByQuantizedMultiplier with its two roundings.
static inline int32_t qi8_apply_tflite(int32_t acc, const qi8_requant_t *r) {
    int32_t x = acc;
    int right = 0;
    if (r->shift > 0) {
        x = (int32_t)((uint32_t)acc << r->shift);
    } else {
        right = -r->shift;
    }
    int64_t ab = (int64_t)x * (int64_t)r->mult;
    int64_t nudge = ab >= 0 ? ((int64_t)1 << 30) : 1 - ((int64_t)1 << 30);
    int32_t h = (int32_t)((ab + nudge) / ((int64_t)1 << 31));
    int32_t mask = (int32_t)(((int64_t)1 << right) - 1);
    int32_t threshold = (mask >> 1) + (h < 0 ? 1 : 0);
    return (h >> right) + ((h & mask) > threshold ? 1 : 0);
}

// qi8_requantize_mode is requantizeMode: round 1 = RoundTFLite. The TFLite
// result can be any int32 (e.g. acc = 2^31−1 at a multiplier near 1), so
// the zero point is added without overflow (an int32 add
// wrapped to the opposite clamp, −128 where Go gives 127).
static inline int8_t qi8_requantize_mode(int32_t acc, const qi8_requant_t *r, int32_t zp, int32_t lo, int32_t hi, int round) {
    if (!round) {
        return qi8_requantize(acc, r, zp, lo, hi);
    }
    return qi8_add_zp_clamp(qi8_apply_tflite(acc, r), zp, lo, hi);
}

#endif
