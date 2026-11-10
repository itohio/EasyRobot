"""Port of TensorFlow Lite's int8 reference arithmetic (default build, i.e.
without TFLITE_SINGLE_ROUNDING), in Python integers with C semantics.

Sources mirrored (tensorflow/lite/kernels/internal/):
  common.h                    MultiplyByQuantizedMultiplier,
                              MultiplyByQuantizedMultiplierSmallerThanOneExp
  quantization_util.cc        QuantizeMultiplier
  kernel_util.cc              CalculateActivationRangeQuantized
  reference/integer_ops/conv.h, depthwise_conv.h, fully_connected.h, add.h,
  reference/integer_ops/pooling.h (AveragePool)
  gemmlowp fixedpoint.h       SaturatingRoundingDoublingHighMul, RoundingDivideByPOT
  kernels/add.cc              int8 Add parameters (left_shift 20)
"""
import math

import numpy as np

I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1


def wrap32(x):
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x & 0x80000000 else x


def cdiv(a, b):
    """C integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def srdhm(a, b):
    if a == b == I32_MIN:
        return I32_MAX
    ab = a * b
    nudge = (1 << 30) if ab >= 0 else 1 - (1 << 30)
    return wrap32(cdiv(ab + nudge, 1 << 31))


def rdbp(x, exponent):
    assert 0 <= exponent <= 31
    mask = (1 << exponent) - 1
    remainder = x & mask  # Python & on negatives = two's complement, as in C for int32
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> exponent) + (1 if remainder > threshold else 0)


def mbqm(x, qm, shift):
    left = shift if shift > 0 else 0
    right = 0 if shift > 0 else -shift
    return rdbp(srdhm(wrap32(x * (1 << left)), qm), right)


def mbqm_smaller_than_one(x, qm, left_shift):
    assert left_shift <= 0
    return rdbp(srdhm(x, qm), -left_shift)


def quantize_multiplier(d):
    """QuantizeMultiplier(double) -> (q_fixed, shift)."""
    if d == 0.0:
        return 0, 0
    q, shift = math.frexp(d)
    q_fixed = math.floor(q * (1 << 31) + 0.5)  # std::round, q > 0
    assert q_fixed <= (1 << 31)
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
    if shift < -31:
        shift, q_fixed = 0, 0
    if shift > 30:
        shift, q_fixed = 30, (1 << 31) - 1
    return q_fixed, shift


def f32(x):
    return np.float32(x)


def act_range(kind, scale, zp):
    """CalculateActivationRangeQuantized for int8 (scale float32)."""
    scale = f32(scale)

    def quantize(f):
        return zp + int(_roundf(f32(f) / scale))  # TfLiteRound(f / scale), float32 division

    qmin, qmax = -128, 127
    if kind == "none":
        return qmin, qmax
    if kind == "relu":
        return max(qmin, quantize(0.0)), qmax
    if kind == "relu6":
        return max(qmin, quantize(0.0)), min(qmax, quantize(6.0))
    if kind == "relun1to1":
        return max(qmin, quantize(-1.0)), min(qmax, quantize(1.0))
    raise ValueError(kind)


def _roundf(v):
    """roundf: half away from zero, on a float32 value."""
    v = float(v)
    return math.floor(v + 0.5) if v >= 0 else -math.floor(-v + 0.5)


def same_or_valid(in_, k, s, pad):
    if pad == "same":
        out = (in_ + s - 1) // s
        total = max((out - 1) * s + k - in_, 0)
        return out, total // 2
    return (in_ - k) // s + 1, 0


def conv(src, w, bias, mult, shift, p):
    H, W, C = p["in"]
    OC, KH, KW, SH, SW = p["outC"], p["kh"], p["kw"], p["sh"], p["sw"]
    oh, pt = same_or_valid(H, KH, SH, p["pad"])
    ow, pl = same_or_valid(W, KW, SW, p["pad"])
    in_off, out_off = -p["inZP"], p["outZP"]
    out = []
    for oy in range(oh):
        for ox in range(ow):
            for oc in range(OC):
                acc = 0
                for ky in range(KH):
                    for kx in range(KW):
                        iy, ix = oy * SH - pt + ky, ox * SW - pl + kx
                        if not (0 <= iy < H and 0 <= ix < W):
                            continue
                        for ic in range(C):
                            x = src[(iy * W + ix) * C + ic]
                            f = w[((oc * KH + ky) * KW + kx) * C + ic]
                            acc += f * (x + in_off)
                if bias is not None:
                    acc += bias[oc]
                acc = wrap32(acc)
                acc = mbqm(acc, mult[oc], shift[oc]) + out_off
                out.append(min(max(acc, p["actMin"]), p["actMax"]))
    return out


def depthwise(src, w, bias, mult, shift, p):
    H, W, C = p["in"]
    KH, KW, SH, SW = p["kh"], p["kw"], p["sh"], p["sw"]
    oh, pt = same_or_valid(H, KH, SH, p["pad"])
    ow, pl = same_or_valid(W, KW, SW, p["pad"])
    in_off, out_off = -p["inZP"], p["outZP"]
    out = []
    for oy in range(oh):
        for ox in range(ow):
            for c in range(C):
                acc = 0
                for ky in range(KH):
                    for kx in range(KW):
                        iy, ix = oy * SH - pt + ky, ox * SW - pl + kx
                        if not (0 <= iy < H and 0 <= ix < W):
                            continue
                        acc += w[(ky * KW + kx) * C + c] * (src[(iy * W + ix) * C + c] + in_off)
                if bias is not None:
                    acc += bias[c]
                acc = wrap32(acc)
                acc = mbqm(acc, mult[c], shift[c]) + out_off
                out.append(min(max(acc, p["actMin"]), p["actMax"]))
    return out


def single_round_away(acc, m, shift):
    """acc·m / 2^(31−shift), one rounding, half away from zero (int64)."""
    s = 31 - shift
    p = acc * m
    half = 1 << (s - 1)
    return (p + half) >> s if p >= 0 else -((-p + half) >> s)


def fully_connected(src, w, bias, mult, shift, p):
    """FULLY_CONNECTED int8 rounds acc·M' once, half away from zero, for
    per-tensor and per-channel weights alike — established against the
    LiteRT 2.3.0 interpreter with the reference resolver (150 of 150 cases;
    double rounding as in MultiplyByQuantizedMultiplier matched 129). CONV_2D
    and DEPTHWISE_CONV_2D do use double rounding (0 mismatches)."""
    n_in, n_out = p["inLen"], p["outLen"]
    out = []
    for o in range(n_out):
        acc = sum(w[o * n_in + i] * (src[i] - p["inZP"]) for i in range(n_in))
        if bias is not None:
            acc += bias[o]
        acc = wrap32(acc)
        acc = single_round_away(acc, mult[o], shift[o]) + p["outZP"]
        out.append(min(max(acc, p["actMin"]), p["actMax"]))
    return out


def add_params(scale_a, scale_b, scale_out):
    """kernels/add.cc Prepare, int8: returns ((mA,sA),(mB,sB),(mO,sO))."""
    sa, sb, so = f32(scale_a), f32(scale_b), f32(scale_out)
    twice_max = float(f32(2) * max(sa, sb))  # 2 * std::max(float, float) -> float
    ra = float(sa) / twice_max
    rb = float(sb) / twice_max
    ro = twice_max / float(f32(1 << 20) * so)
    res = []
    for r in (ra, rb, ro):
        m, s = quantize_multiplier(r)
        assert s <= 0, r  # QuantizeMultiplierSmallerThanOneExp
        res.append((m, s))
    return res


def add(a, b, p, prm):
    (ma, sa), (mb, sb), (mo, so) = prm
    out = []
    for x, y in zip(a, b):
        xa = (x - p["zpA"]) * (1 << 20)
        yb = (y - p["zpB"]) * (1 << 20)
        raw = mbqm_smaller_than_one(xa, ma, sa) + mbqm_smaller_than_one(yb, mb, sb)
        v = mbqm_smaller_than_one(raw, mo, so) + p["zpOut"]
        out.append(min(max(v, p["actMin"]), p["actMax"]))
    return out


def avg_pool(src, p):
    H, W, C = p["in"]
    KH, KW, SH, SW = p["kh"], p["kw"], p["sh"], p["sw"]
    oh, pt = same_or_valid(H, KH, SH, p["pad"])
    ow, pl = same_or_valid(W, KW, SW, p["pad"])
    out = []
    for oy in range(oh):
        for ox in range(ow):
            for c in range(C):
                acc, n = 0, 0
                for ky in range(KH):
                    for kx in range(KW):
                        iy, ix = oy * SH - pt + ky, ox * SW - pl + kx
                        if 0 <= iy < H and 0 <= ix < W:
                            acc += src[(iy * W + ix) * C + c]
                            n += 1
                acc = cdiv(acc + n // 2, n) if acc > 0 else cdiv(acc - n // 2, n)
                out.append(min(max(acc, p.get("actMin", -128)), p.get("actMax", 127)))
    return out


# ---- Softmax (reference_ops::Softmax for int8, gemmlowp fixed point) ----

def sat_mul_pot(x, exponent):
    """gemmlowp SaturatingRoundingMultiplyByPOT."""
    if exponent == 0:
        return x
    if exponent < 0:
        return rdbp(x, -exponent)
    threshold = (1 << (31 - exponent)) - 1
    if x > threshold:
        return I32_MAX
    if x < -threshold:
        return I32_MIN
    return wrap32(x << exponent)


def rounding_half_sum(a, b):
    s = a + b
    sign = 1 if s >= 0 else -1
    return wrap32(cdiv(s + sign, 2))


def exp_on_interval_between_neg_quarter_and_0(a):  # F0 in, F0 out
    constant_term = 1895147668
    constant_1_over_3 = 715827883
    x = wrap32(a + (1 << 28))
    x2 = srdhm(x, x)
    x3 = srdhm(x2, x)
    x4 = srdhm(x2, x2)
    x4_over_4 = sat_mul_pot(x4, -2)
    t = sat_mul_pot(wrap32(srdhm(wrap32(x4_over_4 + x3), constant_1_over_3) + x2), -1)
    return wrap32(constant_term + srdhm(constant_term, wrap32(x + t)))


def exp_on_negative_values(a, int_bits=5):  # Q(int_bits) in, F0 out
    frac = 31 - int_bits
    one_quarter = 1 << (frac - 2)
    mask = one_quarter - 1
    a_mod = wrap32((a & mask) - one_quarter)
    result = exp_on_interval_between_neg_quarter_and_0(sat_mul_pot(a_mod, int_bits))  # Rescale<0>
    remainder = wrap32(a_mod - a)
    for e, m in [(-2, 1672461947), (-1, 1302514674), (0, 790015084), (1, 290630308),
                 (2, 39332535), (3, 720401), (4, 242)]:
        if int_bits > e:
            if remainder & (1 << (frac + e)):
                result = srdhm(result, m)
    if int_bits > 5:
        clamp = -(1 << (36 - int_bits))
        if a < clamp:
            result = 0
    if a == 0:
        result = I32_MAX
    return result


def one_over_one_plus_x_for_x_in_0_1(a):  # F0 in, F0 out
    half_denominator = rounding_half_sum(a, I32_MAX)
    c48_17 = 1515870810
    cn32_17 = -1010580540
    x = wrap32(c48_17 + srdhm(half_denominator, cn32_17))  # F2
    one_f2 = 1 << 29
    for _ in range(3):
        hdx = srdhm(half_denominator, x)  # F2
        one_minus = wrap32(one_f2 - hdx)
        x = wrap32(x + sat_mul_pot(srdhm(x, one_minus), 2))  # F4 -> Rescale<2>
    return sat_mul_pot(x, 1)  # ExactMulByPot<-1> -> F1, Rescale<0>


def clz32(x):
    x &= 0xFFFFFFFF
    return 32 - x.bit_length()


def get_reciprocal(x, x_integer_digits):
    headroom_plus_one = clz32(x)
    num_bits_over_unit = x_integer_digits - headroom_plus_one
    shifted_sum_minus_one = wrap32(((x & 0xFFFFFFFF) << headroom_plus_one) - (1 << 31))
    return one_over_one_plus_x_for_x_in_0_1(shifted_sum_minus_one), num_bits_over_unit


def softmax_params(beta, input_scale):
    """PreprocessSoftmaxScaling + CalculateInputRadius (kScaledDiffIntegerBits = 5)."""
    int_bits = 5
    real = min(float(f32(beta)) * float(f32(input_scale)) * (1 << (31 - int_bits)), (1 << 31) - 1.0)
    mult, left_shift = quantize_multiplier(real)
    assert left_shift >= 0
    max_input_rescaled = 1.0 * ((1 << int_bits) - 1) * (1 << (31 - int_bits)) / (1 << left_shift)
    diff_min = -int(math.floor(max_input_rescaled))
    return mult, left_shift, diff_min


def softmax(src, depth, mult, left_shift, diff_min):
    out = []
    for r in range(0, len(src), depth):
        row = src[r:r + depth]
        mx = max(row)
        sum_of_exps = 0
        for v in row:
            d = v - mx
            if d >= diff_min:
                rescaled = srdhm(wrap32(d * (1 << left_shift)), mult)
                sum_of_exps = wrap32(sum_of_exps + rdbp(exp_on_negative_values(rescaled), 12))
        shifted_scale, nbits = get_reciprocal(sum_of_exps, 12)
        for v in row:
            d = v - mx
            if d >= diff_min:
                rescaled = srdhm(wrap32(d * (1 << left_shift)), mult)
                e = exp_on_negative_values(rescaled)
                unsat = rdbp(srdhm(shifted_scale, e), nbits + 31 - 8)
                out.append(min(max(unsat - 128, -128), 127))
            else:
                out.append(-128)
    return out
