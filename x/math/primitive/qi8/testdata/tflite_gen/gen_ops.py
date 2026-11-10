"""Generate qi8 fixtures for LOGISTIC, MUL, MEAN, QUANTIZE, CONCATENATION,
PAD/PADV2 and RESIZE_NEAREST_NEIGHBOR (set 1, tflite_ops.json.gz) and for
LEAKY_RELU and RESIZE_BILINEAR (set 2, tflite_ops2.json.gz). Inputs are random
(fixed seed per set); every expected output is produced by the real TFLite
interpreter (ai-edge-litert, OpResolverType.BUILTIN_REF = reference kernels)
running a one-operator model. The tflref.py port is used only to *choose*
discriminating inputs (values where candidate arithmetics disagree), never to
compute expected outputs.

usage: .venv/bin/python gen_ops.py OUT.json
       .venv/bin/python gen_ops.py --ops2 OUT2.json
"""
import json
import math
import random
import sys

import numpy as np
from ai_edge_litert import schema_py_generated as S

from interp_check import ACT, M, T_INT8, T_INT32, invoke
import tflref as T

R = random.Random(20261110)
f32 = np.float32
OP = S.BuiltinOperator


def fl(x):
    """A float32 value as a Python float (exact in JSON)."""
    return float(f32(x))


def zp():
    return R.choice([-128, 127, 0, R.randint(-128, 127), R.randint(-128, 127)])


def int8s(n):
    k = R.randrange(8)
    if k == 0:
        return [R.choice([-128, 127]) for _ in range(n)]
    if k == 1:
        return [R.choice([-128, 127])] * n
    if k == 2:  # near a random centre: small differences, frequent ties
        c = R.randint(-120, 120)
        return [max(-128, min(127, c + R.randint(-6, 6))) for _ in range(n)]
    return [R.randint(-128, 127) for _ in range(n)]


def scale():
    k = R.randrange(4)
    if k == 0:
        return fl(2.0 ** -R.randint(0, 10))
    if k == 1:
        return fl(R.choice([6 / 255, 1 / 128, 1 / 255, 0.0235, 0.1]))
    return fl(2.0 ** R.uniform(-11, 1))


# ---- vectorised TFLite MultiplyByQuantizedMultiplier (only to pick inputs)

def mbqm_np(x, qm, shift):
    x = np.asarray(x, dtype=np.int64)
    if shift > 0:
        x = x << shift
        x = ((x + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)
    ab = x * qm
    t = ab + np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    h = np.where(t >= 0, t >> 31, -((-t) >> 31))
    e = max(-shift, 0)
    mask = (1 << e) - 1
    thr = (mask >> 1) + (h < 0)
    return (h >> e) + ((h & mask) > thr)


# ---- LOGISTIC

def logistic_f32_f64_differ(s, z):
    """True when the float32 LUT formula and a float64 evaluation of the
    sigmoid give different int8 tables (rare)."""
    v = np.arange(-128, 128)
    with np.errstate(over="ignore"):
        d = f32(s) * (v - z).astype(np.float32)
        e = np.exp(-d.astype(np.float64)).astype(np.float32)
        t = f32(1) / (f32(1) + e)
        q32 = np.floor(t.astype(np.float64) * 256 + 0.5)
        q64 = np.floor(256 / (1 + np.exp(-(float(f32(s)) * (v - z)))) + 0.5)
    return bool((q32 != q64).any())


LOGISTIC_RANDOM = 300  # then targeted cases where float32 vs float64 matters


def logistic_case(i):
    if i < LOGISTIC_RANDOM:
        s = [fl(2.0 ** R.uniform(-9, 1)), fl(2.0 ** -R.randint(2, 8)), scale()][i % 3]
        z = zp()
    else:
        while True:
            s, z = fl(2.0 ** R.uniform(-12, 3)), R.randint(-128, 127)
            if logistic_f32_f64_differ(s, z):
                break
    m = M()
    x = m.tensor([1, 256], T_INT8, [s], [z])
    y = m.tensor([1, 256], T_INT8, [1.0 / 256], [-128])
    m.op(OP.LOGISTIC, [x], [y], 0, None)
    out = invoke(m.bytes([x], [y]), [list(range(-128, 128))])
    return {"scale": s, "zp": z, "out": out}


# ---- MUL

ACTS = ["none", "relu", "relu6", "relun1to1"]


def mul_scales(i):
    """Scales for case i and the discrimination its inputs target: "round"
    (few-bit multiplier, .5 ties: single vs double rounding), "deriv"
    (float32 vs float64 multiplier derivation) or None (plain random)."""
    k = i % 5
    if k == 0:  # few-bit real multiplier c·2^-e (c odd ≤ 7): exact .5 ties
        sa = fl(R.choice([1, 3, 5, 7]) * 2.0 ** -R.randint(2, 8))
        sb = fl(2.0 ** -R.randint(0, 8))
        so = fl(2.0 ** (math.floor(math.log2(sa * sb)) + R.randint(4, 10)))
        return sa, sb, so, "round"
    if k in (1, 2):
        return scale(), scale(), scale(), "deriv"
    return scale(), scale(), scale(), None


def exact_np(x, qm, shift):
    """round(x·qm·2^(shift−31)), half away from zero (qi8 RoundExact)."""
    x = np.asarray(x, dtype=np.int64)
    t = 31 - shift
    v = (np.abs(x) * qm + (1 << (t - 1))) >> t
    return np.where(x < 0, -v, v)


def targeted_mul_inputs(target, sa, sb, so, za, zb, zo, lo, hi, na, nb):
    """Inputs (a, b[i % nb]) whose final (clamped) outputs differ between the
    two candidate arithmetics of target, or None when there are none."""
    q1, s1 = T.quantize_multiplier(float(f32(f32(f32(sa) * f32(sb)) / f32(so))))
    a = np.arange(-128, 128, dtype=np.int64)
    A, B = np.meshgrid(a, a, indexing="ij")
    P = (A - za) * (B - zb)
    v1 = mbqm_np(P, q1, s1)
    if target == "deriv":
        q2, s2 = T.quantize_multiplier(float(f32(sa)) * float(f32(sb)) / float(f32(so)))
        if (q1, s1) == (q2, s2):
            return None
        v2 = mbqm_np(P, q2, s2)
    else:
        v2 = exact_np(P, q1, s1)
    d = np.clip(v1 + zo, lo, hi) != np.clip(v2 + zo, lo, hi)  # d[ia, ib]
    cols = np.flatnonzero(d.any(axis=0))
    if len(cols) == 0:
        return None
    bi = [int(cols[R.randrange(len(cols))]) for _ in range(nb)]
    ai = []
    for i in range(na):
        rows = np.flatnonzero(d[:, bi[i % nb]])
        ai.append(int(rows[R.randrange(len(rows))]))
    return [int(a[i]) for i in ai], [int(a[j]) for j in bi]


def mul_case(i):
    form = i % 3  # 0 elementwise, 1 per-channel, 2 scalar
    act = ACTS[(i // 3) % 4]
    H, W, C = R.randint(1, 4), R.randint(1, 4), R.randint(1, 8)
    na = H * W * C
    nb = [na, C, 1][form]
    plain = R.random() < 0.2  # some targeted-scale cases keep random inputs
    for _ in range(2000):  # targeted: retry scales until disagreeing inputs exist
        sa, sb, so, target = mul_scales(i)
        za, zb, zo = zp(), zp(), zp()
        lo, hi = T.act_range(act, so, zo)
        if lo > hi:
            continue
        A, B = int8s(na), int8s(nb)
        if not target or plain:
            break
        t = targeted_mul_inputs(target, sa, sb, so, za, zb, zo, lo, hi, na, nb)
        if t is not None:
            A, B = t
            break
    bshape = [[1, H, W, C], [1, 1, 1, C] if R.random() < 0.5 else [C], [1] if R.random() < 0.5 else [1, 1, 1, 1]][form]
    m = M()
    a = m.tensor([1, H, W, C], T_INT8, [sa], [za])
    b = m.tensor(bshape, T_INT8, [sb], [zb])
    y = m.tensor([1, H, W, C], T_INT8, [so], [zo])
    o = S.MulOptionsT()
    o.fusedActivationFunction = ACT[act]
    m.op(OP.MUL, [a, b], [y], S.BuiltinOptions.MulOptions, o)
    out = invoke(m.bytes([a, b], [y]), [A, B])
    return {"scaleA": sa, "scaleB": sb, "scaleOut": so, "zpA": za, "zpB": zb, "zpOut": zo,
            "act": act, "a": A, "b": B, "out": out}


# ---- MEAN

def column(total, n):
    """n random int8 values summing to total (total in [−128n, 127n])."""
    base = total // n
    r = total - base * n
    v = [base + 1] * r + [base] * (n - r)
    for _ in range(2 * n):
        a, b = R.randrange(n), R.randrange(n)
        d = R.randint(0, min(127 - v[a], v[b] + 128))
        v[a] += d
        v[b] -= d
    R.shuffle(v)
    return v


def mean_targeted(n, C, si, so, zi, zo):
    """Channel inputs whose outputs differ between TFLite's truncating fold
    of 1/n into the multiplier and a rounded fold, or None."""
    q, s = T.quantize_multiplier(float(f32(si)) / float(f32(so)))
    k = min(n.bit_length() - 1, 32, 31 + s)
    mt, mr = (q << k) // n, ((q << k) + n // 2) // n
    if mt == mr:
        return None
    sums = np.arange((-128 - zi) * n, (127 - zi) * n + 1, dtype=np.int64)  # Σ(x − zi)
    d = np.clip(mbqm_np(sums, mt, s - k) + zo, -128, 127) != np.clip(mbqm_np(sums, mr, s - k) + zo, -128, 127)
    hits = sums[d]
    if len(hits) == 0:
        return None
    cols = [column(int(hits[R.randrange(len(hits))]) + n * zi, n) for _ in range(C)]
    return [cols[c][j] for j in range(n) for c in range(C)]


def mean_case(i):
    targeted = i % 5 == 4
    for _ in range(5000):
        if i < 8:
            H, W = [(1, 1), (7, 7), (1, 7), (7, 1), (3, 3), (5, 5), (3, 5), (1, 1)][i]
        else:
            H, W = R.randint(1, 7), R.randint(1, 7)
        C = R.randint(1, 40)
        si = scale()
        so = [si, fl(si * 2.0 ** R.randint(-3, 3)), scale(), scale()][i % 4]
        zi, zo = zp(), zp()
        if i % 8 == 0:
            zo = zi
        X = int8s(H * W * C)
        if not targeted:
            break
        t = mean_targeted(H * W, C, si, so, zi, zo)
        if t is not None:
            X = t
            break
    keep = i % 2 == 0
    axes = [1, 2] if (i // 2) % 2 == 0 else [2, 1]
    m = M()
    x = m.tensor([1, H, W, C], T_INT8, [si], [zi])
    ax = m.tensor([2], T_INT32, data=np.array(axes, dtype="<i4"))
    y = m.tensor([1, 1, 1, C] if keep else [1, C], T_INT8, [so], [zo])
    o = S.ReducerOptionsT()
    o.keepDims = keep
    m.op(OP.MEAN, [x, ax], [y], S.BuiltinOptions.ReducerOptions, o)
    out = invoke(m.bytes([x], [y]), [X])
    return {"in": [H, W, C], "inScale": si, "inZP": zi, "outScale": so, "outZP": zo,
            "keepDims": keep, "axes": axes, "src": X, "out": out}


# ---- QUANTIZE (int8 -> int8)

def quantize_case(i):
    si = scale()
    k = i % 4
    if k == 0:  # ratio 1
        so = si
    elif k == 1:  # ratio < 1
        so = fl(si * R.choice([2, 4, 3, 1.5, 2.0 ** R.uniform(0.1, 6)]))
    elif k == 2:  # ratio > 1
        so = fl(si / R.choice([2, 4, 3, 1.5, 2.0 ** R.uniform(0.1, 6)]))
    else:
        so = scale()
    zi, zo = zp(), zp()
    X = list(range(-128, 128))
    m = M()
    x = m.tensor([1, 256], T_INT8, [si], [zi])
    y = m.tensor([1, 256], T_INT8, [so], [zo])
    m.op(OP.QUANTIZE, [x], [y], 0, None)
    out = invoke(m.bytes([x], [y]), [X])
    return {"inScale": si, "inZP": zi, "outScale": so, "outZP": zo, "src": X, "out": out}


# ---- CONCATENATION (int8: equal scale and zero point everywhere)

def concat_case(i):
    n = 2 + i % 3
    axis = 1 + (i // 3) % 3
    base = [1, R.randint(1, 4), R.randint(1, 4), R.randint(1, 6)]
    shapes = []
    for _ in range(n):
        s = list(base)
        s[axis] = R.randint(1, 5)
        shapes.append(s)
    s, z = scale(), zp()
    srcs = [int8s(int(np.prod(sh))) for sh in shapes]
    out_shape = list(base)
    out_shape[axis] = sum(sh[axis] for sh in shapes)
    m = M()
    ins = [m.tensor(sh, T_INT8, [s], [z]) for sh in shapes]
    y = m.tensor(out_shape, T_INT8, [s], [z])
    o = S.ConcatenationOptionsT()
    o.axis = axis if R.random() < 0.7 else axis - 4  # negative axes too
    m.op(OP.CONCATENATION, ins, [y], S.BuiltinOptions.ConcatenationOptions, o)
    out = invoke(m.bytes(ins, [y]), srcs)
    return {"shapes": [sh[1:] for sh in shapes], "axis": axis, "srcs": srcs, "out": out}


# ---- PAD / PADV2

def pad_case(i):
    v2 = i % 2 == 1
    H, W, C = R.randint(1, 5), R.randint(1, 5), R.randint(1, 6)
    pads = [[0, 0]] + [[R.randint(0, 3), R.randint(0, 3)] for _ in range(3)]
    if i % 5 == 0:
        pads[3] = [0, 0]  # H/W padding only (the common case)
    s, z = scale(), zp()
    so, zo = s, z
    if not v2 and i % 6 == 0:  # PAD with a different output quantisation: raw copy
        so, zo = scale(), zp()
    X = int8s(H * W * C)
    m = M()
    x = m.tensor([1, H, W, C], T_INT8, [s], [z])
    p = m.tensor([4, 2], T_INT32, data=np.array(pads, dtype="<i4"))
    ins = [x, p]
    value = zo
    if v2:
        value = R.randint(-128, 127)
        ins.append(m.tensor([1], T_INT8, [so], [zo], data=np.array([value], dtype=np.int8)))
    y = m.tensor([1, H + sum(pads[1]), W + sum(pads[2]), C + sum(pads[3])], T_INT8, [so], [zo])
    if v2:
        m.op(OP.PADV2, ins, [y], S.BuiltinOptions.PadV2Options, S.PadV2OptionsT())
    else:
        m.op(OP.PAD, ins, [y], S.BuiltinOptions.PadOptions, S.PadOptionsT())
    out = invoke(m.bytes([x], [y]), [X])
    return {"in": [H, W, C], "pads": pads[1:], "v2": v2, "outZP": zo, "value": value, "src": X, "out": out}


# ---- RESIZE_NEAREST_NEIGHBOR

def resize_case(i):
    ac, hp = i % 4 in (1, 3), i % 4 in (2, 3)
    H, W, C = R.randint(1, 9), R.randint(1, 9), R.randint(1, 4)
    k = (i // 4) % 3
    if k == 0:  # integer upsampling (YOLO-style 2x)
        f = R.choice([2, 2, 3])
        oh, ow = H * f, W * f
    else:
        oh, ow = R.randint(1, 20), R.randint(1, 20)
    X = int8s(H * W * C)
    m = M()
    x = m.tensor([1, H, W, C], T_INT8, [0.05], [-3])
    sz = m.tensor([2], T_INT32, data=np.array([oh, ow], dtype="<i4"))
    y = m.tensor([1, oh, ow, C], T_INT8, [0.05], [-3])
    o = S.ResizeNearestNeighborOptionsT()
    o.alignCorners, o.halfPixelCenters = ac, hp
    m.op(OP.RESIZE_NEAREST_NEIGHBOR, [x, sz], [y], S.BuiltinOptions.ResizeNearestNeighborOptions, o)
    out = invoke(m.bytes([x], [y]), [X])
    return {"in": [H, W, C], "outH": oh, "outW": ow, "alignCorners": ac, "halfPixel": hp, "src": X, "out": out}


COUNTS = {"logistic": LOGISTIC_RANDOM + 12, "mul": 300, "mean": 250, "quantize": 240,
          "concat": 120, "pad": 120, "resize": 160}


# ---- set 2: LEAKY_RELU

def leaky_port(si, zi, so, zo, alpha, deriv):
    """Port prediction (input selection only): float32 or float64 multiplier
    derivation, MultiplyByQuantizedMultiplier per branch."""
    if deriv == "f32":
        ri, ra = float(f32(f32(si) / f32(so))), float(f32(f32(f32(si) * f32(alpha)) / f32(so)))
    else:
        ri, ra = float(f32(si)) / float(f32(so)), float(f32(si)) * float(f32(alpha)) / float(f32(so))
    qi, s_i = T.quantize_multiplier(ri)
    qa, sa = T.quantize_multiplier(ra)
    v = np.arange(-128, 128) - zi
    out = np.where(v >= 0, mbqm_np(v, qi, s_i), mbqm_np(v, qa, sa))
    return np.clip(out + zo, -128, 127)


def leaky_alpha():
    return fl(R.choice([0.1, 0.2, 0.01, 0.3, R.uniform(0, 1), R.uniform(0, 0.3),
                        1.0, 0.0, R.uniform(1, 4), 2.0, 1e-12]))


def leaky_case(i):
    k = i % 4
    zi, zo = zp(), zp()
    if k == 0:  # equal scales
        si = so = scale()
        alpha = leaky_alpha()
    elif k == 1:  # unequal scales
        si, so, alpha = scale(), scale(), leaky_alpha()
    elif k == 2:  # few-bit multipliers: exact .5 ties (single vs double rounding)
        si = fl(2.0 ** -R.randint(2, 9))
        so = fl(si * 2.0 ** R.randint(-2, 2))
        alpha = fl(R.choice([0.5, 0.25, 0.75, 0.125, 0.375, 1.5, 2.5, 3.0, 0.0]))
    else:  # targeted: float32 vs float64 derivation of identity or alpha
        while True:
            v, t = R.randint(1, 127), R.randint(0, 100) + 0.5
            si = scale()
            if R.random() < 0.5:  # v·si/so close to a .5 tie on the identity branch
                so, alpha = fl(si * v / t), fl(R.uniform(0.05, 0.5))
                zi = R.randint(-128, 127 - v)
            else:  # same on the alpha branch
                so = scale()
                alpha = fl(t * so / (v * si))
                zi = R.randint(-128 + v, 127)
            if not (0 < so and 0 < alpha < 10):
                continue
            if (leaky_port(si, zi, so, zo, alpha, "f32") != leaky_port(si, zi, so, zo, alpha, "f64")).any():
                break
    m = M()
    x = m.tensor([1, 256], T_INT8, [si], [zi])
    y = m.tensor([1, 256], T_INT8, [so], [zo])
    o = S.LeakyReluOptionsT()
    o.alpha = alpha
    m.op(OP.LEAKY_RELU, [x], [y], S.BuiltinOptions.LeakyReluOptions, o)
    out = invoke(m.bytes([x], [y]), [list(range(-128, 128))])
    return {"inScale": si, "inZP": zi, "outScale": so, "outZP": zo, "alpha": alpha, "out": out}


# ---- set 2: RESIZE_BILINEAR (align_corners together with half_pixel_centers
# fails to prepare, so only the three accepted flag combinations)

def bilinear_case(i):
    ac, hp = [(False, False), (True, False), (False, True)][i % 3]
    k = (i // 3) % 6
    C = R.randint(1, 4)
    if k == 0:  # integer upsampling
        H, W = R.randint(1, 8), R.randint(1, 8)
        f = R.choice([2, 2, 3])
        oh, ow = H * f, W * f
    elif k == 1:  # downsampling
        H, W = R.randint(4, 12), R.randint(4, 12)
        oh, ow = R.randint(1, H - 1), R.randint(1, W - 1)
    elif k == 2:  # 1-pixel dimensions
        H, W, oh, ow = R.randint(1, 6), R.randint(1, 6), R.randint(1, 12), R.randint(1, 12)
        j = R.randrange(4)
        H, W, oh, ow = [1 if j == n else d for n, d in enumerate((H, W, oh, ow))]
    elif k == 3:  # many channels, small spatial size
        H, W, oh, ow = R.randint(1, 4), R.randint(1, 4), R.randint(1, 6), R.randint(1, 6)
        C = 40 if i < 18 else R.randint(20, 40)
    else:  # random, odd sizes
        H, W, oh, ow = R.randint(1, 9), R.randint(1, 9), R.randint(1, 20), R.randint(1, 20)
    X = int8s(H * W * C)
    m = M()
    x = m.tensor([1, H, W, C], T_INT8, [0.05], [-3])
    sz = m.tensor([2], T_INT32, data=np.array([oh, ow], dtype="<i4"))
    y = m.tensor([1, oh, ow, C], T_INT8, [0.05], [-3])
    o = S.ResizeBilinearOptionsT()
    o.alignCorners, o.halfPixelCenters = ac, hp
    m.op(OP.RESIZE_BILINEAR, [x, sz], [y], S.BuiltinOptions.ResizeBilinearOptions, o)
    out = invoke(m.bytes([x], [y]), [X])
    return {"in": [H, W, C], "outH": oh, "outW": ow, "alignCorners": ac, "halfPixel": hp, "src": X, "out": out}


COUNTS2 = {"leakyRelu": 320, "bilinear": 216}
BUILD2 = {"leakyRelu": leaky_case, "bilinear": bilinear_case}


def main():
    global R
    if sys.argv[1] == "--ops2":
        R = random.Random(20261111)
        fx = {"source": "ai-edge-litert 2.3.0, OpResolverType.BUILTIN_REF; gen_ops.py --ops2 seed 20261111"}
        for name, n in COUNTS2.items():
            fx[name] = [BUILD2[name](i) for i in range(n)]
        path = sys.argv[2]
    else:
        fx = {"source": "ai-edge-litert 2.3.0, OpResolverType.BUILTIN_REF; gen_ops.py seed 20261110"}
        for name, n in COUNTS.items():
            build = globals()[name + "_case"]
            fx[name] = [build(i) for i in range(n)]
        path = sys.argv[1]
    with open(path, "w") as f:
        json.dump(fx, f, separators=(",", ":"))
        f.write("\n")


if __name__ == "__main__":
    main()
