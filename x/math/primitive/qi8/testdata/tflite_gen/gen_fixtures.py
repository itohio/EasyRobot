"""Generate qi8 TFLite-mode fixtures from the reference-arithmetic port
(tflref.py). Deterministic (seed fixed). Output: JSON.

usage: python3 gen_fixtures.py OUT.json
"""
import json
import math
import random
import sys

import numpy as np

import tflref as T

R = random.Random(20261109)


def zp():
    return R.choice([-128, 127, 0, R.randint(-128, 127), R.randint(-128, 127)])


def int8s(n):
    k = R.randrange(8)
    if k == 0:
        return [R.choice([-128, 127]) for _ in range(n)]
    if k == 1:
        v = R.choice([-128, 127])
        return [v] * n
    return [R.randint(-128, 127) for _ in range(n)]


def scale():
    return float(np.float32(R.choice([R.uniform(0.002, 0.2), 2.0 ** -R.randint(3, 9), 6 / 255, 1 / 128])))



def bias(n):
    k = R.randrange(4)
    if k == 0:
        return None
    if k == 1:
        return [R.randint(-(1 << 24), 1 << 24) for _ in range(n)]
    return [R.randint(-(1 << 16), 1 << 16) for _ in range(n)]


def multipliers(n, s_in, s_out, fan_in, fewbit):
    """Per-channel (mult, shift) and float32 weight scales, derived the
    TFLite way (sIn·sW/sOut in float64 from float32 scales). With fewbit the
    multipliers have 1-3 significant bits (m·2^-k) — exact .5 ties, and so
    the rounding modes' differences, are frequent; s_in and s_out are then
    powers of two so the weight scales are exact. Otherwise weight scales
    are chosen so outputs land mostly inside int8."""
    std = 5400.0 * fan_in ** 0.5  # rough |acc| for random int8 data
    target = 40.0 / std
    if fewbit:
        k = round(math.log2(1 / target))
        reals = [R.choice([1, 3, 5, 7]) * 2.0 ** -(k + R.choice([-1, 0, 1, 2]) + 2) for _ in range(n)]
        ws = [r * s_out / s_in for r in reals]
    else:
        ws = [float(np.float32(target * R.uniform(0.5, 2) * s_out / s_in)) for _ in range(n)]
    reals2 = [float(np.float32(s_in)) * float(np.float32(w)) / float(np.float32(s_out)) for w in ws]
    if fewbit:
        assert reals2 == reals
    mult, shift = zip(*[T.quantize_multiplier(r) for r in reals2])
    return list(mult), list(shift), ws


def act(s_out, zp_out):
    kind = R.choice(["none", "relu", "relu6", "relun1to1"])
    lo, hi = T.act_range(kind, s_out, zp_out)
    if lo > hi:
        kind, (lo, hi) = "none", T.act_range("none", s_out, zp_out)
    return kind, lo, hi


def geometry(depthwise):
    ks = [1, 1, 2, 3, 3, 3, 5]
    p = {"in": [R.randint(1, 7), R.randint(1, 7), R.randint(1, 6)], "kh": R.choice(ks), "sh": R.randint(1, 2)}
    p["kw"], p["sw"] = p["kh"], p["sh"]
    if R.randrange(4) == 0:
        p["kw"], p["sw"] = R.choice(ks), R.randint(1, 3)
    p["pad"] = "valid" if min(p["in"][0], p["in"][1]) >= max(p["kh"], p["kw"]) and R.randrange(3) == 0 else "same"
    if not depthwise:
        p["outC"] = R.randint(1, 5)
    return p


def quant(p, n_out, fan_in):
    fewbit = R.randrange(2) == 0
    if fewbit:
        s_in, s_out = 2.0 ** -R.randint(3, 9), 2.0 ** -R.randint(3, 9)
    else:
        s_in, s_out = scale(), scale()
    p.update(inZP=zp(), outZP=zp(), inScale=s_in, outScale=s_out)
    p["act"], p["actMin"], p["actMax"] = act(s_out, p["outZP"])
    p["mult"], p["shift"], p["wScale"] = multipliers(n_out, s_in, s_out, fan_in, fewbit)


def tie_stress(p, n_w, n_out):
    """A third of the cases: inputs within ±3 of the zero point, weights in
    ±3, small bias and multipliers m·2^-k with k ≤ 6 (scales 1, 1 and m·2^-k),
    so accumulators are small, outputs stay inside int8 and exact .5 ties
    are common — the outputs where TFLite's double rounding differs from a
    single rounding."""
    if R.randrange(3) != 0:
        return False
    p["src"] = [min(max(p["inZP"] + R.randint(-3, 3), -128), 127) for _ in range(len(p["src"]))]
    p["w"] = [R.randint(-3, 3) for _ in range(n_w)]
    p["bias"] = None if R.randrange(2) else [R.randint(-8, 8) for _ in range(n_out)]
    reals = [R.choice([1, 3, 5]) * 2.0 ** -R.randint(1, 6) for _ in range(n_out)]
    p["mult"], p["shift"] = map(list, zip(*[T.quantize_multiplier(r) for r in reals]))
    p["inScale"], p["outScale"], p["wScale"] = 1.0, 1.0, reals
    p["outZP"] = R.randint(-40, 40)
    p["act"], p["actMin"], p["actMax"] = "none", -128, 127
    return True


def conv_case(depthwise):
    p = geometry(depthwise)
    H, W, C = p["in"]
    n_out = C if depthwise else p["outC"]
    quant(p, n_out, p["kh"] * p["kw"] * (1 if depthwise else C))
    p["src"] = int8s(H * W * C)
    p["w"] = int8s(p["kh"] * p["kw"] * C * (1 if depthwise else p["outC"]))
    p["bias"] = bias(n_out)
    tie_stress(p, len(p["w"]), n_out)
    f = T.depthwise if depthwise else T.conv
    p["out"] = f(p["src"], p["w"], p["bias"], p["mult"], p["shift"], p)
    return p


def fc_case():
    """Half the cases have per-tensor weights (one scale, as most TFLite FC
    layers), half per-channel scales (TFLite's FC rounds both once, half
    away from zero; see tflref.fully_connected)."""
    n_in, n_out = R.randint(1, 80), R.randint(1, 10)
    p = {"inLen": n_in, "outLen": n_out}
    quant(p, n_out, n_in)
    per_tensor = R.randrange(2) == 0
    p["src"] = int8s(n_in)
    p["w"] = int8s(n_in * n_out)
    p["bias"] = bias(n_out)
    tie_stress(p, len(p["w"]), n_out)
    if per_tensor or n_out == 1:
        p["mult"], p["shift"] = [p["mult"][0]] * n_out, [p["shift"][0]] * n_out
        p["wScale"] = [p["wScale"][0]] * n_out
    p["perChannel"] = not per_tensor and n_out > 1
    p["out"] = T.fully_connected(p["src"], p["w"], p["bias"], p["mult"], p["shift"], p)
    return p


def add_case():
    n = R.randint(1, 64)
    sa, sb, so = scale(), scale(), scale()
    if R.randrange(4) == 0:
        sb = sa
    if R.randrange(3) == 0:  # power-of-two scales: ties in the output rounding
        e = R.randint(3, 8)
        sa, sb, so = 2.0 ** -e, 2.0 ** -(e + R.randint(0, 2)), 2.0 ** -(e + R.randint(-2, 1))
    p = {"n": n, "scaleA": sa, "scaleB": sb, "scaleOut": so, "zpA": zp(), "zpB": zp(), "zpOut": zp()}
    p["act"], p["actMin"], p["actMax"] = act(so, p["zpOut"])
    prm = T.add_params(sa, sb, so)
    p["mult"] = [m for m, _ in prm]
    p["shift"] = [s for _, s in prm]
    p["a"], p["b"] = int8s(n), int8s(n)
    p["out"] = T.add(p["a"], p["b"], p, prm)
    return p


def pool_case():
    p = geometry(True)
    p["src"] = int8s(p["in"][0] * p["in"][1] * p["in"][2])
    p["out"] = T.avg_pool(p["src"], p)
    return p


def requant_cases():
    """{"acc": [...], "rows": [{mult, shift, out}]}; out[i] is null where
    acc[i]·2^shift overflows int32 (undefined in C, not a TFLite contract)."""
    rows = []
    accs = [0, 1, -1, 2, -2, 3, -3, 5, -5, 7, -7, 1 << 20, -(1 << 20), T.I32_MAX, T.I32_MIN, T.I32_MIN + 1]
    accs += [R.randint(T.I32_MIN, T.I32_MAX) for _ in range(40)]
    accs += [R.randint(-(1 << 16), 1 << 16) for _ in range(40)]
    accs += [R.choice([1, 3, 5]) * (1 << R.randint(0, 24)) * R.choice([1, -1]) for _ in range(40)]
    for shift in range(-31, 31):
        for m in [1 << 30, (1 << 30) + 1, 3 << 29, (1 << 31) - 1, R.randint(1 << 30, (1 << 31) - 1)]:
            lim = T.I32_MAX >> max(shift, 0)
            out = [T.mbqm(x, m, shift) if -lim <= x <= lim else None for x in accs]
            rows.append({"mult": m, "shift": shift, "out": out})
    return {"acc": accs, "rows": rows}


def act_cases():
    out = []
    for _ in range(200):
        s, z = scale(), zp()
        if R.randrange(3) == 0:
            s = float(np.float32(6.0 / (R.randint(1, 300) + 0.5)))  # 6/scale near a .5 tie
        lo, hi = T.act_range("relu6", s, z)
        out.append({"scale": s, "zp": z, "relu6": [lo, hi]})
    return out

def softmax_cases():
    out = []
    for _ in range(300):
        depth, rows = R.randint(1, 24), R.randint(1, 3)
        s = float(np.float32(R.choice([R.uniform(0.01, 0.3), 2.0 ** -R.randint(2, 8), R.uniform(0.0005, 0.01), R.uniform(0.3, 2)])))
        beta = float(np.float32(R.choice([1.0, 1.0, 1.0, R.uniform(0.2, 3)])))
        mult, left, diff_min = T.softmax_params(beta, s)
        src = int8s(depth * rows)
        if R.randrange(4) == 0:  # few distinct values: equal maxima, exact ties
            vals = [R.randint(-128, 127) for _ in range(3)]
            src = [R.choice(vals) for _ in src]
        out.append({"depth": depth, "scale": s, "beta": beta, "mult": mult, "leftShift": left,
                    "diffMin": diff_min, "src": src, "out": T.softmax(src, depth, mult, left, diff_min)})
    return out


def main():
    fx = {
        "source": "Python port of TensorFlow Lite int8 reference kernels (gen_fixtures.py + tflref.py, seed 20261109); "
        "every op case equals the LiteRT 2.3.0 interpreter with the reference resolver (interp_check.py)",
        "requant": requant_cases(),
        "conv": [conv_case(False) for _ in range(150)],
        "depthwise": [conv_case(True) for _ in range(150)],
        "fc": [fc_case() for _ in range(150)],
        "add": [add_case() for _ in range(200)],
        "avgpool": [pool_case() for _ in range(80)],
        "act": act_cases(),
        "softmax": softmax_cases(),
    }
    with open(sys.argv[1], "w") as f:
        json.dump(fx, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
