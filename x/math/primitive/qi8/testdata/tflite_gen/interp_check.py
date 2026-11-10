"""Cross-check fixtures (gen_fixtures.py / tflref.py port) against the real
TFLite interpreter (ai-edge-litert, OpResolverType.BUILTIN_REF = reference
kernels). Every fixture case becomes a one-operator .tflite model built with
the schema's object API; the interpreter's output must equal the fixture's.

usage: .venv/bin/python interp_check.py FIXTURES.json [--model M.tflite INPUT.i8 ...]
"""
import json
import sys

import flatbuffers
import numpy as np
from ai_edge_litert import interpreter as I
from ai_edge_litert import schema_py_generated as S

T_INT8, T_INT32 = S.TensorType.INT8, S.TensorType.INT32
ACT = {"none": S.ActivationFunctionType.NONE, "relu": S.ActivationFunctionType.RELU,
       "relu6": S.ActivationFunctionType.RELU6, "relun1to1": S.ActivationFunctionType.RELU_N1_TO_1}
PAD = {"same": S.Padding.SAME, "valid": S.Padding.VALID}


class M:
    def __init__(self):
        self.buffers = [S.BufferT()]
        self.tensors, self.ops, self.codes = [], [], []

    def buf(self, data):
        b = S.BufferT()
        b.data = np.frombuffer(np.ascontiguousarray(data).tobytes(), dtype=np.uint8)
        self.buffers.append(b)
        return len(self.buffers) - 1

    def tensor(self, shape, typ, scale=None, zp=None, qdim=0, data=None):
        t = S.TensorT()
        t.shape = list(shape)
        t.type = typ
        t.buffer = 0 if data is None else self.buf(data)
        if scale is not None:
            q = S.QuantizationParametersT()
            q.scale = [float(np.float32(s)) for s in scale]
            q.zeroPoint = [int(z) for z in zp]
            q.quantizedDimension = qdim
            t.quantization = q
        self.tensors.append(t)
        return len(self.tensors) - 1

    def op(self, code, ins, outs, otype, opts):
        if code not in self.codes:
            self.codes.append(code)
        o = S.OperatorT()
        o.opcodeIndex = self.codes.index(code)
        o.inputs, o.outputs = list(ins), list(outs)
        o.builtinOptionsType, o.builtinOptions = otype, opts
        self.ops.append(o)

    def bytes(self, ins, outs):
        m = S.ModelT()
        m.version = 3
        m.operatorCodes = []
        for c in self.codes:
            oc = S.OperatorCodeT()
            oc.builtinCode, oc.deprecatedBuiltinCode, oc.version = c, min(c, 127), 1
            m.operatorCodes.append(oc)
        sg = S.SubGraphT()
        sg.tensors, sg.operators, sg.inputs, sg.outputs = self.tensors, self.ops, list(ins), list(outs)
        m.subgraphs, m.buffers = [sg], self.buffers
        b = flatbuffers.Builder(1024)
        b.Finish(m.Pack(b), file_identifier=b"TFL3")
        return bytes(b.Output())


def invoke(model, inputs):
    it = I.Interpreter(model_content=model, experimental_op_resolver_type=getattr(I.OpResolverType, __import__("os").environ.get("RESOLVER", "BUILTIN_REF")))
    it.allocate_tensors()
    for d, x in zip(it.get_input_details(), inputs):
        it.set_tensor(d["index"], np.array(x, dtype=np.int8).reshape(d["shape"]))
    it.invoke()
    return [int(v) for v in it.get_tensor(it.get_output_details()[0]["index"]).reshape(-1)]


def conv_model(c, kind):
    m = M()
    H, W, C = c["in"] if kind != "fc" else (1, 1, c["inLen"])
    n_out = c.get("outC", C) if kind != "fc" else c["outLen"]
    ws = c["wScale"]
    if kind == "fc":
        x = m.tensor([1, c["inLen"]], T_INT8, [c["inScale"]], [c["inZP"]] if False else [c["inZP"]])
        wshape = [n_out, c["inLen"]]
        y_shape = [1, n_out]
        qdim = 0
    else:
        x = m.tensor([1, H, W, C], T_INT8, [c["inScale"]], [c["inZP"]])
        wshape = [n_out, c["kh"], c["kw"], C] if kind == "conv" else [1, c["kh"], c["kw"], C]
        qdim = 0 if kind == "conv" else 3
    per_tensor = kind == "fc" and not c["perChannel"]
    wsc = [ws[0]] if per_tensor else ws
    w = m.tensor(wshape, T_INT8, wsc, [0] * len(wsc), qdim, np.array(c["w"], dtype=np.int8))
    ins = [x, w]
    bias = c["bias"]
    if bias is None and kind != "fc":
        bias = [0] * n_out  # TFLite's int8 conv requires a bias tensor; zero = none
    if bias is not None:
        bsc = [float(np.float32(c["inScale"])) * float(np.float32(s)) for s in wsc]
        ins.append(m.tensor([n_out], T_INT32, bsc, [0] * len(bsc), 0, np.array(bias, dtype="<i4")))
    else:
        ins.append(-1)
    act = ACT[c["act"]]
    if kind == "fc":
        y = m.tensor(y_shape, T_INT8, [c["outScale"]], [c["outZP"]])
        o = S.FullyConnectedOptionsT()
        o.fusedActivationFunction = act
        m.op(S.BuiltinOperator.FULLY_CONNECTED, ins, [y], S.BuiltinOptions.FullyConnectedOptions, o)
        return m.bytes([x], [y])
    import tflref as T
    ohh, _ = T.same_or_valid(H, c["kh"], c["sh"], c["pad"])
    oww, _ = T.same_or_valid(W, c["kw"], c["sw"], c["pad"])
    assert ohh * oww * n_out == len(c["out"])
    y = m.tensor([1, ohh, oww, n_out], T_INT8, [c["outScale"]], [c["outZP"]])
    if kind == "conv":
        o = S.Conv2DOptionsT()
        o.padding, o.strideW, o.strideH, o.fusedActivationFunction = PAD[c["pad"]], c["sw"], c["sh"], act
        o.dilationWFactor = o.dilationHFactor = 1
        m.op(S.BuiltinOperator.CONV_2D, ins, [y], S.BuiltinOptions.Conv2DOptions, o)
    else:
        o = S.DepthwiseConv2DOptionsT()
        o.padding, o.strideW, o.strideH, o.fusedActivationFunction = PAD[c["pad"]], c["sw"], c["sh"], act
        o.depthMultiplier = 1
        o.dilationWFactor = o.dilationHFactor = 1
        m.op(S.BuiltinOperator.DEPTHWISE_CONV_2D, ins, [y], S.BuiltinOptions.DepthwiseConv2DOptions, o)
    return m.bytes([x], [y])


def add_model(c):
    m = M()
    n = c["n"]
    a = m.tensor([1, n], T_INT8, [c["scaleA"]], [c["zpA"]])
    b = m.tensor([1, n], T_INT8, [c["scaleB"]], [c["zpB"]])
    y = m.tensor([1, n], T_INT8, [c["scaleOut"]], [c["zpOut"]])
    o = S.AddOptionsT()
    o.fusedActivationFunction = ACT[c["act"]]
    m.op(S.BuiltinOperator.ADD, [a, b], [y], S.BuiltinOptions.AddOptions, o)
    return m.bytes([a, b], [y])


def pool_model(c):
    import tflref as T
    m = M()
    H, W, C = c["in"]
    x = m.tensor([1, H, W, C], T_INT8, [0.1], [3])
    oh, _ = T.same_or_valid(H, c["kh"], c["sh"], c["pad"])
    ow, _ = T.same_or_valid(W, c["kw"], c["sw"], c["pad"])
    y = m.tensor([1, oh, ow, C], T_INT8, [0.1], [3])
    o = S.Pool2DOptionsT()
    o.padding, o.strideW, o.strideH, o.filterWidth, o.filterHeight = PAD[c["pad"]], c["sw"], c["sh"], c["kw"], c["kh"]
    o.fusedActivationFunction = S.ActivationFunctionType.NONE
    m.op(S.BuiltinOperator.AVERAGE_POOL_2D, [x], [y], S.BuiltinOptions.Pool2DOptions, o)
    return m.bytes([x], [y])


def softmax_model(c):
    m = M()
    n = len(c["src"])
    shape = [n // c["depth"], c["depth"]]
    x = m.tensor(shape, T_INT8, [c["scale"]], [0])
    y = m.tensor(shape, T_INT8, [1.0 / 256], [-128])
    o = S.SoftmaxOptionsT()
    o.beta = float(np.float32(c["beta"]))
    m.op(S.BuiltinOperator.SOFTMAX, [x], [y], S.BuiltinOptions.SoftmaxOptions, o)
    return m.bytes([x], [y])


def relu6_range_model(c):
    """1×1 conv over a ramp -128..127 with a large multiplier: the outputs
    saturate at both ends of the fused ReLU6 range."""
    m = M()
    x = m.tensor([1, 1, 256, 1], T_INT8, [1.0], [0])
    w = m.tensor([1, 1, 1, 1], T_INT8, [float(np.float32(c["scale"]))], [0], 0, np.array([127], dtype=np.int8))
    y = m.tensor([1, 1, 256, 1], T_INT8, [c["scale"]], [c["zp"]])
    o = S.Conv2DOptionsT()
    o.padding, o.strideW, o.strideH, o.fusedActivationFunction = S.Padding.VALID, 1, 1, S.ActivationFunctionType.RELU6
    o.dilationWFactor = o.dilationHFactor = 1
    b = m.tensor([1], T_INT32, [float(np.float32(c["scale"]))], [0], 0, np.array([0], dtype="<i4"))
    m.op(S.BuiltinOperator.CONV_2D, [x, w, b], [y], S.BuiltinOptions.Conv2DOptions, o)
    return m.bytes([x], [y])


def main():
    fx = json.load(open(sys.argv[1]))
    report = {}
    for kind in ("conv", "depthwise", "fc"):
        bad = outs = 0
        for i, c in enumerate(fx[kind]):
            got = invoke(conv_model(c, kind), [c["src"]])
            d = sum(g != w for g, w in zip(got, c["out"]))
            if d and bad < 3:
                print(f"{kind} {i}: {d} differ; got {got[:8]} want {c['out'][:8]}")
            bad += d
            outs += len(got)
        report[kind] = (len(fx[kind]), outs, bad)
    for kind, build, inputs in (
        ("add", add_model, lambda c: [c["a"], c["b"]]),
        ("avgpool", pool_model, lambda c: [c["src"]]),
        ("softmax", softmax_model, lambda c: [c["src"]]),
    ):
        bad = outs = 0
        for i, c in enumerate(fx[kind]):
            got = invoke(build(c), inputs(c))
            d = sum(g != w for g, w in zip(got, c["out"]))
            if d and bad < 3:
                print(f"{kind} {i}: {d} differ; got {got[:8]} want {c['out'][:8]}")
            bad += d
            outs += len(got)
        report[kind] = (len(fx[kind]), outs, bad)
    bad = 0
    for i, c in enumerate(fx["act"]):
        got = invoke(relu6_range_model(c), [list(range(-128, 128))])
        if [min(got), max(got)] != c["relu6"]:
            if bad < 3:
                print(f"act {i}: got [{min(got)},{max(got)}] want {c['relu6']}")
            bad += 1
    report["act"] = (len(fx["act"]), len(fx["act"]) * 2, bad * 2)
    for k, (n, o, b) in report.items():
        print(f"{k:10s} cases {n:4d} outputs {o:6d} mismatches {b}")


if __name__ == "__main__":
    main()
