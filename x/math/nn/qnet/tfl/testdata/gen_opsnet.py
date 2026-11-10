"""Build opsnet.tflite — a small int8 model with float input and output that
uses every operator the loader gained for face detection and YOLO (QUANTIZE
float→int8 and int8→int8, PAD, PADV2, RESIZE_NEAREST_NEIGHBOR in two index
modes, LOGISTIC, MUL with a per-channel constant, two activations and a
scalar constant, CONCATENATION, MEAN, ADD with a constant, LEAKY_RELU,
RESIZE_BILINEAR down and up with both index modes, three DEQUANTIZEd graph
outputs) — run it in the LiteRT interpreter with the reference kernels
and write opsnet.json.gz: per random input (64), the int8 graph input (the first
QUANTIZE's output) and every later operator's int8 output in operator order
(base64), DEQUANTIZEs excluded. Deterministic.

usage: <venv>/bin/python gen_opsnet.py   (ai-edge-litert 2.3.0)
"""
import base64
import gzip
import json

import flatbuffers
import numpy as np
from ai_edge_litert import interpreter as I
from ai_edge_litert import schema_py_generated as S

T_F32, T_I8, T_I32 = S.TensorType.FLOAT32, S.TensorType.INT8, S.TensorType.INT32
OP = S.BuiltinOperator
BO = S.BuiltinOptions
rng = np.random.default_rng(20261110)


class M:
    def __init__(self):
        self.buffers = [S.BufferT()]
        self.tensors, self.ops, self.codes = [], [], []

    def buf(self, data):
        b = S.BufferT()
        b.data = np.frombuffer(np.ascontiguousarray(data).tobytes(), dtype=np.uint8)
        self.buffers.append(b)
        return len(self.buffers) - 1

    def tensor(self, shape, typ, scale=None, zp=None, data=None):
        t = S.TensorT()
        t.shape, t.type = list(shape), typ
        t.buffer = 0 if data is None else self.buf(data)
        if scale is not None:
            q = S.QuantizationParametersT()
            q.scale, q.zeroPoint = [float(np.float32(scale))], [int(zp)]
            t.quantization = q
        self.tensors.append(t)
        return len(self.tensors) - 1

    def act(self, shape, scale, zp):
        return self.tensor(shape, T_I8, scale, zp)

    def const8(self, shape, scale, zp):
        d = rng.integers(-128, 128, size=int(np.prod(shape)), dtype=np.int8)
        return self.tensor(shape, T_I8, scale, zp, d)

    def i32(self, v):
        v = np.array(v, dtype=np.int32)
        return self.tensor(v.shape, T_I32, data=v)

    def op(self, code, ins, outs, otype=0, opts=None):
        if code not in self.codes:
            self.codes.append(code)
        o = S.OperatorT()
        o.opcodeIndex = self.codes.index(code)
        o.inputs, o.outputs = list(ins), list(outs)
        o.builtinOptionsType, o.builtinOptions = otype, opts
        self.ops.append(o)
        return outs[0]

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
        b = flatbuffers.Builder(4096)
        b.Finish(m.Pack(b), file_identifier=b"TFL3")
        return bytes(b.Output())


def opts(cls, **kw):
    o = cls()
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def build():
    m = M()
    x = m.tensor([1, 6, 6, 3], T_F32)
    a = m.op(OP.QUANTIZE, [x], [m.act([1, 6, 6, 3], 0.05, 3)])
    b = m.op(OP.PAD, [a, m.i32([[0, 0], [1, 1], [0, 1], [0, 2]])], [m.act([1, 8, 7, 5], 0.05, 3)])
    c = m.op(OP.RESIZE_NEAREST_NEIGHBOR, [b, m.i32([5, 3])], [m.act([1, 5, 3, 5], 0.05, 3)],
             BO.ResizeNearestNeighborOptions, opts(S.ResizeNearestNeighborOptionsT, alignCorners=False, halfPixelCenters=True))
    d = m.op(OP.RESIZE_NEAREST_NEIGHBOR, [c, m.i32([7, 6])], [m.act([1, 7, 6, 5], 0.05, 3)],
             BO.ResizeNearestNeighborOptions, opts(S.ResizeNearestNeighborOptionsT, alignCorners=True, halfPixelCenters=False))
    e = m.op(OP.LOGISTIC, [d], [m.act([1, 7, 6, 5], 1 / 256, -128)])
    f = m.op(OP.MUL, [e, m.const8([5], 0.011, -7)], [m.act([1, 7, 6, 5], 0.009, -20)],
             BO.MulOptions, opts(S.MulOptionsT, fusedActivationFunction=S.ActivationFunctionType.RELU))
    # scale ratio just below 1/2 (found by search): RoundExact and TFLite's
    # double rounding differ on 128 of the 256 inputs, so the mode shows
    e2 = m.op(OP.QUANTIZE, [e], [m.act([1, 7, 6, 5], 0.007817, -60)])
    g = m.op(OP.MUL, [f, e2], [m.act([1, 7, 6, 5], 0.013, 4)],
             BO.MulOptions, opts(S.MulOptionsT, fusedActivationFunction=S.ActivationFunctionType.NONE))
    h = m.op(OP.QUANTIZE, [d], [m.act([1, 7, 6, 5], 0.013, 4)])
    k = m.op(OP.CONCATENATION, [g, h], [m.act([1, 7, 6, 10], 0.013, 4)],
             BO.ConcatenationOptions, opts(S.ConcatenationOptionsT, axis=3, fusedActivationFunction=0))
    k2 = m.op(OP.CONCATENATION, [k, m.const8([1, 7, 1, 10], 0.013, 4)], [m.act([1, 7, 7, 10], 0.013, 4)],
              BO.ConcatenationOptions, opts(S.ConcatenationOptionsT, axis=-2, fusedActivationFunction=0))
    mn = m.op(OP.MEAN, [k2, m.i32([1, 2])], [m.act([1, 1, 1, 10], 0.007, -3)],
              BO.ReducerOptions, opts(S.ReducerOptionsT, keepDims=True))
    p = m.op(OP.MUL, [mn, m.const8([], 0.02, 1)], [m.act([1, 1, 1, 10], 0.004, 9)],
             BO.MulOptions, opts(S.MulOptionsT, fusedActivationFunction=S.ActivationFunctionType.RELU6))
    q = m.op(OP.ADD, [p, m.const8([1, 1, 1, 10], 0.004, -5)], [m.act([1, 1, 1, 10], 0.006, 0)],
             BO.AddOptions, opts(S.AddOptionsT, fusedActivationFunction=0))
    r = m.op(OP.PADV2, [q, m.i32([[0, 0], [0, 0], [0, 0], [2, 1]]), m.tensor([], T_I8, 0.006, 0, np.array([17], dtype=np.int8))],
             [m.act([1, 1, 1, 13], 0.006, 0)])
    lr = m.op(OP.LEAKY_RELU, [k2], [m.act([1, 7, 7, 10], 0.009, -70)],
              BO.LeakyReluOptions, opts(S.LeakyReluOptionsT, alpha=0.1))
    rb = m.op(OP.RESIZE_BILINEAR, [lr, m.i32([4, 5])], [m.act([1, 4, 5, 10], 0.009, -70)],
              BO.ResizeBilinearOptions, opts(S.ResizeBilinearOptionsT, alignCorners=False, halfPixelCenters=True))
    rb2 = m.op(OP.RESIZE_BILINEAR, [rb, m.i32([9, 11])], [m.act([1, 9, 11, 10], 0.009, -70)],
               BO.ResizeBilinearOptions, opts(S.ResizeBilinearOptionsT, alignCorners=True, halfPixelCenters=False))
    y = m.op(OP.DEQUANTIZE, [r], [m.tensor([1, 1, 1, 13], T_F32)])
    y2 = m.op(OP.DEQUANTIZE, [mn], [m.tensor([1, 1, 1, 10], T_F32)])
    y3 = m.op(OP.DEQUANTIZE, [rb2], [m.tensor([1, 9, 11, 10], T_F32)])
    return m.bytes([x], [y, y2, y3])


def main():
    model = build()
    it = I.Interpreter(model_content=model, experimental_op_resolver_type=I.OpResolverType.BUILTIN_REF,
                       experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    ops = it._get_ops_details()
    kept = [o for o in ops[1:] if o["op_name"] != "DEQUANTIZE"]  # the input QUANTIZE and DEQUANTIZEs are cut
    cases = []
    for _ in range(64):
        x = rng.uniform(-6, 6, size=(1, 6, 6, 3)).astype(np.float32)
        it.set_tensor(it.get_input_details()[0]["index"], x)
        it.invoke()
        enc = lambda v: base64.b64encode(np.asarray(v, dtype=np.int8).reshape(-1).tobytes()).decode()
        cases.append({"input": enc(it.get_tensor(ops[0]["outputs"][0])),
                      "outs": [enc(it.get_tensor(o["outputs"][0])) for o in kept]})
    names = [o["op_name"] for o in kept]
    open("opsnet.tflite", "wb").write(model)
    doc = json.dumps({"generator": "gen_opsnet.py", "litert": "ai-edge-litert 2.3.0, BUILTIN_REF",
                      "ops": names, "cases": cases}, indent=1).encode()
    open("opsnet.json.gz", "wb").write(gzip.compress(doc, compresslevel=9, mtime=0))
    print(len(model), "bytes;", len(cases), "cases;", names)


main()
