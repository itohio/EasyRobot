"""Build yoloxtail.tflite — a small int8 model for the loader's TRANSPOSE,
per-pixel MUL broadcast and int8 → uint8 output QUANTIZE cut, shaped like a
YOLOX head: float input, QUANTIZE, a 1×1 CONV_2D to one channel and LOGISTIC
(objectness) times the input (MUL of an H×W×1 by an H×W×C operand, the
smaller first), then two heads (that product and a RESIZE_NEAREST of it) each
TRANSPOSEd [0,3,1,2] and RESHAPEd to [1,C,H·W], CONCATENATION over the
anchors, TRANSPOSE [0,2,1], then QUANTIZE to a uint8 graph output (same
scale, zero point + 128); a second branch chains TRANSPOSE [0,2,1,3],
[0,1,3,2], [0,3,2,1] and the identity into a second uint8 graph output. Run
in the LiteRT interpreter with the reference kernels; transnet.json.gz holds,
per random input (32): the int8 graph input (the first QUANTIZE's output),
every later operator's int8 output in operator order (the two uint8
QUANTIZEs excluded), and the two uint8 graph outputs. Deterministic.

usage: <venv>/bin/python gen_yoloxtail.py   (ai-edge-litert 2.3.0)
"""
import base64
import gzip
import json

import flatbuffers
import numpy as np
from ai_edge_litert import interpreter as I
from ai_edge_litert import schema_py_generated as S

T_F32, T_I8, T_U8, T_I32 = S.TensorType.FLOAT32, S.TensorType.INT8, S.TensorType.UINT8, S.TensorType.INT32
OP = S.BuiltinOperator
BO = S.BuiltinOptions
rng = np.random.default_rng(20261114)


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

    def act(self, shape, scale=0.05, zp=3):
        return self.tensor(shape, T_I8, scale, zp)

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


def transpose(m, x, shape, perm):
    out = [shape[p] for p in perm]
    return m.op(OP.TRANSPOSE, [x, m.i32(perm)], [m.act(out)], BO.TransposeOptions, S.TransposeOptionsT()), out


def reshape(m, x, shape):
    return m.op(OP.RESHAPE, [x, m.i32(shape)], [m.act(shape)], BO.ReshapeOptions, opts(S.ReshapeOptionsT, newShape=list(shape)))


def build():
    m = M()
    x = m.tensor([1, 5, 6, 7], T_F32)
    a0 = m.op(OP.QUANTIZE, [x], [m.act([1, 5, 6, 7])])
    w = m.tensor([1, 1, 1, 7], T_I8, 0.02, 0, rng.integers(-127, 128, size=7, dtype=np.int8))
    bias = m.tensor([1], T_I32, data=np.array([37], dtype=np.int32))
    m.tensors[bias].quantization = S.QuantizationParametersT()
    m.tensors[bias].quantization.scale, m.tensors[bias].quantization.zeroPoint = [float(np.float32(0.05 * 0.02))], [0]
    obj = m.op(OP.CONV_2D, [a0, w, bias], [m.act([1, 5, 6, 1], 0.07, -10)], BO.Conv2DOptions,
               opts(S.Conv2DOptionsT, padding=S.Padding.VALID, strideW=1, strideH=1, dilationWFactor=1, dilationHFactor=1,
                    fusedActivationFunction=0))
    sig = m.op(OP.LOGISTIC, [obj], [m.act([1, 5, 6, 1], 1 / 256, -128)])
    a = m.op(OP.MUL, [sig, a0], [m.act([1, 5, 6, 7])], BO.MulOptions, opts(S.MulOptionsT, fusedActivationFunction=0))
    b = m.op(OP.RESIZE_NEAREST_NEIGHBOR, [a, m.i32([3, 3])], [m.act([1, 3, 3, 7])],
             BO.ResizeNearestNeighborOptions, opts(S.ResizeNearestNeighborOptionsT, alignCorners=False, halfPixelCenters=False))
    ta, _ = transpose(m, a, [1, 5, 6, 7], [0, 3, 1, 2])
    ra = reshape(m, ta, [1, 7, 30])
    tb, _ = transpose(m, b, [1, 3, 3, 7], [0, 3, 1, 2])
    rb = reshape(m, tb, [1, 7, 9])
    c = m.op(OP.CONCATENATION, [ra, rb], [m.act([1, 7, 39])],
             BO.ConcatenationOptions, opts(S.ConcatenationOptionsT, axis=2, fusedActivationFunction=0))
    t3, _ = transpose(m, c, [1, 7, 39], [0, 2, 1])
    y1 = m.op(OP.QUANTIZE, [t3], [m.tensor([1, 39, 7], T_U8, 0.05, 131)])
    s = [1, 5, 6, 7]
    z = a
    for p in ([0, 2, 1, 3], [0, 1, 3, 2], [0, 3, 2, 1], [0, 1, 2, 3]):
        z, s = transpose(m, z, s, p)
    y2 = m.op(OP.QUANTIZE, [z], [m.tensor(s, T_U8, 0.05, 131)])
    return m.bytes([x], [y1, y2])


def main():
    model = build()
    it = I.Interpreter(model_content=model, experimental_op_resolver_type=I.OpResolverType.BUILTIN_REF,
                       experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    ops = it._get_ops_details()
    td = it.get_tensor_details()
    # The input QUANTIZE and the int8 → uint8 output QUANTIZEs are cut.
    kept = [o for o in ops[1:] if not (o["op_name"] == "QUANTIZE" and td[o["outputs"][0]]["dtype"] == np.uint8)]
    outs = [d["index"] for d in it.get_output_details()]
    cases = []
    for _ in range(32):
        v = rng.uniform(-7, 7, size=(1, 5, 6, 7)).astype(np.float32)
        it.set_tensor(it.get_input_details()[0]["index"], v)
        it.invoke()
        enc = lambda t, dt: base64.b64encode(np.asarray(t, dtype=dt).reshape(-1).tobytes()).decode()
        cases.append({"input": enc(it.get_tensor(ops[0]["outputs"][0]), np.int8),
                      "outs": [enc(it.get_tensor(o["outputs"][0]), np.int8) for o in kept],
                      "graph": [enc(it.get_tensor(i), np.uint8) for i in outs]})
    names = [o["op_name"] for o in kept]
    open("yoloxtail.tflite", "wb").write(model)
    doc = json.dumps({"generator": "gen_yoloxtail.py", "litert": "ai-edge-litert 2.3.0, BUILTIN_REF",
                      "ops": names, "cases": cases}, indent=1).encode()
    open("yoloxtail.json.gz", "wb").write(gzip.compress(doc, compresslevel=9, mtime=0))
    print(len(model), "bytes;", len(cases), "cases;", names)


main()
