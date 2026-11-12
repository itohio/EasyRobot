"""Build u8net.tflite — a small uint8 model in the form of TF's official
"quant" MobileNetV1 exports (uint8 per-tensor activations, asymmetric uint8
weights, int32 bias), using every operator the loader accepts as uint8:
DEPTHWISE_CONV_2D over the one-channel input with multiplier 3 (becomes a
Conv2D), CONV_2D (SAME, stride 2, ReLU6), DEPTHWISE_CONV_2D (multiplier 1,
ReLU), MAX_POOL_2D, AVERAGE_POOL_2D 2×2 (even window: .5 ties), RESHAPE,
FULLY_CONNECTED and SOFTMAX. Power-of-two scales and weights near their zero
points make rounding ties common, so the tests see TFLite's rounding rules.

It runs the model in the LiteRT interpreter with the reference kernels and
writes u8net.json.gz: per input (64; the first ones near the input zero
point) the graph input and every operator's output, each as int8
(uint8 value − 128, the form qnet loads) in base64. Deterministic.

usage: <venv>/bin/python gen_u8net.py   (ai-edge-litert 2.3.0)
"""
import base64
import gzip
import json

import flatbuffers
import numpy as np
from ai_edge_litert import interpreter as I
from ai_edge_litert import schema_py_generated as S

T_U8, T_I32 = S.TensorType.UINT8, S.TensorType.INT32
OP = S.BuiltinOperator
BO = S.BuiltinOptions
ACT = S.ActivationFunctionType
rng = np.random.default_rng(20261112)


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
        return self.tensor(shape, T_U8, scale, zp)

    def weights(self, shape, scale, zp):
        d = np.clip(zp + rng.integers(-6, 7, size=int(np.prod(shape))), 0, 255).astype(np.uint8)
        return self.tensor(shape, T_U8, scale, zp, d)

    def bias(self, n, scale):
        return self.tensor([n], T_I32, scale, 0, rng.integers(-40, 41, size=n).astype(np.int32))

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
    s0, z0 = 2.0 ** -4, 120
    x = m.act([1, 8, 8, 1], s0, z0)
    # depthwise over one channel, multiplier 3 → Conv2D [3][3][3][1]
    s1, z1 = 2.0 ** -3, 131
    w = m.weights([1, 3, 3, 3], 2.0 ** -5, 140)
    a1 = m.op(OP.DEPTHWISE_CONV_2D, [x, w, m.bias(3, s0 * 2.0 ** -5)], [m.act([1, 8, 8, 3], s1, z1)], BO.DepthwiseConv2DOptions,
              opts(S.DepthwiseConv2DOptionsT, padding=S.Padding.SAME, strideW=1, strideH=1, depthMultiplier=3,
                   dilationWFactor=1, dilationHFactor=1, fusedActivationFunction=ACT.NONE))
    s2, z2 = 2.0 ** -2, 3  # ReLU6 output: zero point at the bottom of the range
    w = m.weights([4, 3, 3, 3], 2.0 ** -6, 77)
    a2 = m.op(OP.CONV_2D, [a1, w, m.bias(4, s1 * 2.0 ** -6)], [m.act([1, 4, 4, 4], s2, z2)], BO.Conv2DOptions,
              opts(S.Conv2DOptionsT, padding=S.Padding.SAME, strideW=2, strideH=2, dilationWFactor=1, dilationHFactor=1,
                   fusedActivationFunction=ACT.RELU6))
    s3, z3 = 2.0 ** -3, 10
    w = m.weights([1, 3, 3, 4], 2.0 ** -4, 200)
    a3 = m.op(OP.DEPTHWISE_CONV_2D, [a2, w, m.bias(4, s2 * 2.0 ** -4)], [m.act([1, 4, 4, 4], s3, z3)], BO.DepthwiseConv2DOptions,
              opts(S.DepthwiseConv2DOptionsT, padding=S.Padding.SAME, strideW=1, strideH=1, depthMultiplier=1,
                   dilationWFactor=1, dilationHFactor=1, fusedActivationFunction=ACT.RELU))
    a4 = m.op(OP.MAX_POOL_2D, [a3], [m.act([1, 4, 4, 4], s3, z3)], BO.Pool2DOptions,
              opts(S.Pool2DOptionsT, padding=S.Padding.SAME, strideW=1, strideH=1, filterWidth=2, filterHeight=2,
                   fusedActivationFunction=ACT.NONE))
    a5 = m.op(OP.AVERAGE_POOL_2D, [a4], [m.act([1, 2, 2, 4], s3, z3)], BO.Pool2DOptions,
              opts(S.Pool2DOptionsT, padding=S.Padding.VALID, strideW=2, strideH=2, filterWidth=2, filterHeight=2,
                   fusedActivationFunction=ACT.NONE))
    a6 = m.op(OP.RESHAPE, [a5, m.i32([1, 16])], [m.act([1, 16], s3, z3)], BO.ReshapeOptions,
              opts(S.ReshapeOptionsT, newShape=[1, 16]))
    s7, z7 = 2.0 ** -2, 128
    w = m.weights([5, 16], 2.0 ** -5, 99)
    a7 = m.op(OP.FULLY_CONNECTED, [a6, w, m.bias(5, s3 * 2.0 ** -5)], [m.act([1, 5], s7, z7)], BO.FullyConnectedOptions,
              opts(S.FullyConnectedOptionsT, fusedActivationFunction=ACT.NONE))
    y = m.op(OP.SOFTMAX, [a7], [m.act([1, 5], 1.0 / 256, 0)], BO.SoftmaxOptions, opts(S.SoftmaxOptionsT, beta=1.0))
    return m.bytes([x], [y])


def main():
    model = build()
    it = I.Interpreter(model_content=model, experimental_op_resolver_type=I.OpResolverType.BUILTIN_REF,
                       experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    ops = it._get_ops_details()
    d = it.get_input_details()[0]
    enc = lambda v: base64.b64encode((np.asarray(v).astype(np.int16) - 128).astype(np.int8).reshape(-1).tobytes()).decode()  # noqa: E731
    cases = []
    for k in range(64):
        if k < 32:  # near the zero point: small accumulators, frequent ties
            x = np.clip(120 + rng.integers(-8, 9, size=d["shape"]), 0, 255).astype(np.uint8)
        else:
            x = rng.integers(0, 256, size=d["shape"], dtype=np.uint8)
        it.set_tensor(d["index"], x)
        it.invoke()
        cases.append({"input": enc(x), "outs": [enc(it.get_tensor(o["outputs"][0])) for o in ops]})
    names = [o["op_name"] for o in ops]
    open("u8net.tflite", "wb").write(model)
    doc = json.dumps({"generator": "gen_u8net.py", "litert": "ai-edge-litert 2.3.0, BUILTIN_REF",
                      "form": "uint8 values minus 128 (int8)", "ops": names, "cases": cases}, indent=1).encode()
    open("u8net.json.gz", "wb").write(gzip.compress(doc, compresslevel=9, mtime=0))
    print(len(model), "bytes;", len(cases), "cases;", names)


main()
