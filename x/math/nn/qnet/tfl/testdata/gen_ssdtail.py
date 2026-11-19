"""Build ssdtail.tflite — a small uint8 model shaped like the tail of TF's
quantised SSD-MobileNetV1 COCO export: two feature maps (4×4 and 2×2, two
anchors a cell), each with a box CONV_2D RESHAPEd to [1,N,1,4] and a class
CONV_2D RESHAPEd to [1,N,3] (background + 2 classes); CONCATENATION of the
box heads (output quantisation = the first head's, as in the export) and of
the class heads (a third quantisation), RESHAPE of the boxes to [1,40,4],
LOGISTIC of the scores, and the custom TFLite_Detection_PostProcess with
uint8 anchors and the export's options (fast NMS, one class a box). Weights
and scales make equal scores common, so tie order matters.

It runs the model in the LiteRT interpreter with the reference kernels and
writes ssdtail.json.gz: per input (64) the graph input and every operator
output before the custom one, each as int8 (uint8 value − 128, the form
qnet loads), and the custom operator's four float outputs (boxes, classes,
scores, count). Deterministic.

usage: <venv>/bin/python gen_ssdtail.py   (ai-edge-litert 2.1.5)
"""
import base64
import gzip
import json

import flatbuffers
import numpy as np
from ai_edge_litert import interpreter as I
from ai_edge_litert import schema_py_generated as S
from flatbuffers import flexbuffers

T_U8, T_I32, T_F32 = S.TensorType.UINT8, S.TensorType.INT32, S.TensorType.FLOAT32
OP = S.BuiltinOperator
BO = S.BuiltinOptions
ACT = S.ActivationFunctionType
rng = np.random.default_rng(20261119)
POSTPROCESS = "TFLite_Detection_PostProcess"
OPTIONS = {"max_detections": 5, "max_classes_per_detection": 1, "num_classes": 2,
           "nms_score_threshold": 179 / 256, "nms_iou_threshold": 0.5,
           "y_scale": 10.0, "x_scale": 10.0, "h_scale": 5.0, "w_scale": 5.0, "_output_quantized": True}


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

    def op(self, code, ins, outs, otype=0, opts=None, custom=None):
        key = custom if custom is not None else code
        if key not in self.codes:
            self.codes.append(key)
        o = S.OperatorT()
        o.opcodeIndex = self.codes.index(key)
        o.inputs, o.outputs = list(ins), list(outs)
        if custom is not None:
            o.customOptions = list(flexbuffers.Dumps(OPTIONS))
            o.customOptionsFormat = S.CustomOptionsFormat.FLEXBUFFERS
        else:
            o.builtinOptionsType, o.builtinOptions = otype, opts
        self.ops.append(o)
        return outs[0]

    def bytes(self, ins, outs):
        m = S.ModelT()
        m.version = 3
        m.operatorCodes = []
        for c in self.codes:
            oc = S.OperatorCodeT()
            if isinstance(c, str):
                oc.builtinCode, oc.deprecatedBuiltinCode, oc.customCode = OP.CUSTOM, OP.CUSTOM, c
            else:
                oc.builtinCode, oc.deprecatedBuiltinCode = c, min(c, 127)
            oc.version = 1
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


def conv(m, x, cin, cout, hw, stride, k, s_in, s_out, z_out, act=ACT.NONE):
    w = m.weights([cout, k, k, cin], 2.0 ** -5, 128)
    return m.op(OP.CONV_2D, [x, w, m.bias(cout, s_in * 2.0 ** -5)], [m.act([1, hw, hw, cout], s_out, z_out)], BO.Conv2DOptions,
                opts(S.Conv2DOptionsT, padding=S.Padding.SAME, strideW=stride, strideH=stride, dilationWFactor=1,
                     dilationHFactor=1, fusedActivationFunction=act))


def reshape(m, x, shape, s, z):
    return m.op(OP.RESHAPE, [x, m.i32(shape)], [m.act(shape, s, z)], BO.ReshapeOptions, opts(S.ReshapeOptionsT, newShape=list(shape)))


def anchors():
    a = []
    for n in (4, 2):
        for y in range(n):
            for x in range(n):
                for size in (1.0 / n, 1.5 / n):
                    a.append([(y + 0.5) / n, (x + 0.5) / n, size, size * 0.8])
    return np.array(a, dtype=np.float32)


def build():
    m = M()
    s0, z0 = 2.0 ** -4, 120
    x = m.act([1, 4, 4, 3], s0, z0)
    s1, z1 = 2.0 ** -3, 0
    f1 = conv(m, x, 3, 4, 4, 1, 3, s0, s1, z1, ACT.RELU6)
    qb0, qc0 = (0.0625, 131), (0.125, 140)
    b0 = reshape(m, conv(m, f1, 4, 8, 4, 1, 1, s1, *qb0), [1, 32, 1, 4], *qb0)
    c0 = reshape(m, conv(m, f1, 4, 6, 4, 1, 1, s1, *qc0), [1, 32, 3], *qc0)
    f2 = conv(m, f1, 4, 4, 2, 2, 3, s1, s1, z1, ACT.RELU6)
    qb1, qc1 = (0.04, 126), (0.09, 120)
    b1 = reshape(m, conv(m, f2, 4, 8, 2, 1, 1, s1, *qb1), [1, 8, 1, 4], *qb1)
    c1 = reshape(m, conv(m, f2, 4, 6, 2, 1, 1, s1, *qc1), [1, 8, 3], *qc1)
    bc = m.op(OP.CONCATENATION, [b0, b1], [m.act([1, 40, 1, 4], *qb0)], BO.ConcatenationOptions,
              opts(S.ConcatenationOptionsT, axis=1, fusedActivationFunction=ACT.NONE))
    qcc = (0.11, 133)
    cc = m.op(OP.CONCATENATION, [c0, c1], [m.act([1, 40, 3], *qcc)], BO.ConcatenationOptions,
              opts(S.ConcatenationOptionsT, axis=1, fusedActivationFunction=ACT.NONE))
    boxes = reshape(m, bc, [1, 40, 4], *qb0)
    scores = m.op(OP.LOGISTIC, [cc], [m.act([1, 40, 3], 1.0 / 256, 0)])
    sa = 1.0 / 128
    an = m.tensor([40, 4], T_U8, sa, 0, np.clip(np.round(anchors() / sa), 0, 255).astype(np.uint8))
    outs = [m.tensor([1, 5, 4], T_F32), m.tensor([1, 5], T_F32), m.tensor([1, 5], T_F32), m.tensor([1], T_F32)]
    m.op(None, [boxes, scores, an], outs, custom=POSTPROCESS)
    return m.bytes([x], outs)


def main():
    model = build()
    it = I.Interpreter(model_content=model, experimental_op_resolver_type=I.OpResolverType.BUILTIN_REF,
                       experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    ops = it._get_ops_details()
    kept = [o for o in ops if o["op_name"] != POSTPROCESS]
    d = it.get_input_details()[0]
    outs = [o["index"] for o in it.get_output_details()]
    i8 = lambda v: base64.b64encode((np.asarray(v).astype(np.int16) - 128).astype(np.int8).reshape(-1).tobytes()).decode()  # noqa: E731
    f32 = lambda v: base64.b64encode(np.asarray(v, dtype="<f4").reshape(-1).tobytes()).decode()  # noqa: E731
    cases = []
    for _ in range(64):
        x = rng.integers(0, 256, size=d["shape"], dtype=np.uint8)
        it.set_tensor(d["index"], x)
        it.invoke()
        cases.append({"input": i8(x), "outs": [i8(it.get_tensor(o["outputs"][0])) for o in kept],
                      "post": [f32(it.get_tensor(i)) for i in outs]})
    names = [o["op_name"] for o in kept]
    open("ssdtail.tflite", "wb").write(model)
    doc = json.dumps({"generator": "gen_ssdtail.py", "litert": "ai-edge-litert 2.1.5, BUILTIN_REF",
                      "ops": names, "cases": cases}, indent=1).encode()
    open("ssdtail.json.gz", "wb").write(gzip.compress(doc, compresslevel=9, mtime=0))
    print(len(model), "bytes;", len(cases), "cases;", names)
    print("detections per case:", [int(np.frombuffer(base64.b64decode(c["post"][3]), "<f4")[0]) for c in cases])


main()
