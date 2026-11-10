package tfl

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"slices"
	"testing"
)

// TestFBWriterFields checks the low-level writer against the reader: every
// field kind, vtable layout, absent/explicit-zero fields and alignment.
func TestFBWriterFields(t *testing.T) {
	const big = -1<<40 - 5
	root := (&tableSpec{}).
		u8(0, 200).
		boolean(1, true).
		i32(2, -123456).
		u32(3, 0xDEADBEEF).
		f32(4, 3.25).
		i64(5, big).
		table(6, (&tableSpec{}).i32(0, -7)).
		union(7, 9, (&tableSpec{}).f32(1, 2.5)).
		str(9, "hello").
		vecI32(10, []int32{1, -2, 3}).
		vecF32(11, []float32{0.5, -1.5}).
		vecI64(12, []int64{-1, 1 << 40}).
		vecU8(13, []byte{1, 2, 3}, 16).
		vecTables(14, []*tableSpec{(&tableSpec{}).i32(0, 10), {}, (&tableSpec{}).str(2, "x")}).
		i32(16, 0).
		i32(17, 5).i32(17, 6). // replaced
		u8(18, 1).drop(18)     // removed
	b := fbFinish(root, "TEST")
	if string(b[4:8]) != "TEST" {
		t.Fatalf("identifier %q", b[4:8])
	}
	f := fb{b: b}
	r := f.root()
	if !r.ok {
		t.Fatal("root table not ok")
	}
	if want := 4 + 2*18; r.vtLen != want {
		t.Errorf("vtable size %d, want %d", r.vtLen, want)
	}
	if r.pos%8 != 0 {
		t.Errorf("table with an int64 field at %d, not 8-aligned", r.pos)
	}
	tsize := int(binary.LittleEndian.Uint16(b[r.vt+2:]))
	for s := range 18 {
		o := int(binary.LittleEndian.Uint16(b[r.vt+4+2*s:]))
		if o != 0 && (o < 4 || o >= tsize) {
			t.Errorf("slot %d offset %d outside table size %d", s, o, tsize)
		}
	}

	if got := f.fieldU8(r, 0, 0); got != 200 {
		t.Errorf("u8 %d", got)
	}
	if !f.fieldBool(r, 1) {
		t.Error("bool false")
	}
	if got := f.fieldI32(r, 2, 0); got != -123456 {
		t.Errorf("i32 %d", got)
	}
	if got := f.fieldU32(r, 3, 0); got != 0xDEADBEEF {
		t.Errorf("u32 %#x", got)
	}
	if got := f.fieldF32(r, 4, 0); got != 3.25 {
		t.Errorf("f32 %v", got)
	}
	if p := f.field(r, 5); p < 0 || p%8 != 0 {
		t.Errorf("i64 field at %d", p)
	} else if got := int64(binary.LittleEndian.Uint64(b[p:])); got != big {
		t.Errorf("i64 %d", got)
	}
	if st := f.fieldTable(r, 6); !st.ok || f.fieldI32(st, 0, 0) != -7 {
		t.Errorf("nested table %+v", st)
	}
	if got := f.fieldU8(r, 7, 0); got != 9 {
		t.Errorf("union type %d", got)
	}
	ut := f.fieldTable(r, 8)
	if !ut.ok || f.fieldF32(ut, 1, 0) != 2.5 || f.fieldF32(ut, 0, 99) != 99 {
		t.Errorf("union value %+v", ut)
	}
	sv := f.fieldVec(r, 9, 1)
	if string(f.bytes(sv)) != "hello" || sv.pos%4 != 0 || b[sv.pos+sv.n] != 0 {
		t.Errorf("string at %d %q", sv.pos, f.bytes(sv))
	}
	iv := f.fieldVec(r, 10, 4)
	if iv.n != 3 || iv.pos%4 != 0 || f.vecI32(iv, 0) != 1 || f.vecI32(iv, 1) != -2 || f.vecI32(iv, 2) != 3 {
		t.Errorf("int32 vector %+v", iv)
	}
	fv := f.fieldVec(r, 11, 4)
	if fv.n != 2 || fv.pos%4 != 0 || f.vecF32(fv, 0) != 0.5 || f.vecF32(fv, 1) != -1.5 {
		t.Errorf("float32 vector %+v", fv)
	}
	lv := f.fieldVec(r, 12, 8)
	if lv.n != 2 || lv.pos%8 != 0 || f.vecI64(lv, 0) != -1 || f.vecI64(lv, 1) != 1<<40 {
		t.Errorf("int64 vector %+v", lv)
	}
	bv := f.fieldVec(r, 13, 1)
	if bv.pos%16 != 0 || !bytes.Equal(f.bytes(bv), []byte{1, 2, 3}) {
		t.Errorf("byte vector at %d %v", bv.pos, f.bytes(bv))
	}
	tv := f.fieldVec(r, 14, 4)
	if tv.n != 3 {
		t.Fatalf("table vector length %d", tv.n)
	}
	if e := f.vecTable(tv, 0); f.fieldI32(e, 0, 0) != 10 {
		t.Error("table vector element 0")
	}
	if e := f.vecTable(tv, 1); !e.ok || e.vtLen != 4 || f.fieldI32(e, 0, 42) != 42 {
		t.Errorf("empty table element %+v", e)
	}
	if e := f.vecTable(tv, 2); string(f.fieldString(e, 2)) != "x" || f.field(e, 0) != -1 {
		t.Error("table vector element 2")
	}
	if f.field(r, 15) != -1 {
		t.Error("unset slot 15 present")
	}
	if f.field(r, 16) < 0 || f.fieldI32(r, 16, 1) != 0 {
		t.Error("explicit zero in slot 16 not written")
	}
	if got := f.fieldI32(r, 17, 0); got != 6 {
		t.Errorf("replaced slot 17 = %d", got)
	}
	if f.field(r, 18) != -1 || f.field(r, 40) != -1 {
		t.Error("dropped/out-of-vtable slot present")
	}
	if f.err != nil {
		t.Fatal(f.err)
	}
}

// rtTensor and rtOp describe the round-trip model.
type rtTensor struct {
	name  string
	shape []int32
	typ   uint8
	data  []byte // nil: buffer 0
	scale []float32
	zp    []int64
	qdim  int32
}

type rtOp struct {
	code    int32
	in, out []int32
	optType uint8
	opts    *tableSpec
}

func le32(v ...int32) []byte {
	b := make([]byte, 0, 4*len(v))
	for _, x := range v {
		b = binary.LittleEndian.AppendUint32(b, uint32(x))
	}
	return b
}

func seq(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i*7 - 50)
	}
	return b
}

const opBuiltin145 = 145 // a builtin code ≥ 127 (deprecated_builtin_code 127)

var (
	rtTensors = []rtTensor{
		{"input", []int32{1, 4, 4, 2}, typeInt8, nil, []float32{0.5}, []int64{-3}, 0},
		{"conv_w", []int32{3, 1, 1, 2}, typeInt8, []byte{1, 0xfe, 3, 0xfc, 5, 0xfa}, []float32{0.1, 0.2, 0.3}, []int64{0, 0, 0}, 0},
		{"conv_b", []int32{3}, typeInt32, le32(100, -200, 300), []float32{0.05, 0.1, 0.15}, []int64{0, 0, 0}, 0},
		{"conv_out", []int32{1, 4, 4, 3}, typeInt8, nil, []float32{0.25}, []int64{5}, 0},
		{"dw_w", []int32{1, 3, 3, 3}, typeInt8, seq(27), []float32{0.01, 0.02, 0.03}, []int64{0, 0, 0}, 3},
		{"dw_out", []int32{1, 4, 4, 3}, typeInt8, nil, []float32{0.125}, []int64{-1}, 0},
		{"pool_out", []int32{1, 2, 2, 3}, typeInt8, nil, []float32{0.125}, []int64{-1}, 0},
		{"shape", []int32{2}, typeInt32, le32(1, 12), nil, nil, 0},
		{"reshape_out", []int32{1, 12}, typeInt8, nil, []float32{0.125}, []int64{-1}, 0},
		{"fc_w", []int32{4, 12}, typeInt8, seq(48), []float32{0.02}, []int64{0}, 0},
		{"fc_out", []int32{1, 4}, typeInt8, nil, []float32{0.5}, []int64{2}, 0},
		{"softmax_out", []int32{1, 4}, typeInt8, nil, []float32{1.0 / 256}, []int64{-128}, 0},
		{"add_out", []int32{1, 4}, typeInt8, nil, []float32{1.0 / 128}, []int64{-128}, 0},
		{"add2_out", []int32{1, 4}, typeInt8, nil, []float32{1.0 / 128}, []int64{-128}, 0},
		{"output", []int32{1, 4}, typeInt8, nil, []float32{1.0 / 128}, []int64{-128}, 0},
	}
	rtOps = []rtOp{
		{opConv2D, []int32{0, 1, 2}, []int32{3}, optConv2D, optsConv2D(padSame, 1, 1, actRelu6, 1, 1)},
		{opDepthwiseConv2D, []int32{3, 4, -1}, []int32{5}, optDepthwiseConv2D, optsDepthwiseConv2D(padValid, 2, 3, 4, actRelu, 5, 6)},
		{opMaxPool2D, []int32{5}, []int32{6}, optPool2D, optsPool2D(padValid, 2, 3, 4, 5, actReluN1To1)},
		{opReshape, []int32{6, 7}, []int32{8}, optReshape, optsReshape([]int32{1, 12})},
		{opFullyConnected, []int32{8, 9, -1}, []int32{10}, optFullyConnected, optsFullyConnected(actRelu)},
		{opSoftmax, []int32{10}, []int32{11}, optSoftmax, optsSoftmax(1.0)},
		{opAdd, []int32{11, 11}, []int32{12}, optAdd, optsAdd(actRelu6)},
		{opAdd, []int32{12, 11}, []int32{13}, optAdd, optsAdd(actNone)},
		{opBuiltin145, []int32{13}, []int32{14}, 0, nil},
	}
	rtCodes = []int32{opConv2D, opDepthwiseConv2D, opMaxPool2D, opReshape, opFullyConnected, opSoftmax, opAdd, opBuiltin145}
)

// buildRoundtrip serialises rtTensors/rtOps and returns the bytes and each
// tensor's buffer index.
func buildRoundtrip() ([]byte, []int) {
	b := newTFLBuilder()
	bufs := make([]int, len(rtTensors))
	for i, tt := range rtTensors {
		if tt.data != nil {
			bufs[i] = b.buffer(tt.data)
		}
		if got := b.tensor(tt.name, tt.shape, tt.typ, bufs[i], tt.scale, tt.zp, tt.qdim); got != i {
			panic("tensor index")
		}
	}
	for i, o := range rtOps {
		if got := b.op(o.code, o.in, o.out, o.optType, o.opts); got != i {
			panic("op index")
		}
	}
	b.io([]int32{0}, []int32{14})
	return b.bytes(), bufs
}

func vecEqual(f *fb, v vec, want []int32) bool {
	if v.n != len(want) {
		return false
	}
	for k, w := range want {
		if f.vecI32(v, k) != w {
			return false
		}
	}
	return true
}

func TestBuilderRoundtrip(t *testing.T) {
	data, bufs := buildRoundtrip()
	if string(data[4:8]) != "TFL3" {
		t.Fatalf("identifier %q", data[4:8])
	}
	m, err := Open(data)
	if err != nil {
		t.Fatal(err)
	}
	f := &m.f
	if m.Version() != 3 || m.NumTensors() != len(rtTensors) || m.NumOperators() != len(rtOps) {
		t.Fatalf("version %d tensors %d ops %d", m.Version(), m.NumTensors(), m.NumOperators())
	}
	if !vecEqual(f, m.inputs, []int32{0}) || !vecEqual(f, m.outputs, []int32{14}) {
		t.Errorf("subgraph io: %d inputs, %d outputs", m.inputs.n, m.outputs.n)
	}

	// Tensors.
	for i, want := range rtTensors {
		tt, err := m.tensor(i)
		if err != nil {
			t.Fatalf("tensor %d: %v", i, err)
		}
		if string(tt.name) != want.name || !slices.Equal(tt.shape, want.shape) || tt.typ != want.typ ||
			tt.buffer != bufs[i] || tt.qdim != want.qdim || tt.sparse {
			t.Errorf("tensor %d: %q %v type %d buf %d qdim %d sparse %v; want %+v buf %d",
				i, tt.name, tt.shape, tt.typ, tt.buffer, tt.qdim, tt.sparse, want, bufs[i])
		}
		if tt.scale.n != len(want.scale) || tt.zp.n != len(want.zp) {
			t.Errorf("tensor %d: %d scales %d zps", i, tt.scale.n, tt.zp.n)
			continue
		}
		for k := range want.scale {
			if got := m.scaleAt(&tt, k); got != want.scale[k] {
				t.Errorf("tensor %d scale[%d] = %v, want %v", i, k, got, want.scale[k])
			}
		}
		for k := range want.zp {
			if got := m.zpAt(&tt, k); got != want.zp[k] {
				t.Errorf("tensor %d zp[%d] = %d, want %d", i, k, got, want.zp[k])
			}
		}
		raw := f.vecTable(m.tensors, i)
		hasQ := f.field(raw, tensorQuantization) >= 0
		if hasQ != (want.scale != nil) {
			t.Errorf("tensor %d: quantization present %v", i, hasQ)
		}
		if hasQ && f.field(f.fieldTable(raw, tensorQuantization), quantDimension) < 0 {
			t.Errorf("tensor %d: quantized_dimension absent", i)
		}
	}

	// Buffers: 0 is the empty buffer; data is 16-byte aligned and aliased.
	nbuf := 1
	for _, tt := range rtTensors {
		if tt.data != nil {
			nbuf++
		}
	}
	if m.buffers.n != nbuf {
		t.Fatalf("%d buffers, want %d", m.buffers.n, nbuf)
	}
	if d, err := m.bufferData(0); d != nil || err != nil || f.field(f.vecTable(m.buffers, 0), bufferData) >= 0 {
		t.Errorf("buffer 0: %v %v", d, err)
	}
	for i, tt := range rtTensors {
		if tt.data == nil {
			continue
		}
		d, err := m.bufferData(bufs[i])
		if err != nil || !bytes.Equal(d, tt.data) {
			t.Errorf("buffer %d (tensor %d): %v %v", bufs[i], i, d, err)
			continue
		}
		v := f.fieldVec(f.vecTable(m.buffers, bufs[i]), bufferData, 1)
		if v.pos%16 != 0 || &d[0] != &data[v.pos] {
			t.Errorf("buffer %d data at %d: not 16-byte aligned or not aliased", bufs[i], v.pos)
		}
	}

	// Operator codes: deduplicated, in order of first use; ≥ 127 split.
	if m.codes.n != len(rtCodes) {
		t.Fatalf("%d operator codes, want %d", m.codes.n, len(rtCodes))
	}
	for ci, code := range rtCodes {
		c := f.vecTable(m.codes, ci)
		dep := f.fieldU8(c, opcodeDeprecatedBuiltin, 0xff)
		bi := f.fieldI32(c, opcodeBuiltin, -1)
		if int32(dep) != min(code, 127) || bi != code || f.fieldI32(c, opcodeVersion, 0) != 1 {
			t.Errorf("operator code %d: deprecated %d builtin %d, want %d", ci, dep, bi, code)
		}
	}

	// Operators.
	for i, want := range rtOps {
		o, err := m.operator(i)
		if err != nil {
			t.Fatalf("operator %d: %v", i, err)
		}
		raw := f.vecTable(m.ops, i)
		ci := int(f.fieldU32(raw, operatorOpcodeIndex, 99))
		if o.code != want.code || o.version != 1 || len(o.custom) != 0 || rtCodes[ci] != want.code {
			t.Errorf("operator %d: code %d version %d opcode index %d, want %d", i, o.code, o.version, ci, want.code)
		}
		if !vecEqual(f, o.inputs, want.in) || !vecEqual(f, o.outputs, want.out) {
			t.Errorf("operator %d: inputs/outputs", i)
		}
		if o.optType != want.optType || o.opts.ok != (want.opts != nil) {
			t.Errorf("operator %d: optType %d opts %v", i, o.optType, o.opts.ok)
		}
		if want.opts == nil && f.field(raw, operatorOptionsType) >= 0 {
			t.Errorf("operator %d: options type written without options", i)
		}
	}
	if a, b := f.fieldU32(f.vecTable(m.ops, 6), operatorOpcodeIndex, 0), f.fieldU32(f.vecTable(m.ops, 7), operatorOpcodeIndex, 1); a != b {
		t.Errorf("ADD ops use opcode indices %d and %d", a, b)
	}

	// Options, read with the schema slots. Defaults differ from the expected
	// values so absent fields fail.
	opts := func(i int) table {
		o, err := m.operator(i)
		if err != nil {
			t.Fatal(err)
		}
		return o.opts
	}
	u8 := func(op int, o table, slot int, want uint8) {
		t.Helper()
		if got := f.fieldU8(o, slot, ^want); got != want {
			t.Errorf("op %d slot %d: %d, want %d", op, slot, got, want)
		}
	}
	i32 := func(op int, o table, slot int, want int32) {
		t.Helper()
		if got := f.fieldI32(o, slot, ^want); got != want {
			t.Errorf("op %d slot %d: %d, want %d", op, slot, got, want)
		}
	}
	nan := float32(math.NaN())

	o := opts(0)
	u8(0, o, convPadding, padSame)
	i32(0, o, convStrideW, 1)
	i32(0, o, convStrideH, 1)
	u8(0, o, convAct, actRelu6)
	i32(0, o, convDilW, 1)
	i32(0, o, convDilH, 1)

	o = opts(1)
	u8(1, o, dwPadding, padValid)
	i32(1, o, dwStrideW, 2)
	i32(1, o, dwStrideH, 3)
	i32(1, o, dwMultiplier, 4)
	u8(1, o, dwAct, actRelu)
	i32(1, o, dwDilW, 5)
	i32(1, o, dwDilH, 6)

	o = opts(2)
	u8(2, o, poolPadding, padValid)
	i32(2, o, poolStrideW, 2)
	i32(2, o, poolStrideH, 3)
	i32(2, o, poolFilterW, 4)
	i32(2, o, poolFilterH, 5)
	u8(2, o, poolAct, actReluN1To1)

	if ns := f.fieldVec(opts(3), tbReshapeNewShape, 4); !vecEqual(f, ns, []int32{1, 12}) {
		t.Errorf("reshape new_shape: %d entries", ns.n)
	}
	u8(4, opts(4), fcAct, actRelu)
	if beta := f.fieldF32(opts(5), softmaxBeta, nan); beta != 1.0 {
		t.Errorf("softmax beta %v", beta)
	}
	u8(6, opts(6), addAct, actRelu6)
	u8(7, opts(7), addAct, actNone)

	if f.err != nil {
		t.Fatal(f.err)
	}
}

// walkModel opens b and reads everything the reader exposes, returning the
// first error.
func walkModel(b []byte) error {
	m, err := Open(b)
	if err != nil {
		return err
	}
	f := &m.f
	for k := range m.inputs.n {
		f.vecI32(m.inputs, k)
	}
	for k := range m.outputs.n {
		f.vecI32(m.outputs, k)
	}
	for i := range m.NumTensors() {
		tt, err := m.tensor(i)
		if err != nil {
			return err
		}
		for k := range tt.scale.n {
			m.scaleAt(&tt, k)
		}
		for k := range tt.zp.n {
			m.zpAt(&tt, k)
		}
	}
	for i := range m.NumOperators() {
		o, err := m.operator(i)
		if err != nil {
			return err
		}
		for k := range o.inputs.n {
			f.vecI32(o.inputs, k)
		}
		for k := range o.outputs.n {
			f.vecI32(o.outputs, k)
		}
		for s := range 8 {
			f.fieldI32(o.opts, s, 0)
		}
	}
	for i := range m.buffers.n {
		if _, err := m.bufferData(i); err != nil {
			return err
		}
	}
	return f.err
}

// TestBuilderTruncated cuts the model at every length: the reader must report
// ErrFormat (the last bytes are buffer data, so every cut loses read bytes)
// and never panic.
func TestBuilderTruncated(t *testing.T) {
	data, _ := buildRoundtrip()
	if err := walkModel(data); err != nil {
		t.Fatalf("full model: %v", err)
	}
	for n := range len(data) {
		cut := append([]byte(nil), data[:n]...)
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("len %d: panic %v", n, r)
				}
			}()
			if err := walkModel(cut); !errors.Is(err, ErrFormat) {
				t.Errorf("len %d of %d: %v, want ErrFormat", n, len(data), err)
			}
		}()
	}
}

func TestBuilderTwoSubgraphs(t *testing.T) {
	b := newTFLBuilder()
	b.tensor("a", []int32{1, 4}, typeInt8, 0, []float32{0.5}, []int64{0}, 0)
	b.io([]int32{0}, []int32{0})
	if k := b.newSubgraph(); k != 1 {
		t.Fatalf("second subgraph index %d", k)
	}
	b.tensor("b0", []int32{1, 4}, typeInt8, 0, nil, nil, 0)
	b.tensor("b1", []int32{1, 4}, typeInt8, 0, nil, nil, 0)
	b.op(opAdd, []int32{0, 0}, []int32{1}, optAdd, optsAdd(actNone))
	b.io([]int32{0}, []int32{1})
	data := b.bytes()

	if _, err := Open(data); !errors.Is(err, ErrUnsupported) {
		t.Errorf("Open: %v, want ErrUnsupported", err)
	}
	f := fb{b: data}
	sgs := f.fieldVec(f.root(), modelSubgraphs, 4)
	if sgs.n != 2 {
		t.Fatalf("%d subgraphs", sgs.n)
	}
	sg0, sg1 := f.vecTable(sgs, 0), f.vecTable(sgs, 1)
	if n := f.fieldVec(sg0, subgraphTensors, 4).n; n != 1 {
		t.Errorf("subgraph 0: %d tensors", n)
	}
	if n := f.fieldVec(sg1, subgraphTensors, 4).n; n != 2 {
		t.Errorf("subgraph 1: %d tensors", n)
	}
	if n := f.fieldVec(sg1, subgraphOperators, 4).n; n != 1 {
		t.Errorf("subgraph 1: %d operators", n)
	}
	if !vecEqual(&f, f.fieldVec(sg1, subgraphOutputs, 4), []int32{1}) {
		t.Error("subgraph 1 outputs")
	}
	if f.err != nil {
		t.Fatal(f.err)
	}
}

func TestBuilderSparsityAndQuantDetails(t *testing.T) {
	b := newTFLBuilder()
	shape := []int32{1, 4}
	b.tensor("dense", shape, typeInt8, 0, []float32{0.5}, []int64{0}, 0)
	b.sparse(b.tensor("sparse", shape, typeInt8, 0, []float32{0.5}, []int64{0}, 0))
	b.quantDetails(b.tensor("custom_q", shape, typeInt8, 0, []float32{0.5}, []int64{0}, 0), 1)
	b.quantDetails(b.tensor("custom_noq", shape, typeInt8, 0, nil, nil, 0), 1)
	b.io([]int32{0}, []int32{0})
	m, err := Open(b.bytes())
	if err != nil {
		t.Fatal(err)
	}
	if tt, err := m.tensor(0); err != nil || tt.sparse {
		t.Errorf("dense: sparse %v err %v", tt.sparse, err)
	}
	if tt, err := m.tensor(1); err != nil || !tt.sparse {
		t.Errorf("sparse: sparse %v err %v", tt.sparse, err)
	}
	for i := 2; i < 4; i++ {
		if _, err := m.tensor(i); !errors.Is(err, ErrUnsupported) {
			t.Errorf("tensor %d with details_type: %v, want ErrUnsupported", i, err)
		}
		f := &m.f
		q := f.fieldTable(f.vecTable(m.tensors, i), tensorQuantization)
		if f.fieldU8(q, quantDetails, 0) != 1 || !f.fieldTable(q, quantDetails+1).ok {
			t.Errorf("tensor %d: details union not written", i)
		}
	}
	if f := &m.f; f.fieldVec(f.fieldTable(f.vecTable(m.tensors, 2), tensorQuantization), quantScale, 4).n != 1 {
		t.Error("custom_q: scale lost")
	}
}
