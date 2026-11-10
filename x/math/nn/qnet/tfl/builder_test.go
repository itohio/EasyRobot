package tfl

import (
	"encoding/binary"
	"math"
	"slices"
)

// A minimal FlatBuffers writer for tests: builds TFLite models in memory so
// the loader is exercised without downloaded .tflite files.
//
// A table is described by a tableSpec (typed fields by vtable slot) and
// serialised front to back: vtable, then the table's inline fields, then the
// objects its offset fields reference (depth first, in slot order), so every
// uoffset points forward as FlatBuffers requires. Scalars are naturally
// aligned within the output; vectors are aligned to their element size (or a
// forced alignment, Buffer.data uses 16). Little-endian throughout.

// tbReshapeNewShape is ReshapeOptions.new_shape (schema.go has no constant).
const tbReshapeNewShape = 0

type fbKind uint8

const (
	fbU8     fbKind = iota // u8 / bool / byte enums
	fbI32                  // int32 / uint32 / float32 bits
	fbI64                  // int64
	fbTable                // offset to a table
	fbString               // offset to a string
	fbVector               // offset to a vector of scalars (raw LE bytes)
	fbTables               // offset to a vector of tables
)

// fbField is one table field.
type fbField struct {
	slot  int
	kind  fbKind
	bits  uint64       // scalar value
	sub   *tableSpec   // fbTable
	subs  []*tableSpec // fbTables
	data  []byte       // fbString, fbVector
	n     int          // fbVector element count
	align int          // fbVector element-data alignment
}

// size is the field's inline size in the table.
func (f *fbField) size() int {
	switch f.kind {
	case fbU8:
		return 1
	case fbI64:
		return 8
	}
	return 4 // 32-bit scalars and uoffsets
}

// tableSpec describes a table: fields keyed by vtable slot. Every field set is
// written (no default elision), so explicit zeros are present in the vtable.
// Setters replace an existing field in the same slot and return t.
type tableSpec struct {
	fields []fbField
}

func (t *tableSpec) put(f fbField) *tableSpec {
	for i := range t.fields {
		if t.fields[i].slot == f.slot {
			t.fields[i] = f
			return t
		}
	}
	t.fields = append(t.fields, f)
	return t
}

func (t *tableSpec) u8(slot int, v uint8) *tableSpec {
	return t.put(fbField{slot: slot, kind: fbU8, bits: uint64(v)})
}

func (t *tableSpec) boolean(slot int, v bool) *tableSpec {
	var b uint8
	if v {
		b = 1
	}
	return t.u8(slot, b)
}

func (t *tableSpec) i32(slot int, v int32) *tableSpec { return t.u32(slot, uint32(v)) }

func (t *tableSpec) u32(slot int, v uint32) *tableSpec {
	return t.put(fbField{slot: slot, kind: fbI32, bits: uint64(v)})
}

func (t *tableSpec) f32(slot int, v float32) *tableSpec {
	return t.u32(slot, math.Float32bits(v))
}

func (t *tableSpec) i64(slot int, v int64) *tableSpec {
	return t.put(fbField{slot: slot, kind: fbI64, bits: uint64(v)})
}

// table sets a nested table.
func (t *tableSpec) table(slot int, sub *tableSpec) *tableSpec {
	return t.put(fbField{slot: slot, kind: fbTable, sub: sub})
}

// union sets a union: the type byte in typeSlot and the value table in
// typeSlot+1 (omitted when sub is nil).
func (t *tableSpec) union(typeSlot int, typ uint8, sub *tableSpec) *tableSpec {
	t.u8(typeSlot, typ)
	if sub != nil {
		t.table(typeSlot+1, sub)
	}
	return t
}

func (t *tableSpec) str(slot int, s string) *tableSpec {
	return t.put(fbField{slot: slot, kind: fbString, data: []byte(s)})
}

func (t *tableSpec) vecI32(slot int, v []int32) *tableSpec {
	d := make([]byte, 4*len(v))
	for k, x := range v {
		binary.LittleEndian.PutUint32(d[4*k:], uint32(x))
	}
	return t.put(fbField{slot: slot, kind: fbVector, data: d, n: len(v), align: 4})
}

func (t *tableSpec) vecF32(slot int, v []float32) *tableSpec {
	d := make([]byte, 4*len(v))
	for k, x := range v {
		binary.LittleEndian.PutUint32(d[4*k:], math.Float32bits(x))
	}
	return t.put(fbField{slot: slot, kind: fbVector, data: d, n: len(v), align: 4})
}

func (t *tableSpec) vecI64(slot int, v []int64) *tableSpec {
	d := make([]byte, 8*len(v))
	for k, x := range v {
		binary.LittleEndian.PutUint64(d[8*k:], uint64(x))
	}
	return t.put(fbField{slot: slot, kind: fbVector, data: d, n: len(v), align: 8})
}

// vecU8 sets a byte vector whose data starts at a multiple of align.
func (t *tableSpec) vecU8(slot int, v []byte, align int) *tableSpec {
	return t.put(fbField{slot: slot, kind: fbVector, data: slices.Clone(v), n: len(v), align: align})
}

func (t *tableSpec) vecTables(slot int, v []*tableSpec) *tableSpec {
	return t.put(fbField{slot: slot, kind: fbTables, subs: v})
}

// sub returns the nested table in slot, or nil.
func (t *tableSpec) sub(slot int) *tableSpec {
	for i := range t.fields {
		if t.fields[i].slot == slot && t.fields[i].kind == fbTable {
			return t.fields[i].sub
		}
	}
	return nil
}

// drop removes the field in slot (absent from the vtable).
func (t *tableSpec) drop(slot int) *tableSpec {
	t.fields = slices.DeleteFunc(t.fields, func(f fbField) bool { return f.slot == slot })
	return t
}

// fbWriter appends FlatBuffers objects front to back.
type fbWriter struct {
	b []byte
}

// pad appends zeros until len+extra is a multiple of align.
func (w *fbWriter) pad(align, extra int) {
	for (len(w.b)+extra)%align != 0 {
		w.b = append(w.b, 0)
	}
}

func (w *fbWriter) u16(v uint16) { w.b = binary.LittleEndian.AppendUint16(w.b, v) }
func (w *fbWriter) u32(v uint32) { w.b = binary.LittleEndian.AppendUint32(w.b, v) }

// patch stores at off the uoffset to target.
func (w *fbWriter) patch(off, target int) {
	binary.LittleEndian.PutUint32(w.b[off:], uint32(target-off))
}

// table writes t (vtable, inline fields, then referenced objects) and returns
// the table's position.
func (w *fbWriter) table(t *tableSpec) int {
	fields := slices.Clone(t.fields)
	// Largest first: 8-byte fields sit 8-aligned right after the soffset.
	slices.SortStableFunc(fields, func(a, b fbField) int { return b.size() - a.size() })
	nslots, align, end := 0, 4, 4
	offs := make([]int, len(fields))
	for i := range fields {
		f := &fields[i]
		if f.slot < 0 {
			panic("tableSpec: negative slot")
		}
		nslots = max(nslots, f.slot+1)
		s := f.size()
		align = max(align, s)
		end = (end + s - 1) / s * s
		offs[i] = end
		end += s
	}
	vtSize := 4 + 2*nslots
	w.pad(align, vtSize) // the table follows the vtable, aligned
	vt := len(w.b)
	w.u16(uint16(vtSize))
	w.u16(uint16(end))
	slot := make([]uint16, nslots)
	for i := range fields {
		slot[fields[i].slot] = uint16(offs[i])
	}
	for _, o := range slot {
		w.u16(o)
	}
	pos := len(w.b)
	w.b = append(w.b, make([]byte, end)...)
	binary.LittleEndian.PutUint32(w.b[pos:], uint32(pos-vt)) // soffset to vtable
	type ref struct {
		at int
		f  *fbField
	}
	var refs []ref
	for i := range fields {
		f := &fields[i]
		p := pos + offs[i]
		switch f.kind {
		case fbU8:
			w.b[p] = byte(f.bits)
		case fbI32:
			binary.LittleEndian.PutUint32(w.b[p:], uint32(f.bits))
		case fbI64:
			binary.LittleEndian.PutUint64(w.b[p:], f.bits)
		default:
			refs = append(refs, ref{p, f})
		}
	}
	slices.SortFunc(refs, func(a, b ref) int { return a.f.slot - b.f.slot })
	for _, r := range refs {
		w.patch(r.at, w.object(r.f))
	}
	return pos
}

// object writes what an offset field references and returns its position.
func (w *fbWriter) object(f *fbField) int {
	switch f.kind {
	case fbTable:
		return w.table(f.sub)
	case fbString:
		w.pad(4, 0)
		p := len(w.b)
		w.u32(uint32(len(f.data)))
		w.b = append(w.b, f.data...)
		w.b = append(w.b, 0)
		return p
	case fbVector:
		w.pad(max(f.align, 4), 4) // length prefix 4-aligned, data align-aligned
		p := len(w.b)
		w.u32(uint32(f.n))
		w.b = append(w.b, f.data...)
		return p
	case fbTables:
		w.pad(4, 0)
		p := len(w.b)
		w.u32(uint32(len(f.subs)))
		w.b = append(w.b, make([]byte, 4*len(f.subs))...)
		for k, s := range f.subs {
			w.patch(p+4+4*k, w.table(s))
		}
		return p
	}
	panic("tableSpec: bad field kind")
}

// fbFinish serialises root: the root uoffset at byte 0, the 4-byte file
// identifier at bytes 4..7, then the object graph.
func fbFinish(root *tableSpec, ident string) []byte {
	if len(ident) != 4 {
		panic("fbFinish: identifier must be 4 bytes")
	}
	w := fbWriter{b: make([]byte, 8, 4096)}
	copy(w.b[4:], ident)
	w.patch(0, w.table(root))
	return w.b
}

// tflSubgraph collects one subgraph's tables.
type tflSubgraph struct {
	tensors, ops    []*tableSpec
	inputs, outputs []int32
}

// tflBuilder assembles a TFLite model. Tensor, op and io calls go to the
// current (last created) subgraph; buffers and operator codes are shared.
type tflBuilder struct {
	codes   []*tableSpec
	codeIdx map[int32]int
	buffers []*tableSpec
	sgs     []*tflSubgraph
}

// newTFLBuilder starts a model with the conventional empty buffer 0 and one
// subgraph.
func newTFLBuilder() *tflBuilder {
	b := &tflBuilder{codeIdx: map[int32]int{}, buffers: []*tableSpec{{}}}
	b.newSubgraph()
	return b
}

// newSubgraph appends a subgraph, makes it current and returns its index.
func (b *tflBuilder) newSubgraph() int {
	b.sgs = append(b.sgs, &tflSubgraph{})
	return len(b.sgs) - 1
}

func (b *tflBuilder) cur() *tflSubgraph { return b.sgs[len(b.sgs)-1] }

// buffer appends a Buffer (data 16-byte aligned; nil data = no data field)
// and returns its index.
func (b *tflBuilder) buffer(data []byte) int {
	t := &tableSpec{}
	if data != nil {
		t.vecU8(bufferData, data, 16)
	}
	b.buffers = append(b.buffers, t)
	return len(b.buffers) - 1
}

// tensor appends a Tensor and returns its index. A QuantizationParameters
// table is written when scale or zp is non-nil.
func (b *tflBuilder) tensor(name string, shape []int32, typ uint8, buffer int, scale []float32, zp []int64, qdim int32) int {
	t := (&tableSpec{}).
		vecI32(tensorShape, shape).
		u8(tensorType, typ).
		u32(tensorBuffer, uint32(buffer)).
		str(tensorName, name)
	if scale != nil || zp != nil {
		q := &tableSpec{}
		if scale != nil {
			q.vecF32(quantScale, scale)
		}
		if zp != nil {
			q.vecI64(quantZeroPoint, zp)
		}
		q.i32(quantDimension, qdim)
		t.table(tensorQuantization, q)
	}
	sg := b.cur()
	sg.tensors = append(sg.tensors, t)
	return len(sg.tensors) - 1
}

// sparse gives tensor ti a (empty) sparsity table.
func (b *tflBuilder) sparse(ti int) {
	b.tensorTable(ti).table(tensorSparsity, &tableSpec{})
}

// quantDetails sets tensor ti's QuantizationParameters.details union to type
// typ with an empty value table (creating the quantization table if absent).
func (b *tflBuilder) quantDetails(ti int, typ uint8) {
	t := b.tensorTable(ti)
	q := t.sub(tensorQuantization)
	if q == nil {
		q = &tableSpec{}
		t.table(tensorQuantization, q)
	}
	q.union(quantDetails, typ, &tableSpec{})
}

// op appends an Operator and returns its index. Operator codes are
// deduplicated; codes ≥ 127 store deprecated_builtin_code 127 and the real
// code in builtin_code. Options are written when optType != 0 or opts != nil.
func (b *tflBuilder) op(code int32, inputs, outputs []int32, optType uint8, opts *tableSpec) int {
	ci, ok := b.codeIdx[code]
	if !ok {
		c := (&tableSpec{}).
			u8(opcodeDeprecatedBuiltin, uint8(min(code, 127))).
			i32(opcodeBuiltin, code).
			i32(opcodeVersion, 1)
		ci = len(b.codes)
		b.codes = append(b.codes, c)
		b.codeIdx[code] = ci
	}
	o := (&tableSpec{}).
		u32(operatorOpcodeIndex, uint32(ci)).
		vecI32(operatorInputs, inputs).
		vecI32(operatorOutputs, outputs)
	if optType != 0 || opts != nil {
		o.union(operatorOptionsType, optType, opts)
	}
	sg := b.cur()
	sg.ops = append(sg.ops, o)
	return len(sg.ops) - 1
}

// io sets the current subgraph's inputs and outputs.
func (b *tflBuilder) io(inputs, outputs []int32) {
	sg := b.cur()
	sg.inputs, sg.outputs = inputs, outputs
}

// tensorTable, opTable (current subgraph), bufferTable and opcodeTable expose
// the specs for tests that need to corrupt or extend a model.
func (b *tflBuilder) tensorTable(i int) *tableSpec { return b.cur().tensors[i] }
func (b *tflBuilder) opTable(i int) *tableSpec     { return b.cur().ops[i] }
func (b *tflBuilder) bufferTable(i int) *tableSpec { return b.buffers[i] }

// opcodeTable returns the OperatorCode table for builtin code, or nil.
func (b *tflBuilder) opcodeTable(code int32) *tableSpec {
	if ci, ok := b.codeIdx[code]; ok {
		return b.codes[ci]
	}
	return nil
}

// modelSpec returns the Model root table (version 3).
func (b *tflBuilder) modelSpec() *tableSpec {
	sgs := make([]*tableSpec, len(b.sgs))
	for k, s := range b.sgs {
		sgs[k] = (&tableSpec{}).
			vecTables(subgraphTensors, s.tensors).
			vecI32(subgraphInputs, s.inputs).
			vecI32(subgraphOutputs, s.outputs).
			vecTables(subgraphOperators, s.ops)
	}
	return (&tableSpec{}).
		u32(modelVersion, 3).
		vecTables(modelOperatorCodes, b.codes).
		vecTables(modelSubgraphs, sgs).
		vecTables(modelBuffers, b.buffers)
}

// bytes serialises the model with file identifier "TFL3".
func (b *tflBuilder) bytes() []byte { return fbFinish(b.modelSpec(), "TFL3") }

// Option tables (BuiltinOptions union members).

func optsConv2D(padding uint8, strideW, strideH int32, act uint8, dilW, dilH int32) *tableSpec {
	return (&tableSpec{}).
		u8(convPadding, padding).
		i32(convStrideW, strideW).
		i32(convStrideH, strideH).
		u8(convAct, act).
		i32(convDilW, dilW).
		i32(convDilH, dilH)
}

func optsDepthwiseConv2D(padding uint8, strideW, strideH, depthMultiplier int32, act uint8, dilW, dilH int32) *tableSpec {
	return (&tableSpec{}).
		u8(dwPadding, padding).
		i32(dwStrideW, strideW).
		i32(dwStrideH, strideH).
		i32(dwMultiplier, depthMultiplier).
		u8(dwAct, act).
		i32(dwDilW, dilW).
		i32(dwDilH, dilH)
}

func optsPool2D(padding uint8, strideW, strideH, filterW, filterH int32, act uint8) *tableSpec {
	return (&tableSpec{}).
		u8(poolPadding, padding).
		i32(poolStrideW, strideW).
		i32(poolStrideH, strideH).
		i32(poolFilterW, filterW).
		i32(poolFilterH, filterH).
		u8(poolAct, act)
}

func optsFullyConnected(act uint8) *tableSpec { return (&tableSpec{}).u8(fcAct, act) }

func optsSoftmax(beta float32) *tableSpec { return (&tableSpec{}).f32(softmaxBeta, beta) }

func optsAdd(act uint8) *tableSpec { return (&tableSpec{}).u8(addAct, act) }

func optsReshape(newShape []int32) *tableSpec {
	return (&tableSpec{}).vecI32(tbReshapeNewShape, newShape)
}
