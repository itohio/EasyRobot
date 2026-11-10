package tfl

import (
	"errors"
	"strconv"
)

// Errors. ErrFormat: not a valid TFLite flatbuffer (or truncated);
// ErrUnsupported: valid but outside what the loader handles (an *OpError
// names the operator).
var (
	ErrFormat      = errors.New("tfl: malformed TFLite model")
	ErrUnsupported = errors.New("tfl: unsupported model")
)

// OpError reports an operator the loader cannot convert. It wraps
// ErrUnsupported.
type OpError struct {
	Index  int    // operator index in the subgraph
	Op     string // TFLite builtin name, or "BUILTIN_<code>" / "CUSTOM:<name>"
	Reason string
}

func (e *OpError) Error() string {
	return "tfl: operator " + strconv.Itoa(e.Index) + " " + e.Op + ": " + e.Reason
}

func (e *OpError) Unwrap() error { return ErrUnsupported }

// Model is a parsed view of a TFLite flatbuffer. It aliases the bytes it was
// opened from (weights are not copied).
type Model struct {
	f       fb
	version uint32
	codes   vec
	buffers vec
	sg      table
	tensors vec
	ops     vec
	inputs  vec
	outputs vec
}

// Open parses the model header and its first subgraph.
func Open(b []byte) (*Model, error) {
	if len(b) < 8 || string(b[4:8]) != "TFL3" {
		return nil, ErrFormat
	}
	m := &Model{f: fb{b: b}}
	f := &m.f
	root := f.root()
	m.version = f.fieldU32(root, modelVersion, 0)
	m.codes = f.fieldVec(root, modelOperatorCodes, 4)
	m.buffers = f.fieldVec(root, modelBuffers, 4)
	sgs := f.fieldVec(root, modelSubgraphs, 4)
	if f.err == nil && sgs.n < 1 {
		return nil, ErrFormat
	}
	if f.err == nil && sgs.n > 1 {
		return nil, ErrUnsupported // control flow (WHILE/IF) subgraphs
	}
	m.sg = f.vecTable(sgs, 0)
	m.tensors = f.fieldVec(m.sg, subgraphTensors, 4)
	m.ops = f.fieldVec(m.sg, subgraphOperators, 4)
	m.inputs = f.fieldVec(m.sg, subgraphInputs, 4)
	m.outputs = f.fieldVec(m.sg, subgraphOutputs, 4)
	if f.err != nil {
		return nil, f.err
	}
	return m, nil
}

// Version returns the schema version (3 for current models).
func (m *Model) Version() uint32 { return m.version }

// NumTensors and NumOperators count the first subgraph's tensors and ops.
func (m *Model) NumTensors() int   { return m.tensors.n }
func (m *Model) NumOperators() int { return m.ops.n }

// tensor is one tensor's metadata.
type tensor struct {
	shape  []int32 // aliases nothing: copied (rank ≤ 8)
	typ    uint8
	buffer int
	name   []byte
	scale  vec // float32
	zp     vec // int64
	qdim   int32
	sparse bool
}

func (m *Model) tensor(i int) (tensor, error) {
	f := &m.f
	if i < 0 || i >= m.tensors.n {
		return tensor{}, ErrFormat
	}
	t := f.vecTable(m.tensors, i)
	sv := f.fieldVec(t, tensorShape, 4)
	if sv.n > 8 {
		return tensor{}, ErrUnsupported
	}
	shape := make([]int32, sv.n)
	for k := range shape {
		shape[k] = f.vecI32(sv, k)
	}
	q := f.fieldTable(t, tensorQuantization)
	out := tensor{
		shape:  shape,
		typ:    f.fieldU8(t, tensorType, typeFloat32),
		buffer: int(f.fieldU32(t, tensorBuffer, 0)),
		name:   f.fieldString(t, tensorName),
		sparse: f.field(t, tensorSparsity) >= 0,
	}
	if q.ok {
		out.scale = f.fieldVec(q, quantScale, 4)
		out.zp = f.fieldVec(q, quantZeroPoint, 8)
		out.qdim = f.fieldI32(q, quantDimension, 0)
		if f.fieldU8(q, quantDetails, 0) != 0 {
			return tensor{}, ErrUnsupported // custom quantisation
		}
	}
	return out, f.err
}

// scaleAt and zpAt read quantisation entry k of t.
func (m *Model) scaleAt(t *tensor, k int) float32 { return m.f.vecF32(t.scale, k) }
func (m *Model) zpAt(t *tensor, k int) int64      { return m.f.vecI64(t.zp, k) }

// bufferData returns the constant data of buffer i (nil for none). Buffers
// stored outside the flatbuffer (offset/size, models > 2 GB) are rejected.
func (m *Model) bufferData(i int) ([]byte, error) {
	f := &m.f
	if i < 0 || i >= m.buffers.n {
		return nil, ErrFormat
	}
	b := f.vecTable(m.buffers, i)
	if f.field(b, bufferOffset) >= 0 && f.fieldU32(b, bufferOffset, 0) > 1 {
		return nil, ErrUnsupported
	}
	d := f.bytes(f.fieldVec(b, bufferData, 1))
	return d, f.err
}

// operator is one operator's references.
type operator struct {
	code    int32 // builtin code
	custom  []byte
	version int32
	inputs  vec // int32 tensor indices (−1 = optional absent)
	outputs vec
	optType uint8
	opts    table
}

func (m *Model) operator(i int) (operator, error) {
	f := &m.f
	if i < 0 || i >= m.ops.n {
		return operator{}, ErrFormat
	}
	o := f.vecTable(m.ops, i)
	ci := int(f.fieldU32(o, operatorOpcodeIndex, 0))
	if ci >= m.codes.n {
		return operator{}, ErrFormat
	}
	c := f.vecTable(m.codes, ci)
	code := int32(f.fieldU8(c, opcodeDeprecatedBuiltin, 0))
	if b := f.fieldI32(c, opcodeBuiltin, 0); b > code {
		code = b
	}
	return operator{
		code:    code,
		custom:  f.fieldString(c, opcodeCustom),
		version: f.fieldI32(c, opcodeVersion, 1),
		inputs:  f.fieldVec(o, operatorInputs, 4),
		outputs: f.fieldVec(o, operatorOutputs, 4),
		optType: f.fieldU8(o, operatorOptionsType, 0),
		opts:    f.fieldTable(o, operatorOptions),
	}, f.err
}

// name returns the operator's display name.
func (o *operator) name() string {
	if len(o.custom) > 0 {
		return "CUSTOM:" + string(o.custom)
	}
	if n := opName(o.code); n != "" {
		return n
	}
	return "BUILTIN_" + strconv.Itoa(int(o.code))
}
