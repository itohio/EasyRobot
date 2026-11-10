package tfl

import (
	"encoding/binary"
	"math"
)

// A minimal, bounds-checked FlatBuffers reader over a byte slice: tables,
// vtables, scalars, vectors and strings. Every access is checked against the
// buffer; a malformed file yields ErrFormat, never a panic or an allocation.
// Little-endian, as FlatBuffers specifies.

// fb is the whole file.
type fb struct {
	b   []byte
	err error // first error; reads after an error return zero values
}

// table is a FlatBuffers table: its position and its vtable.
type table struct {
	pos   int // table start (soffset to the vtable lives here)
	vt    int // vtable start
	vtLen int // vtable length in bytes
	ok    bool
}

func (f *fb) fail() {
	if f.err == nil {
		f.err = ErrFormat
	}
}

func (f *fb) u32(off int) uint32 {
	if off < 0 || off+4 > len(f.b) {
		f.fail()
		return 0
	}
	return binary.LittleEndian.Uint32(f.b[off:])
}

func (f *fb) u16(off int) uint16 {
	if off < 0 || off+2 > len(f.b) {
		f.fail()
		return 0
	}
	return binary.LittleEndian.Uint16(f.b[off:])
}

// tableAt returns the table at absolute position pos.
func (f *fb) tableAt(pos int) table {
	if f.err != nil || pos < 0 || pos+4 > len(f.b) {
		f.fail()
		return table{}
	}
	vt := pos - int(int32(f.u32(pos)))
	if vt < 0 || vt+4 > len(f.b) {
		f.fail()
		return table{}
	}
	vtLen := int(f.u16(vt))
	if vtLen < 4 || vtLen&1 != 0 || vt+vtLen > len(f.b) {
		f.fail()
		return table{}
	}
	return table{pos: pos, vt: vt, vtLen: vtLen, ok: true}
}

// deref follows the uoffset stored at off.
func (f *fb) deref(off int) int {
	v := f.u32(off)
	if f.err != nil || v > uint32(len(f.b)) {
		f.fail()
		return -1
	}
	return off + int(v)
}

// root returns the root table.
func (f *fb) root() table { return f.tableAt(f.deref(0)) }

// field returns the absolute position of field i of t, or -1 when absent.
func (f *fb) field(t table, i int) int {
	if !t.ok {
		return -1
	}
	slot := 4 + 2*i
	if slot+2 > t.vtLen {
		return -1
	}
	o := int(f.u16(t.vt + slot))
	if o == 0 {
		return -1
	}
	return t.pos + o
}

func (f *fb) fieldU8(t table, i int, def uint8) uint8 {
	p := f.field(t, i)
	if p < 0 {
		return def
	}
	if p >= len(f.b) {
		f.fail()
		return def
	}
	return f.b[p]
}

func (f *fb) fieldBool(t table, i int) bool { return f.fieldU8(t, i, 0) != 0 }

func (f *fb) fieldI32(t table, i int, def int32) int32 {
	p := f.field(t, i)
	if p < 0 {
		return def
	}
	return int32(f.u32(p))
}

func (f *fb) fieldU32(t table, i int, def uint32) uint32 {
	p := f.field(t, i)
	if p < 0 {
		return def
	}
	return f.u32(p)
}

func (f *fb) fieldF32(t table, i int, def float32) float32 {
	p := f.field(t, i)
	if p < 0 {
		return def
	}
	return math.Float32frombits(f.u32(p))
}

// fieldTable returns the sub-table in field i (ok false when absent).
func (f *fb) fieldTable(t table, i int) table {
	p := f.field(t, i)
	if p < 0 {
		return table{}
	}
	return f.tableAt(f.deref(p))
}

// vec is a vector: n elements of size elem starting at pos.
type vec struct {
	pos, n, elem int
}

// fieldVec returns the vector in field i with element size elem (n == 0
// when absent).
func (f *fb) fieldVec(t table, i, elem int) vec {
	p := f.field(t, i)
	if p < 0 {
		return vec{}
	}
	start := f.deref(p)
	if f.err != nil {
		return vec{}
	}
	n := int(f.u32(start))
	data := start + 4
	if f.err != nil || n < 0 || n > (len(f.b)-data)/elem {
		f.fail()
		return vec{}
	}
	return vec{pos: data, n: n, elem: elem}
}

// bytes returns the vector's raw bytes (aliasing the file).
func (f *fb) bytes(v vec) []byte {
	if v.n == 0 {
		return nil
	}
	return f.b[v.pos : v.pos+v.n*v.elem]
}

func (f *fb) vecI32(v vec, k int) int32 {
	if k < 0 || k >= v.n {
		f.fail()
		return 0
	}
	return int32(f.u32(v.pos + 4*k))
}

func (f *fb) vecI64(v vec, k int) int64 {
	if k < 0 || k >= v.n {
		f.fail()
		return 0
	}
	lo := uint64(f.u32(v.pos + 8*k))
	hi := uint64(f.u32(v.pos + 8*k + 4))
	return int64(hi<<32 | lo)
}

func (f *fb) vecF32(v vec, k int) float32 {
	if k < 0 || k >= v.n {
		f.fail()
		return 0
	}
	return math.Float32frombits(f.u32(v.pos + 4*k))
}

// vecTable returns table element k of a vector of tables.
func (f *fb) vecTable(v vec, k int) table {
	if k < 0 || k >= v.n {
		f.fail()
		return table{}
	}
	return f.tableAt(f.deref(v.pos + 4*k))
}

// fieldString returns the string in field i (aliasing the file).
func (f *fb) fieldString(t table, i int) []byte {
	return f.bytes(f.fieldVec(t, i, 1))
}
