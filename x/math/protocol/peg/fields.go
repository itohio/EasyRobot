package peg

type FieldType int

const (
	FieldUint8 FieldType = iota
	FieldInt8
	FieldUint16LE
	FieldUint16BE
	FieldInt16LE
	FieldInt16BE
	FieldUint32LE
	FieldUint32BE
	FieldInt32LE
	FieldInt32BE
	FieldUint64LE
	FieldUint64BE
	FieldInt64LE
	FieldInt64BE
	FieldFloat32
	FieldFloat64
	FieldVarintU
	FieldVarintS
	FieldStringPascal
	FieldStringC
	FieldStringFixed
)

type FieldKind int

const (
	FieldKindNormal FieldKind = iota
	FieldKindLength
)

type FieldSpec struct {
	Name  string
	Type  FieldType
	Kind  FieldKind
	Count int
	Size  int // used for fixed strings
}

func (f FieldSpec) TotalBytes() int {
	switch f.Type {
	case FieldUint8:
		return 1
	case FieldInt8:
		return 1
	case FieldUint16LE, FieldUint16BE, FieldInt16LE, FieldInt16BE:
		return 2
	case FieldUint32LE, FieldUint32BE, FieldInt32LE, FieldInt32BE, FieldFloat32:
		return 4
	case FieldUint64LE, FieldUint64BE, FieldInt64LE, FieldInt64BE, FieldFloat64:
		return 8
	case FieldVarintU, FieldVarintS:
		return 0 // Variable length (1-10 bytes), caller must handle
	case FieldStringFixed:
		if f.Size <= 0 {
			return 0
		}
		return f.Size
	default:
		return 0
	}
}
