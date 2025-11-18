package peg

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestMatchLiteralOp(t *testing.T) {
	tests := []struct {
		name     string
		value    []byte
		packet   []byte
		expected Decision
		complete bool
	}{
		{
			name:     "exact match",
			value:    []byte{0x55, 0xAA},
			packet:   []byte{0x55, 0xAA, 0x12},
			expected: DecisionContinue,
			complete: true,
		},
		{
			name:     "mismatch",
			value:    []byte{0x55, 0xAA},
			packet:   []byte{0x55, 0xBB, 0x12},
			expected: DecisionDrop,
			complete: true,
		},
		{
			name:     "need more bytes",
			value:    []byte{0x55, 0xAA},
			packet:   []byte{0x55},
			expected: DecisionContinue,
			complete: false,
		},
		{
			name:     "empty packet",
			value:    []byte{0x55},
			packet:   []byte{},
			expected: DecisionContinue,
			complete: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := NewMatchLiteralOp(tt.value)
			state := NewDefaultState()
			state.SetPacket(tt.packet)
			state.SetOffset(0)

			decision, complete := op(state, nil)
			require.Equal(t, tt.expected, decision)
			require.Equal(t, tt.complete, complete)
			require.Equal(t, tt.expected, state.Decision())
		})
	}
}

func TestMatchWildcardOp(t *testing.T) {
	tests := []struct {
		name     string
		count    int
		packet   []byte
		expected Decision
		complete bool
		offset   int
	}{
		{
			name:     "exact bytes",
			count:    3,
			packet:   []byte{0x01, 0x02, 0x03, 0x04},
			expected: DecisionContinue,
			complete: true,
			offset:   3,
		},
		{
			name:     "need more bytes",
			count:    3,
			packet:   []byte{0x01, 0x02},
			expected: DecisionContinue,
			complete: false,
			offset:   0,
		},
		{
			name:     "empty packet",
			count:    1,
			packet:   []byte{},
			expected: DecisionContinue,
			complete: false,
			offset:   0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := NewMatchWildcardOp(tt.count)
			state := NewDefaultState()
			state.SetPacket(tt.packet)
			state.SetOffset(0)

			decision, complete := op(state, nil)
			require.Equal(t, tt.expected, decision)
			require.Equal(t, tt.complete, complete)
			require.Equal(t, tt.offset, state.Offset())
		})
	}
}

func TestDecodeFieldOp(t *testing.T) {
	tests := []struct {
		name     string
		spec     FieldSpec
		packet   []byte
		offset   int
		expected Decision
		complete bool
		value    interface{}
	}{
		{
			name:     "uint8",
			spec:     FieldSpec{Type: FieldUint8, Name: "val"},
			packet:   []byte{0x42},
			offset:   0,
			expected: DecisionContinue,
			complete: true,
			value:    byte(0x42),
		},
		{
			name:     "uint16 LE",
			spec:     FieldSpec{Type: FieldUint16LE, Name: "val"},
			packet:   []byte{0x34, 0x12},
			offset:   0,
			expected: DecisionContinue,
			complete: true,
			value:    uint16(0x1234),
		},
		{
			name:     "uint16 BE",
			spec:     FieldSpec{Type: FieldUint16BE, Name: "val"},
			packet:   []byte{0x12, 0x34},
			offset:   0,
			expected: DecisionContinue,
			complete: true,
			value:    uint16(0x1234),
		},
		{
			name:     "need more bytes",
			spec:     FieldSpec{Type: FieldUint16LE, Name: "val"},
			packet:   []byte{0x34},
			offset:   0,
			expected: DecisionContinue,
			complete: false,
		},
		{
			name:     "length field",
			spec:     FieldSpec{Type: FieldUint8, Name: "len", Kind: FieldKindLength},
			packet:   []byte{0x05},
			offset:   0,
			expected: DecisionContinue,
			complete: true,
			value:    byte(0x05),
		},
		{
			name:     "pascal string",
			spec:     FieldSpec{Type: FieldStringPascal, Name: "str"},
			packet:   []byte{0x03, 'a', 'b', 'c'},
			offset:   0,
			expected: DecisionContinue,
			complete: true,
			value:    "abc",
		},
		{
			name:     "pascal string need more",
			spec:     FieldSpec{Type: FieldStringPascal, Name: "str"},
			packet:   []byte{0x03, 'a', 'b'},
			offset:   0,
			expected: DecisionContinue,
			complete: false,
		},
		{
			name:     "c string",
			spec:     FieldSpec{Type: FieldStringC, Name: "str"},
			packet:   []byte{'a', 'b', 'c', 0x00, 0x01},
			offset:   0,
			expected: DecisionContinue,
			complete: true,
			value:    "abc",
		},
		{
			name:     "c string need more",
			spec:     FieldSpec{Type: FieldStringC, Name: "str"},
			packet:   []byte{'a', 'b', 'c'},
			offset:   0,
			expected: DecisionContinue,
			complete: false,
		},
		{
			name:     "fixed string",
			spec:     FieldSpec{Type: FieldStringFixed, Name: "str", Size: 5},
			packet:   []byte{'a', 'b', 0x00, 0x00, 0x00},
			offset:   0,
			expected: DecisionContinue,
			complete: true,
			value:    "ab",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := NewDecodeFieldOp(tt.spec)
			state := NewDefaultState()
			state.SetPacket(tt.packet)
			state.SetOffset(tt.offset)

			decision, complete := op(state, nil)
			require.Equal(t, tt.expected, decision)
			require.Equal(t, tt.complete, complete)

			if tt.value != nil {
				fields := state.Fields()
				require.Len(t, fields, 1)
				require.Equal(t, tt.value, fields[0].Value)
			}

			if tt.spec.Kind == FieldKindLength {
				require.Equal(t, int(tt.value.(byte)), state.DeclaredLength())
			}
		})
	}
}

