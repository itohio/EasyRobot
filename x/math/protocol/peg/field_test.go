package peg

import (
	"encoding/hex"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestInt64Fields(t *testing.T) {
	require := require.New(t)
	tests := []struct {
		name     string
		pattern  string
		data     []byte
		expected interface{}
	}{
		{
			name:     "int64 little-endian",
			pattern:  "^%iiiiii$",
			data:     []byte{0x78, 0x56, 0x34, 0x12, 0x00, 0x00, 0x00, 0x00},
			expected: int64(0x12345678),
		},
		{
			name:     "int64 big-endian",
			pattern:  "^%IIIIII$",
			data:     []byte{0x00, 0x00, 0x00, 0x00, 0x12, 0x34, 0x56, 0x78},
			expected: int64(0x12345678),
		},
		{
			name:     "uint64 little-endian",
			pattern:  "^%uuuuuu$",
			data:     []byte{0x78, 0x56, 0x34, 0x12, 0xEF, 0xCD, 0xAB, 0x90},
			expected: uint64(0x90ABCDEF12345678),
		},
		{
			name:     "uint64 big-endian",
			pattern:  "^%UUUUUU$",
			data:     []byte{0x90, 0xAB, 0xCD, 0xEF, 0x12, 0x34, 0x56, 0x78},
			expected: uint64(0x90ABCDEF12345678),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prog := mustCompile(t, tt.pattern)
			state := NewDefaultState()
			for _, b := range tt.data {
				state.AppendPacket(b)
			}
			decision, err := prog.Decide(state)
			require.NoError(err)
			require.Equal(DecisionEmit, decision)
			fields := state.Fields()
			require.Len(fields, 1)
			require.Equal(tt.expected, fields[0].Value)
		})
	}
}

func TestVarintFields(t *testing.T) {
	require := require.New(t)
	tests := []struct {
		name     string
		pattern  string
		data     []byte
		expected interface{}
	}{
		{
			name:     "unsigned varint small",
			pattern:  "^%v$",
			data:     []byte{0x05},
			expected: uint64(5),
		},
		{
			name:     "unsigned varint medium",
			pattern:  "^%v$",
			data:     []byte{0xAC, 0x02},
			expected: uint64(300),
		},
		{
			name:     "unsigned varint large",
			pattern:  "^%v$",
			data:     []byte{0xFF, 0xFF, 0xFF, 0xFF, 0x0F},
			expected: uint64(0xFFFFFFFF),
		},
		{
			name:     "signed varint positive",
			pattern:  "^%V$",
			data:     []byte{0x0A}, // zigzag(5) = 10
			expected: int64(5),
		},
		{
			name:     "signed varint negative",
			pattern:  "^%V$",
			data:     []byte{0x01}, // zigzag(-1) = 1
			expected: int64(-1),
		},
		{
			name:     "signed varint zero",
			pattern:  "^%V$",
			data:     []byte{0x00},
			expected: int64(0),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			prog := mustCompile(t, tt.pattern)
			state := NewDefaultState()
			for _, b := range tt.data {
				state.AppendPacket(b)
			}
			decision, err := prog.Decide(state)
			require.NoError(err)
			require.Equal(DecisionEmit, decision)
			fields := state.Fields()
			require.Len(fields, 1)
			require.Equal(tt.expected, fields[0].Value)
		})
	}
}

func TestVarintIncremental(t *testing.T) {
	require := require.New(t)
	prog := mustCompile(t, "^%v$")
	state := NewDefaultState()

	// Multi-byte varint, feed byte by byte
	state.AppendPacket(0xAC)
	decision, err := prog.Decide(state)
	require.NoError(err)
	require.Equal(DecisionContinue, decision) // Need more bytes

	state.AppendPacket(0x02)
	decision, err = prog.Decide(state)
	require.NoError(err)
	require.Equal(DecisionEmit, decision)
	fields := state.Fields()
	require.Len(fields, 1)
	require.Equal(uint64(300), fields[0].Value)
}

func TestVarintEdgeCases(t *testing.T) {
	require := require.New(t)
	t.Run("incomplete varint with continuation bit", func(t *testing.T) {
		op := NewDecodeFieldOp(FieldSpec{Type: FieldVarintU})
		state := NewDefaultState()
		state.SetPacket([]byte{0x80}) // continuation bit set, but only 1 byte
		state.SetOffset(0)
		decision, complete := op(state, nil)
		require.Equal(DecisionContinue, decision)
		require.False(complete)
	})

	t.Run("varint with many continuation bits", func(t *testing.T) {
		op := NewDecodeFieldOp(FieldSpec{Type: FieldVarintU})
		state := NewDefaultState()
		// Create a varint with continuation bits that would cause shift >= 64
		// After 9 bytes with continuation, shift would be 63, 10th byte would make it 70
		data := make([]byte, 10)
		for i := 0; i < 9; i++ {
			data[i] = 0x80 // continuation bit
		}
		data[9] = 0x80 // 10th byte also has continuation (invalid)
		state.SetPacket(data)
		state.SetOffset(0)
		decision, complete := op(state, nil)
		// Should return need more bytes since we check maxLen < 10 first
		require.Equal(DecisionContinue, decision)
		require.False(complete)
	})
}

func TestPascalStringField(t *testing.T) {
	require := require.New(t)
	prog := mustCompile(t, "^AA%s$")
	state := NewDefaultState()
	payload, _ := hex.DecodeString("AA03626172")
	for _, b := range payload {
		state.AppendPacket(b)
	}
	decision, err := prog.Decide(state)
	require.NoError(err)
	require.Equal(DecisionEmit, decision)
	fields := state.Fields()
	require.Len(fields, 1)
	require.Equal("bar", fields[0].Value.(string))
}

func TestDecodeFieldErrors(t *testing.T) {
	require := require.New(t)
	_, err := decodeField([]byte{0x01}, FieldSpec{Type: FieldUint16LE})
	require.Error(err)
}

func TestDecodeFieldAllTypes(t *testing.T) {
	require := require.New(t)
	tests := []struct {
		name     string
		spec     FieldSpec
		data     []byte
		expected interface{}
	}{
		{"int8", FieldSpec{Type: FieldInt8}, []byte{0x80}, int8(-128)},
		{"int16 LE", FieldSpec{Type: FieldInt16LE}, []byte{0x34, 0x12}, int16(0x1234)},
		{"int16 BE", FieldSpec{Type: FieldInt16BE}, []byte{0x12, 0x34}, int16(0x1234)},
		{"int32 LE", FieldSpec{Type: FieldInt32LE}, []byte{0x78, 0x56, 0x34, 0x12}, int32(0x12345678)},
		{"int32 BE", FieldSpec{Type: FieldInt32BE}, []byte{0x12, 0x34, 0x56, 0x78}, int32(0x12345678)},
		{"uint32 LE", FieldSpec{Type: FieldUint32LE}, []byte{0x78, 0x56, 0x34, 0x12}, uint32(0x12345678)},
		{"uint32 BE", FieldSpec{Type: FieldUint32BE}, []byte{0x12, 0x34, 0x56, 0x78}, uint32(0x12345678)},
		{"float32", FieldSpec{Type: FieldFloat32}, []byte{0x00, 0x00, 0x80, 0x3F}, float32(1.0)},
		{"float64", FieldSpec{Type: FieldFloat64}, []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F}, float64(1.0)},
		{"length uint32", FieldSpec{Type: FieldUint32LE, Kind: FieldKindLength}, []byte{0x05, 0x00, 0x00, 0x00}, uint32(5)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := NewDecodeFieldOp(tt.spec)
			state := NewDefaultState()
			state.SetPacket(tt.data)
			state.SetOffset(0)

			decision, complete := op(state, nil)
			require.Equal(DecisionContinue, decision)
			require.True(complete)

			fields := state.Fields()
			require.Len(fields, 1)
			require.Equal(tt.expected, fields[0].Value)
		})
	}
}

func TestDecodeFieldErrorCases(t *testing.T) {
	require := require.New(t)
	t.Run("decodeField error - need more bytes", func(t *testing.T) {
		op := NewDecodeFieldOp(FieldSpec{Type: FieldUint16LE})
		state := NewDefaultState()
		state.SetPacket([]byte{0x01}) // only 1 byte, need 2
		state.SetOffset(0)
		decision, complete := op(state, nil)
		require.Equal(DecisionContinue, decision)
		require.False(complete)
	})
}

func TestDecodeFieldLengthFieldTypes(t *testing.T) {
	require := require.New(t)
	tests := []struct {
		name     string
		spec     FieldSpec
		data     []byte
		expected int
	}{
		{"uint8 length", FieldSpec{Type: FieldUint8, Kind: FieldKindLength}, []byte{0x05}, 5},
		{"uint16 LE length", FieldSpec{Type: FieldUint16LE, Kind: FieldKindLength}, []byte{0x05, 0x00}, 5},
		{"uint32 LE length", FieldSpec{Type: FieldUint32LE, Kind: FieldKindLength}, []byte{0x05, 0x00, 0x00, 0x00}, 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := NewDecodeFieldOp(tt.spec)
			state := NewDefaultState()
			state.SetPacket(tt.data)
			state.SetOffset(0)
			decision, complete := op(state, nil)
			require.Equal(DecisionContinue, decision)
			require.True(complete)
			require.Equal(tt.expected, state.DeclaredLength())
		})
	}
}


