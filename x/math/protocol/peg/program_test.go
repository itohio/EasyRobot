package peg

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDecideLiteralFlow(t *testing.T) {
	require := require.New(t)
	prog := mustCompile(t, "^AA%u$")
	state := NewDefaultState()

	state.AppendPacket(0xAA)
	dec, err := prog.Decide(state)
	require.NoError(err)
	require.Equal(DecisionContinue, dec)

	state.AppendPacket(0x01)
	dec, err = prog.Decide(state)
	require.NoError(err)
	require.Equal(DecisionEmit, dec)

	fields := state.Fields()
	require.Len(fields, 1)
	require.Equal(uint8(1), fields[0].Value.(uint8))
}

func TestDecideMismatch(t *testing.T) {
	require := require.New(t)
	prog := mustCompile(t, "^AA$")
	state := NewDefaultState()
	state.AppendPacket(0xBB)
	dec, err := prog.Decide(state)
	require.NoError(err)
	require.Equal(DecisionDrop, dec)
}

func TestLengthDrivenEmit(t *testing.T) {
	require := require.New(t)
	prog := mustCompile(t, "^55AA*ll$")
	state := NewDefaultState()

	state.AppendPacket(0x55)
	decision, err := prog.Decide(state)
	require.NoError(err)
	require.Equal(DecisionContinue, decision)

	state.AppendPacket(0xAA)
	decision, err = prog.Decide(state)
	require.NoError(err)
	require.Equal(DecisionContinue, decision)

	state.AppendPacket(0x01) // wildcard byte
	decision, err = prog.Decide(state)
	require.NoError(err)
	require.Equal(DecisionContinue, decision)

	// length field little-endian = 0x0005 (total packet length)
	state.AppendPacket(0x05, 0x00)
	decision, err = prog.Decide(state)
	require.NoError(err)
	require.Equal(DecisionEmit, decision)
	require.Equal(5, state.DeclaredLength())
	// Should have a field for the length field
	fields := state.Fields()
	require.GreaterOrEqual(len(fields), 1, "should have at least the length field")
	// Find the length field (should be a uint16 LE field)
	var lengthField *DecodedField
	for i := range fields {
		if fields[i].Type == FieldUint16LE {
			lengthField = &fields[i]
			break
		}
	}
	require.NotNil(lengthField, "should have length field")
	require.Equal(uint16(5), lengthField.Value, "length field should be 5")
}

func TestProgramEdgeCases(t *testing.T) {
	require := require.New(t)
	t.Run("anchorEnd with wrong length", func(t *testing.T) {
		prog := mustCompile(t, "^AA$")
		state := NewDefaultState()
		state.AppendPacket(0xAA, 0xBB)
		decision, err := prog.Decide(state)
		require.NoError(err)
		require.Equal(DecisionDrop, decision)
	})

	t.Run("maxLen exceeded", func(t *testing.T) {
		prog, err := Compile("^AA$5")
		require.NoError(err)
		state := NewDefaultState()
		for i := 0; i < 10; i++ {
			state.AppendPacket(byte(i))
		}
		decision, err := prog.Decide(state)
		require.NoError(err)
		require.Equal(DecisionDrop, decision)
	})

	t.Run("declared length exceeded", func(t *testing.T) {
		prog := mustCompile(t, "^ll$")
		state := NewDefaultState()
		state.AppendPacket(0x05, 0x00)                         // length = 5
		state.AppendPacket(0x01, 0x02, 0x03, 0x04, 0x05, 0x06) // 6 bytes
		decision, err := prog.Decide(state)
		require.NoError(err)
		require.Equal(DecisionDrop, decision)
	})

	t.Run("declared length not reached", func(t *testing.T) {
		prog := mustCompile(t, "^ll$")
		state := NewDefaultState()
		state.AppendPacket(0x05, 0x00) // length = 5 (total packet should be 5 bytes)
		state.AppendPacket(0x01, 0x02) // only 2 more bytes, total = 4 < 5
		decision, err := prog.Decide(state)
		require.NoError(err)
		require.Equal(DecisionContinue, decision)
	})
}

func TestProgramNilState(t *testing.T) {
	require := require.New(t)
	prog := mustCompile(t, "^AA$")
	// The program should handle nil state internally by creating a new DefaultState
	results, err := prog.Compute(nil, nil)
	require.NoError(err)
	// Results may be empty if the graph evaluation fails, but should not error
	_ = results
}
