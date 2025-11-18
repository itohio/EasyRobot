package peg

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNamedFields(t *testing.T) {
	prog := mustCompile(t, "^%temperature:uu%voltage:uu$")
	state := NewDefaultState()
	state.AppendPacket(0x34, 0x12, 0x78, 0x56)

	decision, err := prog.Decide(state)
	require.NoError(t, err)
	require.Equal(t, DecisionEmit, decision)

	fields := state.Fields()
	require.Len(t, fields, 2)
	require.Equal(t, "temperature", fields[0].Name)
	require.Equal(t, uint16(0x1234), fields[0].Value)
	require.Equal(t, "voltage", fields[1].Name)
	require.Equal(t, uint16(0x5678), fields[1].Value)
}

func TestSeparatorLiteral(t *testing.T) {
	prog := mustCompile(t, "^AA#2;BB$")
	state := NewDefaultState()
	state.AppendPacket(0xAA, 0x00, 0xBB)

	decision, err := prog.Decide(state)
	require.NoError(t, err)
	require.Equal(t, DecisionEmit, decision)
}

func TestExpressionScaling(t *testing.T) {
	prog := mustCompile(t, "^%(uu/360.0)$")
	state := NewDefaultState()
	// 0x0168 == 360
	state.AppendPacket(0x68, 0x01)

	decision, err := prog.Decide(state)
	require.NoError(t, err)
	require.Equal(t, DecisionEmit, decision)

	fields := state.Fields()
	require.Len(t, fields, 1)
	require.InDelta(t, 1.0, fields[0].Value.(float64), 1e-6)
}

func TestExpressionRangeCondition(t *testing.T) {
	pattern := "^%(50<uu && uu<1500)$"
	prog := mustCompile(t, pattern)

	t.Run("within range", func(t *testing.T) {
		state := NewDefaultState()
		state.AppendPacket(0x64, 0x00) // 100
		decision, err := prog.Decide(state)
		require.NoError(t, err)
		require.Equal(t, DecisionEmit, decision)
	})

	t.Run("outside range", func(t *testing.T) {
		state := NewDefaultState()
		state.AppendPacket(0x0A, 0x00) // 10
		decision, err := prog.Decide(state)
		require.NoError(t, err)
		require.Equal(t, DecisionDrop, decision)
	})
}

func TestConditionalGuard(t *testing.T) {
	prog := mustCompile(t, "^%start:uu%end:uu@(end>start)$")

	t.Run("condition satisfied", func(t *testing.T) {
		state := NewDefaultState()
		state.AppendPacket(0x01, 0x00, 0x05, 0x00)
		decision, err := prog.Decide(state)
		require.NoError(t, err)
		require.Equal(t, DecisionEmit, decision)
	})

	t.Run("condition violated", func(t *testing.T) {
		state := NewDefaultState()
		state.AppendPacket(0x05, 0x00, 0x01, 0x00)
		decision, err := prog.Decide(state)
		require.NoError(t, err)
		require.Equal(t, DecisionDrop, decision)
	})
}

func TestArrayWithStride(t *testing.T) {
	prog := mustCompile(t, "^%3:2u$")
	state := NewDefaultState()
	state.AppendPacket(0x11, 0x00, 0x22, 0x00, 0x33, 0x00)

	decision, err := prog.Decide(state)
	require.NoError(t, err)
	require.Equal(t, DecisionEmit, decision)

	fields := state.Fields()
	require.Len(t, fields, 3)
	require.Equal(t, uint8(0x11), fields[0].Value)
	require.Equal(t, uint8(0x22), fields[1].Value)
	require.Equal(t, uint8(0x33), fields[2].Value)
}

func TestStarArrayNamed(t *testing.T) {
	prog := mustCompile(t, "^%*data:uu$")
	state := NewDefaultState()
	state.AppendPacket(0x34, 0x12, 0x78, 0x56)

	decision, err := prog.Decide(state)
	require.NoError(t, err)
	require.Equal(t, DecisionEmit, decision)

	fields := state.Fields()
	require.Len(t, fields, 2)
	require.Equal(t, "data_0", fields[0].Name)
	require.Equal(t, uint16(0x1234), fields[0].Value)
	require.Equal(t, "data_1", fields[1].Name)
	require.Equal(t, uint16(0x5678), fields[1].Value)
}

func TestStarArrayOfStructs(t *testing.T) {
	prog := mustCompile(t, "^%*{uu,u}$")
	state := NewDefaultState()
	state.AppendPacket(
		0x34, 0x12, 0x01,
		0x78, 0x56, 0x02,
	)

	decision, err := prog.Decide(state)
	require.NoError(t, err)
	require.Equal(t, DecisionEmit, decision)

	fields := state.Fields()
	require.Len(t, fields, 4)
	require.Equal(t, uint16(0x1234), fields[0].Value)
	require.Equal(t, uint8(0x01), fields[1].Value)
	require.Equal(t, uint16(0x5678), fields[2].Value)
	require.Equal(t, uint8(0x02), fields[3].Value)
}
