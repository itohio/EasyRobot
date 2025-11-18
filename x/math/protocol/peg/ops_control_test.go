package peg

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSetMaxLengthOp(t *testing.T) {
	op := NewSetMaxLengthOp(100)
	state := NewDefaultState()

	decision, complete := op(state, nil)
	require.Equal(t, DecisionContinue, decision)
	require.True(t, complete)
	require.Equal(t, 100, state.MaxLength())
}

func TestCheckLengthOp(t *testing.T) {
	tests := []struct {
		name        string
		declaredLen int
		currentLen  int
		expected    Decision
		complete    bool
	}{
		{
			name:        "exact match",
			declaredLen: 5,
			currentLen:  5,
			expected:    DecisionEmit,
			complete:    true,
		},
		{
			name:        "need more",
			declaredLen: 5,
			currentLen:  3,
			expected:    DecisionContinue,
			complete:    false,
		},
		{
			name:        "too long",
			declaredLen: 5,
			currentLen:  6,
			expected:    DecisionDrop,
			complete:    true,
		},
		{
			name:        "no declared length",
			declaredLen: 0,
			currentLen:  3,
			expected:    DecisionContinue,
			complete:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := NewCheckLengthOp()
			state := NewDefaultState()
			state.SetDeclaredLength(tt.declaredLen)
			state.SetPacket(make([]byte, tt.currentLen))

			decision, complete := op(state, nil)
			require.Equal(t, tt.expected, decision)
			require.Equal(t, tt.complete, complete)
		})
	}
}

func TestCheckMaxLengthOp(t *testing.T) {
	tests := []struct {
		name       string
		maxLen     int
		currentLen int
		expected   Decision
		complete   bool
	}{
		{
			name:       "within limit",
			maxLen:     100,
			currentLen: 50,
			expected:   DecisionContinue,
			complete:   true,
		},
		{
			name:       "exceeds limit",
			maxLen:     100,
			currentLen: 101,
			expected:   DecisionDrop,
			complete:   true,
		},
		{
			name:       "no max length",
			maxLen:     0,
			currentLen: 1000,
			expected:   DecisionContinue,
			complete:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op := NewCheckMaxLengthOp()
			state := NewDefaultState()
			state.SetMaxLength(tt.maxLen)
			state.SetPacket(make([]byte, tt.currentLen))

			decision, complete := op(state, nil)
			require.Equal(t, tt.expected, decision)
			require.Equal(t, tt.complete, complete)
		})
	}
}

// TestSequenceOp tests sequence operation
func TestSequenceOp(t *testing.T) {
	require := require.New(t)
	op := NewSequenceOp()

	t.Run("all continue", func(t *testing.T) {
		state := NewDefaultState()
		childOutputs := map[int64]Decision{
			1: DecisionContinue,
			2: DecisionContinue,
		}
		decision, complete := op(state, childOutputs)
		require.Equal(DecisionContinue, decision)
		require.False(complete)
	})

	t.Run("one drop", func(t *testing.T) {
		state := NewDefaultState()
		childOutputs := map[int64]Decision{
			1: DecisionContinue,
			2: DecisionDrop,
		}
		decision, complete := op(state, childOutputs)
		require.Equal(DecisionDrop, decision)
		require.True(complete)
	})

	t.Run("all emit", func(t *testing.T) {
		state := NewDefaultState()
		childOutputs := map[int64]Decision{
			1: DecisionEmit,
			2: DecisionEmit,
		}
		decision, complete := op(state, childOutputs)
		require.Equal(DecisionEmit, decision)
		require.True(complete)
	})

	t.Run("mixed continue and emit", func(t *testing.T) {
		state := NewDefaultState()
		childOutputs := map[int64]Decision{
			1: DecisionContinue,
			2: DecisionEmit,
		}
		decision, complete := op(state, childOutputs)
		require.Equal(DecisionContinue, decision)
		require.False(complete)
	})
}

// TestChoiceOp tests choice operation
func TestChoiceOp(t *testing.T) {
	require := require.New(t)
	op := NewChoiceOp()

	t.Run("one continue", func(t *testing.T) {
		state := NewDefaultState()
		childOutputs := map[int64]Decision{
			1: DecisionDrop,
			2: DecisionContinue,
		}
		decision, complete := op(state, childOutputs)
		require.Equal(DecisionContinue, decision)
		require.True(complete)
	})

	t.Run("one emit", func(t *testing.T) {
		state := NewDefaultState()
		childOutputs := map[int64]Decision{
			1: DecisionDrop,
			2: DecisionEmit,
		}
		decision, complete := op(state, childOutputs)
		require.Equal(DecisionEmit, decision)
		require.True(complete)
	})

	t.Run("all drop", func(t *testing.T) {
		state := NewDefaultState()
		childOutputs := map[int64]Decision{
			1: DecisionDrop,
			2: DecisionDrop,
		}
		decision, complete := op(state, childOutputs)
		require.Equal(DecisionDrop, decision)
		require.True(complete)
	})
}

