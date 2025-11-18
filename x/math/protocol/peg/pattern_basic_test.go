package peg

import (
	"testing"

	"github.com/stretchr/testify/require"
)

// TestGroupAlternation tests groups with alternation (A|B)
func TestGroupAlternation(t *testing.T) {
	require := require.New(t)
	tests := []struct {
		name     string
		pattern  string
		data     []byte
		expected Decision
		fields   int
	}{
		{
			name:     "match first branch",
			pattern:  "^(55AA|BBCC)%u$",
			data:     []byte{0x55, 0xAA, 0x42},
			expected: DecisionEmit,
			fields:   1,
		},
		{
			name:     "match second branch",
			pattern:  "^(55AA|BBCC)%u$",
			data:     []byte{0xBB, 0xCC, 0x42},
			expected: DecisionEmit,
			fields:   1,
		},
		{
			name:     "no match",
			pattern:  "^(55AA|BBCC)%u$",
			data:     []byte{0x11, 0x22, 0x42},
			expected: DecisionDrop,
			fields:   0,
		},
		{
			name:     "three branches",
			pattern:  "^(AA|BB|CC)%u$",
			data:     []byte{0xBB, 0x42},
			expected: DecisionEmit,
			fields:   1,
		},
		{
			name:     "group with fields",
			pattern:  "^(55AA%u|BBCC%uu)$",
			data:     []byte{0x55, 0xAA, 0x42},
			expected: DecisionEmit,
			fields:   1,
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
			require.Equal(tt.expected, decision)
			fields := state.Fields()
			require.Len(fields, tt.fields)
		})
	}
}

// TestSkipNBytes tests skipping N bytes (*N)
func TestSkipNBytes(t *testing.T) {
	require := require.New(t)
	tests := []struct {
		name     string
		pattern  string
		data     []byte
		expected Decision
		fields   int
	}{
		{
			name:     "skip 3 bytes",
			pattern:  "^AA*3%u$",
			data:     []byte{0xAA, 0x11, 0x22, 0x33, 0x42},
			expected: DecisionEmit,
			fields:   1,
		},
		{
			name:     "skip 0 bytes",
			pattern:  "^AA*0%u$",
			data:     []byte{0xAA, 0x42},
			expected: DecisionEmit,
			fields:   1,
		},
		{
			name:     "skip with field before and after",
			pattern:  "^%u*5%u$",
			data:     []byte{0x01, 0x11, 0x22, 0x33, 0x44, 0x55, 0x02},
			expected: DecisionEmit,
			fields:   2,
		},
		{
			name:     "need more bytes for skip",
			pattern:  "^AA*5%u$",
			data:     []byte{0xAA, 0x11, 0x22},
			expected: DecisionContinue,
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
			require.Equal(tt.expected, decision)
			fields := state.Fields()
			require.Len(fields, tt.fields)
		})
	}
}
