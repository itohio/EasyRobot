package peg

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDefaultStateClone(t *testing.T) {
	require := require.New(t)
	s := NewDefaultState()
	s.AddField(DecodedField{Name: "a"})
	clone := s.Clone()
	s.AddField(DecodedField{Name: "b"})
	require.Len(clone.Fields(), 1)
}

func TestStateEdgeCases(t *testing.T) {
	require := require.New(t)
	t.Run("SetPacket with capacity reuse", func(t *testing.T) {
		s := NewDefaultState()
		s.SetPacket([]byte{0x01, 0x02, 0x03})
		require.Len(s.Packet(), 3)
		s.SetPacket([]byte{0x04, 0x05})
		require.Len(s.Packet(), 2)
		require.Equal(byte(0x04), s.Packet()[0])
	})

	t.Run("SetDeclaredLength with maxLength clamping", func(t *testing.T) {
		s := NewDefaultState()
		s.SetMaxLength(10)
		s.SetDeclaredLength(15)
		require.Equal(10, s.DeclaredLength())
	})

	t.Run("SetMaxLength with declaredLength clamping", func(t *testing.T) {
		s := NewDefaultState()
		s.SetDeclaredLength(15)
		s.SetMaxLength(10)
		require.Equal(10, s.DeclaredLength())
		require.Equal(10, s.MaxLength())
	})

	t.Run("SetDeclaredLength zero or negative", func(t *testing.T) {
		s := NewDefaultState()
		s.SetDeclaredLength(0)
		require.Equal(0, s.DeclaredLength())
		s.SetDeclaredLength(-5)
		require.Equal(0, s.DeclaredLength())
	})

	t.Run("SetMaxLength zero or negative", func(t *testing.T) {
		s := NewDefaultState()
		s.SetMaxLength(0)
		require.Equal(0, s.MaxLength())
		s.SetMaxLength(-5)
		require.Equal(0, s.MaxLength())
	})

	t.Run("ResetFields", func(t *testing.T) {
		s := NewDefaultState()
		s.AddField(DecodedField{Name: "a"})
		s.AddField(DecodedField{Name: "b"})
		require.Len(s.Fields(), 2)
		s.ResetFields()
		require.Len(s.Fields(), 0)
	})

	t.Run("Merge preserves offset", func(t *testing.T) {
		s1 := NewDefaultState()
		s1.SetOffset(5)
		s2 := NewDefaultState()
		s2.Merge(s1)
		require.Equal(5, s2.Offset())
	})
}

func TestStateMergeNonDefaultState(t *testing.T) {
	require := require.New(t)
	s1 := NewDefaultState()
	s1.SetOffset(5)
	s1.AddField(DecodedField{Name: "test"})

	// Create a mock state that embeds DefaultState (so type assertion succeeds)
	type mockState struct {
		*DefaultState
	}
	s2 := &mockState{DefaultState: NewDefaultState()}
	s2.Merge(s1)
	// Since mockState embeds DefaultState, the type assertion succeeds and merge works
	require.Len(s2.Fields(), 1)
	require.Equal(5, s2.Offset())
}

