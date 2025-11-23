package generics

import (
	"io"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuffer_WriteRead(t *testing.T) {
	tests := []struct {
		name     string
		write    []int
		read     int
		wantRead []int
		wantErr  error
	}{
		{
			name:     "simple write and read",
			write:    []int{1, 2, 3},
			read:     3,
			wantRead: []int{1, 2, 3},
		},
		{
			name:     "read more than available",
			write:    []int{1, 2, 3},
			read:     5,
			wantRead: []int{1, 2, 3},
			wantErr:  io.EOF,
		},
		{
			name:     "multiple writes",
			write:    []int{1, 2, 3},
			read:     2,
			wantRead: []int{1, 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backing := make([]int, 0, 10)
			buf := NewBuffer(backing)

			n, err := buf.Write(tt.write)
			require.NoError(t, err)
			assert.Equal(t, len(tt.write), n)
			assert.Equal(t, len(tt.write), buf.Len())

			readBuf := make([]int, tt.read)
			n, err = buf.Read(readBuf)
			if tt.wantErr != nil {
				assert.ErrorIs(t, err, tt.wantErr)
			} else {
				require.NoError(t, err)
			}
			assert.Equal(t, len(tt.wantRead), n)
			assert.Equal(t, tt.wantRead, readBuf[:n])
		})
	}
}

func TestBuffer_WriteByteReadByte(t *testing.T) {
	backing := make([]byte, 0, 10)
	buf := NewBuffer(backing)

	// Write bytes
	for i := byte(0); i < 5; i++ {
		err := buf.WriteItem(i)
		require.NoError(t, err)
	}
	assert.Equal(t, 5, buf.Len())

	// Read bytes
	for i := byte(0); i < 5; i++ {
		b, err := buf.ReadItem()
		require.NoError(t, err)
		assert.Equal(t, i, b)
	}

	// Read from empty buffer
	_, err := buf.ReadItem()
	assert.ErrorIs(t, err, io.EOF)
}

func TestBuffer_Bytes(t *testing.T) {
	backing := make([]int, 0, 10)
	buf := NewBuffer(backing)

	data := []int{1, 2, 3, 4, 5}
	_, err := buf.Write(data)
	require.NoError(t, err)

	bytes := buf.Items()
	assert.Equal(t, data, bytes)
	assert.Equal(t, 5, len(bytes))

	// Read some
	readBuf := make([]int, 2)
	_, err = buf.Read(readBuf)
	require.NoError(t, err)

	// Bytes should reflect unread portion
	bytes = buf.Items()
	assert.Equal(t, []int{3, 4, 5}, bytes)
}

func TestBuffer_Next(t *testing.T) {
	backing := make([]int, 0, 10)
	buf := NewBuffer(backing)

	data := []int{1, 2, 3, 4, 5}
	_, err := buf.Write(data)
	require.NoError(t, err)

	// Get next 3 elements
	next := buf.Next(3)
	assert.Equal(t, []int{1, 2, 3}, next)
	assert.Equal(t, 2, buf.Len())

	// Get remaining
	next = buf.Next(10)
	assert.Equal(t, []int{4, 5}, next)
	assert.Equal(t, 0, buf.Len())
}

func TestBuffer_UnreadByte(t *testing.T) {
	backing := make([]byte, 0, 10)
	buf := NewBuffer(backing)

	data := []byte{1, 2, 3}
	_, err := buf.Write(data)
	require.NoError(t, err)

	// Read a byte
	b, err := buf.ReadItem()
	require.NoError(t, err)
	assert.Equal(t, byte(1), b)
	assert.Equal(t, 2, buf.Len())

	// Unread it
	err = buf.UnreadByte()
	require.NoError(t, err)
	assert.Equal(t, 3, buf.Len())

	// Read again
	b, err = buf.ReadItem()
	require.NoError(t, err)
	assert.Equal(t, byte(1), b)

	// Try to unread at start
	buf.Reset()
	err = buf.UnreadByte()
	assert.ErrorIs(t, err, io.ErrUnexpectedEOF)
}

func TestBuffer_Reset(t *testing.T) {
	backing := make([]int, 0, 10)
	buf := NewBuffer(backing)

	data := []int{1, 2, 3, 4, 5}
	_, err := buf.Write(data)
	require.NoError(t, err)

	// Read some
	readBuf := make([]int, 2)
	_, err = buf.Read(readBuf)
	require.NoError(t, err)

	// Reset
	buf.Reset()

	assert.Equal(t, 0, buf.Len())
	assert.Equal(t, 10, buf.Cap())

	// Should be able to write again
	_, err = buf.Write(data)
	require.NoError(t, err)
	assert.Equal(t, 5, buf.Len())
}

func TestBuffer_Grow(t *testing.T) {
	backing := make([]int, 0, 4)
	buf := NewBuffer(backing)

	initialCap := buf.Cap()
	assert.Equal(t, 4, initialCap)

	// Grow by 10
	buf.Grow(10)
	newCap := buf.Cap()
	assert.GreaterOrEqual(t, newCap, initialCap+10)

	// Write data that would require growth
	data := make([]int, 20)
	for i := range data {
		data[i] = i
	}
	_, err := buf.Write(data)
	require.NoError(t, err)
	assert.Equal(t, 20, buf.Len())
}

func TestBuffer_Truncate(t *testing.T) {
	backing := make([]int, 0, 10)
	buf := NewBuffer(backing)

	data := []int{1, 2, 3, 4, 5}
	_, err := buf.Write(data)
	require.NoError(t, err)

	// Truncate to 3 elements
	buf.Truncate(3)
	assert.Equal(t, 3, buf.Len())

	readBuf := make([]int, 3)
	n, err := buf.Read(readBuf)
	require.NoError(t, err)
	assert.Equal(t, 3, n)
	assert.Equal(t, []int{1, 2, 3}, readBuf[:n])
}

func TestBuffer_ByteIO(t *testing.T) {
	backing := make([]byte, 0, 10)
	buf := NewBuffer(backing)

	// Test io.Writer interface
	data := []byte("hello")
	n, err := buf.Write(data)
	require.NoError(t, err)
	assert.Equal(t, len(data), n)

	// Test io.Reader interface
	readBuf := make([]byte, 10)
	n, err = buf.Read(readBuf)
	require.NoError(t, err)
	assert.Equal(t, len(data), n)
	assert.Equal(t, data, readBuf[:n])
}

func TestBuffer_GenericTypes(t *testing.T) {
	// Test with float32
	backing := make([]float32, 0, 10)
	buf := NewBuffer(backing)

	data := []float32{1.1, 2.2, 3.3}
	_, err := buf.Write(data)
	require.NoError(t, err)

	readBuf := make([]float32, 3)
	n, err := buf.Read(readBuf)
	require.NoError(t, err)
	assert.Equal(t, 3, n)
	assert.Equal(t, data, readBuf)
}

func TestBuffer_MultipleWrites(t *testing.T) {
	backing := make([]int, 0, 5)
	buf := NewBuffer(backing)

	// Write multiple times
	_, err := buf.Write([]int{1, 2})
	require.NoError(t, err)

	_, err = buf.Write([]int{3, 4, 5})
	require.NoError(t, err)

	assert.Equal(t, 5, buf.Len())

	// Read all
	readBuf := make([]int, 5)
	n, err := buf.Read(readBuf)
	require.NoError(t, err)
	assert.Equal(t, 5, n)
	assert.Equal(t, []int{1, 2, 3, 4, 5}, readBuf)
}

func TestBuffer_EmptyRead(t *testing.T) {
	backing := make([]int, 0, 10)
	buf := NewBuffer(backing)

	readBuf := make([]int, 5)
	n, err := buf.Read(readBuf)
	assert.ErrorIs(t, err, io.EOF)
	assert.Equal(t, 0, n)
}
