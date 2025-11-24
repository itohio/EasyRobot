package buffer

import (
	"io"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRingBuffer_WriteRead(t *testing.T) {
	tests := []struct {
		name     string
		capacity int
		write    []int
		read     int
		wantRead []int
		wantErr  error
	}{
		{
			name:     "simple write and read",
			capacity: 10,
			write:    []int{1, 2, 3},
			read:     3,
			wantRead: []int{1, 2, 3},
		},
		{
			name:     "write more than capacity",
			capacity: 5,
			write:    []int{1, 2, 3, 4, 5, 6, 7},
			read:     5,
			wantRead: []int{1, 2, 3, 4, 5},
		},
		{
			name:     "read more than available",
			capacity: 10,
			write:    []int{1, 2, 3},
			read:     5,
			wantRead: []int{1, 2, 3},
			wantErr:  io.EOF,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			backing := make([]int, 0, tt.capacity)
			rb := NewRing(backing)

			n, err := rb.Write(tt.write)
			require.NoError(t, err)
			// For "write more than capacity" test, expect partial write (up to capacity)
			expectedWrite := len(tt.write)
			if tt.name == "write more than capacity" {
				expectedWrite = tt.capacity
			}
			assert.Equal(t, expectedWrite, n)
			assert.Equal(t, expectedWrite, rb.Len())

			readBuf := make([]int, tt.read)
			n, err = rb.Read(readBuf)
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

func TestRingBuffer_WrapAround(t *testing.T) {
	backing := make([]int, 0, 5)
	rb := NewRing(backing)

	// Fill buffer
	_, err := rb.Write([]int{1, 2, 3, 4, 5})
	require.NoError(t, err)

	// Read some
	readBuf := make([]int, 3)
	n, err := rb.Read(readBuf)
	require.NoError(t, err)
	assert.Equal(t, 3, n)
	assert.Equal(t, []int{1, 2, 3}, readBuf)

	// Write more (should wrap around)
	_, err = rb.Write([]int{6, 7, 8})
	require.NoError(t, err)
	assert.Equal(t, 5, rb.Len())

	// Read remaining
	readBuf = make([]int, 5)
	n, err = rb.Read(readBuf)
	require.NoError(t, err)
	assert.Equal(t, 5, n)
	assert.Equal(t, []int{4, 5, 6, 7, 8}, readBuf[:n])
}

func TestRingBuffer_WriteByteReadByte(t *testing.T) {
	backing := make([]byte, 0, 10)
	rb := NewRing(backing)

	// Write bytes
	for i := byte(0); i < 5; i++ {
		err := rb.WriteItem(i)
		require.NoError(t, err)
	}
	assert.Equal(t, 5, rb.Len())

	// Read bytes
	for i := byte(0); i < 5; i++ {
		b, err := rb.ReadItem()
		require.NoError(t, err)
		assert.Equal(t, i, b)
	}

	// Read from empty buffer
	_, err := rb.ReadItem()
	assert.ErrorIs(t, err, io.EOF)
}

func TestRingBuffer_Available(t *testing.T) {
	backing := make([]int, 0, 10)
	rb := NewRing(backing)

	assert.Equal(t, 10, rb.Available())
	assert.Equal(t, 10, rb.Cap())
	assert.Equal(t, 0, rb.Len())

	_, err := rb.Write([]int{1, 2, 3})
	require.NoError(t, err)

	assert.Equal(t, 7, rb.Available())
	assert.Equal(t, 3, rb.Len())
}

func TestRingBuffer_FullEmpty(t *testing.T) {
	backing := make([]int, 0, 5)
	rb := NewRing(backing)

	assert.True(t, rb.Empty())
	assert.False(t, rb.Full())

	// Fill buffer
	_, err := rb.Write([]int{1, 2, 3, 4, 5})
	require.NoError(t, err)

	assert.False(t, rb.Empty())
	assert.True(t, rb.Full())

	// Try to write when full
	_, err = rb.Write([]int{6})
	assert.ErrorIs(t, err, io.ErrShortWrite)
}

func TestRingBuffer_Reset(t *testing.T) {
	backing := make([]int, 0, 10)
	rb := NewRing(backing)

	_, err := rb.Write([]int{1, 2, 3, 4, 5})
	require.NoError(t, err)

	rb.Reset()

	assert.Equal(t, 0, rb.Len())
	assert.Equal(t, 10, rb.Available())
	assert.True(t, rb.Empty())
}

func TestRingBuffer_ByteIO(t *testing.T) {
	backing := make([]byte, 0, 10)
	rb := NewRing(backing)

	// Test io.Writer interface - calls WriteItem in a loop
	data := []byte("hello")
	n, err := rb.Write(data)
	require.NoError(t, err)
	assert.Equal(t, len(data), n) // 5 bytes written

	// Test io.Reader interface - calls ReadItem in a loop
	readBuf := make([]byte, 5)
	n, err = rb.Read(readBuf)
	require.NoError(t, err)
	assert.Equal(t, len(data), n) // 5 bytes read
	assert.Equal(t, data, readBuf)
}

func TestRingBuffer_GenericTypes(t *testing.T) {
	// Test with float32
	backing := make([]float32, 0, 5)
	rb := NewRing(backing)

	data := []float32{1.1, 2.2, 3.3}
	_, err := rb.Write(data)
	require.NoError(t, err)

	readBuf := make([]float32, 3)
	n, err := rb.Read(readBuf)
	require.NoError(t, err)
	assert.Equal(t, 3, n)
	assert.Equal(t, data, readBuf)
}

func TestRingBuffer_ZeroCapacity(t *testing.T) {
	backing := make([]int, 0, 0)
	rb := NewRing(backing)

	assert.Equal(t, 0, rb.Cap())
	assert.Equal(t, 0, rb.Available())

	_, err := rb.Write([]int{1})
	assert.ErrorIs(t, err, io.ErrShortWrite)
}
