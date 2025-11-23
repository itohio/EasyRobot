package generics

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestBitBuffer_Basic(t *testing.T) {
	bb := NewBitBuffer[int64](3)

	// Write values: 5, 2, 7
	bb.Write(5)
	bb.Write(2)
	bb.Write(7)

	// Reset to read from beginning
	bb.Reset()

	// Read first value: 5
	value, ok := bb.Read()
	assert.True(t, ok)
	assert.Equal(t, int64(5), value)

	// Read second value: 2
	value, ok = bb.Read()
	assert.True(t, ok)
	assert.Equal(t, int64(2), value)

	// Read third value: 7
	value, ok = bb.Read()
	assert.True(t, ok)
	assert.Equal(t, int64(7), value)

	// No more complete values
	_, ok = bb.Read()
	assert.False(t, ok)
}

func TestBitBuffer_PartialValue(t *testing.T) {
	// Create bytes with only 1 byte: 0x80 = 10000000
	// This gives us 8 bits: 1,0,0,0,0,0,0,0
	bb := NewBitBuffer[int64](4)
	bb.ReadFromBytes([]byte{0x80})

	// Should be able to read first value: 1000 = 8
	value, ok := bb.Read()
	assert.True(t, ok)
	assert.Equal(t, int64(8), value)

	// Should be able to read second value: 0000 = 0
	value, ok = bb.Read()
	assert.True(t, ok)
	assert.Equal(t, int64(0), value)

	// No more values
	_, ok = bb.Read()
	assert.False(t, ok)
}

func TestBitBuffer_MultipleValues(t *testing.T) {
	// Create bytes: 00011011
	// Bits: 00 (0), 01 (1), 10 (2), 11 (3)
	bb := NewBitBuffer[int64](2)
	bb.ReadFromBytes([]byte{0x1B}) // 00011011

	values := []int64{0, 1, 2, 3}
	for _, expected := range values {
		value, ok := bb.Read()
		assert.True(t, ok, "Should read value %d", expected)
		assert.Equal(t, expected, value)
	}

	// No more values
	_, ok := bb.Read()
	assert.False(t, ok)
}

func TestBitBuffer_GetRemainingBits(t *testing.T) {
	bb := NewBitBuffer[int64](3)
	bb.ReadFromBytes([]byte{0xFF, 0xFF})

	// Initially at position 0, 16 bits available
	assert.Equal(t, 16, bb.GetRemainingBits())

	// Read one value (3 bits)
	bb.Read()
	// Should have 13 bits remaining
	assert.Equal(t, 13, bb.GetRemainingBits())
}

func TestBitBuffer_Roundtrip(t *testing.T) {
	// Test roundtrip: write values, then read them back
	bb := NewBitBuffer[int64](3)

	// Write values: 5, 2, 7
	bb.Write(5)
	bb.Write(2)
	bb.Write(7)

	// Reset to read from beginning
	bb.Reset()

	expected := []int64{5, 2, 7}
	for _, exp := range expected {
		value, ok := bb.Read()
		assert.True(t, ok, "Should read value %d", exp)
		assert.Equal(t, exp, value)
	}
}

func TestBitBuffer_Roundtrip_MultipleBitLengths(t *testing.T) {
	bitLengths := []int{1, 2, 3, 4, 5, 6, 7, 8}
	for _, bitLen := range bitLengths {
		t.Run(fmt.Sprintf("%d_bits", bitLen), func(t *testing.T) {
			bb := NewBitBuffer[int64](bitLen)

			// Write test values
			maxValue := int64((1 << uint(bitLen)) - 1)
			testValues := []int64{0, maxValue / 4, maxValue / 2, maxValue}

			for _, val := range testValues {
				bb.Write(val)
			}

			// Reset to read
			bb.Reset()

			for _, exp := range testValues {
				value, ok := bb.Read()
				assert.True(t, ok, "Should read value %d for bitLength %d", exp, bitLen)
				assert.Equal(t, exp, value, "Value mismatch for bitLength %d", bitLen)
			}
		})
	}
}

func TestBitBuffer_ToBytes(t *testing.T) {
	bb := NewBitBuffer[int64](3)

	// Write values: 5 (101), 2 (010), 7 (111)
	bb.Write(5)
	bb.Write(2)
	bb.Write(7)

	// Convert to bytes
	bytes := bb.ToBytes()

	// Bits: 101 010 111 = 10101011 1...
	// First byte: 10101011 = 0xAB
	// Second byte: 11100000 = 0xE0 (with padding)
	// Actually: 10101011 11100000 = 0xAB 0xE0
	// But with padding: 10101011 10000000 = 0xAB 0x80
	expected := []byte{0xAB, 0x80}
	assert.Equal(t, expected, bytes)
}

func TestBitBuffer_MultipleValues_ToBytes(t *testing.T) {
	bb := NewBitBuffer[int64](2)

	// Write values: 0 (00), 1 (01), 2 (10), 3 (11)
	bb.Write(0)
	bb.Write(1)
	bb.Write(2)
	bb.Write(3)

	// Convert to bytes: 00011011 = 0x1B
	bytes := bb.ToBytes()
	expected := []byte{0x1B}
	assert.Equal(t, expected, bytes)
}

func TestBitBuffer_LargeValue(t *testing.T) {
	bb := NewBitBuffer[int64](8)

	// Write value 255 (all bits set)
	bb.Write(255)

	// Convert to bytes - should have 0xFF
	bytes := bb.ToBytes()
	assert.Equal(t, []byte{0xFF}, bytes)
}

func TestBitBuffer_ValueMasking(t *testing.T) {
	bb := NewBitBuffer[int64](3)

	// Write value 15 (1111 in binary) - should be masked to 7 (111)
	bb.Write(15)

	// Convert to bytes - should have 11100000 = 0xE0 (with padding)
	bytes := bb.ToBytes()
	assert.Equal(t, []byte{0xE0}, bytes)
}

func TestBitBuffer_PartialByte(t *testing.T) {
	bb := NewBitBuffer[int64](3)

	// Write one value (3 bits) - not a complete byte
	bb.Write(5)
	assert.Equal(t, 3, bb.GetWrittenBits())

	// Convert to bytes - should pad with zeros
	bytes := bb.ToBytes()
	// Should have one byte: 10100000 = 0xA0
	assert.Equal(t, []byte{0xA0}, bytes)
}

func TestBitBuffer_FromSlice(t *testing.T) {
	// Create a pre-existing slice
	data := make([]byte, 0, 16)
	bb := NewBitBuffer[int64](3, data...)

	// Write some values
	bb.Write(5)
	bb.Write(2)

	// The original slice should be modified
	assert.Greater(t, len(data), 0)
}

func TestBitBuffer_Position(t *testing.T) {
	bb := NewBitBuffer[int64](3)

	bb.Write(5)
	bb.Write(2)
	bb.Write(7)

	// Position should be at end
	assert.Equal(t, 9, bb.Position())

	// Set position to beginning
	bb.SetPosition(0)
	assert.Equal(t, 0, bb.Position())

	// Can read from beginning
	value, ok := bb.Read()
	assert.True(t, ok)
	assert.Equal(t, int64(5), value)
}

func TestBitBuffer_Reset(t *testing.T) {
	bb := NewBitBuffer[int64](3)

	bb.Write(5)
	bb.Write(2)
	bb.Write(7)

	// Read one value
	bb.Reset()
	value, ok := bb.Read()
	assert.True(t, ok)
	assert.Equal(t, int64(5), value)

	// Reset again
	bb.Reset()
	value, ok = bb.Read()
	assert.True(t, ok)
	assert.Equal(t, int64(5), value)
}

func TestBitBuffer_Clear(t *testing.T) {
	bb := NewBitBuffer[int64](3)

	bb.Write(5)
	bb.Write(2)
	bb.Write(7)

	bb.Clear()

	// Should be empty
	assert.Equal(t, 0, bb.GetWrittenBits())
	assert.Equal(t, 0, bb.Position())
	assert.False(t, bb.HasValue())
}

func TestBitBuffer_Compact(t *testing.T) {
	bb := NewBitBuffer[int64](3)

	// Write many values
	for i := 0; i < 20; i++ {
		bb.Write(int64(i % 8))
	}

	// Read half of them
	for i := 0; i < 10; i++ {
		bb.Read()
	}

	// Compact should move remaining data to beginning
	oldPos := bb.Position()
	bb.Compact()
	assert.Equal(t, 0, bb.Position())
	assert.Equal(t, bb.GetWrittenBits()-oldPos, bb.GetRemainingBits())
}

func TestBitBuffer_GenericTypes(t *testing.T) {
	// Test with different integer types
	t.Run("int8", func(t *testing.T) {
		bb := NewBitBuffer[int8](3)
		bb.Write(int8(5))
		bb.Reset()
		value, ok := bb.Read()
		assert.True(t, ok)
		assert.Equal(t, int8(5), value)
	})

	t.Run("uint16", func(t *testing.T) {
		bb := NewBitBuffer[uint16](4)
		bb.Write(uint16(15))
		bb.Reset()
		value, ok := bb.Read()
		assert.True(t, ok)
		assert.Equal(t, uint16(15), value)
	})

	t.Run("uint8", func(t *testing.T) {
		bb := NewBitBuffer[uint8](8)
		bb.Write(uint8(255))
		bb.Reset()
		value, ok := bb.Read()
		assert.True(t, ok)
		assert.Equal(t, uint8(255), value)
	})
}

func TestBitBuffer_SignExtension(t *testing.T) {
	// Test sign extension for signed types
	bb := NewBitBuffer[int8](3)

	// Write 7 (111 in 3 bits) - MSB is set, should sign extend when read
	bb.Write(int8(7))
	bb.Reset()

	value, ok := bb.Read()
	assert.True(t, ok)
	// With sign extension, 111 in 3 bits becomes -1 in int8
	assert.Equal(t, int8(-1), value)
}

func TestBitBuffer_ReadFromBytes(t *testing.T) {
	bb := NewBitBuffer[int64](4)

	// Read from bytes
	bb.ReadFromBytes([]byte{0x80}) // 10000000

	// Should be able to read two 4-bit values
	value, ok := bb.Read()
	assert.True(t, ok)
	assert.Equal(t, int64(8), value) // 1000 = 8

	value, ok = bb.Read()
	assert.True(t, ok)
	assert.Equal(t, int64(0), value) // 0000 = 0
}

func TestBitBuffer_SharedState(t *testing.T) {
	// Test that reading and writing share the same position
	bb := NewBitBuffer[int64](3)

	// Write a value
	bb.Write(5)

	// Position is now at 3
	assert.Equal(t, 3, bb.Position())

	// Reset and read
	bb.Reset()
	value, ok := bb.Read()
	assert.True(t, ok)
	assert.Equal(t, int64(5), value)

	// Position is now at 3 again
	assert.Equal(t, 3, bb.Position())

	// Can write more at current position
	bb.Write(2)
	assert.Equal(t, 6, bb.Position())

	// Reset and read both
	bb.Reset()
	value, ok = bb.Read()
	assert.True(t, ok)
	assert.Equal(t, int64(5), value)

	value, ok = bb.Read()
	assert.True(t, ok)
	assert.Equal(t, int64(2), value)
}

func TestBitBuffer_CompatibilityWithBytesBuffer(t *testing.T) {
	// Test compatibility pattern: write to BitBuffer, convert to bytes, read from bytes
	bb := NewBitBuffer[int64](3)

	// Write values
	bb.Write(5)
	bb.Write(2)
	bb.Write(7)

	// Convert to bytes
	bytes := bb.ToBytes()

	// Create new buffer and read from bytes
	bb2 := NewBitBuffer[int64](3)
	bb2.ReadFromBytes(bytes)
	bb2.Reset()

	// Should read same values
	expected := []int64{5, 2, 7}
	for _, exp := range expected {
		value, ok := bb2.Read()
		assert.True(t, ok, "Should read value %d", exp)
		assert.Equal(t, exp, value)
	}
}
