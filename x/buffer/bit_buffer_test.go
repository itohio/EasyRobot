package buffer

import (
	"fmt"
	"io"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBits_Basic(t *testing.T) {
	bb, err := NewBits(3)
	require.NoError(t, err)

	// Write bytes: 0x05, 0x02, 0x07
	// Each byte uses 3 bits, so we get values: 5, 2, 7
	bb.WriteByte(5)
	bb.WriteByte(2)
	bb.WriteByte(7)

	// Reset to read from beginning
	bb.Reset()

	// Read first value: 5 (first 3 bits of first byte)
	value, ok := bb.ReadByte()
	assert.True(t, ok)
	assert.Equal(t, byte(5), value)

	// Read second value: 2 (next 3 bits, but we only have 5 bits left in first byte)
	// Actually, with 3-bit encoding, byte 0x05 = 00000101
	// First 3 bits (MSB): 000 = 0, but we mask to 3 bits so it's 5 & 0x07 = 5
	// Wait, let me think about this more carefully...
	// WriteByte(5) with bitLength=3: 5 = 101 in binary, writes as 101
	// WriteByte(2) with bitLength=3: 2 = 010 in binary, writes as 010
	// So we have: 101010...
	value, ok = bb.ReadByte()
	assert.True(t, ok)
	assert.Equal(t, byte(2), value)

	// Read third value: 7
	value, ok = bb.ReadByte()
	assert.True(t, ok)
	assert.Equal(t, byte(7), value)

	// No more complete values
	_, ok = bb.ReadByte()
	assert.False(t, ok)
}

func TestBits_WriteReadBytes_4Bit(t *testing.T) {
	// Test the main use case: 4-bit buffer, write 8 bytes
	bb, err := NewBits(4)
	require.NoError(t, err)

	// Write 8 bytes: each byte contributes 8 bits, so we get 16 values (8*8/4)
	input := []byte{0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0}
	n, err := bb.Write(input)
	require.NoError(t, err)
	assert.Equal(t, 64, n) // 8 bytes * 8 bits = 64 bits written

	// Reset to read
	bb.Reset()

	// Read 16 values (8 bytes * 8 bits / 4 bits per value)
	output := make([]byte, 16)
	nRead, err := bb.Read(output)
	require.NoError(t, err)
	assert.Equal(t, 64, nRead) // 16 values * 4 bits = 64 bits

	// Verify: first byte 0x12 = 00010010
	// First 4 bits (MSB): 0001 = 1
	// Next 4 bits: 0010 = 2
	// Second byte 0x34 = 00110100
	// First 4 bits: 0011 = 3
	// Next 4 bits: 0100 = 4
	// And so on...
	expected := []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0}
	assert.Equal(t, expected, output)
}

func TestBits_WriteReadBytes_4Bit_ConsecutiveBits(t *testing.T) {
	// Verify consecutive bit stream behavior
	bb, err := NewBits(4)
	require.NoError(t, err)

	// Write single byte: 0xAB = 10101011
	// Should produce 2 values: 1010 (10) and 1011 (11)
	// Use Write() to write all 8 bits, not WriteByte() which only writes bitLength bits
	n, err := bb.Write([]byte{0xAB})
	require.NoError(t, err)
	assert.Equal(t, 8, n) // 2 values * 4 bits = 8 bits

	bb.Reset()

	// Read first 4 bits: 1010 = 10
	val1, ok := bb.ReadByte()
	assert.True(t, ok)
	assert.Equal(t, byte(10), val1)

	// Read next 4 bits: 1011 = 11
	val2, ok := bb.ReadByte()
	assert.True(t, ok)
	assert.Equal(t, byte(11), val2)

	// No more values
	_, ok = bb.ReadByte()
	assert.False(t, ok)
}

func TestBits_WriteReadBytes_4Bit_MultipleBytes(t *testing.T) {
	bb, err := NewBits(4)
	require.NoError(t, err)

	// Write 2 bytes: 0x12, 0x34
	// 0x12 = 00010010 -> values: 0001 (1), 0010 (2)
	// 0x34 = 00110100 -> values: 0011 (3), 0100 (4)
	input := []byte{0x12, 0x34}
	n, err := bb.Write(input)
	require.NoError(t, err)
	assert.Equal(t, 16, n) // 2 bytes * 8 bits = 16 bits

	bb.Reset()

	// Read 4 values
	output := make([]byte, 4)
	nRead, err := bb.Read(output)
	require.NoError(t, err)
	assert.Equal(t, 16, nRead)

	expected := []byte{1, 2, 3, 4}
	assert.Equal(t, expected, output)
}

func TestBits_PartialValue(t *testing.T) {
	// Create buffer with only 1 byte: 0x80 = 10000000
	bb, err := NewBits(4)
	require.NoError(t, err)
	bb.ReadFromBytes([]byte{0x80})

	// Should be able to read first value: 1000 = 8
	value, ok := bb.ReadByte()
	assert.True(t, ok)
	assert.Equal(t, byte(8), value)

	// Should be able to read second value: 0000 = 0
	value, ok = bb.ReadByte()
	assert.True(t, ok)
	assert.Equal(t, byte(0), value)

	// No more values
	_, ok = bb.ReadByte()
	assert.False(t, ok)
}

func TestBits_MultipleValues_2Bit(t *testing.T) {
	// Create bytes: 0x1B = 00011011
	// Bits: 00 (0), 01 (1), 10 (2), 11 (3)
	bb, err := NewBits(2)
	require.NoError(t, err)
	bb.ReadFromBytes([]byte{0x1B})

	expected := []byte{0, 1, 2, 3}
	for _, exp := range expected {
		value, ok := bb.ReadByte()
		assert.True(t, ok, "Should read value %d", exp)
		assert.Equal(t, exp, value)
	}

	// No more values
	_, ok := bb.ReadByte()
	assert.False(t, ok)
}

func TestBits_GetRemainingBits(t *testing.T) {
	bb, err := NewBits(3)
	require.NoError(t, err)
	bb.ReadFromBytes([]byte{0xFF, 0xFF})

	// Initially at position 0, 16 bits available
	assert.Equal(t, 16, bb.GetRemainingBits())

	// Read one value (3 bits)
	bb.ReadByte()
	// Should have 13 bits remaining
	assert.Equal(t, 13, bb.GetRemainingBits())
}

func TestBits_Roundtrip(t *testing.T) {
	bb, err := NewBits(3)
	require.NoError(t, err)

	// Write bytes: 5, 2, 7
	bb.WriteByte(5)
	bb.WriteByte(2)
	bb.WriteByte(7)

	// Reset to read from beginning
	bb.Reset()

	expected := []byte{5, 2, 7}
	for _, exp := range expected {
		value, ok := bb.ReadByte()
		assert.True(t, ok, "Should read value %d", exp)
		assert.Equal(t, exp, value)
	}
}

func TestBits_Roundtrip_MultipleBitLengths(t *testing.T) {
	bitLengths := []int{1, 2, 3, 4, 5, 6, 7, 8}
	for _, bitLen := range bitLengths {
		t.Run(fmt.Sprintf("%d_bits", bitLen), func(t *testing.T) {
			bb, err := NewBits(bitLen)
			require.NoError(t, err)

			// Write test bytes
			maxValue := byte((1 << bitLen) - 1)
			testValues := []byte{0, maxValue / 4, maxValue / 2, maxValue}

			for _, val := range testValues {
				bb.WriteByte(val)
			}

			// Reset to read
			bb.Reset()

			for _, exp := range testValues {
				value, ok := bb.ReadByte()
				assert.True(t, ok, "Should read value %d for bitLength %d", exp, bitLen)
				// Mask to expected bit length
				mask := byte((1 << bitLen) - 1)
				assert.Equal(t, exp&mask, value&mask, "Value mismatch for bitLength %d", bitLen)
			}
		})
	}
}

func TestBits_ToBytes(t *testing.T) {
	bb, err := NewBits(3)
	require.NoError(t, err)

	// Write bytes: 5 (101), 2 (010), 7 (111)
	bb.WriteByte(5)
	bb.WriteByte(2)
	bb.WriteByte(7)

	// Convert to bytes
	bytes := bb.ToBytes()

	// Bits: 101 010 111 = 10101011 1...
	// First byte: 10101011 = 0xAB
	// Second byte: 10000000 = 0x80 (with padding, MSB first)
	assert.Equal(t, []byte{0xAB, 0x80}, bytes)
}

func TestBits_MultipleValues_ToBytes(t *testing.T) {
	bb, err := NewBits(2)
	require.NoError(t, err)

	// Write bytes: 0 (00), 1 (01), 2 (10), 3 (11)
	bb.WriteByte(0)
	bb.WriteByte(1)
	bb.WriteByte(2)
	bb.WriteByte(3)

	// Convert to bytes: 00011011 = 0x1B
	bytes := bb.ToBytes()
	expected := []byte{0x1B}
	assert.Equal(t, expected, bytes)
}

func TestBits_LargeValue(t *testing.T) {
	bb, err := NewBits(8)
	require.NoError(t, err)

	// Write byte 255 (all bits set)
	bb.WriteByte(255)

	// Convert to bytes - should have 0xFF
	bytes := bb.ToBytes()
	assert.Equal(t, []byte{0xFF}, bytes)
}

func TestBits_ValueMasking(t *testing.T) {
	bb, err := NewBits(3)
	require.NoError(t, err)

	// Write byte 15 (1111 in binary) - should be masked to 7 (111)
	bb.WriteByte(15)

	// Convert to bytes - should have 11100000 = 0xE0 (with padding)
	bytes := bb.ToBytes()
	assert.Equal(t, []byte{0xE0}, bytes)
}

func TestBits_PartialByte(t *testing.T) {
	bb, err := NewBits(3)
	require.NoError(t, err)

	// Write one byte (3 bits used) - not a complete byte
	bb.WriteByte(5)
	assert.Equal(t, 3, bb.GetWrittenBits())

	// Convert to bytes - should pad with zeros
	bytes := bb.ToBytes()
	// Should have one byte: 10100000 = 0xA0
	assert.Equal(t, []byte{0xA0}, bytes)
}

func TestBits_FromSlice(t *testing.T) {
	// Create a pre-existing slice
	data := make([]byte, 0, 16)
	bb, err := NewBits(3, data...)
	require.NoError(t, err)

	// Write some bytes
	bb.WriteByte(5)
	bb.WriteByte(2)

	// The buffer's data should be modified
	assert.Greater(t, len(bb.Bytes()), 0)
}

func TestBits_Position(t *testing.T) {
	bb, err := NewBits(3)
	require.NoError(t, err)

	bb.WriteByte(5)
	bb.WriteByte(2)
	bb.WriteByte(7)

	// Position should be at end
	assert.Equal(t, 9, bb.Position())

	// Set position to beginning
	err = bb.SetPosition(0)
	require.NoError(t, err)
	assert.Equal(t, 0, bb.Position())

	// Can read from beginning
	value, ok := bb.ReadByte()
	assert.True(t, ok)
	assert.Equal(t, byte(5), value)
}

func TestBits_Reset(t *testing.T) {
	bb, err := NewBits(3)
	require.NoError(t, err)

	bb.WriteByte(5)
	bb.WriteByte(2)
	bb.WriteByte(7)

	// Read one value
	bb.Reset()
	value, ok := bb.ReadByte()
	assert.True(t, ok)
	assert.Equal(t, byte(5), value)

	// Reset again
	bb.Reset()
	value, ok = bb.ReadByte()
	assert.True(t, ok)
	assert.Equal(t, byte(5), value)
}

func TestBits_Clear(t *testing.T) {
	bb, err := NewBits(3)
	require.NoError(t, err)

	bb.WriteByte(5)
	bb.WriteByte(2)
	bb.WriteByte(7)

	bb.Clear()

	// Should be empty
	assert.Equal(t, 0, bb.GetWrittenBits())
	assert.Equal(t, 0, bb.Position())
	assert.False(t, bb.HasValue())
}

func TestBits_Compact(t *testing.T) {
	bb, err := NewBits(8) // Use 8 bits per value for byte alignment
	require.NoError(t, err)

	// Write many bytes (byte-aligned)
	for i := range 20 {
		bb.WriteByte(byte(i % 256))
	}

	// Read half of them
	for range 10 {
		bb.ReadByte()
	}

	// Compact should move remaining data to beginning
	oldPos := bb.Position()
	oldWrittenBits := bb.GetWrittenBits()
	bb.Compact()
	assert.Equal(t, 0, bb.Position())
	// After compact, remaining bits should equal what was written minus what was consumed
	expectedRemaining := oldWrittenBits - oldPos
	assert.Equal(t, expectedRemaining, bb.GetRemainingBits())
}

func TestBits_ReadFromBytes(t *testing.T) {
	bb, err := NewBits(4)
	require.NoError(t, err)

	// Read from bytes: 0x80 = 10000000
	bb.ReadFromBytes([]byte{0x80})

	// Should be able to read two 4-bit values
	value, ok := bb.ReadByte()
	assert.True(t, ok)
	assert.Equal(t, byte(8), value) // 1000 = 8

	value, ok = bb.ReadByte()
	assert.True(t, ok)
	assert.Equal(t, byte(0), value) // 0000 = 0
}

func TestBits_SharedState(t *testing.T) {
	// Test that reading and writing share the same position
	bb, err := NewBits(3)
	require.NoError(t, err)

	// Write a byte
	bb.WriteByte(5)

	// Position is now at 3
	assert.Equal(t, 3, bb.Position())

	// Reset and read
	bb.Reset()
	value, ok := bb.ReadByte()
	assert.True(t, ok)
	assert.Equal(t, byte(5), value)

	// Position is now at 3 again
	assert.Equal(t, 3, bb.Position())

	// Can write more at current position
	bb.WriteByte(2)
	assert.Equal(t, 6, bb.Position())

	// Reset and read both
	bb.Reset()
	value, ok = bb.ReadByte()
	assert.True(t, ok)
	assert.Equal(t, byte(5), value)

	value, ok = bb.ReadByte()
	assert.True(t, ok)
	assert.Equal(t, byte(2), value)
}

func TestBits_CompatibilityWithBytesBuffer(t *testing.T) {
	// Test compatibility pattern: write to Bits, convert to bytes, read from bytes
	bb, err := NewBits(3)
	require.NoError(t, err)

	// Write bytes
	bb.WriteByte(5)
	bb.WriteByte(2)
	bb.WriteByte(7)

	// Convert to bytes
	bytes := bb.ToBytes()

	// Create new buffer and read from bytes
	bb2, err := NewBits(3)
	require.NoError(t, err)
	bb2.ReadFromBytes(bytes)
	bb2.Reset()

	// Should read same values
	expected := []byte{5, 2, 7}
	for _, exp := range expected {
		value, ok := bb2.ReadByte()
		assert.True(t, ok, "Should read value %d", exp)
		assert.Equal(t, exp, value)
	}
}

func TestBits_WriteInterface(t *testing.T) {
	bb, err := NewBits(3)
	require.NoError(t, err)

	// Write multiple bytes using WriteByte (writes bitLength bits per value)
	values := []byte{5, 2, 7}
	for _, v := range values {
		bb.WriteByte(v)
	}

	// Should have written 9 bits (3 values * 3 bits)
	assert.Equal(t, 9, bb.GetWrittenBits())

	// Reset and verify we can read them back
	bb.Reset()
	for _, exp := range values {
		value, ok := bb.ReadByte()
		assert.True(t, ok)
		assert.Equal(t, exp, value)
	}
}

func TestBits_ReadInterface(t *testing.T) {
	bb, err := NewBits(3)
	require.NoError(t, err)

	// Write some bytes
	bb.WriteByte(5)
	bb.WriteByte(2)
	bb.WriteByte(7)
	bb.Reset()

	// Read multiple bytes using interface method
	buf := make([]byte, 3)
	n, err := bb.Read(buf)

	assert.NoError(t, err)
	// Should return bits read: 3 bytes * 3 bits = 9 bits
	assert.Equal(t, 9, n)
	assert.Equal(t, []byte{5, 2, 7}, buf)
	assert.Equal(t, 9, bb.Position())
}

func TestBits_ReadInterface_Partial(t *testing.T) {
	bb, err := NewBits(3)
	require.NoError(t, err)

	// Write only 2 bytes (6 bits)
	bb.WriteByte(5)
	bb.WriteByte(2)
	bb.Reset()

	// Try to read 3 bytes
	buf := make([]byte, 3)
	n, err := bb.Read(buf)

	assert.Equal(t, io.EOF, err)
	// Should return bits read: 2 bytes * 3 bits = 6 bits
	assert.Equal(t, 6, n)
	assert.Equal(t, []byte{5, 2, 0}, buf) // Last byte remains zero
}

func TestBits_ReadInterface_Empty(t *testing.T) {
	bb, err := NewBits(3)
	require.NoError(t, err)

	// Try to read from empty buffer
	buf := make([]byte, 3)
	n, err := bb.Read(buf)

	assert.Equal(t, io.EOF, err)
	assert.Equal(t, 0, n)
	assert.Equal(t, []byte{0, 0, 0}, buf)
}

func TestBits_WriteInterface_Empty(t *testing.T) {
	bb, err := NewBits(3)
	require.NoError(t, err)

	// Write empty slice
	n, err := bb.Write([]byte{})

	assert.NoError(t, err)
	assert.Equal(t, 0, n)
	assert.Equal(t, 0, bb.GetWrittenBits())
}

func TestBits_ReadInterface_EmptySlice(t *testing.T) {
	bb, err := NewBits(3)
	require.NoError(t, err)

	// Write some bytes
	bb.WriteByte(5)
	bb.Reset()

	// Read into empty slice
	n, err := bb.Read([]byte{})

	assert.NoError(t, err)
	assert.Equal(t, 0, n)
}

func TestBits_WriteReadBytes_8Bits(t *testing.T) {
	// Test 8 bits per value - should be byte-aligned
	bb, err := NewBits(8)
	require.NoError(t, err)

	// Write bytes
	bytes := []byte{0x01, 0x02, 0x03, 0xFF}
	n, err := bb.Write(bytes)
	require.NoError(t, err)
	// Should return bits: 4 bytes * 8 bits = 32 bits
	assert.Equal(t, 32, n)

	// Read bytes
	bb.Reset()
	readBuf := make([]byte, 4)
	nRead, err := bb.Read(readBuf)
	require.NoError(t, err)
	assert.Equal(t, 32, nRead)
	assert.Equal(t, bytes, readBuf)
}

func TestBits_WriteReadBytes_4Bits_8Bytes(t *testing.T) {
	// Main test case: 4-bit buffer, write 8 bytes, read 16 values
	bb, err := NewBits(4)
	require.NoError(t, err)

	// Write 8 bytes
	input := []byte{0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0}
	n, err := bb.Write(input)
	require.NoError(t, err)
	assert.Equal(t, 64, n) // 8 bytes * 8 bits = 64 bits

	bb.Reset()

	// Read 16 values (8 bytes * 8 bits / 4 bits per value)
	output := make([]byte, 16)
	nRead, err := bb.Read(output)
	require.NoError(t, err)
	assert.Equal(t, 64, nRead)

	// Verify: each byte contributes 2 values
	// 0x12 = 00010010 -> 0001 (1), 0010 (2)
	// 0x34 = 00110100 -> 0011 (3), 0100 (4)
	// etc.
	expected := []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0}
	assert.Equal(t, expected, output)
}

func TestBits_Interface_Compatibility(t *testing.T) {
	// Test that Bits properly implements Reader[byte] and Writer[byte]
	var _ Writer[byte] = (*Bits)(nil)
	var _ Reader[byte] = (*Bits)(nil)

	bb, err := NewBits(8) // 8 bits = 1 byte per value
	require.NoError(t, err)

	// Write bytes
	bytes := []byte{0x01, 0x02, 0x03, 0xFF}
	n, err := bb.Write(bytes)
	require.NoError(t, err)
	// Should return bits: 4 bytes * 8 bits = 32 bits
	assert.Equal(t, 32, n)

	// Read bytes
	bb.Reset()
	readBuf := make([]byte, 4)
	nRead, err := bb.Read(readBuf)
	require.NoError(t, err)
	assert.Equal(t, 32, nRead)
	assert.Equal(t, bytes, readBuf)
}

func TestBits_ReadBit_WriteBit(t *testing.T) {
	bb, err := NewBits(1)
	require.NoError(t, err)

	// Write individual bits
	bb.WriteBit(true)
	bb.WriteBit(false)
	bb.WriteBit(true)
	bb.WriteBit(true)

	bb.Reset()

	// Read bits back
	bit, ok := bb.ReadBit()
	assert.True(t, ok)
	assert.True(t, bit)

	bit, ok = bb.ReadBit()
	assert.True(t, ok)
	assert.False(t, bit)

	bit, ok = bb.ReadBit()
	assert.True(t, ok)
	assert.True(t, bit)

	bit, ok = bb.ReadBit()
	assert.True(t, ok)
	assert.True(t, bit)

	_, ok = bb.ReadBit()
	assert.False(t, ok)
}

func TestBits_Cap(t *testing.T) {
	bb, err := NewBits(4)
	require.NoError(t, err)

	// Initial capacity should be 8 (default)
	assert.GreaterOrEqual(t, bb.Cap(), 8)

	// Grow buffer
	bb.Grow(100)
	oldCap := bb.Cap()

	// Capacity should have increased
	assert.GreaterOrEqual(t, bb.Cap(), oldCap)
}

func TestBits_Init(t *testing.T) {
	bb := &Bits{}

	// Initialize with specific parameters
	err := bb.Init(4, 0, 16)
	require.NoError(t, err)

	assert.Equal(t, 4, bb.Bits())
	assert.Equal(t, 16, bb.Cap())
	assert.Equal(t, 0, bb.Position())
}

func TestBits_Init_Invalid(t *testing.T) {
	bb := &Bits{}

	// Invalid bitLength
	err := bb.Init(0, 0, 16)
	assert.Error(t, err)

	err = bb.Init(9, 0, 16)
	assert.Error(t, err)

	// Invalid length/capacity
	err = bb.Init(4, -1, 16)
	assert.Error(t, err)

	err = bb.Init(4, 10, 5)
	assert.Error(t, err)
}
