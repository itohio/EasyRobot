package buffer

import (
	"fmt"
	"io"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBitRing_Basic(t *testing.T) {
	br, err := NewBitRing(3, 10)
	require.NoError(t, err)

	// Write bytes: 5, 2, 7
	err = br.WriteByte(5)
	require.NoError(t, err)
	err = br.WriteByte(2)
	require.NoError(t, err)
	err = br.WriteByte(7)
	require.NoError(t, err)

	assert.Equal(t, 3, br.Len())
	assert.Equal(t, 7, br.Available())

	// Read bytes back
	value, err := br.ReadByte()
	require.NoError(t, err)
	assert.Equal(t, byte(5), value)

	value, err = br.ReadByte()
	require.NoError(t, err)
	assert.Equal(t, byte(2), value)

	value, err = br.ReadByte()
	require.NoError(t, err)
	assert.Equal(t, byte(7), value)

	assert.Equal(t, 0, br.Len())
	assert.True(t, br.Empty())
}

func TestBitRing_WriteReadBytes_4Bit(t *testing.T) {
	// Test 4-bit buffer, write 8 bytes, read 8 values
	br, err := NewBitRing(4, 20)
	require.NoError(t, err)

	// Write 8 bytes - writes 4 bits from each byte
	input := []byte{0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0}
	n, err := br.Write(input)
	require.NoError(t, err)
	assert.Equal(t, 32, n) // 8 bytes * 4 bits = 32 bits

	// Read 8 values (4 bits each)
	output := make([]byte, 8)
	nRead, err := br.Read(output)
	require.NoError(t, err)
	assert.Equal(t, 32, nRead)

	// Verify: each byte is masked to 4 bits
	expected := []byte{0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E, 0x00}
	assert.Equal(t, expected, output)
}

func TestBitRing_WriteReadInterface(t *testing.T) {
	br, err := NewBitRing(3, 10)
	require.NoError(t, err)

	// Write multiple bytes using WriteByte (writes bitLength bits per byte)
	values := []byte{5, 2, 7}
	for _, v := range values {
		err := br.WriteByte(v)
		require.NoError(t, err)
	}

	// Read multiple bytes using interface
	buf := make([]byte, 3)
	nRead, err := br.Read(buf)
	require.NoError(t, err)
	assert.Equal(t, 9, nRead) // 3 values * 3 bits = 9 bits
	assert.Equal(t, values, buf)
}

func TestBitRing_Capacity(t *testing.T) {
	br, err := NewBitRing(3, 5)
	require.NoError(t, err)

	// Fill buffer using WriteByte (writes bitLength bits per value)
	values := []byte{0, 1, 2, 3, 4}
	for _, v := range values {
		err := br.WriteByte(v)
		require.NoError(t, err)
	}

	assert.True(t, br.Full())
	assert.False(t, br.Empty())
	assert.Equal(t, 5, br.Len())
	assert.Equal(t, 0, br.Available())

	// Try to write when full
	err = br.WriteByte(5)
	assert.ErrorIs(t, err, io.ErrShortWrite)
}

func TestBitRing_WriteMoreThanCapacity(t *testing.T) {
	br, err := NewBitRing(3, 5)
	require.NoError(t, err)

	// Write more than capacity using WriteByte
	values := []byte{0, 1, 2, 3, 4, 5, 6}
	for i, v := range values {
		err := br.WriteByte(v)
		if i < 5 {
			require.NoError(t, err)
		} else {
			assert.ErrorIs(t, err, io.ErrShortWrite)
		}
	}

	assert.True(t, br.Full())
}

func TestBitRing_ReadEmpty(t *testing.T) {
	br, err := NewBitRing(3, 5)
	require.NoError(t, err)

	// Read from empty buffer
	value, err := br.ReadByte()
	assert.ErrorIs(t, err, io.EOF)
	assert.Equal(t, byte(0), value)

	buf := make([]byte, 3)
	n, err := br.Read(buf)
	assert.ErrorIs(t, err, io.EOF)
	assert.Equal(t, 0, n)
}

func TestBitRing_WrapAround(t *testing.T) {
	br, err := NewBitRing(3, 5)
	require.NoError(t, err)

	// Fill buffer using WriteByte
	values := []byte{0, 1, 2, 3, 4}
	for _, v := range values {
		err := br.WriteByte(v)
		require.NoError(t, err)
	}

	// Read some values
	buf := make([]byte, 3)
	n, err := br.Read(buf)
	require.NoError(t, err)
	assert.Equal(t, 9, n) // 3 values * 3 bits
	assert.Equal(t, []byte{0, 1, 2}, buf)

	// Write more (should wrap around) using WriteByte
	values2 := []byte{5, 6}
	for _, v := range values2 {
		err := br.WriteByte(v)
		require.NoError(t, err)
	}
	assert.Equal(t, 4, br.Len())

	// Read remaining
	buf = make([]byte, 4)
	n, err = br.Read(buf)
	require.NoError(t, err)
	assert.Equal(t, 12, n) // 4 values * 3 bits
	assert.Equal(t, []byte{3, 4, 5, 6}, buf)
}

func TestBitRing_WriteReadBytes_8Bits(t *testing.T) {
	// Test 8 bits per value - should be byte-aligned
	br, err := NewBitRing(8, 10)
	require.NoError(t, err)

	values := []byte{0, 102, 255, 128}

	// Write using interface
	n, err := br.Write(values)
	require.NoError(t, err)
	assert.Equal(t, 32, n) // 4 bytes * 8 bits = 32 bits

	// Read back
	buf := make([]byte, 4)
	nRead, err := br.Read(buf)
	require.NoError(t, err)
	assert.Equal(t, 32, nRead)
	assert.Equal(t, values, buf)
}

func TestBitRing_WriteReadBytes_8Bits_WrapAround(t *testing.T) {
	br, err := NewBitRing(8, 4)
	require.NoError(t, err)

	// Fill buffer
	values := []byte{0, 1, 2, 3}
	_, err = br.Write(values)
	require.NoError(t, err)

	// Read 2 bytes
	buf := make([]byte, 2)
	_, err = br.Read(buf)
	require.NoError(t, err)
	assert.Equal(t, []byte{0, 1}, buf)

	// Write 2 more (should wrap around)
	values2 := []byte{4, 5}
	_, err = br.Write(values2)
	require.NoError(t, err)

	// Read remaining
	buf = make([]byte, 4)
	_, err = br.Read(buf)
	require.NoError(t, err)
	assert.Equal(t, []byte{2, 3, 4, 5}, buf)
}

func TestBitRing_Reset(t *testing.T) {
	br, err := NewBitRing(3, 10)
	require.NoError(t, err)

	// Write some values using WriteByte
	values := []byte{1, 2, 3}
	for _, v := range values {
		err := br.WriteByte(v)
		require.NoError(t, err)
	}

	br.Reset()

	assert.Equal(t, 0, br.Len())
	assert.Equal(t, 10, br.Available())
	assert.True(t, br.Empty())
	assert.False(t, br.Full())

	// Should be able to write again
	for _, v := range values {
		err := br.WriteByte(v)
		require.NoError(t, err)
	}
}

func TestBitRing_PartialRead(t *testing.T) {
	br, err := NewBitRing(3, 10)
	require.NoError(t, err)

	// Write 2 values using WriteByte
	values := []byte{1, 2}
	for _, v := range values {
		err := br.WriteByte(v)
		require.NoError(t, err)
	}

	// Try to read 5 values
	buf := make([]byte, 5)
	n, err := br.Read(buf)
	assert.ErrorIs(t, err, io.EOF)
	assert.Equal(t, 6, n) // 2 values * 3 bits
	assert.Equal(t, []byte{1, 2, 0, 0, 0}, buf)
}

func TestBitRing_VariousBitLengths(t *testing.T) {
	bitLengths := []int{1, 2, 3, 4, 8}

	for _, bitLen := range bitLengths {
		t.Run(fmt.Sprintf("%d_bits", bitLen), func(t *testing.T) {
			br, err := NewBitRing(bitLen, 10)
			require.NoError(t, err)

			maxValue := byte((1 << bitLen) - 1)
			testValues := []byte{0, maxValue / 4, maxValue / 2, maxValue}

			// Write using WriteByte (writes bitLength bits per value)
			for _, v := range testValues {
				err := br.WriteByte(v)
				require.NoError(t, err)
			}

			// Read back
			buf := make([]byte, len(testValues))
			nRead, err := br.Read(buf)
			require.NoError(t, err)
			assert.Equal(t, len(testValues)*bitLen, nRead)
			// Mask values to expected bit length
			for i := range testValues {
				mask := byte((1 << bitLen) - 1)
				assert.Equal(t, testValues[i]&mask, buf[i]&mask)
			}
		})
	}
}

func TestBitRing_Interface_Compatibility(t *testing.T) {
	// Test that BitRing properly implements Reader[byte] and Writer[byte]
	var _ Writer[byte] = (*BitRing)(nil)
	var _ Reader[byte] = (*BitRing)(nil)

	br, err := NewBitRing(8, 10) // 8 bits = 1 byte per value
	require.NoError(t, err)

	// Write bytes
	bytes := []byte{0x01, 0x02, 0x03, 0xFF}
	n, err := br.Write(bytes)
	require.NoError(t, err)
	assert.Equal(t, 32, n) // 4 bytes * 8 bits = 32 bits

	// Read bytes
	buf := make([]byte, 4)
	nRead, err := br.Read(buf)
	require.NoError(t, err)
	assert.Equal(t, 32, nRead)
	assert.Equal(t, bytes, buf)
}

func TestBitRing_ZeroCapacity(t *testing.T) {
	br, err := NewBitRing(3, 0)
	require.NoError(t, err)

	assert.Equal(t, 0, br.Cap())
	assert.Equal(t, 0, br.Available())
	assert.True(t, br.Empty())
	assert.True(t, br.Full())

	err = br.WriteByte(1)
	assert.ErrorIs(t, err, io.ErrShortWrite)

	_, err = br.ReadByte()
	assert.ErrorIs(t, err, io.EOF)
}

func TestBitRing_ValueMasking(t *testing.T) {
	br, err := NewBitRing(3, 10)
	require.NoError(t, err)

	// Write byte 15 (1111 in binary) - should be masked to 7 (111)
	err = br.WriteByte(15)
	require.NoError(t, err)

	value, err := br.ReadByte()
	require.NoError(t, err)
	assert.Equal(t, byte(7), value)
}

func TestBitRing_ContinuousWriteRead(t *testing.T) {
	br, err := NewBitRing(4, 5)
	require.NoError(t, err)

	// Continuously write and read
	for i := range 20 {
		err := br.WriteByte(byte(i % 16))
		if err != nil {
			// Buffer full, read one value
			_, err := br.ReadByte()
			require.NoError(t, err)
			// Try again
			err = br.WriteByte(byte(i % 16))
			require.NoError(t, err)
		}
	}

	// Read all remaining values
	values := make([]byte, 0, 5)
	for !br.Empty() {
		val, err := br.ReadByte()
		if err == io.EOF {
			break
		}
		require.NoError(t, err)
		values = append(values, val)
	}

	assert.Equal(t, 5, len(values))
	// Last 5 values written should be 15, 0, 1, 2, 3 (i % 16 for i = 15..19)
	expected := []byte{15, 0, 1, 2, 3}
	assert.Equal(t, expected, values)
}

func TestBitRing_WriteReadBytes_4Bit_ConsecutiveBits(t *testing.T) {
	// Verify consecutive bit stream behavior with ring buffer
	br, err := NewBitRing(4, 10)
	require.NoError(t, err)

	// Write single byte: 0xAB = 10101011
	// Write() writes 4 bits from this byte (0xAB & 0x0F = 0x0B = 11)
	n, err := br.Write([]byte{0xAB})
	require.NoError(t, err)
	assert.Equal(t, 4, n) // 1 byte * 4 bits = 4 bits

	// Read 1 value (4 bits): 0x0B = 11
	val1, err := br.ReadByte()
	require.NoError(t, err)
	assert.Equal(t, byte(0x0B), val1) // 0xAB & 0x0F = 0x0B

	// No more values
	_, err = br.ReadByte()
	assert.ErrorIs(t, err, io.EOF)
}

func TestBitRing_WriteReadBytes_4Bit_MultipleBytes(t *testing.T) {
	br, err := NewBitRing(4, 10)
	require.NoError(t, err)

	// Write 2 bytes: 0x12, 0x34
	// Write() writes 4 bits from each byte
	input := []byte{0x12, 0x34}
	n, err := br.Write(input)
	require.NoError(t, err)
	assert.Equal(t, 8, n) // 2 bytes * 4 bits = 8 bits

	// Read 2 values (4 bits each)
	output := make([]byte, 2)
	nRead, err := br.Read(output)
	require.NoError(t, err)
	assert.Equal(t, 8, nRead)

	// Each byte is masked to 4 bits: 0x12 & 0x0F = 0x02, 0x34 & 0x0F = 0x04
	expected := []byte{0x02, 0x04}
	assert.Equal(t, expected, output)
}

func TestBitRing_ReadBit_WriteBit(t *testing.T) {
	br, err := NewBitRing(1, 10)
	require.NoError(t, err)

	// Write individual bits
	err = br.WriteBit(true)
	require.NoError(t, err)
	err = br.WriteBit(false)
	require.NoError(t, err)
	err = br.WriteBit(true)
	require.NoError(t, err)
	err = br.WriteBit(true)
	require.NoError(t, err)

	// Read bits back
	bit, err := br.ReadBit()
	require.NoError(t, err)
	assert.True(t, bit)

	bit, err = br.ReadBit()
	require.NoError(t, err)
	assert.False(t, bit)

	bit, err = br.ReadBit()
	require.NoError(t, err)
	assert.True(t, bit)

	bit, err = br.ReadBit()
	require.NoError(t, err)
	assert.True(t, bit)

	_, err = br.ReadBit()
	assert.ErrorIs(t, err, io.EOF)
}

func TestBitRing_Cap(t *testing.T) {
	br, err := NewBitRing(4, 10)
	require.NoError(t, err)

	assert.Equal(t, 10, br.Cap()) // Cap returns capacity in values
}

// TestBitRing_CapLenBitsBitsLength verifies the meaning of Cap, Len, Bits, and BitsLength
func TestBitRing_CapLenBitsBitsLength(t *testing.T) {
	tests := []struct {
		name      string
		bitLength int
		capacity  int
	}{
		{"1_bit", 1, 10},
		{"2_bits", 2, 10},
		{"4_bits", 4, 10},
		{"8_bits", 8, 10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			br, err := NewBitRing(tt.bitLength, tt.capacity)
			require.NoError(t, err)

			// Initial state
			assert.Equal(t, tt.capacity, br.Cap(), "Cap() should return capacity in values")
			assert.Equal(t, 0, br.Len(), "Len() should return number of values currently stored")
			assert.Equal(t, 0, br.Bits(), "Bits() should return number of bits written (0 values * bitLength = 0)")
			assert.Equal(t, tt.bitLength, br.BitsLength(), "BitsLength() should return bits per value")

			// Write some values
			valuesToWrite := 3
			for i := 0; i < valuesToWrite; i++ {
				err := br.WriteByte(byte(i))
				require.NoError(t, err)
			}

			// After writing
			assert.Equal(t, tt.capacity, br.Cap(), "Cap() should remain constant (capacity in values)")
			assert.Equal(t, valuesToWrite, br.Len(), "Len() should return number of values written")
			assert.Equal(t, valuesToWrite*tt.bitLength, br.Bits(), "Bits() should return bits written (values * bitLength)")
			assert.Equal(t, tt.bitLength, br.BitsLength(), "BitsLength() should remain constant (bits per value)")

			// Read some values
			valuesToRead := 2
			for i := 0; i < valuesToRead; i++ {
				_, err := br.ReadByte()
				require.NoError(t, err)
			}

			// After reading
			remainingValues := valuesToWrite - valuesToRead
			assert.Equal(t, tt.capacity, br.Cap(), "Cap() should remain constant")
			assert.Equal(t, remainingValues, br.Len(), "Len() should return remaining values")
			assert.Equal(t, remainingValues*tt.bitLength, br.Bits(), "Bits() should return bits for remaining values")
			assert.Equal(t, tt.bitLength, br.BitsLength(), "BitsLength() should remain constant")

			// Fill to capacity
			for br.Len() < br.Cap() {
				err := br.WriteByte(0xFF)
				require.NoError(t, err)
			}

			// At capacity
			assert.Equal(t, tt.capacity, br.Cap(), "Cap() should remain constant")
			assert.Equal(t, tt.capacity, br.Len(), "Len() should equal capacity when full")
			assert.Equal(t, tt.capacity*tt.bitLength, br.Bits(), "Bits() should be capacity * bitLength when full")
			assert.Equal(t, tt.bitLength, br.BitsLength(), "BitsLength() should remain constant")
		})
	}
}

func TestBitRing_Init(t *testing.T) {
	br := &BitRing{}

	// Initialize with specific parameters
	err := br.Init(4, 0, 16)
	require.NoError(t, err)

	assert.Equal(t, 4, br.BitsLength()) // BitsLength returns bits per value
	assert.Equal(t, 0, br.Bits())      // Bits returns bits written (0 values * 4 bits = 0)
	assert.Equal(t, 0, br.Len())        // Len returns number of values
}

func TestBitRing_Init_Invalid(t *testing.T) {
	br := &BitRing{}

	// Invalid bitLength
	err := br.Init(0, 0, 16)
	assert.Error(t, err)

	err = br.Init(9, 0, 16)
	assert.Error(t, err)

	// Invalid length/capacity
	err = br.Init(4, -1, 16)
	assert.Error(t, err)

	err = br.Init(4, 10, 5)
	assert.Error(t, err)
}

func TestBitRing_Grow(t *testing.T) {
	br, err := NewBitRing(4, 5)
	require.NoError(t, err)

	// Fill buffer
	values := []byte{0, 1, 2, 3, 4}
	_, err = br.Write(values)
	require.NoError(t, err)
	assert.True(t, br.Full())

	// Grow buffer
	oldCap := br.Cap()
	br.Grow(5)
	assert.Greater(t, br.Cap(), oldCap)
	assert.Equal(t, 5, br.Len()) // Existing data preserved

	// Should be able to write more
	err = br.WriteByte(5)
	require.NoError(t, err)
	assert.Equal(t, 6, br.Len())
}
