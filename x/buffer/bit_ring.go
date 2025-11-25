package buffer

import (
	"io"
)

var _ Writer[byte] = (*BitRing)(nil)
var _ Reader[byte] = (*BitRing)(nil)
var _ BitsReader = (*BitRing)(nil)
var _ BitsWriter = (*BitRing)(nil)
var _ Initializer = (*BitRing)(nil)

// BitRing is a circular buffer for bit-level data with a fixed capacity.
// Works on bytes only - bits are packed into bytes (8 bits per byte).
type BitRing struct {
	// data is the underlying byte slice with bits packed (8 bits per byte)
	data []byte

	// bitLength is the number of bits per value (1-8)
	bitLength int

	// capacity is the number of values the ring can hold
	capacity int

	// totalBits is the total bit capacity (capacity * bitLength)
	totalBits int

	// headPos is the bit position where the next write will occur (0-based)
	headPos int

	// tailPos is the bit position where the next read will occur (0-based)
	tailPos int

	// len is the number of values currently stored
	len int
}

// NewBitRing creates a new BitRing with the specified bit length per value and capacity in values.
//
// Parameters:
//   - bitLength: number of bits per value (1-8)
//   - capacity: number of values the ring can hold
//
// Returns:
//   - BitRing buffer initialized with the given parameters
//   - error if parameters are invalid
func NewBitRing(bitLength, capacity int) (*BitRing, error) {
	br := &BitRing{}

	if bitLength < 1 || bitLength > 8 {
		return nil, &InitError{Message: "bitLength must be between 1 and 8"}
	}
	if capacity < 0 {
		return nil, &InitError{Message: "capacity must be non-negative"}
	}

	totalBits := capacity * bitLength
	bytesNeeded := (totalBits + 7) >> 3

	if err := br.Init(bitLength, 0, bytesNeeded); err != nil {
		return nil, err
	}

	br.capacity = capacity
	br.totalBits = totalBits

	return br, nil
}

// Init initializes the ring buffer with the given parameters.
// For BitRing buffer, numBits is the bitLength per value.
// Returns an error if parameters are invalid.
func (br *BitRing) Init(numBits, bufLen, bufCap int) error {
	if numBits < 1 || numBits > 8 {
		return &InitError{Message: "bitLength must be between 1 and 8"}
	}
	if bufLen < 0 {
		return &InitError{Message: "length must be non-negative"}
	}
	if bufCap < bufLen {
		return &InitError{Message: "capacity must be >= length"}
	}

	br.bitLength = numBits
	if bufCap > 0 {
		br.data = make([]byte, bufCap)
	}
	br.headPos = 0
	br.tailPos = 0
	br.len = 0

	return nil
}

// Bits returns the number of bits per value.
func (br *BitRing) BitsLength() int {
	return br.bitLength
}

// Bits returns the number of bits written.
func (br *BitRing) Bits() int {
	return br.len * br.bitLength
}

// WriteBit writes a single bit to the ring buffer.
// Returns an error if the buffer is full.
func (br *BitRing) WriteBit(bit bool) error {
	if br.len >= br.capacity {
		return io.ErrShortWrite
	}

	br.writeBitAt(br.headPos, bit)
	br.headPos = (br.headPos + 1) % br.totalBits
	br.len++

	return nil
}

// WriteByte writes a byte value using bitLength bits to the ring buffer.
// Returns an error if the buffer is full.
func (br *BitRing) WriteByte(value byte) error {
	if br.len >= br.capacity {
		return io.ErrShortWrite
	}

	mask := byte((1 << br.bitLength) - 1)
	value &= mask

	pos := br.headPos
	for i := br.bitLength - 1; i >= 0; i-- {
		bit := (value >> uint(i)) & 1
		br.writeBitAt(pos, bit != 0)
		pos = (pos + 1) % br.totalBits
	}

	br.headPos = (br.headPos + br.bitLength) % br.totalBits
	br.len++

	return nil
}

// Write implements Writer[byte] interface by writing bytes as a bit stream.
// Each byte is written using bitLength bits (calls WriteByte in a loop).
// Returns the number of bits written and any error encountered.
func (br *BitRing) Write(p []byte) (n int, err error) {
	for _, b := range p {
		if err := br.WriteByte(b); err != nil {
			return n, err
		}
		n += br.bitLength
	}
	return n, nil
}

// ReadBit reads a single bit from the ring buffer.
// Returns the bit and an error if the buffer is empty.
func (br *BitRing) ReadBit() (bool, error) {
	if br.len == 0 {
		return false, io.EOF
	}

	bit := br.readBitAt(br.tailPos)
	br.tailPos = (br.tailPos + 1) % br.totalBits
	br.len--

	return bit, nil
}

// ReadByte reads a byte value using bitLength bits from the ring buffer.
// Returns the value and an error if the buffer is empty.
func (br *BitRing) ReadByte() (byte, error) {
	if br.len == 0 {
		return 0, io.EOF
	}

	var value byte
	pos := br.tailPos
	for range br.bitLength {
		value <<= 1
		if br.readBitAt(pos) {
			value |= 1
		}
		pos = (pos + 1) % br.totalBits
	}

	br.tailPos = (br.tailPos + br.bitLength) % br.totalBits
	br.len--

	return value, nil
}

// Read implements Reader[byte] interface by reading bytes as a bit stream.
// Each byte is read using bitLength bits.
// Returns the number of bits read and io.EOF if fewer bytes were read than requested.
func (br *BitRing) Read(p []byte) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}
	if br.len == 0 {
		return 0, io.EOF
	}

	valuesRead := 0
	for i := range p {
		if br.len == 0 {
			if valuesRead > 0 {
				return valuesRead * br.bitLength, io.EOF
			}
			return 0, io.EOF
		}

		value, err := br.ReadByte()
		if err != nil {
			if valuesRead > 0 {
				return valuesRead * br.bitLength, err
			}
			return 0, err
		}
		p[i] = value
		valuesRead++
	}

	return valuesRead * br.bitLength, nil
}

// writeBitAt writes a single bit at the specified bit position (with wrapping).
func (br *BitRing) writeBitAt(bitPos int, bit bool) {
	byteIdx := (bitPos >> 3) % len(br.data)
	bitOffset := bitPos & 7
	bitIdx := 7 - bitOffset // MSB first within byte

	if bit {
		br.data[byteIdx] |= 1 << bitIdx
	} else {
		br.data[byteIdx] &^= 1 << bitIdx
	}
}

// readBitAt reads a single bit at the specified bit position (with wrapping).
func (br *BitRing) readBitAt(bitPos int) bool {
	byteIdx := (bitPos >> 3) % len(br.data)
	bitOffset := bitPos & 7
	bitIdx := 7 - bitOffset // MSB first within byte

	return (br.data[byteIdx]>>bitIdx)&1 != 0
}

// Len returns the number of values currently in the buffer.
func (br *BitRing) Len() int {
	return br.len
}

// Cap returns the capacity of the ring buffer in values.
func (br *BitRing) Cap() int {
	return br.capacity
}

// Available returns the number of values that can be written without overwriting unread data.
func (br *BitRing) Available() int {
	return br.capacity - br.len
}

// Full returns true if the buffer is full (no space available).
func (br *BitRing) Full() bool {
	return br.len >= br.capacity
}

// Empty returns true if the buffer is empty.
func (br *BitRing) Empty() bool {
	return br.len == 0
}

// Grow grows the buffer's capacity to guarantee space for n more values.
// After Grow(n), at least n values can be written without another allocation.
// Note: Growing a ring buffer requires reallocating and copying existing data.
func (br *BitRing) Grow(n int) {
	if n < 0 {
		panic("BitRing.Grow: negative count")
	}
	if n == 0 {
		return
	}

	if br.Available() >= n {
		return // Already have enough space
	}

	// Calculate new capacity and total bits
	newCapacity := br.capacity + n
	newTotalBits := newCapacity * br.bitLength
	newBytesNeeded := (newTotalBits + 7) >> 3

	// Allocate new buffer
	newBuf := make([]byte, newBytesNeeded)

	// Copy existing data preserving bit order
	if br.len > 0 {
		valuesToCopy := br.len
		readPos := br.tailPos

		// Copy bits value by value
		for i := range valuesToCopy {
			// Read value from old buffer
			var value byte
			pos := readPos
			for range br.bitLength {
				value <<= 1
				if br.readBitAt(pos) {
					value |= 1
				}
				pos = (pos + 1) % br.totalBits
			}

			// Write value to new buffer at position i
			writePos := i * br.bitLength
			val := value
			mask := byte((1 << br.bitLength) - 1)
			val &= mask

			for j := br.bitLength - 1; j >= 0; j-- {
				bit := (val >> uint(j)) & 1
				byteIdx := (writePos >> 3) % len(newBuf)
				bitOffset := writePos & 7
				bitIdx := 7 - bitOffset
				if bit != 0 {
					newBuf[byteIdx] |= 1 << bitIdx
				} else {
					newBuf[byteIdx] &^= 1 << bitIdx
				}
				writePos++
			}

			readPos = (readPos + br.bitLength) % br.totalBits
		}
	}

	// Update buffer and positions
	br.data = newBuf
	br.capacity = newCapacity
	br.totalBits = newTotalBits
	br.tailPos = 0
	br.headPos = br.len * br.bitLength
}

// Reset clears the buffer by resetting head, tail positions and length.
func (br *BitRing) Reset() {
	br.headPos = 0
	br.tailPos = 0
	br.len = 0
}
