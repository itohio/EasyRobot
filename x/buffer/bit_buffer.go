package buffer

import "io"

var _ Writer[byte] = (*Bits)(nil)
var _ Reader[byte] = (*Bits)(nil)
var _ BitsReader = (*Bits)(nil)
var _ BitsWriter = (*Bits)(nil)
var _ Initializer = (*Bits)(nil)

// Bits provides unified bit-level reading and writing with shared state.
// Works on bytes only - bits are packed into bytes (8 bits per byte).
// Maintains a single bit buffer with a shared position for both reading and writing.
type Bits struct {
	// data is the underlying byte slice with bits packed (8 bits per byte)
	data []byte

	// pos is the shared bit position for both reading and writing (0-based)
	// byte index = pos / 8, bit index within byte = pos % 8
	pos int

	// writtenBits is the total number of bits written (never decreases, except on Clear)
	// Used to determine how many bits are available for reading
	writtenBits int

	// bitLength is the number of bits per value (1-8)
	bitLength int
}

// NewBits creates a new Bits buffer.
// Can be initialized with a pre-existing byte slice (varargs) to minimize allocations.
// If bytes are provided, they are used directly (no copy) - modifications will affect the original slice.
// If no bytes are provided, an internal buffer is allocated.
//
// Parameters:
//   - bitLength: number of bits per value (1-8)
//   - data: optional pre-existing byte slice (bits are packed, 8 bits per byte)
//
// Returns:
//   - Bits buffer initialized with the given parameters
//   - error if bitLength is invalid or initialization fails
func NewBits(bitLength int, data ...byte) (*Bits, error) {
	bb := &Bits{}

	var bufLen, bufCap int
	var writtenBits int
	if len(data) > 0 {
		bufLen = len(data)
		bufCap = cap(data)
		writtenBits = len(data) << 3
	} else {
		bufLen = 0
		bufCap = 8 // Pre-allocate capacity for 8 bytes (64 bits)
		writtenBits = 0
	}

	if err := bb.Init(bitLength, bufLen, bufCap); err != nil {
		return nil, err
	}

	if len(data) > 0 {
		bb.writtenBits = writtenBits
		bb.data = data
	}

	return bb, nil
}

// Bits returns the number of bits per value.
func (bb *Bits) BitsLength() int {
	return bb.bitLength
}

// Length returns the number of bits written.
func (bb *Bits) Bits() int {
	return bb.writtenBits
}

// Init initializes the buffer with the given number of bits, length, and capacity.
// Returns an error if numBits is invalid (must be between 1 and 8).
func (bb *Bits) Init(numBits, bufLen, bufCap int) error {
	if numBits < 1 || numBits > 8 {
		return &InitError{Message: "bitLength must be between 1 and 8"}
	}
	if bufLen < 0 {
		return &InitError{Message: "length must be non-negative"}
	}
	if bufCap < bufLen {
		return &InitError{Message: "capacity must be >= length"}
	}

	bb.bitLength = numBits
	if bufCap > 0 {
		bb.data = make([]byte, bufLen, bufCap)
	}
	bb.pos = 0
	bb.writtenBits = 0
	return nil
}

// Grow grows the buffer's capacity to guarantee space for n more bits.
func (bb *Bits) Grow(n int) {
	if n < 0 {
		panic("Bits.Grow: negative count")
	}
	bytesNeeded := (bb.pos + n + 7) >> 3
	if bytesNeeded > cap(bb.data) {
		newCap := 2*cap(bb.data) + ((n + 7) >> 3)
		if newCap < 4 {
			newCap = 4
		}
		newBuf := make([]byte, len(bb.data), newCap)
		copy(newBuf, bb.data)
		bb.data = newBuf
	}
}

// WriteBit writes a single bit to the buffer.
func (bb *Bits) WriteBit(bit bool) {
	bitsNeeded := bb.pos + 1
	bytesNeeded := (bitsNeeded + 7) >> 3
	if bytesNeeded > len(bb.data) {
		bb.Grow(1)
		bb.data = bb.data[:bytesNeeded]
	}

	byteIdx := bb.pos >> 3
	bitOffset := bb.pos & 7
	bitIdx := 7 - bitOffset // MSB first within byte

	if bit {
		bb.data[byteIdx] |= 1 << bitIdx
	} else {
		bb.data[byteIdx] &^= 1 << bitIdx
	}

	bb.pos++
	if bb.pos > bb.writtenBits {
		bb.writtenBits = bb.pos
	}
}

// WriteByte writes a byte value using bitLength bits.
func (bb *Bits) WriteByte(value byte) {
	mask := byte((1 << bb.bitLength) - 1)
	value &= mask

	bitsNeeded := bb.pos + bb.bitLength
	bytesNeeded := (bitsNeeded + 7) >> 3
	if bytesNeeded > len(bb.data) {
		bb.Grow(bb.bitLength)
		bb.data = bb.data[:bytesNeeded]
	}

	byteIdx := bb.pos >> 3
	bitOffset := bb.pos & 7

	// Write bits MSB first
	for i := bb.bitLength - 1; i >= 0; i-- {
		bit := (value >> uint(i)) & 1
		bitIdx := 7 - bitOffset
		if bit != 0 {
			bb.data[byteIdx] |= 1 << bitIdx
		} else {
			bb.data[byteIdx] &^= 1 << bitIdx
		}
		bitOffset++
		if bitOffset == 8 {
			bitOffset = 0
			byteIdx++
		}
		bb.pos++
	}

	if bb.pos > bb.writtenBits {
		bb.writtenBits = bb.pos
	}
}

// Write implements Writer[byte] interface by writing bytes as a bit stream.
// Each byte is written using bitLength bits (calls WriteByte in a loop).
// Returns the number of bits written and any error encountered.
func (bb *Bits) Write(p []byte) (n int, err error) {
	for _, b := range p {
		bb.WriteByte(b)
		n += bb.bitLength
	}
	return n, nil
}

// ReadBit reads a single bit from the buffer.
// Returns the bit and true if a bit was read, false otherwise.
func (bb *Bits) ReadBit() (bool, bool) {
	if bb.pos >= bb.writtenBits {
		return false, false
	}

	byteIdx := bb.pos >> 3
	bitOffset := bb.pos & 7
	bitIdx := 7 - bitOffset // MSB first within byte

	bit := (bb.data[byteIdx]>>bitIdx)&1 != 0
	bb.pos++

	return bit, true
}

// ReadByte reads a byte value using bitLength bits from the buffer.
// Returns the value and true if a complete value was read, false otherwise.
func (bb *Bits) ReadByte() (byte, bool) {
	bitsAvailable := bb.writtenBits - bb.pos
	if bitsAvailable < bb.bitLength {
		return 0, false
	}

	var value byte
	byteIdx := bb.pos >> 3
	bitOffset := bb.pos & 7

	// Read bits MSB first
	for range bb.bitLength {
		value <<= 1
		bitIdx := 7 - bitOffset
		if (bb.data[byteIdx]>>bitIdx)&1 != 0 {
			value |= 1
		}
		bitOffset++
		if bitOffset == 8 {
			bitOffset = 0
			byteIdx++
		}
		bb.pos++
	}

	return value, true
}

// Read implements Reader[byte] interface by reading bytes as a bit stream.
// Each byte is read using bitLength bits (calls ReadByte in a loop).
// Returns the number of bits read and io.EOF if fewer bytes were read than requested.
func (bb *Bits) Read(p []byte) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}

	valuesRead := 0
	for i := range p {
		value, ok := bb.ReadByte()
		if !ok {
			if valuesRead > 0 {
				return valuesRead * bb.bitLength, io.EOF
			}
			return 0, io.EOF
		}
		p[i] = value
		valuesRead++
	}

	return valuesRead * bb.bitLength, nil
}

// HasValue checks if there's at least one complete value available for reading.
func (bb *Bits) HasValue() bool {
	bitsAvailable := bb.writtenBits - bb.pos
	return bitsAvailable >= bb.bitLength
}

// GetRemainingBits returns the number of remaining bits available for reading.
func (bb *Bits) GetRemainingBits() int {
	bitsAvailable := bb.writtenBits - bb.pos
	if bitsAvailable < 0 {
		return 0
	}
	return bitsAvailable
}

// GetWrittenBits returns the total number of bits written.
func (bb *Bits) GetWrittenBits() int {
	return bb.writtenBits
}

// Position returns the current shared position.
func (bb *Bits) Position() int {
	return bb.pos
}

// SetPosition sets the shared position.
func (bb *Bits) SetPosition(pos int) error {
	if pos < 0 || pos > bb.writtenBits {
		return &InitError{Message: "position out of bounds"}
	}
	bb.pos = pos
	return nil
}

// Cap returns the capacity of the buffer's underlying byte slice.
func (bb *Bits) Cap() int {
	return cap(bb.data)
}

func (bb *Bits) Len() int {
	return len(bb.data)
}

// Bytes returns the underlying byte slice.
// Bits are packed (8 bits per byte).
// The returned slice is a view into the internal buffer - modifications will affect the buffer.
func (bb *Bits) Bytes() []byte {
	return bb.data
}

// Reset resets the position to the beginning of the buffer.
func (bb *Bits) Reset() {
	bb.pos = 0
}

// Clear clears all written data and resets the position.
func (bb *Bits) Clear() {
	bb.pos = 0
	bb.writtenBits = 0
	bb.data = bb.data[:0]
}

// Compact removes consumed bits from the beginning of the buffer.
// Only compacts if more than half the buffer has been consumed.
func (bb *Bits) Compact() {
	bytesConsumed := bb.pos >> 3
	if bytesConsumed > 0 && bytesConsumed > len(bb.data)>>1 {
		bitsRemaining := bb.pos & 7
		if bitsRemaining == 0 {
			copy(bb.data, bb.data[bytesConsumed:])
			bb.data = bb.data[:len(bb.data)-bytesConsumed]
			bitsConsumed := bytesConsumed << 3
			bb.pos = 0
			bb.writtenBits -= bitsConsumed
		}
	}
}

// ToBytes returns all written bytes.
// If the position is not byte-aligned, the last byte includes padding bits (zeros).
func (bb *Bits) ToBytes() []byte {
	if len(bb.data) == 0 {
		return []byte{}
	}

	result := make([]byte, len(bb.data))
	copy(result, bb.data)

	bitsWritten := bb.writtenBits
	completeBytes := bitsWritten >> 3
	remainingBits := bitsWritten & 7

	if remainingBits > 0 && completeBytes < len(bb.data) {
		mask := byte(0xFF) << (8 - remainingBits)
		result[completeBytes] &= mask
	}

	if remainingBits > 0 {
		return result[:completeBytes+1]
	}
	return result[:completeBytes]
}

// ReadFromBytes appends bytes to the buffer as a bit stream.
// Bits are unpacked and written starting at the current position.
func (bb *Bits) ReadFromBytes(src []byte) {
	if len(src) == 0 {
		return
	}

	bitsNeeded := bb.pos + (len(src) << 3)
	bytesNeeded := (bitsNeeded + 7) >> 3
	if bytesNeeded > len(bb.data) {
		bb.Grow(len(src) << 3)
		bb.data = bb.data[:bytesNeeded]
	}

	byteIdx := bb.pos >> 3
	bitOffset := bb.pos & 7

	if bitOffset == 0 {
		copy(bb.data[byteIdx:], src)
		bb.pos += len(src) << 3
	} else {
		shiftRight := bitOffset
		shiftLeft := 8 - bitOffset
		for _, b := range src {
			bb.data[byteIdx] |= b >> shiftRight
			byteIdx++
			if byteIdx < len(bb.data) {
				bb.data[byteIdx] = b << shiftLeft
			}
		}
		bb.pos += len(src) << 3
	}

	if bb.pos > bb.writtenBits {
		bb.writtenBits = bb.pos
	}
	bb.pos = 0
}
