package buffer

import "github.com/itohio/EasyRobot/x/types"

// Bits provides unified bit-level reading and writing with shared state.
// Can be instantiated from a pre-existing slice to minimize allocations.
// Maintains a single bit buffer with a shared position for both reading and writing.
// Bits are packed into bytes (8 bits per byte).
// Generic over integer types for efficient type-specific operations.
type Bits[T types.Unsigned] struct {
	// data is the underlying byte slice with bits packed (8 bits per byte)
	// Can be provided externally or allocated internally
	data []byte

	// pos is the shared bit position for both reading and writing (0-based)
	// byte index = pos / 8, bit index within byte = pos % 8
	pos int

	// writtenBits is the total number of bits written (never decreases, except on Clear)
	// Used to determine how many bits are available for reading
	writtenBits int

	// bitLength is the number of bits per value (1-64)
	bitLength int
}

// NewBitBuffer creates a new BitBuffer.
// Can be initialized with a pre-existing byte slice (varargs) to minimize allocations.
// If bytes are provided, they are used directly (no copy) - modifications will affect the original slice.
// If no bytes are provided, an internal buffer is allocated.
//
// Parameters:
//   - bitLength: number of bits per value (1-64)
//   - data: optional pre-existing byte slice (bits are packed, 8 bits per byte)
func NewBits[T types.Unsigned](bitLength int, data ...byte) Bits[T] {
	if bitLength < 1 || bitLength > 64 {
		panic("bitLength must be between 1 and 64")
	}

	var buf []byte
	var writtenBits int
	if len(data) > 0 {
		// Use provided slice directly (no copy)
		// Assume all bits in provided data are written
		buf = data
		writtenBits = len(data) << 3
	} else {
		// Allocate internal buffer
		buf = make([]byte, 0, 8) // Pre-allocate capacity for 8 bytes (64 bits)
		writtenBits = 0
	}

	return Bits[T]{
		data:       buf,
		pos:        0,
		writtenBits: writtenBits,
		bitLength:  bitLength,
	}
}

// Grow grows the buffer's capacity to guarantee space for n more bits.
// After Grow(n), at least n bits can be written without another allocation.
func (bb *Bits[T]) Grow(n int) {
	if n < 0 {
		panic("BitBuffer.Grow: negative count")
	}
	// Calculate bytes needed (round up)
	bytesNeeded := (bb.pos + n + 7) >> 3
	if bytesNeeded > cap(bb.data) {
		// Double the capacity, but ensure we have at least n more bits
		newCap := 2*cap(bb.data) + ((n + 7) >> 3)
		if newCap < 4 {
			newCap = 4
		}
		newBuf := make([]byte, len(bb.data), newCap)
		copy(newBuf, bb.data)
		bb.data = newBuf
	}
}

// Write writes a value of bitLength bits to the buffer.
// Values are written MSB first (most significant bit first).
// Values larger than what can fit in bitLength are masked.
// Writing advances the shared position.
func (bb *Bits[T]) Write(value T) {
	// Convert to uint64 for bit operations
	val := uint64(value)

	// Mask value to fit in bitLength bits
	mask := uint64((1 << uint(bb.bitLength)) - 1)
	val &= mask

	// Calculate bytes needed
	bitsNeeded := bb.pos + bb.bitLength
	bytesNeeded := (bitsNeeded + 7) >> 3
	if bytesNeeded > len(bb.data) {
		bb.Grow(bb.bitLength)
		bb.data = bb.data[:bytesNeeded]
	}

	// Write bits MSB first, packing into bytes
	// Hot path: use bit shifts and masks instead of division/modulo
	byteIdx := bb.pos >> 3
	bitOffset := bb.pos & 7
	for i := bb.bitLength - 1; i >= 0; i-- {
		bit := (val >> uint(i)) & 1
		bitIdx := 7 - bitOffset // MSB first within byte
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
	// Update writtenBits to track maximum position reached
	if bb.pos > bb.writtenBits {
		bb.writtenBits = bb.pos
	}
}

// Read reads a value of bitLength bits from the buffer.
// Returns the value and true if a complete value was read, false otherwise.
// Values are read MSB first.
// Reading advances the shared position.
// For signed types, sign extension is applied if the MSB is set.
func (bb *Bits[T]) Read() (T, bool) {
	// Check if we have enough bits available (use writtenBits, not len(data))
	bitsAvailable := bb.writtenBits - bb.pos
	if bitsAvailable < bb.bitLength {
		var zero T
		return zero, false
	}

	// Read the value as uint64 first, unpacking from bytes
	// Hot path: use bit shifts and masks instead of division/modulo
	var value uint64
	byteIdx := bb.pos >> 3
	bitOffset := bb.pos & 7
	for i := 0; i < bb.bitLength; i++ {
		value <<= 1
		bitIdx := 7 - bitOffset // MSB first within byte
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

	// Convert to type T
	// For signed types, apply sign extension if MSB is set
	var result T
	var zero T
	switch any(zero).(type) {
	case int8, int16, int32, int64, int:
		// Signed type - apply sign extension
		msbMask := uint64(1 << uint(bb.bitLength-1))
		if (value & msbMask) != 0 {
			// Sign extend: set all bits above bitLength
			signExtendMask := ^uint64((1 << uint(bb.bitLength)) - 1)
			value |= signExtendMask
		}
		// Convert via int64 to handle all signed types
		result = T(int64(value))
	default:
		// Unsigned type - direct conversion
		result = T(value)
	}

	return result, true
}

// HasValue checks if there's at least one complete value available for reading.
func (bb *Bits[T]) HasValue() bool {
	bitsAvailable := bb.writtenBits - bb.pos
	return bitsAvailable >= bb.bitLength
}

// GetRemainingBits returns the number of remaining bits available for reading.
func (bb *Bits[T]) GetRemainingBits() int {
	bitsAvailable := bb.writtenBits - bb.pos
	if bitsAvailable < 0 {
		return 0
	}
	return bitsAvailable
}

// GetWrittenBits returns the total number of bits written.
func (bb *Bits[T]) GetWrittenBits() int {
	return bb.writtenBits
}

// Position returns the current shared position.
func (bb *Bits[T]) Position() int {
	return bb.pos
}

// SetPosition sets the shared position.
func (bb *Bits[T]) SetPosition(pos int) {
	if pos < 0 || pos > bb.writtenBits {
		panic("position out of bounds")
	}
	bb.pos = pos
}

// Bytes returns the underlying byte slice.
// Bits are packed (8 bits per byte).
// The returned slice is a view into the internal buffer - modifications will affect the buffer.
func (bb *Bits[T]) Bytes() []byte {
	return bb.data
}

// Reset resets the position to the beginning of the buffer.
// Allows re-reading all written data.
func (bb *Bits[T]) Reset() {
	bb.pos = 0
}

// Clear clears all written data and resets the position.
func (bb *Bits[T]) Clear() {
	bb.pos = 0
	bb.writtenBits = 0
	bb.data = bb.data[:0]
}

// Compact removes consumed bits from the beginning of the buffer.
// Useful to free memory when many bits have been read.
// Only compacts if more than half the buffer has been consumed.
func (bb *Bits[T]) Compact() {
	bytesConsumed := bb.pos >> 3
	if bytesConsumed > 0 && bytesConsumed > len(bb.data)>>1 {
		// Shift remaining bytes to the beginning
		bitsRemaining := bb.pos & 7
		if bitsRemaining == 0 {
			// Byte-aligned, simple case
			copy(bb.data, bb.data[bytesConsumed:])
			bb.data = bb.data[:len(bb.data)-bytesConsumed]
			bb.pos = 0
		} else {
			// Not byte-aligned, need to shift bits
			// This is complex - for now, only compact if byte-aligned
			// TODO: implement bit-level shifting for non-aligned case
		}
	}
}

// ToBytes returns all written bytes.
// If the position is not byte-aligned, the last byte includes padding bits (zeros).
// Does not modify the position.
func (bb *Bits[T]) ToBytes() []byte {
	if len(bb.data) == 0 {
		return []byte{}
	}

	// Copy all complete bytes
	result := make([]byte, len(bb.data))
	copy(result, bb.data)

	// If position is not byte-aligned, we need to handle the last byte
	// The last byte may have unwritten bits that should be zero
	bitsWritten := bb.writtenBits
	completeBytes := bitsWritten >> 3
	remainingBits := bitsWritten & 7

	if remainingBits > 0 && completeBytes < len(bb.data) {
		// Clear unwritten bits in the last byte
		mask := byte(0xFF) << (8 - remainingBits)
		result[completeBytes] &= mask
	}

	if remainingBits > 0 {
		return result[:completeBytes+1]
	}
	return result[:completeBytes]
}

// ReadFromBytes appends bytes to the buffer.
// Bits are unpacked and written starting at the current position.
// Useful for reading from io.Reader or existing byte slices.
func (bb *Bits[T]) ReadFromBytes(src []byte) {
	if len(src) == 0 {
		return
	}

	// Calculate bytes needed
	bitsNeeded := bb.pos + (len(src) << 3)
	bytesNeeded := (bitsNeeded + 7) >> 3
	if bytesNeeded > len(bb.data) {
		bb.Grow(len(src) << 3)
		bb.data = bb.data[:bytesNeeded]
	}

	// Write bytes directly (they're already packed)
	byteIdx := bb.pos >> 3
	bitOffset := bb.pos & 7

	if bitOffset == 0 {
		// Byte-aligned, simple case
		copy(bb.data[byteIdx:], src)
		bb.pos += len(src) << 3
	} else {
		// Not byte-aligned, need to shift bits
		shiftRight := bitOffset
		shiftLeft := 8 - bitOffset
		for _, b := range src {
			// Write byte with bit shifting
			bb.data[byteIdx] |= b >> shiftRight
			byteIdx++
			if byteIdx < len(bb.data) {
				bb.data[byteIdx] = b << shiftLeft
			}
		}
		bb.pos += len(src) << 3
	}
	// Update writtenBits to track maximum position reached
	if bb.pos > bb.writtenBits {
		bb.writtenBits = bb.pos
	}
}
