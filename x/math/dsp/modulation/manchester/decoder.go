package manchester

// Decoder handles Manchester line coding decoding.
// Decodes Manchester-encoded bytes back to original bytes.
type Decoder struct {
	// IEEE 802.3 variant: if true, 0 = high-to-low (10), 1 = low-to-high (01)
	ieee8023 bool
	// Lookup table reference (16 bytes)
	decodeTable *[16]byte
	// Bit alignment offset (0-7) for handling misaligned signals
	bitOffset int
}

// NewDecoder creates a new Manchester decoder with standard encoding.
// Standard: 0 = 01, 1 = 10
func NewDecoder() *Decoder {
	return &Decoder{
		ieee8023:    false,
		decodeTable: &decodeTable4Std,
		bitOffset:   0,
	}
}

// NewDecoderIEEE8023 creates a new Manchester decoder with IEEE 802.3 encoding.
// IEEE 802.3: 0 = 10, 1 = 01
func NewDecoderIEEE8023() *Decoder {
	return &Decoder{
		ieee8023:    true,
		decodeTable: &decodeTable4IEEE,
		bitOffset:   0,
	}
}

// SetBitOffset sets the bit alignment offset for misaligned signals.
// Offset range: 0-7 bits. This allows decoding when the signal doesn't start
// at a byte boundary. The offset is applied to the input stream before decoding.
//
// How Misalignment Works:
//
// Manchester encoding guarantees a transition in the middle of each bit period,
// which helps with clock recovery. However, if decoding starts at the wrong bit
// position (misaligned), the decoder may interpret Manchester symbols incorrectly.
//
// Example of misalignment:
//   - Correct alignment: [01][10][01][10]... (Manchester symbols)
//   - Misaligned by 1 bit: [1][01][10][01]... (starts in middle of symbol)
//
// The bitOffset parameter tells the decoder to skip the first N bits before
// starting to decode Manchester symbols. This is useful when:
//  1. The signal stream doesn't start at a byte boundary
//  2. Clock recovery has identified the correct symbol boundary
//  3. Processing partial byte streams
//
// Note: For best results, use clock recovery/synchronization to find the correct
// symbol boundary before decoding. The bitOffset is a simple mechanism for
// handling known misalignments.
func (m *Decoder) SetBitOffset(offset int) {
	if offset < 0 || offset > 7 {
		offset = 0
	}
	m.bitOffset = offset
}

// Decode decodes Manchester-encoded bytes to original bytes using destination pattern.
// Each 2 bits in input represent 1 bit in output.
// Input: []byte of Manchester-encoded data
// Output: []byte of decoded data (half the size of input)
// Returns the number of bytes written to dst.
// Optimized using 4-bit lookup tables (16 bytes per table, 64 bytes total).
// Decode implements modulation.ByteDecoder interface
func (m *Decoder) Decode(dst, src []byte) int {
	if len(src) == 0 {
		return 0
	}

	requiredLen := (len(src) + 1) / 2 // Round up
	if dst == nil {
		return requiredLen
	}

	if len(dst) < requiredLen {
		return 0
	}

	dstIdx := 0
	srcIdx := 0
	var currentByte byte
	bitPos := 7 // Start at MSB

	// Handle bit offset for misaligned signals
	if m.bitOffset > 0 {
		srcIdx, dstIdx, currentByte, bitPos = m.decodeWithBitOffset(dst, src, srcIdx, dstIdx, currentByte, bitPos)
		if dstIdx >= len(dst) {
			return dstIdx
		}
	}

	// Fast path: process bytes when aligned (bitOffset == 0)
	if m.bitOffset == 0 && bitPos == 7 {
		srcIdx, dstIdx = m.decodeFastPath(dst, src, srcIdx, dstIdx)
		if dstIdx >= len(dst) {
			return dstIdx
		}
	}

	// Continue with remaining bytes using lookup table
	srcIdx, dstIdx, currentByte, bitPos = m.decodeRemaining(dst, src, srcIdx, dstIdx, currentByte, bitPos)

	// Write final byte if complete
	if bitPos < 7 && dstIdx < len(dst) {
		dst[dstIdx] = currentByte
		dstIdx++
	}

	return dstIdx
}

// decodeWithBitOffset handles decoding with bit alignment offset for misaligned signals.
func (m *Decoder) decodeWithBitOffset(dst, src []byte, srcIdx, dstIdx int, currentByte byte, bitPos int) (int, int, byte, int) {
	firstByte := src[srcIdx]
	srcIdx++
	bitOffset := m.bitOffset

	// Pre-compute shift amounts to avoid uint() casts in hot path
	shift7 := uint(7)
	shift6 := uint(6)

	for bitOffset < 8 && srcIdx < len(src) {
		if bitOffset+1 >= 8 {
			// Need to get second bit from next byte
			if srcIdx >= len(src) {
				break
			}
			bit1 := (firstByte >> (shift7 - uint(bitOffset))) & 1
			bit2 := (src[srcIdx] >> shift7) & 1
			bitOffset = 0
			firstByte = src[srcIdx]
			srcIdx++

			decodedBit := m.decodeManchesterSymbol(bit1, bit2)
			if decodedBit != 0xFF {
				currentByte |= (decodedBit << uint(bitPos))
				bitPos--
				if bitPos < 0 {
					if dstIdx < len(dst) {
						dst[dstIdx] = currentByte
						dstIdx++
					}
					currentByte = 0
					bitPos = 7
					if dstIdx >= len(dst) {
						return srcIdx, dstIdx, currentByte, bitPos
					}
				}
			}
		} else {
			bit1 := (firstByte >> (shift7 - uint(bitOffset))) & 1
			bit2 := (firstByte >> (shift6 - uint(bitOffset))) & 1
			bitOffset += 2

			decodedBit := m.decodeManchesterSymbol(bit1, bit2)
			if decodedBit != 0xFF {
				currentByte |= (decodedBit << uint(bitPos))
				bitPos--
				if bitPos < 0 {
					if dstIdx < len(dst) {
						dst[dstIdx] = currentByte
						dstIdx++
					}
					currentByte = 0
					bitPos = 7
					if dstIdx >= len(dst) {
						return srcIdx, dstIdx, currentByte, bitPos
					}
				}
			}
		}
	}

	return srcIdx, dstIdx, currentByte, bitPos
}

// decodeFastPath processes aligned bytes efficiently (2 input bytes -> 1 output byte).
func (m *Decoder) decodeFastPath(dst, src []byte, srcIdx, dstIdx int) (int, int) {
	for srcIdx+1 < len(src) && dstIdx < len(dst) {
		// Decode 2 input bytes (8 Manchester symbols) -> 1 output byte
		b0 := src[srcIdx]
		b1 := src[srcIdx+1]
		srcIdx += 2

		// Decode b0: high 4 bits -> 2 bits, low 4 bits -> 2 bits = 4 bits total
		high4 := (b0 >> 4) & 0x0F
		decodedHigh := m.decodeTable[high4] // 2 bits in positions [1:0]
		low4 := b0 & 0x0F
		decodedLow := m.decodeTable[low4] // 2 bits in positions [1:0]

		// Pack: decodedHigh[1:0] at [7:6], decodedLow[1:0] at [5:4]
		decoded0 := (decodedHigh << 6) | (decodedLow << 4)

		// Decode b1: high 4 bits -> 2 bits, low 4 bits -> 2 bits = 4 bits total
		high4 = (b1 >> 4) & 0x0F
		decodedHigh = m.decodeTable[high4]
		low4 = b1 & 0x0F
		decodedLow = m.decodeTable[low4]

		// Pack: decodedHigh[1:0] at [3:2], decodedLow[1:0] at [1:0]
		decoded1 := (decodedHigh << 2) | decodedLow

		// Combine: decoded0[7:4] | decoded1[3:0] -> output byte
		outputByte := (decoded0 & 0xF0) | (decoded1 & 0x0F)
		dst[dstIdx] = outputByte
		dstIdx++
		if dstIdx >= len(dst) {
			return srcIdx, dstIdx
		}
	}

	return srcIdx, dstIdx
}

// decodeRemaining processes remaining bytes using lookup table with bit packing.
func (m *Decoder) decodeRemaining(dst, src []byte, srcIdx, dstIdx int, currentByte byte, bitPos int) (int, int, byte, int) {
	// Pre-compute shift amount to avoid uint() casts in hot path
	shift3 := uint(3)

	for srcIdx < len(src) && dstIdx < len(dst) {
		b := src[srcIdx]
		srcIdx++

		// Decode high 4 bits (2 Manchester symbols) -> 2 bits
		high4 := (b >> 4) & 0x0F
		decodedHigh := m.decodeTable[high4] // 2 bits in positions [1:0]

		// Decode low 4 bits (2 Manchester symbols) -> 2 bits
		low4 := b & 0x0F
		decodedLow := m.decodeTable[low4] // 2 bits in positions [1:0]

		// Pack 4 decoded bits: decodedHigh[1:0] | decodedLow[1:0]
		decodedBits := (decodedHigh << 2) | decodedLow // 4 bits in positions [3:0]

		// Add to current output byte (unroll loop to avoid uint() casts)
		// Bit 3
		decodedBit := (decodedBits >> shift3) & 1
		currentByte |= (decodedBit << uint(bitPos))
		bitPos--
		if bitPos < 0 {
			dst[dstIdx] = currentByte
			dstIdx++
			currentByte = 0
			bitPos = 7
			if dstIdx >= len(dst) {
				return srcIdx, dstIdx, currentByte, bitPos
			}
		}

		// Bit 2
		decodedBit = (decodedBits >> 2) & 1
		currentByte |= (decodedBit << uint(bitPos))
		bitPos--
		if bitPos < 0 {
			dst[dstIdx] = currentByte
			dstIdx++
			currentByte = 0
			bitPos = 7
			if dstIdx >= len(dst) {
				return srcIdx, dstIdx, currentByte, bitPos
			}
		}

		// Bit 1
		decodedBit = (decodedBits >> 1) & 1
		currentByte |= (decodedBit << uint(bitPos))
		bitPos--
		if bitPos < 0 {
			dst[dstIdx] = currentByte
			dstIdx++
			currentByte = 0
			bitPos = 7
			if dstIdx >= len(dst) {
				return srcIdx, dstIdx, currentByte, bitPos
			}
		}

		// Bit 0
		decodedBit = decodedBits & 1
		currentByte |= (decodedBit << uint(bitPos))
		bitPos--
		if bitPos < 0 {
			dst[dstIdx] = currentByte
			dstIdx++
			currentByte = 0
			bitPos = 7
			if dstIdx >= len(dst) {
				return srcIdx, dstIdx, currentByte, bitPos
			}
		}
	}

	return srcIdx, dstIdx, currentByte, bitPos
}

// decodeManchesterSymbol decodes a 2-bit Manchester symbol to a single bit.
// Returns 0xFF for invalid symbols (00 or 11).
// Used for misaligned signals where we can't use the lookup table.
func (m *Decoder) decodeManchesterSymbol(bit1, bit2 byte) byte {
	// Check if valid Manchester symbol (must be 01 or 10, not 00 or 11)
	if bit1 == bit2 {
		return 0xFF // Invalid symbol
	}

	if m.ieee8023 {
		// IEEE 802.3: 10 = 0, 01 = 1
		// bit1=1, bit2=0 -> 0
		// bit1=0, bit2=1 -> 1
		return bit2
	} else {
		// Standard: 01 = 0, 10 = 1
		// bit1=0, bit2=1 -> 0
		// bit1=1, bit2=0 -> 1
		return bit1
	}
}

// Reset resets the decoder state.
// Reset implements modulation.ByteDecoder interface
func (m *Decoder) Reset() {
	m.bitOffset = 0
}

