package manchester

// Encoder handles Manchester line coding encoding.
// Manchester encoding: each bit is encoded as a transition.
// Standard Manchester: 0 = low-to-high (01), 1 = high-to-low (10)
type Encoder struct {
	// IEEE 802.3 variant: if true, 0 = high-to-low (10), 1 = low-to-high (01)
	ieee8023 bool
	// Lookup table reference (16 bytes)
	encodeTable *[16]byte
}

// NewEncoder creates a new Manchester encoder with standard encoding.
// Standard: 0 = 01, 1 = 10
func NewEncoder() *Encoder {
	return &Encoder{
		ieee8023:    false,
		encodeTable: &encodeTable4Std,
	}
}

// NewEncoderIEEE8023 creates a new Manchester encoder with IEEE 802.3 encoding.
// IEEE 802.3: 0 = 10, 1 = 01
func NewEncoderIEEE8023() *Encoder {
	return &Encoder{
		ieee8023:    true,
		encodeTable: &encodeTable4IEEE,
	}
}

// Encode encodes input bytes to Manchester-encoded bytes using destination pattern.
// Each input bit is encoded as 2 bits in the output.
// Input: []byte of data
// Output: []byte of Manchester-encoded data (2x the size of input)
// Returns the number of bytes written to dst.
// Optimized using 4-bit lookup tables (16 bytes per table, 64 bytes total).
// Encode implements modulation.ByteEncoder interface
func (m *Encoder) Encode(dst, src []byte) int {
	requiredLen := len(src) * 2
	if len(dst) < requiredLen {
		return 0
	}

	dstIdx := 0
	for _, b := range src {
		// Process high 4 bits -> first output byte
		high4 := (b >> 4) & 0x0F
		dst[dstIdx] = m.encodeTable[high4]

		// Process low 4 bits -> second output byte
		low4 := b & 0x0F
		dst[dstIdx+1] = m.encodeTable[low4]

		dstIdx += 2
	}

	return dstIdx
}

// Reset resets the encoder state (no-op for Manchester encoding).
// Reset implements modulation.ByteEncoder interface
func (m *Encoder) Reset() {
	// No state to reset
}
