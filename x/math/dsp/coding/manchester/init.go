package manchester

// Small lookup tables for Manchester encoding/decoding (4-bit chunks)
// This reduces memory from 1.5KB to just 64 bytes while maintaining speed
var (
	// encodeTable4Std maps 4 bits to 8 bits (4 Manchester symbols) - standard encoding
	// 16 entries × 1 byte = 16 bytes
	encodeTable4Std [16]byte
	// encodeTable4IEEE maps 4 bits to 8 bits (4 Manchester symbols) - IEEE 802.3 encoding
	// 16 entries × 1 byte = 16 bytes
	encodeTable4IEEE [16]byte
	// decodeTable4Std maps 4 bits (2 Manchester symbols) to 2 decoded bits - standard
	// 16 entries × 1 byte = 16 bytes
	decodeTable4Std [16]byte
	// decodeTable4IEEE maps 4 bits (2 Manchester symbols) to 2 decoded bits - IEEE 802.3
	// 16 entries × 1 byte = 16 bytes
	decodeTable4IEEE [16]byte
)

func init() {
	// Initialize 4-bit encoding lookup tables
	// Each 4-bit input becomes 8 bits (4 Manchester symbols)
	for b := 0; b < 16; b++ {
		var stdOut, ieeeOut byte

		// Process each of the 4 bits
		for i := 0; i < 4; i++ {
			bit := (byte(b) >> uint(3-i)) & 1
			manchesterPos := i * 2 // Position in the 8-bit output

			// Standard: 0 = 01, 1 = 10
			if bit == 0 {
				// 01: bit0=0, bit1=1
				stdOut |= (0 << uint(7-manchesterPos))
				stdOut |= (1 << uint(6-manchesterPos))
			} else {
				// 10: bit0=1, bit1=0
				stdOut |= (1 << uint(7-manchesterPos))
				stdOut |= (0 << uint(6-manchesterPos))
			}

			// IEEE 802.3: 0 = 10, 1 = 01
			if bit == 0 {
				// 10: bit0=1, bit1=0
				ieeeOut |= (1 << uint(7-manchesterPos))
				ieeeOut |= (0 << uint(6-manchesterPos))
			} else {
				// 01: bit0=0, bit1=1
				ieeeOut |= (0 << uint(7-manchesterPos))
				ieeeOut |= (1 << uint(6-manchesterPos))
			}
		}

		encodeTable4Std[b] = stdOut
		encodeTable4IEEE[b] = ieeeOut
	}

	// Initialize 4-bit decoding lookup tables
	// Each 4-bit input (2 Manchester symbols) becomes 2 decoded bits
	for b := 0; b < 16; b++ {
		var stdDecoded, ieeeDecoded byte

		// Process 2 Manchester symbols (4 bits total)
		for i := 0; i < 2; i++ {
			bit1 := (byte(b) >> uint(3-i*2)) & 1 // First bit of Manchester symbol
			bit2 := (byte(b) >> uint(2-i*2)) & 1 // Second bit of Manchester symbol

			// Standard: 01 = 0, 10 = 1
			if bit1 == 0 && bit2 == 1 {
				stdDecoded |= (0 << uint(1-i))
			} else if bit1 == 1 && bit2 == 0 {
				stdDecoded |= (1 << uint(1-i))
			}
			// Invalid symbols (00, 11) result in 0 bit (already set)

			// IEEE 802.3: 10 = 0, 01 = 1
			if bit1 == 1 && bit2 == 0 {
				ieeeDecoded |= (0 << uint(1-i))
			} else if bit1 == 0 && bit2 == 1 {
				ieeeDecoded |= (1 << uint(1-i))
			}
			// Invalid symbols (00, 11) result in 0 bit (already set)
		}

		decodeTable4Std[b] = stdDecoded
		decodeTable4IEEE[b] = ieeeDecoded
	}
}

