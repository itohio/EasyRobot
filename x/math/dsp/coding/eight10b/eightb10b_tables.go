package ecc

// 8b/10b encoding tables
// Based on standard 8b/10b encoding used in PCIe, SATA, DisplayPort, etc.
// Maps 8-bit data bytes to 10-bit symbols with DC balance and bounded run length

// eightB10BDataTable maps 5-bit data (0-31) to 6-bit code (for 5b/6b encoding)
// Index: 5-bit data value, Value: 6-bit code
var eightB10BDataTable = [32]uint16{
	0b100111, // 0x00 -> D.0
	0b011101, // 0x01 -> D.1
	0b101101, // 0x02 -> D.2
	0b110001, // 0x03 -> D.3
	0b110101, // 0x04 -> D.4
	0b101001, // 0x05 -> D.5
	0b011001, // 0x06 -> D.6
	0b111000, // 0x07 -> D.7
	0b111001, // 0x08 -> D.8
	0b100101, // 0x09 -> D.9
	0b010101, // 0x0A -> D.10
	0b110100, // 0x0B -> D.11
	0b001101, // 0x0C -> D.12
	0b101100, // 0x0D -> D.13
	0b011100, // 0x0E -> D.14
	0b010111, // 0x0F -> D.15
	0b011011, // 0x10 -> D.16
	0b100011, // 0x11 -> D.17
	0b010011, // 0x12 -> D.18
	0b110010, // 0x13 -> D.19
	0b001011, // 0x14 -> D.20
	0b101010, // 0x15 -> D.21
	0b011010, // 0x16 -> D.22
	0b111010, // 0x17 -> D.23
	0b110011, // 0x18 -> D.24
	0b100110, // 0x19 -> D.25
	0b010110, // 0x1A -> D.26
	0b110110, // 0x1B -> D.27
	0b001110, // 0x1C -> D.28
	0b101110, // 0x1D -> D.29
	0b011110, // 0x1E -> D.30
	0b101011, // 0x1F -> D.31
}

// eightB10BControlTable maps 3-bit control (0-7) to 4-bit code (for 3b/4b encoding)
// Index: 3-bit control value, Value: 4-bit code
var eightB10BControlTable = [8]uint16{
	0b1011, // K.0
	0b1001, // K.1
	0b0101, // K.2
	0b1100, // K.3
	0b1101, // K.4
	0b1010, // K.5
	0b0110, // K.6
	0b1110, // K.7 (or 0b0111 for -)
}

// Simplified 8b/10b: For each 8-bit byte, we'll use a simplified mapping
// In practice, 8b/10b uses running disparity (RD) to maintain DC balance
// This is a simplified version that maps directly without RD tracking
// For full implementation, RD tracking would be needed

// buildEightB10BEncodeTable builds a simplified 8b/10b encoding table
// Maps 8-bit value to 10-bit symbol (stored as uint16, lower 10 bits)
func buildEightB10BEncodeTable() [256]uint16 {
	var table [256]uint16
	// Simplified: use 5b/6b + 3b/4b combination
	// For each byte: high 5 bits -> 6-bit code, low 3 bits -> 4-bit code
	for i := 0; i < 256; i++ {
		high5 := (i >> 3) & 0x1F
		low3 := i & 0x07
		sixBit := eightB10BDataTable[high5]
		fourBit := eightB10BControlTable[low3]
		// Combine: 6-bit (bits 9-4) + 4-bit (bits 3-0) = 10-bit symbol
		table[i] = (sixBit << 4) | fourBit
	}
	return table
}

var eightB10BEncodeTable = buildEightB10BEncodeTable()

// buildEightB10BDecodeTable builds reverse lookup (10-bit -> 8-bit)
func buildEightB10BDecodeTable() [1024]byte {
	var table [1024]byte
	// Initialize with invalid marker
	for i := range table {
		table[i] = 0xFF
	}
	// Fill valid mappings
	for i := 0; i < 256; i++ {
		symbol := eightB10BEncodeTable[i]
		table[symbol] = byte(i)
	}
	return table
}

var eightB10BDecodeTable = buildEightB10BDecodeTable()

