package dsss

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"math"
)

// Code represents a spreading code as a sequence of chips
// Chips are represented as bits: 0 or 1, packed 8 chips per byte
// Bits are stored in little-endian order within each byte (LSB first)
type Code struct {
	bits   []byte // Packed bits: 8 chips per byte
	length int    // Number of chips (may be less than len(bits)*8)
}

// CodeTable represents a set of orthogonal codes for M-ary DSSS
// Each code is indexed by symbol value
type CodeTable []Code

// NewCode creates a new Code with the specified number of chips
func NewCode(length int) Code {
	bytesNeeded := (length + 7) / 8
	return Code{
		bits:   make([]byte, bytesNeeded),
		length: length,
	}
}

// CodeFromHex creates a Code from a hex string representation
// hexStr: hex string representing the code bits (e.g., "A5" = 10100101)
// length: number of chips in the code (must match or be less than bits in hex string)
// The hex string is interpreted as bytes, with bits stored in little-endian order within each byte (LSB first)
func CodeFromHex(hexStr string, length int) (Code, error) {
	if length <= 0 {
		return Code{}, fmt.Errorf("code length must be positive, got %d", length)
	}

	// Decode hex string to bytes
	bytes, err := hex.DecodeString(hexStr)
	if err != nil {
		return Code{}, fmt.Errorf("invalid hex string: %w", err)
	}

	// Calculate minimum bytes needed
	bytesNeeded := (length + 7) / 8
	if len(bytes) < bytesNeeded {
		return Code{}, fmt.Errorf("hex string too short: need at least %d bytes for length %d, got %d", bytesNeeded, length, len(bytes))
	}

	// Create code with specified length
	code := Code{
		bits:   make([]byte, bytesNeeded),
		length: length,
	}

	// Copy bytes (only what we need)
	copy(code.bits, bytes[:bytesNeeded])

	return code, nil
}

// Length returns the number of chips in the code
func (c Code) Length() int {
	return c.length
}

// GetBit returns the bit value (0 or 1) at the specified chip index
func (c Code) GetBit(index int) byte {
	if index < 0 || index >= c.length {
		panic(fmt.Sprintf("Code.GetBit: index %d out of range [0, %d)", index, c.length))
	}
	byteIdx := index / 8
	bitIdx := index % 8
	return (c.bits[byteIdx] >> bitIdx) & 1
}

// SetBit sets the bit value (0 or 1) at the specified chip index
func (c Code) SetBit(index int, value byte) {
	if index < 0 || index >= c.length {
		panic(fmt.Sprintf("Code.SetBit: index %d out of range [0, %d)", index, c.length))
	}
	if value != 0 && value != 1 {
		panic(fmt.Sprintf("Code.SetBit: value must be 0 or 1, got %d", value))
	}
	byteIdx := index / 8
	bitIdx := index % 8
	if value == 0 {
		c.bits[byteIdx] &^= 1 << bitIdx // Clear bit
	} else {
		c.bits[byteIdx] |= 1 << bitIdx // Set bit
	}
}

// GetBipolar returns the bipolar value (+1 or -1) for the chip at index
// This is used directly in modulation/demodulation without allocation
func (c Code) GetBipolar(index int) float32 {
	bit := c.GetBit(index)
	if bit == 0 {
		return -1.0
	}
	return +1.0
}

// GenerateMSequence generates an m-sequence (maximal-length LFSR sequence)
// n: register length (sequence length will be L = 2^n - 1)
// taps: tap positions (indices in the LFSR register, 0-based)
// seed: initial state (must not be all zeros)
// Returns a code (bits: 0/1) of length 2^n - 1
func GenerateMSequence(n int, taps []int, seed uint) Code {
	if n < 2 || n > 16 {
		panic(fmt.Sprintf("m-sequence: n must be between 2 and 16, got %d", n))
	}
	if seed == 0 {
		panic("m-sequence: seed must not be zero")
	}
	if len(taps) == 0 {
		panic("m-sequence: taps must not be empty")
	}

	length := (1 << n) - 1
	code := NewCode(length)
	register := seed

	// Generate sequence
	for i := 0; i < length; i++ {
		// Output current bit (LSB)
		bit := byte(register & 1)
		code.SetBit(i, bit)

		// Compute feedback: XOR of tapped bits
		feedback := uint(0)
		for _, tap := range taps {
			if tap < 0 || tap >= n {
				panic(fmt.Sprintf("m-sequence: tap position %d out of range [0, %d)", tap, n))
			}
			feedback ^= (register >> tap) & 1
		}

		// Shift register and insert feedback
		register = (register >> 1) | (feedback << (n - 1))
	}

	return code
}

// GenerateGoldCode generates a Gold code by combining two m-sequences
// a, b: two m-sequences of the same length (bits: 0/1)
// shift: cyclic shift applied to sequence b (0 <= shift < b.Length())
// Returns a Gold code (bits: 0/1)
func GenerateGoldCode(a, b Code, shift int) Code {
	if a.length != b.length {
		panic(fmt.Sprintf("Gold code: sequences must have same length, got %d and %d", a.length, b.length))
	}
	if shift < 0 || shift >= b.length {
		panic(fmt.Sprintf("Gold code: shift must be in [0, %d), got %d", b.length, shift))
	}

	length := a.length
	code := NewCode(length)

	// Combine: XOR the sequences (with shift on b)
	for i := 0; i < length; i++ {
		bIdx := (i + shift) % length
		// XOR: a[i] ^ b[bIdx]
		bitA := a.GetBit(i)
		bitB := b.GetBit(bIdx)
		code.SetBit(i, bitA^bitB)
	}

	return code
}

// GenerateWalshCode generates a Walsh code from Hadamard matrix
// order: order of Walsh code (must be power of 2, length = order)
// row: which row of the Hadamard matrix to use (0 <= row < order)
// Returns a Walsh code (bits: 0/1)
func GenerateWalshCode(order, row int) Code {
	if order < 2 || (order&(order-1)) != 0 {
		panic(fmt.Sprintf("Walsh code: order must be power of 2, got %d", order))
	}
	if row < 0 || row >= order {
		panic(fmt.Sprintf("Walsh code: row must be in [0, %d), got %d", order, row))
	}

	// Generate Hadamard matrix recursively
	hadamard := generateHadamardMatrix(order)

	// Extract row and convert to bits
	code := NewCode(order)
	for i := 0; i < order; i++ {
		code.SetBit(i, byte(hadamard[row][i]))
	}

	return code
}

// generateHadamardMatrix generates a Hadamard matrix of given order
// Returns a matrix where each element is 0 or 1
func generateHadamardMatrix(order int) [][]int {
	if order == 1 {
		return [][]int{{0}}
	}

	half := order / 2
	halfMatrix := generateHadamardMatrix(half)

	// Construct full matrix using Sylvester construction
	matrix := make([][]int, order)
	for i := range matrix {
		matrix[i] = make([]int, order)
	}

	// Top-left: halfMatrix
	for i := 0; i < half; i++ {
		for j := 0; j < half; j++ {
			matrix[i][j] = halfMatrix[i][j]
		}
	}

	// Top-right: halfMatrix
	for i := 0; i < half; i++ {
		for j := 0; j < half; j++ {
			matrix[i][j+half] = halfMatrix[i][j]
		}
	}

	// Bottom-left: halfMatrix
	for i := 0; i < half; i++ {
		for j := 0; j < half; j++ {
			matrix[i+half][j] = halfMatrix[i][j]
		}
	}

	// Bottom-right: complement of halfMatrix
	for i := 0; i < half; i++ {
		for j := 0; j < half; j++ {
			matrix[i+half][j+half] = 1 - halfMatrix[i][j]
		}
	}

	return matrix
}

// GenerateWalshCodeTable generates a complete Walsh code table
// order: order of Walsh codes (must be power of 2)
// Returns a CodeTable with 'order' codes, each of length 'order'
func GenerateWalshCodeTable(order int) CodeTable {
	if order < 2 || (order&(order-1)) != 0 {
		panic(fmt.Sprintf("Walsh code table: order must be power of 2, got %d", order))
	}

	table := make(CodeTable, order)
	for i := range table {
		table[i] = GenerateWalshCode(order, i)
	}

	return table
}

// GenerateMSequenceCodeTable generates a code table from m-sequences
// n: register length
// taps: tap positions for LFSR
// numCodes: number of codes to generate (using different seeds)
// useRandomSeeds: if true, uses crypto/rand to generate random seeds for unique codes
//
//	if false, uses sequential seeds starting from 1 (deterministic)
//
// # Returns a CodeTable with numCodes codes
//
// Note: For unique codes across different sessions, set useRandomSeeds=true.
//
//	For reproducible codes, set useRandomSeeds=false.
func GenerateMSequenceCodeTable(n int, taps []int, numCodes int, useRandomSeeds bool) CodeTable {
	if numCodes <= 0 {
		panic("m-sequence code table: numCodes must be positive")
	}

	table := make(CodeTable, numCodes)
	length := (1 << n) - 1
	maxSeed := uint((1 << n) - 1) // Maximum valid seed

	// Generate codes with different seeds
	if useRandomSeeds {
		// Use random seeds for unique codes
		usedSeeds := make(map[uint]bool)
		for i := range table {
			var seed uint
			attempts := 0
			for {
				// Generate random seed
				var seedBytes [4]byte
				if _, err := rand.Read(seedBytes[:]); err != nil {
					panic(fmt.Sprintf("m-sequence code table: failed to generate random seed: %v", err))
				}
				seed = uint(seedBytes[0]) | (uint(seedBytes[1]) << 8) | (uint(seedBytes[2]) << 16) | (uint(seedBytes[3]) << 24)
				seed = (seed % maxSeed) + 1 // Ensure seed is in [1, maxSeed]

				// Check if we've used this seed before
				if !usedSeeds[seed] {
					usedSeeds[seed] = true
					break
				}
				attempts++
				if attempts > 1000 {
					// Fallback to sequential if too many collisions
					seed = uint(i) + 1
					break
				}
			}
			table[i] = GenerateMSequence(n, taps, seed)
		}
	} else {
		// Use sequential seeds (deterministic)
		seed := uint(1) // Start with 1 (not 0)
		for i := range table {
			// Find a valid seed (non-zero)
			for seed == 0 {
				seed++
			}
			table[i] = GenerateMSequence(n, taps, seed)
			seed++
		}
	}

	// Verify all codes have same length
	for i, code := range table {
		if code.length != length {
			panic(fmt.Sprintf("m-sequence code table: code %d has unexpected length %d", i, code.length))
		}
	}

	return table
}

// GenerateGoldCodeTable generates a Gold code table
// n: register length for m-sequences
// tapsA, tapsB: tap positions for the two m-sequences
// seedA, seedB: seeds for the two m-sequences
// numCodes: number of Gold codes to generate (using different shifts)
// useRandomShifts: if true, uses crypto/rand to generate random shifts for unique codes
//
//	if false, uses sequential shifts (deterministic)
//
// # Returns a CodeTable with numCodes Gold codes
//
// Note: For unique codes across different sessions, set useRandomShifts=true.
//
//	For reproducible codes, set useRandomShifts=false.
func GenerateGoldCodeTable(n int, tapsA, tapsB []int, seedA, seedB uint, numCodes int, useRandomShifts bool) CodeTable {
	if numCodes <= 0 {
		panic("Gold code table: numCodes must be positive")
	}

	// Generate the two m-sequences
	seqA := GenerateMSequence(n, tapsA, seedA)
	seqB := GenerateMSequence(n, tapsB, seedB)

	length := seqA.length
	table := make(CodeTable, numCodes)

	// Generate Gold codes with different shifts
	if useRandomShifts {
		// Use random shifts for unique codes
		usedShifts := make(map[int]bool)
		for i := range table {
			var shift int
			attempts := 0
			for {
				// Generate random shift
				var shiftBytes [4]byte
				if _, err := rand.Read(shiftBytes[:]); err != nil {
					panic(fmt.Sprintf("Gold code table: failed to generate random shift: %v", err))
				}
				shift = int(shiftBytes[0]) | (int(shiftBytes[1]) << 8) | (int(shiftBytes[2]) << 16) | (int(shiftBytes[3]) << 24)
				shift = shift % length
				if shift < 0 {
					shift = -shift
				}

				// Check if we've used this shift before
				if !usedShifts[shift] {
					usedShifts[shift] = true
					break
				}
				attempts++
				if attempts > 1000 {
					// Fallback to sequential if too many collisions
					shift = i % length
					break
				}
			}
			table[i] = GenerateGoldCode(seqA, seqB, shift)
		}
	} else {
		// Use sequential shifts (deterministic)
		for i := range table {
			shift := i % length
			table[i] = GenerateGoldCode(seqA, seqB, shift)
		}
	}

	return table
}

// NormalizeCode normalizes a code to unit energy
// Divides each chip by sqrt(L) where L is code length
// Returns normalized bipolar values
func NormalizeCode(code Code) []float32 {
	length := code.Length()
	normalized := make([]float32, length)
	normFactor := 1.0 / float32(math.Sqrt(float64(length)))
	for i := 0; i < length; i++ {
		normalized[i] = code.GetBipolar(i) * normFactor
	}
	return normalized
}

// NormalizeCodeTable normalizes all codes in a table to unit energy
// Returns normalized bipolar values (not bits)
func NormalizeCodeTable(table CodeTable) [][]float32 {
	normalized := make([][]float32, len(table))
	for i := range normalized {
		normalized[i] = NormalizeCode(table[i])
	}
	return normalized
}

// VerifyOrthogonality checks if codes in a table are orthogonal
// Returns true if all pairs are orthogonal (inner product ≈ 0)
// tolerance: allowed deviation from zero
// Codes are converted to bipolar for orthogonality check
func VerifyOrthogonality(table CodeTable, tolerance float32) bool {
	for i := range table {
		for j := i + 1; j < len(table); j++ {
			innerProduct := computeInnerProduct(table[i], table[j])
			if math.Abs(float64(innerProduct)) > float64(tolerance) {
				return false
			}
		}
	}
	return true
}

// computeInnerProduct computes the inner product of two codes
// Works directly with bits, converting to bipolar on-the-fly
func computeInnerProduct(a, b Code) float32 {
	if a.length != b.length {
		return 0
	}
	// Compute inner product directly without allocation
	sum := float32(0)
	for i := 0; i < a.length; i++ {
		sum += a.GetBipolar(i) * b.GetBipolar(i)
	}
	return sum
}

// GetUniqueSymbolCodeTable generates a unique code table for M-ary DSSS
// This ensures each symbol maps to a unique spreading code
// method: "walsh", "m-sequence", or "gold"
// order: modulation order (number of symbols, must be power of 2 for Walsh)
// codeLength: desired code length (for m-sequence/gold, will be adjusted to 2^n-1)
// useRandom: if true, uses random seeds/shifts for unique codes across sessions
//
//	if false, uses deterministic seeds/shifts (reproducible)
//
// Returns a CodeTable with 'order' unique codes
//
// Generating Unique Codes:
// ========================
// To generate unique codes that differ across sessions or instances:
//
// 1. For m-sequences:
//   - Set useRandom=true to use crypto/rand for seed generation
//   - Each call will generate different codes
//
// 2. For Gold codes:
//   - Set useRandom=true to use crypto/rand for shift selection
//   - You can also use random seeds for the m-sequences:
//     seedA := randomSeed()
//     seedB := randomSeed()
//     table := GenerateGoldCodeTable(n, tapsA, tapsB, seedA, seedB, numCodes, true)
//
// 3. For Walsh codes:
//   - Walsh codes are deterministic by design (orthogonal set)
//   - To get different sets, use different orders or select different rows
//   - For randomization, you can permute the code table after generation
//
// Example - Unique m-sequence codes:
//
//	table := GetUniqueSymbolCodeTable("m-sequence", 8, 15, true)
//
// Example - Reproducible Gold codes:
//
//	table := GetUniqueSymbolCodeTable("gold", 8, 15, false)
func GetUniqueSymbolCodeTable(method string, order int, codeLength int, useRandom bool) CodeTable {
	switch method {
	case "walsh":
		// For Walsh, order must be power of 2 and code length = order
		if (order & (order - 1)) != 0 {
			panic(fmt.Sprintf("Walsh codes require order to be power of 2, got %d", order))
		}
		if codeLength != order {
			panic(fmt.Sprintf("Walsh codes require codeLength == order, got codeLength=%d, order=%d", codeLength, order))
		}
		return GenerateWalshCodeTable(order)

	case "m-sequence":
		// For m-sequence, find n such that 2^n - 1 >= codeLength
		n := 2
		for (1<<n)-1 < codeLength {
			n++
			if n > 16 {
				panic(fmt.Sprintf("m-sequence: codeLength %d too large", codeLength))
			}
		}
		actualLength := (1 << n) - 1

		// Use standard taps for m-sequences (primitive polynomials)
		taps := getStandardTaps(n)
		if len(taps) == 0 {
			panic(fmt.Sprintf("m-sequence: no standard taps for n=%d", n))
		}

		// Generate table with different seeds (random or sequential)
		table := GenerateMSequenceCodeTable(n, taps, order, useRandom)
		if len(table) != order {
			panic(fmt.Sprintf("m-sequence: generated %d codes, expected %d", len(table), order))
		}

		// Verify all codes have same length
		for i, code := range table {
			if code.length != actualLength {
				panic(fmt.Sprintf("m-sequence: code %d has length %d, expected %d", i, code.length, actualLength))
			}
		}

		return table

	case "gold":
		// For Gold codes, find n such that 2^n - 1 >= codeLength
		n := 2
		for (1<<n)-1 < codeLength {
			n++
			if n > 16 {
				panic(fmt.Sprintf("Gold codes: codeLength %d too large", codeLength))
			}
		}

		// Use standard taps for both sequences
		tapsA := getStandardTaps(n)
		tapsB := getStandardTaps(n)
		if len(tapsA) == 0 || len(tapsB) == 0 {
			panic(fmt.Sprintf("Gold codes: no standard taps for n=%d", n))
		}

		// Use different taps for B if available
		if len(tapsB) > 1 {
			// Use alternative taps for better cross-correlation
			tapsB = tapsB[1:]
		}

		// Generate seeds (random or deterministic)
		var seedA, seedB uint
		if useRandom {
			// Generate random seeds
			var seedBytes [4]byte
			if _, err := rand.Read(seedBytes[:]); err != nil {
				panic(fmt.Sprintf("Gold codes: failed to generate random seedA: %v", err))
			}
			seedA = uint(seedBytes[0]) | (uint(seedBytes[1]) << 8) | (uint(seedBytes[2]) << 16) | (uint(seedBytes[3]) << 24)
			seedA = (seedA % uint((1<<n)-1)) + 1

			if _, err := rand.Read(seedBytes[:]); err != nil {
				panic(fmt.Sprintf("Gold codes: failed to generate random seedB: %v", err))
			}
			seedB = uint(seedBytes[0]) | (uint(seedBytes[1]) << 8) | (uint(seedBytes[2]) << 16) | (uint(seedBytes[3]) << 24)
			seedB = (seedB % uint((1<<n)-1)) + 1
		} else {
			seedA = uint(1)
			seedB = uint(1)
		}

		table := GenerateGoldCodeTable(n, tapsA, tapsB, seedA, seedB, order, useRandom)
		return table

	default:
		panic(fmt.Sprintf("unknown code generation method: %s (must be 'walsh', 'm-sequence', or 'gold')", method))
	}
}

// getStandardTaps returns standard tap positions for m-sequences
// These correspond to primitive polynomials
func getStandardTaps(n int) []int {
	// Standard tap positions for primitive polynomials
	// Format: [n, tap1, tap2, ...] where taps are 0-based positions
	standardTaps := map[int][]int{
		2:  {1, 0},            // x^2 + x + 1
		3:  {2, 1, 0},         // x^3 + x + 1
		4:  {3, 2, 0},         // x^4 + x + 1
		5:  {4, 2, 0},         // x^5 + x^2 + 1
		6:  {5, 4, 0},         // x^6 + x + 1
		7:  {6, 3, 0},         // x^7 + x + 1
		8:  {7, 5, 4, 3, 0},   // x^8 + x^4 + x^3 + x^2 + 1
		9:  {8, 4, 0},         // x^9 + x^4 + 1
		10: {9, 6, 0},         // x^10 + x^3 + 1
		11: {10, 8, 0},        // x^11 + x^2 + 1
		12: {11, 10, 9, 3, 0}, // x^12 + x^6 + x^4 + x + 1
		13: {12, 10, 9, 0},    // x^13 + x^4 + x^3 + x + 1
		14: {13, 11, 1, 0},    // x^14 + x^10 + x^6 + x + 1
		15: {14, 13, 0},       // x^15 + x + 1
		16: {15, 13, 12, 0},   // x^16 + x^12 + x^3 + x + 1
	}

	taps, ok := standardTaps[n]
	if !ok {
		return nil
	}

	// Return taps excluding the first element (which is n itself)
	if len(taps) > 1 {
		return taps[1:]
	}
	return []int{0}
}
