package dsss

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGenerateMSequence(t *testing.T) {
	// Test m-sequence with n=3, taps=[2,1,0] (x^3 + x + 1)
	n := 3
	taps := []int{2, 1, 0}
	seed := uint(1)

	code := GenerateMSequence(n, taps, seed)

	// Length should be 2^n - 1 = 7
	expectedLength := (1 << n) - 1
	assert.Equal(t, expectedLength, code.Length(), "m-sequence length should be 2^n - 1")

	// All values should be 0 or 1 (bits)
	for i := 0; i < code.Length(); i++ {
		bit := code.GetBit(i)
		assert.True(t, bit == 0 || bit == 1, "m-sequence chips should be 0 or 1")
	}

	// Verify it's not all the same value
	allSame := true
	firstBit := code.GetBit(0)
	for i := 1; i < code.Length(); i++ {
		if code.GetBit(i) != firstBit {
			allSame = false
			break
		}
	}
	assert.False(t, allSame, "m-sequence should not be constant")
}

func TestGenerateMSequence_InvalidInput(t *testing.T) {
	// Test invalid n
	assert.Panics(t, func() {
		GenerateMSequence(1, []int{0}, 1)
	}, "should panic for n < 2")

	assert.Panics(t, func() {
		GenerateMSequence(17, []int{0}, 1)
	}, "should panic for n > 16")

	// Test zero seed
	assert.Panics(t, func() {
		GenerateMSequence(3, []int{2, 1, 0}, 0)
	}, "should panic for zero seed")

	// Test empty taps
	assert.Panics(t, func() {
		GenerateMSequence(3, []int{}, 1)
	}, "should panic for empty taps")

	// Test invalid tap position
	assert.Panics(t, func() {
		GenerateMSequence(3, []int{5, 0}, 1)
	}, "should panic for tap out of range")
}

func TestGenerateGoldCode(t *testing.T) {
	// Generate two m-sequences
	n := 3
	taps := []int{2, 1, 0}
	seqA := GenerateMSequence(n, taps, 1)
	seqB := GenerateMSequence(n, taps, 2)

	// Generate Gold code
	shift := 0
	goldCode := GenerateGoldCode(seqA, seqB, shift)

	// Length should match
	assert.Equal(t, seqA.Length(), goldCode.Length(), "Gold code length should match m-sequence length")

	// All values should be 0 or 1 (bits)
	for i := 0; i < goldCode.Length(); i++ {
		bit := goldCode.GetBit(i)
		assert.True(t, bit == 0 || bit == 1, "Gold code chips should be 0 or 1")
	}
}

func TestGenerateGoldCode_InvalidInput(t *testing.T) {
	seqA := GenerateMSequence(3, []int{2, 1, 0}, 1)
	seqB := GenerateMSequence(4, []int{3, 2, 0}, 1)

	// Test different lengths
	assert.Panics(t, func() {
		GenerateGoldCode(seqA, seqB, 0)
	}, "should panic for sequences of different lengths")

	// Test invalid shift
	seqB2 := GenerateMSequence(3, []int{2, 1, 0}, 2)
	assert.Panics(t, func() {
		GenerateGoldCode(seqA, seqB2, seqA.Length())
	}, "should panic for shift >= length")
}

func TestGenerateWalshCode(t *testing.T) {
	// Test Walsh code of order 4
	order := 4
	row := 0

	code := GenerateWalshCode(order, row)

	// Length should equal order
	assert.Equal(t, order, code.Length(), "Walsh code length should equal order")

	// All values should be 0 or 1 (bits)
	for i := 0; i < code.Length(); i++ {
		bit := code.GetBit(i)
		assert.True(t, bit == 0 || bit == 1, "Walsh code chips should be 0 or 1")
	}
}

func TestGenerateWalshCode_InvalidInput(t *testing.T) {
	// Test non-power-of-2 order
	assert.Panics(t, func() {
		GenerateWalshCode(3, 0)
	}, "should panic for non-power-of-2 order")

	// Test invalid row
	assert.Panics(t, func() {
		GenerateWalshCode(4, 4)
	}, "should panic for row >= order")
}

func TestGenerateWalshCodeTable(t *testing.T) {
	order := 4
	table := GenerateWalshCodeTable(order)

	// Should have 'order' codes
	assert.Equal(t, order, len(table), "code table should have 'order' codes")

	// All codes should have length 'order'
	for i, code := range table {
		assert.Equal(t, order, code.Length(), "code %d should have length %d", i, order)
	}

	// Verify orthogonality (codes should be orthogonal)
	tolerance := float32(0.01)
	assert.True(t, VerifyOrthogonality(table, tolerance), "Walsh codes should be orthogonal")
}

func TestGenerateWalshCodeTable_InvalidInput(t *testing.T) {
	// Test non-power-of-2 order
	assert.Panics(t, func() {
		GenerateWalshCodeTable(3)
	}, "should panic for non-power-of-2 order")
}

func TestNormalizeCode(t *testing.T) {
	// Create a simple code (bits: 0/1)
	code := NewCode(4)
	code.SetBit(0, 1)
	code.SetBit(1, 0)
	code.SetBit(2, 1)
	code.SetBit(3, 0)
	normalized := NormalizeCode(code)

	// Length should be same
	assert.Equal(t, code.Length(), len(normalized), "normalized code should have same length")

	// Energy should be approximately 1
	energy := float32(0)
	for _, chip := range normalized {
		energy += chip * chip
	}
	assert.InDelta(t, 1.0, energy, 0.01, "normalized code should have unit energy")
}

func TestVerifyOrthogonality(t *testing.T) {
	// Generate orthogonal Walsh codes
	table := GenerateWalshCodeTable(4)

	// Should be orthogonal
	assert.True(t, VerifyOrthogonality(table, 0.1), "Walsh codes should be orthogonal")

	// Create non-orthogonal codes (all same)
	code1 := NewCode(4)
	code1.SetBit(0, 1)
	code1.SetBit(1, 1)
	code1.SetBit(2, 1)
	code1.SetBit(3, 1)
	code2 := NewCode(4)
	code2.SetBit(0, 1)
	code2.SetBit(1, 1)
	code2.SetBit(2, 1)
	code2.SetBit(3, 1)
	nonOrthoTable := CodeTable{code1, code2}

	// Should not be orthogonal
	assert.False(t, VerifyOrthogonality(nonOrthoTable, 0.1), "identical codes should not be orthogonal")
}

func TestGetUniqueSymbolCodeTable_Walsh(t *testing.T) {
	order := 4
	codeLength := 4

	table := GetUniqueSymbolCodeTable("walsh", order, codeLength, false)

	// Should have 'order' codes
	assert.Equal(t, order, len(table), "code table should have 'order' codes")

	// All codes should have length 'codeLength'
	for i, code := range table {
		assert.Equal(t, codeLength, code.Length(), "code %d should have length %d", i, codeLength)
	}

	// Should be orthogonal
	assert.True(t, VerifyOrthogonality(table, 0.1), "Walsh codes should be orthogonal")
}

func TestGetUniqueSymbolCodeTable_MSequence(t *testing.T) {
	order := 4
	codeLength := 7 // Will use n=3, length=7

	table := GetUniqueSymbolCodeTable("m-sequence", order, codeLength, false)

	// Should have 'order' codes
	assert.Equal(t, order, len(table), "code table should have 'order' codes")

	// All codes should have same length
	expectedLength := 7
	for i, code := range table {
		assert.Equal(t, expectedLength, code.Length(), "code %d should have length %d", i, expectedLength)
	}
}

func TestGetUniqueSymbolCodeTable_Gold(t *testing.T) {
	order := 4
	codeLength := 7 // Will use n=3, length=7

	table := GetUniqueSymbolCodeTable("gold", order, codeLength, false)

	// Should have 'order' codes
	assert.Equal(t, order, len(table), "code table should have 'order' codes")

	// All codes should have same length
	expectedLength := 7
	for i, code := range table {
		assert.Equal(t, expectedLength, code.Length(), "code %d should have length %d", i, expectedLength)
	}
}

func TestGetUniqueSymbolCodeTable_InvalidInput(t *testing.T) {
	// Test invalid method
	assert.Panics(t, func() {
		GetUniqueSymbolCodeTable("invalid", 4, 4, false)
	}, "should panic for invalid method")

	// Test Walsh with non-power-of-2 order
	assert.Panics(t, func() {
		GetUniqueSymbolCodeTable("walsh", 3, 3, false)
	}, "should panic for Walsh with non-power-of-2 order")

	// Test Walsh with mismatched codeLength
	assert.Panics(t, func() {
		GetUniqueSymbolCodeTable("walsh", 4, 8, false)
	}, "should panic for Walsh with codeLength != order")
}
