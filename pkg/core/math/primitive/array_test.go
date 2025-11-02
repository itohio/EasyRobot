package primitive

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSumArr(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	dst := make([]float32, 4)

	SumArr(dst, a, b, 4, 1, 1)
	assert.Equal(t, []float32{6, 8, 10, 12}, dst)
}

func TestDotProduct(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}

	result := DotProduct(a, b, 3, 1, 1)
	assert.Equal(t, float32(32), result) // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
}

func TestSum(t *testing.T) {
	a := []float32{1, 2, 3, 4}
	result := Sum(a, 4, 1)
	assert.Equal(t, float32(10), result)
}

func TestSqrSum(t *testing.T) {
	a := []float32{1, 2, 3}
	result := SqrSum(a, 3, 1)
	assert.Equal(t, float32(14), result) // 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
}

func TestMulArrConst(t *testing.T) {
	src := []float32{1, 2, 3, 4}
	dst := make([]float32, 4)
	MulArrConst(dst, src, 2.0, 4, 1)
	assert.Equal(t, []float32{2, 4, 6, 8}, dst)
}
