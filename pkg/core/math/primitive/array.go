package primitive

import (
	"sort"

	"github.com/chewxy/math32"
)

// SumArr computes dst[i] = a[i] + b[i] for all i
// Element-wise addition for tensor operations
// num: number of elements
// strideA, strideB: strides for a and b (default 1)
func SumArr(dst, a, b []float32, num int, strideA, strideB int) {
	if num == 0 {
		return
	}

	pa := 0
	pb := 0
	pd := 0

	for i := 0; i < num; i++ {
		dst[pd] = a[pa] + b[pb]
		pa += strideA
		pb += strideB
		pd++
	}
}

// DiffArr computes dst[i] = a[i] - b[i] for all i
// Element-wise subtraction for tensor operations
func DiffArr(dst, a, b []float32, num int, strideA, strideB int) {
	if num == 0 {
		return
	}

	pa := 0
	pb := 0
	pd := 0

	for i := 0; i < num; i++ {
		dst[pd] = a[pa] - b[pb]
		pa += strideA
		pb += strideB
		pd++
	}
}

// MulArr computes dst[i] = a[i] * b[i] for all i
// Element-wise multiplication for tensor operations
func MulArr(dst, a, b []float32, num int, strideA, strideB int) {
	if num == 0 {
		return
	}

	pa := 0
	pb := 0
	pd := 0

	for i := 0; i < num; i++ {
		dst[pd] = a[pa] * b[pb]
		pa += strideA
		pb += strideB
		pd++
	}
}

// DivArr computes dst[i] = a[i] / b[i] for all i
// Element-wise division for tensor operations
func DivArr(dst, a, b []float32, num int, strideA, strideB int) {
	if num == 0 {
		return
	}

	pa := 0
	pb := 0
	pd := 0

	for i := 0; i < num; i++ {
		dst[pd] = a[pa] / b[pb]
		pa += strideA
		pb += strideB
		pd++
	}
}

// Sum computes sum of array elements
// Utility function for statistics
func Sum(a []float32, num int, stride int) float32 {
	if num == 0 {
		return 0
	}

	acc := float32(0.0)
	pa := 0

	for i := 0; i < num; i++ {
		acc += a[pa]
		pa += stride
	}

	return acc
}

// SqrSum computes sum of squares of array elements
// Utility function for statistics
func SqrSum(a []float32, num int, stride int) float32 {
	if num == 0 {
		return 0
	}

	acc := float32(0.0)
	pa := 0

	for i := 0; i < num; i++ {
		val := a[pa]
		acc += val * val
		pa += stride
	}

	return acc
}

// StatsArr computes min, max, mean, and standard deviation of array elements in one pass
// Uses numerically stable algorithm for mean and stddev
// min, max, mean, stddev: output parameters
// a: input array
// num: number of elements
// stride: access stride
func StatsArr(min, max, mean, stddev *float32, a []float32, num int, stride int) {
	if num == 0 {
		*min = 0
		*max = 0
		*mean = 0
		*stddev = 0
		return
	}

	if num == 1 {
		val := a[0]
		*min = val
		*max = val
		*mean = val
		*stddev = 0
		return
	}

	// Initialize min and max
	pa := 0
	*min = math32.MaxFloat32
	*max = -math32.MaxFloat32

	// Numerically stable algorithm for mean and stddev
	K := a[0]
	n := float32(num)
	Ex := float32(0.0)
	Ex2 := float32(0.0)

	for i := 0; i < num; i++ {
		val := a[pa]

		// Update min/max
		if val < *min {
			*min = val
		}
		if val > *max {
			*max = val
		}

		// Accumulate for mean and stddev
		tmp := val - K
		Ex += tmp
		Ex2 += tmp * tmp

		pa += stride
	}

	*mean = K + Ex/n
	*stddev = math32.Sqrt((Ex2 - (Ex*Ex)/n) / n)
}

// PercentileArr computes percentile value and sum of values above percentile
// p: percentile value (0.0 to 1.0), e.g., 0.5 for median (p50)
// sumAboveP: output parameter for sum of values > percentile
// a: input array
// num: number of elements
// stride: access stride
// Returns: percentile value
func PercentileArr(p float32, sumAboveP *float32, a []float32, num int, stride int) float32 {
	if num == 0 {
		*sumAboveP = 0
		return 0
	}

	if num == 1 {
		*sumAboveP = 0
		return a[0]
	}

	// Copy and collect values
	values := make([]float32, num)
	pa := 0
	for i := 0; i < num; i++ {
		values[i] = a[pa]
		pa += stride
	}

	// Sort to find percentile
	sort.Slice(values, func(i, j int) bool {
		return values[i] < values[j]
	})

	// Calculate percentile index
	percentileIdx := int(float32(num-1) * p)
	if percentileIdx < 0 {
		percentileIdx = 0
	}
	if percentileIdx >= num {
		percentileIdx = num - 1
	}
	percentileVal := values[percentileIdx]

	// Calculate sum of values above percentile
	*sumAboveP = float32(0.0)
	pa = 0
	for i := 0; i < num; i++ {
		val := a[pa]
		if val > percentileVal {
			*sumAboveP += val
		}
		pa += stride
	}

	return percentileVal
}

// SumArrAdd computes dst[i] += src[i] + c for all i (accumulate)
// DEPRECATED: Use Axpy from level1.go for better performance: Axpy(dst, stride, num, 1.0, src, stride) then SumArrInPlace
// Kept for backward compatibility with vec.go
func SumArrAdd(dst, src []float32, c float32, num int, stride int) {
	if num == 0 {
		return
	}

	// Use Axpy for dst += src, then add c
	Axpy(dst, src, stride, stride, num, 1.0)
	if c != 0 {
		SumArrInPlace(dst, c, num)
	}
}

// DiffArrInPlace computes dst[i] -= c for all i (in-place)
// Utility function for scalar subtraction
func DiffArrInPlace(dst []float32, c float32, num int) {
	if num == 0 {
		return
	}

	for i := 0; i < num; i++ {
		dst[i] -= c
	}
}

// MulArrAdd computes dst[i] += src[i] * c for all i (accumulate)
// DEPRECATED: Use Axpy from level1.go: Axpy(dst, src, stride, stride, num, c)
// Kept for backward compatibility with vec.go
func MulArrAdd(dst, src []float32, c float32, num int, stride int) {
	if num == 0 {
		return
	}

	Axpy(dst, src, stride, stride, num, c)
}

// DivArrInPlace computes dst[i] /= c for all i (in-place)
// DEPRECATED: Use Scal from level1.go: Scal(dst, stride, num, 1.0/c)
// Kept for backward compatibility with vec.go
func DivArrInPlace(dst []float32, c float32, num int) {
	if num == 0 {
		return
	}

	if c == 0 {
		return // Avoid division by zero
	}

	Scal(dst, 1, num, 1.0/c)
}
