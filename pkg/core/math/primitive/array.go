package primitive

import (
	"github.com/chewxy/math32"
)

// SumArr computes dst[i] = a[i] + b[i] for all i
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

// SumArrConst computes dst[i] = src[i] + c for all i
func SumArrConst(dst, src []float32, c float32, num int, stride int) {
	if num == 0 {
		return
	}

	ps := 0
	pd := 0

	for i := 0; i < num; i++ {
		dst[pd] = src[ps] + c
		ps += stride
		pd++
	}
}

// DiffArrConst computes dst[i] = src[i] - c for all i
func DiffArrConst(dst, src []float32, c float32, num int, stride int) {
	if num == 0 {
		return
	}

	ps := 0
	pd := 0

	for i := 0; i < num; i++ {
		dst[pd] = src[ps] - c
		ps += stride
		pd++
	}
}

// MulArrConst computes dst[i] = src[i] * c for all i
func MulArrConst(dst, src []float32, c float32, num int, stride int) {
	if num == 0 {
		return
	}

	ps := 0
	pd := 0

	for i := 0; i < num; i++ {
		dst[pd] = src[ps] * c
		ps += stride
		pd++
	}
}

// DivArrConst computes dst[i] = src[i] / c for all i
func DivArrConst(dst, src []float32, c float32, num int, stride int) {
	if num == 0 {
		return
	}

	ps := 0
	pd := 0

	for i := 0; i < num; i++ {
		dst[pd] = src[ps] / c
		ps += stride
		pd++
	}
}

// SumArrInPlace computes dst[i] += c for all i (in-place)
func SumArrInPlace(dst []float32, c float32, num int) {
	if num == 0 {
		return
	}

	for i := 0; i < num; i++ {
		dst[i] += c
	}
}

// DiffArrInPlace computes dst[i] -= c for all i (in-place)
func DiffArrInPlace(dst []float32, c float32, num int) {
	if num == 0 {
		return
	}

	for i := 0; i < num; i++ {
		dst[i] -= c
	}
}

// MulArrInPlace computes dst[i] *= c for all i (in-place)
func MulArrInPlace(dst []float32, c float32, num int) {
	if num == 0 {
		return
	}

	for i := 0; i < num; i++ {
		dst[i] *= c
	}
}

// DivArrInPlace computes dst[i] /= c for all i (in-place)
func DivArrInPlace(dst []float32, c float32, num int) {
	if num == 0 {
		return
	}

	for i := 0; i < num; i++ {
		dst[i] /= c
	}
}

// SumArrAdd computes dst[i] += src[i] + c for all i (accumulate)
func SumArrAdd(dst, src []float32, c float32, num int, stride int) {
	if num == 0 {
		return
	}

	ps := 0
	pd := 0

	for i := 0; i < num; i++ {
		dst[pd] += src[ps] + c
		ps += stride
		pd++
	}
}

// DiffArrAdd computes dst[i] += src[i] - c for all i (accumulate)
func DiffArrAdd(dst, src []float32, c float32, num int, stride int) {
	if num == 0 {
		return
	}

	ps := 0
	pd := 0

	for i := 0; i < num; i++ {
		dst[pd] += src[ps] - c
		ps += stride
		pd++
	}
}

// MulArrAdd computes dst[i] += src[i] * c for all i (accumulate)
func MulArrAdd(dst, src []float32, c float32, num int, stride int) {
	if num == 0 {
		return
	}

	ps := 0
	pd := 0

	for i := 0; i < num; i++ {
		dst[pd] += src[ps] * c
		ps += stride
		pd++
	}
}

// DivArrAdd computes dst[i] += src[i] / c for all i (accumulate)
func DivArrAdd(dst, src []float32, c float32, num int, stride int) {
	if num == 0 {
		return
	}

	ps := 0
	pd := 0

	for i := 0; i < num; i++ {
		dst[pd] += src[ps] / c
		ps += stride
		pd++
	}
}

// Sum computes sum of array elements
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

// MinArr finds minimum value in array, returns value and index
func MinArr(a []float32, num int, stride int) (min float32, index int) {
	if num == 0 {
		return 0, -1
	}

	min = math32.MaxFloat32
	index = 0
	pa := 0

	for i := 0; i < num; i++ {
		val := a[pa]
		if val < min {
			min = val
			index = i
		}
		pa += stride
	}

	return min, index
}

// MaxArr finds maximum value in array, returns value and index
func MaxArr(a []float32, num int, stride int) (max float32, index int) {
	if num == 0 {
		return 0, -1
	}

	max = -math32.MaxFloat32
	index = 0
	pa := 0

	for i := 0; i < num; i++ {
		val := a[pa]
		if val > max {
			max = val
			index = i
		}
		pa += stride
	}

	return max, index
}

// MeanArr computes mean of array elements
func MeanArr(a []float32, num int, stride int) float32 {
	if num == 0 {
		return 0
	}

	return Sum(a, num, stride) / float32(num)
}

// MomentsArr computes mean and standard deviation of array
// Uses numerically stable algorithm
func MomentsArr(mean, stddev *float32, a []float32, num int, stride int) {
	if num < 2 {
		if num == 1 {
			*mean = a[0]
			*stddev = 0
		}
		return
	}

	K := a[0]
	n := float32(num)
	Ex := float32(0.0)
	Ex2 := float32(0.0)
	pa := 0

	for i := 0; i < num; i++ {
		tmp := a[pa] - K
		Ex += tmp
		Ex2 += tmp * tmp
		pa += stride
	}

	*mean = K + Ex/n
	*stddev = math32.Sqrt((Ex2 - (Ex*Ex)/n) / n)
}

// WeightedMomentsArr computes weighted mean and standard deviation
func WeightedMomentsArr(mean, stddev *float32, a, w []float32, num int, stride int) {
	if num < 2 {
		if num == 1 {
			*mean = a[0] * w[0]
			*stddev = 0
		}
		return
	}

	K := a[0] * w[0]
	n := float32(num)
	Ex := float32(0.0)
	Ex2 := float32(0.0)
	pa := 0
	pw := 0

	for i := 0; i < num; i++ {
		tmp := a[pa]*w[pw] - K
		Ex += tmp
		Ex2 += tmp * tmp
		pa += stride
		pw += stride
	}

	*mean = K + Ex/n
	*stddev = math32.Sqrt((Ex2 - (Ex*Ex)/n) / n)
}
