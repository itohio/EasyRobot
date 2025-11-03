package tensor

import (
	"errors"
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
)

// Add adds another tensor element-wise in-place.
// Uses fp32 primitive.Axpy for efficient computation.
// Returns the tensor itself for method chaining.
func (t *Tensor) Add(other *Tensor) *Tensor {
	if t == nil || other == nil {
		return t
	}

	if !t.sameShape(other) {
		panic(fmt.Sprintf("tensor.Add: shape mismatch: %v vs %v", t.Dim, other.Dim))
	}

	if !t.isContiguous() || !other.isContiguous() {
		// Handle strided case
		t.addStrided(other)
		return t
	}

	// Use fp32.Axpy for contiguous case
	size := t.Size()
	fp32.Axpy(t.Data, other.Data, 1, 1, size, 1.0)
	return t
}

// Sub subtracts another tensor element-wise in-place.
// Uses fp32 primitive.Axpy with alpha=-1.
func (t *Tensor) Sub(other *Tensor) *Tensor {
	if t == nil || other == nil {
		return t
	}

	if !t.sameShape(other) {
		panic(fmt.Sprintf("tensor.Sub: shape mismatch: %v vs %v", t.Dim, other.Dim))
	}

	if !t.isContiguous() || !other.isContiguous() {
		// Handle strided case
		t.subStrided(other)
		return t
	}

	// Use fp32.Axpy for contiguous case
	size := t.Size()
	fp32.Axpy(t.Data, other.Data, 1, 1, size, -1.0)
	return t
}

// Mul multiplies another tensor element-wise in-place.
func (t *Tensor) Mul(other *Tensor) *Tensor {
	if t == nil || other == nil {
		return t
	}

	if !t.sameShape(other) {
		panic(fmt.Sprintf("tensor.Mul: shape mismatch: %v vs %v", t.Dim, other.Dim))
	}

	t.mulStrided(other)
	return t
}

// Div divides by another tensor element-wise in-place.
func (t *Tensor) Div(other *Tensor) *Tensor {
	if t == nil || other == nil {
		return t
	}

	if !t.sameShape(other) {
		panic(fmt.Sprintf("tensor.Div: shape mismatch: %v vs %v", t.Dim, other.Dim))
	}

	t.divStrided(other)
	return t
}

// Scale multiplies the tensor by a scalar in-place.
// Uses fp32 primitive.Scal for efficient computation.
func (t *Tensor) Scale(scalar float32) *Tensor {
	if t == nil {
		return t
	}

	if !t.isContiguous() {
		// Handle strided case
		t.scaleStrided(scalar)
		return t
	}

	// Use fp32.Scal for contiguous case
	size := t.Size()
	fp32.Scal(t.Data, 1, size, scalar)
	return t
}

// AddTo adds two tensors and stores result in dst (or creates new tensor if dst is nil).
// Returns the destination tensor.
func (t *Tensor) AddTo(other *Tensor, dst *Tensor) *Tensor {
	if t == nil || other == nil {
		return nil
	}

	if !t.sameShape(other) {
		panic(fmt.Sprintf("tensor.AddTo: shape mismatch: %v vs %v", t.Dim, other.Dim))
	}

	if dst == nil {
		dst = t.Clone()
		dst.Add(other)
		return dst
	}

	if !dst.sameShape(t) {
		panic(fmt.Sprintf("tensor.AddTo: destination shape mismatch: %v vs %v", dst.Dim, t.Dim))
	}

	// Copy t to dst, then add other
	t.copyTo(dst)
	dst.Add(other)
	return dst
}

// MulTo multiplies two tensors element-wise and stores result in dst (or creates new tensor if dst is nil).
func (t *Tensor) MulTo(other *Tensor, dst *Tensor) *Tensor {
	if t == nil || other == nil {
		return nil
	}

	if !t.sameShape(other) {
		panic(fmt.Sprintf("tensor.MulTo: shape mismatch: %v vs %v", t.Dim, other.Dim))
	}

	if dst == nil {
		dst = t.Clone()
		dst.Mul(other)
		return dst
	}

	if !dst.sameShape(t) {
		panic(fmt.Sprintf("tensor.MulTo: destination shape mismatch: %v vs %v", dst.Dim, t.Dim))
	}

	// Copy t to dst, then multiply by other
	t.copyTo(dst)
	dst.Mul(other)
	return dst
}

// BroadcastTo broadcasts the tensor to a new shape.
// Currently creates a view-like operation (future: implement efficient broadcasting).
func (t *Tensor) BroadcastTo(shape []int) (*Tensor, error) {
	if t == nil {
		return nil, errors.New("tensor.BroadcastTo: nil tensor")
	}

	if len(shape) < len(t.Dim) {
		return nil, fmt.Errorf("tensor.BroadcastTo: target shape %v has fewer dimensions than %v", shape, t.Dim)
	}

	// Check if broadcasting is possible
	for i := 0; i < len(t.Dim); i++ {
		targetDim := shape[len(shape)-len(t.Dim)+i]
		if t.Dim[i] != 1 && t.Dim[i] != targetDim {
			return nil, fmt.Errorf("tensor.BroadcastTo: cannot broadcast dimension %d: %d to %d", i, t.Dim[i], targetDim)
		}
	}

	// For now, if shapes match exactly, return clone
	// Future: implement efficient broadcasting without copying
	if t.sameShapeInt(shape) {
		return t.Clone(), nil
	}

	// Create broadcasted tensor (simplified implementation)
	// TODO: Implement efficient broadcasting
	broadcasted := &Tensor{
		Dim:  make([]int, len(shape)),
		Data: t.Data,
	}
	copy(broadcasted.Dim, shape)
	return broadcasted, nil
}

// Sum computes sum along specified dimensions.
// If no dimensions specified, sums all elements.
func (t *Tensor) Sum(dims ...int) *Tensor {
	if t == nil {
		return nil
	}

	// If no dimensions specified, sum all elements
	if len(dims) == 0 {
		sum := float32(0)
		if t.isContiguous() {
			size := t.Size()
			sum = fp32.Asum(t.Data, 1, size)
		} else {
			sum = t.sumAllStrided()
		}
		// Return scalar tensor
		return &Tensor{
			Dim:  []int{1},
			Data: []float32{sum},
		}
	}

	// Sum along specified dimensions
	return t.sumDims(dims)
}

// Mean computes mean along specified dimensions.
func (t *Tensor) Mean(dims ...int) *Tensor {
	if t == nil {
		return nil
	}

	sum := t.Sum(dims...)
	if sum == nil {
		return nil
	}

	// Compute count of elements averaged
	count := 1
	if len(dims) == 0 {
		count = t.Size()
	} else {
		for _, dim := range dims {
			if dim >= 0 && dim < len(t.Dim) {
				count *= t.Dim[dim]
			}
		}
	}

	if count > 0 {
		sum.Scale(1.0 / float32(count))
	}

	return sum
}

// Max computes maximum along specified dimensions.
func (t *Tensor) Max(dims ...int) *Tensor {
	if t == nil {
		return nil
	}

	// If no dimensions specified, max of all elements
	if len(dims) == 0 {
		max := t.maxAll()
		return &Tensor{
			Dim:  []int{1},
			Data: []float32{max},
		}
	}

	// Max along specified dimensions
	return t.maxDims(dims)
}

// Min computes minimum along specified dimensions.
func (t *Tensor) Min(dims ...int) *Tensor {
	if t == nil {
		return nil
	}

	// If no dimensions specified, min of all elements
	if len(dims) == 0 {
		min := t.minAll()
		return &Tensor{
			Dim:  []int{1},
			Data: []float32{min},
		}
	}

	// Min along specified dimensions
	return t.minDims(dims)
}

// ArgMax returns indices of maximum elements along specified dimension.
// Uses fp32 primitive.Iamax for vector case.
func (t *Tensor) ArgMax(dim int) *Tensor {
	if t == nil {
		return nil
	}

	if dim < 0 || dim >= len(t.Dim) {
		panic(fmt.Sprintf("tensor.ArgMax: dimension %d out of range for shape %v", dim, t.Dim))
	}

	// For 1D tensor, use primitive.Iamax
	if len(t.Dim) == 1 {
		if !t.isContiguous() {
			// Handle strided case
			return t.argMaxStrided(dim)
		}
		idx := fp32.Iamax(t.Data, 1, t.Size())
		return &Tensor{
			Dim:  []int{1},
			Data: []float32{float32(idx)},
		}
	}

	// Multi-dimensional case
	return t.argMaxDim(dim)
}

// Helper functions

func (t *Tensor) sameShape(other *Tensor) bool {
	if len(t.Dim) != len(other.Dim) {
		return false
	}
	for i := range t.Dim {
		if t.Dim[i] != other.Dim[i] {
			return false
		}
	}
	return true
}

func (t *Tensor) sameShapeInt(shape []int) bool {
	if len(t.Dim) != len(shape) {
		return false
	}
	for i := range t.Dim {
		if t.Dim[i] != shape[i] {
			return false
		}
	}
	return true
}

func (t *Tensor) addStrided(other *Tensor) {
	strides := computeStrides(t.Dim)
	otherStrides := computeStrides(other.Dim)
	indices := make([]int, len(t.Dim))
	t.addStridedRecursive(other, indices, strides, otherStrides, 0)
}

func (t *Tensor) addStridedRecursive(other *Tensor, indices []int, strides, otherStrides []int, dim int) {
	if dim == len(t.Dim) {
		idx := t.elementIndex(indices, strides)
		otherIdx := other.elementIndex(indices, otherStrides)
		t.Data[idx] += other.Data[otherIdx]
		return
	}

	for i := 0; i < t.Dim[dim]; i++ {
		indices[dim] = i
		t.addStridedRecursive(other, indices, strides, otherStrides, dim+1)
	}
}

func (t *Tensor) subStrided(other *Tensor) {
	strides := computeStrides(t.Dim)
	otherStrides := computeStrides(other.Dim)
	indices := make([]int, len(t.Dim))
	t.subStridedRecursive(other, indices, strides, otherStrides, 0)
}

func (t *Tensor) subStridedRecursive(other *Tensor, indices []int, strides, otherStrides []int, dim int) {
	if dim == len(t.Dim) {
		idx := t.elementIndex(indices, strides)
		otherIdx := other.elementIndex(indices, otherStrides)
		t.Data[idx] -= other.Data[otherIdx]
		return
	}

	for i := 0; i < t.Dim[dim]; i++ {
		indices[dim] = i
		t.subStridedRecursive(other, indices, strides, otherStrides, dim+1)
	}
}

func (t *Tensor) mulStrided(other *Tensor) {
	strides := computeStrides(t.Dim)
	otherStrides := computeStrides(other.Dim)
	indices := make([]int, len(t.Dim))
	t.mulStridedRecursive(other, indices, strides, otherStrides, 0)
}

func (t *Tensor) mulStridedRecursive(other *Tensor, indices []int, strides, otherStrides []int, dim int) {
	if dim == len(t.Dim) {
		idx := t.elementIndex(indices, strides)
		otherIdx := other.elementIndex(indices, otherStrides)
		t.Data[idx] *= other.Data[otherIdx]
		return
	}

	for i := 0; i < t.Dim[dim]; i++ {
		indices[dim] = i
		t.mulStridedRecursive(other, indices, strides, otherStrides, dim+1)
	}
}

func (t *Tensor) divStrided(other *Tensor) {
	strides := computeStrides(t.Dim)
	otherStrides := computeStrides(other.Dim)
	indices := make([]int, len(t.Dim))
	t.divStridedRecursive(other, indices, strides, otherStrides, 0)
}

func (t *Tensor) divStridedRecursive(other *Tensor, indices []int, strides, otherStrides []int, dim int) {
	if dim == len(t.Dim) {
		idx := t.elementIndex(indices, strides)
		otherIdx := other.elementIndex(indices, otherStrides)
		if other.Data[otherIdx] != 0 {
			t.Data[idx] /= other.Data[otherIdx]
		}
		return
	}

	for i := 0; i < t.Dim[dim]; i++ {
		indices[dim] = i
		t.divStridedRecursive(other, indices, strides, otherStrides, dim+1)
	}
}

func (t *Tensor) scaleStrided(scalar float32) {
	strides := computeStrides(t.Dim)
	indices := make([]int, len(t.Dim))
	t.scaleStridedRecursive(scalar, indices, strides, 0)
}

func (t *Tensor) scaleStridedRecursive(scalar float32, indices []int, strides []int, dim int) {
	if dim == len(t.Dim) {
		idx := t.elementIndex(indices, strides)
		t.Data[idx] *= scalar
		return
	}

	for i := 0; i < t.Dim[dim]; i++ {
		indices[dim] = i
		t.scaleStridedRecursive(scalar, indices, strides, dim+1)
	}
}

func (t *Tensor) copyTo(dst *Tensor) {
	if t.isContiguous() && dst.isContiguous() && t.Size() == dst.Size() {
		size := t.Size()
		fp32.Copy(dst.Data, t.Data, 1, 1, size)
		return
	}

	// Strided copy
	strides := computeStrides(t.Dim)
	dstStrides := computeStrides(dst.Dim)
	indices := make([]int, len(t.Dim))
	t.copyStridedRecursive(dst, indices, strides, dstStrides, 0)
}

func (t *Tensor) copyStridedRecursive(dst *Tensor, indices []int, strides, dstStrides []int, dim int) {
	if dim == len(t.Dim) {
		idx := t.elementIndex(indices, strides)
		dstIdx := dst.elementIndex(indices, dstStrides)
		dst.Data[dstIdx] = t.Data[idx]
		return
	}

	for i := 0; i < t.Dim[dim]; i++ {
		indices[dim] = i
		t.copyStridedRecursive(dst, indices, strides, dstStrides, dim+1)
	}
}

func (t *Tensor) sumAllStrided() float32 {
	strides := computeStrides(t.Dim)
	indices := make([]int, len(t.Dim))
	var sum float32
	t.sumAllStridedRecursive(&sum, indices, strides, 0)
	return sum
}

func (t *Tensor) sumAllStridedRecursive(sum *float32, indices []int, strides []int, dim int) {
	if dim == len(t.Dim) {
		idx := t.elementIndex(indices, strides)
		*sum += t.Data[idx]
		return
	}

	for i := 0; i < t.Dim[dim]; i++ {
		indices[dim] = i
		t.sumAllStridedRecursive(sum, indices, strides, dim+1)
	}
}

func (t *Tensor) sumDims(dims []int) *Tensor {
	// Create output shape (remove summed dimensions)
	newShape := make([]int, 0, len(t.Dim))
	dimSet := make(map[int]bool)
	for _, d := range dims {
		dimSet[d] = true
	}

	for i, d := range t.Dim {
		if !dimSet[i] {
			newShape = append(newShape, d)
		}
	}

	if len(newShape) == 0 {
		newShape = []int{1}
	}

	result := &Tensor{
		Dim:  newShape,
		Data: make([]float32, sizeFromShape(newShape)),
	}

	// Compute sums
	indices := make([]int, len(t.Dim))
	resultIndices := make([]int, len(newShape))
	t.sumDimsRecursive(result, indices, resultIndices, dims, dimSet, 0, 0)

	return result
}

func (t *Tensor) sumDimsRecursive(result *Tensor, indices, resultIndices []int, dims []int, dimSet map[int]bool, dim, resultDim int) {
	if dim == len(t.Dim) {
		// Map result indices
		ri := 0
		for i := 0; i < len(t.Dim); i++ {
			if !dimSet[i] {
				if ri < len(resultIndices) {
					resultIndices[ri] = indices[i]
					ri++
				}
			}
		}
		resultStrides := computeStrides(result.Dim)
		resultIdx := result.elementIndex(resultIndices, resultStrides)
		tStrides := computeStrides(t.Dim)
		tIdx := t.elementIndex(indices, tStrides)
		result.Data[resultIdx] += t.Data[tIdx]
		return
	}

	for i := 0; i < t.Dim[dim]; i++ {
		indices[dim] = i
		t.sumDimsRecursive(result, indices, resultIndices, dims, dimSet, dim+1, resultDim)
	}
}

func (t *Tensor) maxAll() float32 {
	if t.Size() == 0 {
		return 0
	}
	max := t.Data[0]
	for i := 1; i < len(t.Data); i++ {
		if t.Data[i] > max {
			max = t.Data[i]
		}
	}
	return max
}

func (t *Tensor) minAll() float32 {
	if t.Size() == 0 {
		return 0
	}
	min := t.Data[0]
	for i := 1; i < len(t.Data); i++ {
		if t.Data[i] < min {
			min = t.Data[i]
		}
	}
	return min
}

func (t *Tensor) maxDims(dims []int) *Tensor {
	// Similar to sumDims but compute max instead
	newShape := make([]int, 0, len(t.Dim))
	dimSet := make(map[int]bool)
	for _, d := range dims {
		dimSet[d] = true
	}

	for i, d := range t.Dim {
		if !dimSet[i] {
			newShape = append(newShape, d)
		}
	}

	if len(newShape) == 0 {
		newShape = []int{1}
	}

	result := &Tensor{
		Dim:  newShape,
		Data: make([]float32, sizeFromShape(newShape)),
	}

	// Initialize with very small values
	for i := range result.Data {
		result.Data[i] = -1e30
	}

	indices := make([]int, len(t.Dim))
	resultIndices := make([]int, len(newShape))
	t.maxDimsRecursive(result, indices, resultIndices, dims, dimSet, 0, 0)

	return result
}

func (t *Tensor) maxDimsRecursive(result *Tensor, indices, resultIndices []int, dims []int, dimSet map[int]bool, dim, resultDim int) {
	if dim == len(t.Dim) {
		ri := 0
		for i := 0; i < len(t.Dim); i++ {
			if !dimSet[i] {
				if ri < len(resultIndices) {
					resultIndices[ri] = indices[i]
					ri++
				}
			}
		}
		resultStrides := computeStrides(result.Dim)
		resultIdx := result.elementIndex(resultIndices, resultStrides)
		tStrides := computeStrides(t.Dim)
		tIdx := t.elementIndex(indices, tStrides)
		if t.Data[tIdx] > result.Data[resultIdx] {
			result.Data[resultIdx] = t.Data[tIdx]
		}
		return
	}

	for i := 0; i < t.Dim[dim]; i++ {
		indices[dim] = i
		t.maxDimsRecursive(result, indices, resultIndices, dims, dimSet, dim+1, resultDim)
	}
}

func (t *Tensor) minDims(dims []int) *Tensor {
	newShape := make([]int, 0, len(t.Dim))
	dimSet := make(map[int]bool)
	for _, d := range dims {
		dimSet[d] = true
	}

	for i, d := range t.Dim {
		if !dimSet[i] {
			newShape = append(newShape, d)
		}
	}

	if len(newShape) == 0 {
		newShape = []int{1}
	}

	result := &Tensor{
		Dim:  newShape,
		Data: make([]float32, sizeFromShape(newShape)),
	}

	// Initialize with very large values
	for i := range result.Data {
		result.Data[i] = 1e30
	}

	indices := make([]int, len(t.Dim))
	resultIndices := make([]int, len(newShape))
	t.minDimsRecursive(result, indices, resultIndices, dims, dimSet, 0, 0)

	return result
}

func (t *Tensor) minDimsRecursive(result *Tensor, indices, resultIndices []int, dims []int, dimSet map[int]bool, dim, resultDim int) {
	if dim == len(t.Dim) {
		ri := 0
		for i := 0; i < len(t.Dim); i++ {
			if !dimSet[i] {
				if ri < len(resultIndices) {
					resultIndices[ri] = indices[i]
					ri++
				}
			}
		}
		resultStrides := computeStrides(result.Dim)
		resultIdx := result.elementIndex(resultIndices, resultStrides)
		tStrides := computeStrides(t.Dim)
		tIdx := t.elementIndex(indices, tStrides)
		if t.Data[tIdx] < result.Data[resultIdx] {
			result.Data[resultIdx] = t.Data[tIdx]
		}
		return
	}

	for i := 0; i < t.Dim[dim]; i++ {
		indices[dim] = i
		t.minDimsRecursive(result, indices, resultIndices, dims, dimSet, dim+1, resultDim)
	}
}

func (t *Tensor) argMaxStrided(dim int) *Tensor {
	// For 1D case with striding
	strides := computeStrides(t.Dim)
	maxVal := t.Data[0]
	maxIdx := 0
	for i := 1; i < t.Size(); i++ {
		val := t.Data[i*strides[0]]
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}
	return &Tensor{
		Dim:  []int{1},
		Data: []float32{float32(maxIdx)},
	}
}

func (t *Tensor) argMaxDim(dim int) *Tensor {
	// Create output shape (remove the dimension we're argmaxing over)
	newShape := make([]int, 0, len(t.Dim)-1)
	for i, d := range t.Dim {
		if i != dim {
			newShape = append(newShape, d)
		}
	}

	if len(newShape) == 0 {
		newShape = []int{1}
	}

	result := &Tensor{
		Dim:  newShape,
		Data: make([]float32, sizeFromShape(newShape)),
	}

	// Initialize with zeros and track max indices
	maxVals := make([]float32, sizeFromShape(newShape))
	for i := range maxVals {
		maxVals[i] = -1e30
	}

	indices := make([]int, len(t.Dim))
	resultIndices := make([]int, len(newShape))
	t.argMaxDimRecursive(result, maxVals, indices, resultIndices, dim, 0)

	return result
}

func (t *Tensor) argMaxDimRecursive(result *Tensor, maxVals []float32, indices, resultIndices []int, argMaxDim, dim int) {
	if dim == len(t.Dim) {
		// Build result indices (excluding argMaxDim)
		ri := 0
		for i := 0; i < len(t.Dim); i++ {
			if i != argMaxDim {
				if ri < len(resultIndices) {
					resultIndices[ri] = indices[i]
					ri++
				}
			}
		}
		resultStrides := computeStrides(result.Dim)
		resultIdx := result.elementIndex(resultIndices, resultStrides)

		tStrides := computeStrides(t.Dim)
		tIdx := t.elementIndex(indices, tStrides)
		val := t.Data[tIdx]

		if val > maxVals[resultIdx] {
			maxVals[resultIdx] = val
			result.Data[resultIdx] = float32(indices[argMaxDim])
		}
		return
	}

	if dim == argMaxDim {
		// Iterate over argMaxDim dimension
		for i := 0; i < t.Dim[dim]; i++ {
			indices[dim] = i
			t.argMaxDimRecursive(result, maxVals, indices, resultIndices, argMaxDim, dim+1)
		}
	} else {
		// Iterate over other dimensions
		for i := 0; i < t.Dim[dim]; i++ {
			indices[dim] = i
			t.argMaxDimRecursive(result, maxVals, indices, resultIndices, argMaxDim, dim+1)
		}
	}
}

func sizeFromShape(shape []int) int {
	if len(shape) == 0 {
		return 1
	}
	size := 1
	for _, d := range shape {
		size *= d
	}
	return size
}
