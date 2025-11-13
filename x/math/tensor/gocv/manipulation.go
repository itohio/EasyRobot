package gocv

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
	cv "gocv.io/x/gocv"
)

// Clone performs a deep copy of the underlying Mat.
func (t Tensor) Clone() types.Tensor {
	h, err := t.getHandle()
	if err != nil {
		panic(err)
	}

	cloned := h.mat.Clone()
	return Tensor{
		handle: &tensorHandle{
			mat:   &cloned,
			shape: append(types.Shape(nil), h.shape...),
			dtype: h.dtype,
			owns:  true,
		},
	}
}

// Copy copies contents from src into this tensor. Only GoCV tensors with the
// same dtype and shape are supported.
func (t Tensor) Copy(src types.Tensor) types.Tensor {
	if src == nil {
		return t
	}

	dstHandle, err := t.getHandle()
	if err != nil {
		panic(err)
	}

	srcTensor, ok := src.(Tensor)
	if !ok {
		panic(fmt.Errorf("%w: Copy expects gocv.Tensor source", ErrUnsupported))
	}

	srcHandle, err := srcTensor.getHandle()
	if err != nil {
		panic(err)
	}

	if !dstHandle.shape.Equal(srcHandle.shape) {
		panic(fmt.Errorf("gocv tensor: Copy shape mismatch %v vs %v", dstHandle.shape, srcHandle.shape))
	}

	if dstHandle.dtype != srcHandle.dtype {
		panic(fmt.Errorf("gocv tensor: Copy dtype mismatch %v vs %v", dstHandle.dtype, srcHandle.dtype))
	}

	srcHandle.mat.CopyTo(dstHandle.mat)
	return t
}

// Reshape currently supports returning the same tensor when newShape matches
// the existing shape. Other reshape scenarios panic with ErrUnsupported.
func (t Tensor) Reshape(dst types.Tensor, newShape types.Shape) types.Tensor {
	h, err := t.getHandle()
	if err != nil {
		panic(err)
	}

	if !h.shape.Equal(newShape) {
		panicUnsupported("Reshape (only identical shapes supported)")
	}

	if dst == nil {
		return t
	}

	dstTensor, ok := dst.(Tensor)
	if !ok {
		panic(fmt.Errorf("%w: Reshape destination must be gocv.Tensor", ErrUnsupported))
	}

	dstHandle, err := dstTensor.getHandle()
	if err != nil {
		panic(err)
	}

	if !dstHandle.shape.Equal(h.shape) {
		panic(fmt.Errorf("gocv tensor: Reshape dst shape mismatch %v vs %v", dstHandle.shape, h.shape))
	}

	return dst
}

// Slice is not yet implemented for GoCV tensors.
func (t Tensor) Slice(dst types.Tensor, dim int, start int, length int) types.Tensor {
	panicUnsupported("Slice")
	return nil
}

// Transpose swaps the first two dimensions ([H, W, C] -> [W, H, C]).
func (t Tensor) Transpose(dst types.Tensor, dims []int) types.Tensor {
	h, err := t.getHandle()
	if err != nil {
		panic(err)
	}

	if len(h.shape) != 3 {
		panicUnsupported("Transpose (rank != 3)")
	}

	permutation := []int{1, 0, 2}
	if len(dims) > 0 {
		if len(dims) != 3 {
			panicUnsupported("Transpose (dims length mismatch)")
		}
		permutation = dims
	}

	if permutation[0] != 1 || permutation[1] != 0 || permutation[2] != 2 {
		panicUnsupported("Transpose (unsupported permutation)")
	}

	outShape := types.Shape{h.shape[1], h.shape[0], h.shape[2]}

	var target Tensor
	if dst != nil {
		var ok bool
		target, ok = dst.(Tensor)
		if !ok {
			panic(fmt.Errorf("%w: Transpose destination must be gocv.Tensor", ErrUnsupported))
		}
		dstHandle, err := target.getHandle()
		if err != nil {
			panic(err)
		}
		if !dstHandle.shape.Equal(outShape) {
			panic(fmt.Errorf("gocv tensor: Transpose dst shape mismatch %v vs %v", dstHandle.shape, outShape))
		}
	} else {
		matType, err := dataTypeToMatType(h.dtype, outShape[2])
		if err != nil {
			panic(err)
		}
		mat := cv.NewMatWithSize(outShape[0], outShape[1], matType)
		target = Tensor{
			handle: &tensorHandle{
				mat:   &mat,
				shape: outShape,
				dtype: h.dtype,
				owns:  true,
			},
		}
	}

	dstHandle, err := target.getHandle()
	if err != nil {
		panic(err)
	}

	cv.Transpose(*h.mat, dstHandle.mat)
	return target
}

// Permute currently supports only the identity permutation.
func (t Tensor) Permute(dst types.Tensor, dims []int) types.Tensor {
	h, err := t.getHandle()
	if err != nil {
		panic(err)
	}

	if len(dims) == 0 || (len(dims) == len(h.shape) && isIdentityPermutation(dims)) {
		if dst == nil {
			return t
		}
		dstTensor, ok := dst.(Tensor)
		if !ok {
			panic(fmt.Errorf("%w: Permute destination must be gocv.Tensor", ErrUnsupported))
		}
		return dstTensor.Copy(t)
	}

	panicUnsupported("Permute")
	return nil
}

func isIdentityPermutation(dims []int) bool {
	for i, v := range dims {
		if i != v {
			return false
		}
	}
	return true
}

// BroadcastTo is not supported for GoCV tensors.
func (t Tensor) BroadcastTo(dst types.Tensor, shape types.Shape) types.Tensor {
	panicUnsupported("BroadcastTo")
	return nil
}

// Fill writes a scalar value into the tensor using Mat.SetTo.
func (t Tensor) Fill(dst types.Tensor, value float64) types.Tensor {
	if dst != nil {
		panicUnsupported("Fill (dst parameter)")
	}

	h, err := t.getHandle()
	if err != nil {
		panic(err)
	}

	scalar := cv.NewScalar(value, value, value, value)
	h.mat.SetTo(scalar)
	return t
}

// FillFunc fills the tensor by invoking callback for each element (slow path).
func (t Tensor) FillFunc(dst types.Tensor, f func() float64) types.Tensor {
	if dst != nil {
		panicUnsupported("FillFunc (dst parameter)")
	}

	if f == nil {
		return t
	}

	for i := 0; i < t.Size(); i++ {
		t.SetAt(f(), i)
	}
	return t
}

// Pad is not supported.
func (t Tensor) Pad(dst types.Tensor, padding []int, value float64) types.Tensor {
	panicUnsupported("Pad")
	return nil
}

// Unpad is not supported.
func (t Tensor) Unpad(dst types.Tensor, padding []int) types.Tensor {
	panicUnsupported("Unpad")
	return nil
}
