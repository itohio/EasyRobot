package gocv

import (
	"fmt"
	"sync/atomic"
	"unsafe"

	"github.com/itohio/EasyRobot/x/math/tensor/types"
	cv "gocv.io/x/gocv"
)

// Tensor is a value receiver wrapper around a gocv.Mat handle. The handle is
// shared across copies of Tensor, so Release is safe to call on any copy.
type Tensor struct {
	handle *tensorHandle
}

type tensorHandle struct {
	mat      *cv.Mat
	shape    types.Shape
	dtype    types.DataType
	owns     bool
	released atomic.Bool
}

var _ types.Tensor = Tensor{}

// NewImage creates a new image tensor with the specified dimensions and channels.
// The tensor is initialized with zeros (black image).
// rows and cols specify the image dimensions, channels must be 1, 3, or 4.
func NewImage(rows, cols, channels int, opts ...Option) (types.Tensor, error) {
	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("gocv tensor: invalid dimensions: rows=%d cols=%d", rows, cols)
	}
	if channels < 1 || channels > 4 {
		return nil, fmt.Errorf("gocv tensor: invalid channels: %d (must be 1-4)", channels)
	}

	matType, err := dataTypeToMatType(types.UINT8, channels)
	if err != nil {
		return nil, err
	}

	mat := cv.NewMatWithSize(rows, cols, matType)
	mat.SetTo(cv.NewScalar(0, 0, 0, 0))

	cfg := constructorConfig{ownership: ownershipAdopt}
	for _, opt := range opts {
		opt(&cfg)
	}

	shape := types.Shape{rows, cols, channels}
	return Tensor{
		handle: &tensorHandle{
			mat:   &mat,
			shape: shape,
			dtype: types.UINT8,
			owns:  cfg.ownership == ownershipAdopt,
		},
	}, nil
}

// FromMat constructs a tensor from an existing Mat. By default the Mat is
// cloned so the tensor owns its memory. Use WithAdoptedMat or WithSharedMat to
// override.
func FromMat(mat cv.Mat, opts ...Option) (types.Tensor, error) {
	cfg := constructorConfig{ownership: ownershipClone}
	for _, opt := range opts {
		opt(&cfg)
	}

	var (
		target cv.Mat
		owns   bool
	)

	switch cfg.ownership {
	case ownershipClone:
		target = mat.Clone()
		owns = true
	case ownershipAdopt:
		target = mat
		owns = true
	case ownershipShare:
		target = mat
		owns = false
	default:
		return nil, fmt.Errorf("%w: unknown ownership mode %d", ErrUnsupported, cfg.ownership)
	}

	if target.Empty() {
		return Tensor{handle: &tensorHandle{
			mat:   &target,
			shape: nil,
			dtype: types.DT_UNKNOWN,
			owns:  owns,
		}}, nil
	}

	shape, dtype, err := inferMetadata(target)
	if err != nil {
		if owns {
			target.Close()
		}
		return nil, err
	}

	return Tensor{
		handle: &tensorHandle{
			mat:   &target,
			shape: shape,
			dtype: dtype,
			owns:  owns,
		},
	}, nil
}

func inferMetadata(mat cv.Mat) (types.Shape, types.DataType, error) {
	mt := mat.Type()
	dtype, err := matTypeToDataType(mt)
	if err != nil {
		return nil, types.DT_UNKNOWN, err
	}

	rows, cols := mat.Rows(), mat.Cols()
	ch := mat.Channels()
	if rows <= 0 || cols <= 0 || ch <= 0 {
		return nil, types.DT_UNKNOWN, fmt.Errorf("gocv tensor: invalid mat dimensions: rows=%d cols=%d channels=%d", rows, cols, ch)
	}

	shape := types.Shape{rows, cols, ch}
	return shape, dtype, nil
}

func matTypeToDataType(mt cv.MatType) (types.DataType, error) {
	switch mt {
	case cv.MatTypeCV8UC1, cv.MatTypeCV8UC2, cv.MatTypeCV8UC3, cv.MatTypeCV8UC4:
		return types.UINT8, nil
	case cv.MatTypeCV32FC1, cv.MatTypeCV32FC2, cv.MatTypeCV32FC3, cv.MatTypeCV32FC4:
		return types.FP32, nil
	default:
		return types.DT_UNKNOWN, fmt.Errorf("%w: %v", ErrUnsupportedDepth, mt)
	}
}

func dataTypeToMatType(dtype types.DataType, channels int) (cv.MatType, error) {
	switch dtype {
	case types.UINT8:
		switch channels {
		case 1:
			return cv.MatTypeCV8UC1, nil
		case 2:
			return cv.MatTypeCV8UC2, nil
		case 3:
			return cv.MatTypeCV8UC3, nil
		case 4:
			return cv.MatTypeCV8UC4, nil
		default:
			return 0, fmt.Errorf("%w: uint8 channels=%d", ErrUnsupported, channels)
		}
	case types.FP32:
		switch channels {
		case 1:
			return cv.MatTypeCV32FC1, nil
		case 2:
			return cv.MatTypeCV32FC2, nil
		case 3:
			return cv.MatTypeCV32FC3, nil
		case 4:
			return cv.MatTypeCV32FC4, nil
		default:
			return 0, fmt.Errorf("%w: fp32 channels=%d", ErrUnsupported, channels)
		}
	default:
		return 0, fmt.Errorf("%w: dataTypeToMatType(%v)", ErrUnsupported, dtype)
	}
}

func (t Tensor) getHandle() (*tensorHandle, error) {
	if t.handle == nil || t.handle.mat == nil {
		return nil, ErrNilMat
	}
	if t.handle.released.Load() {
		return nil, ErrReleased
	}
	return t.handle, nil
}

// Empty reports whether the tensor has no underlying data.
func (t Tensor) Empty() bool {
	if t.handle == nil || t.handle.mat == nil {
		return true
	}
	if t.handle.released.Load() {
		return true
	}
	return t.handle.shape == nil || t.handle.mat.Empty()
}

// ID returns a unique identifier derived from the underlying Mat pointer.
func (t Tensor) ID() uintptr {
	h, err := t.getHandle()
	if err != nil || h.mat == nil {
		return 0
	}
	return uintptr(unsafe.Pointer(h.mat))
}

// DataType returns the tensor data type.
func (t Tensor) DataType() types.DataType {
	h, err := t.getHandle()
	if err != nil {
		return types.DT_UNKNOWN
	}
	return h.dtype
}

// Shape returns the tensor shape in [rows, cols, channels] format.
func (t Tensor) Shape() types.Shape {
	h, err := t.getHandle()
	if err != nil {
		return nil
	}
	if h.shape == nil {
		return nil
	}
	return append(types.Shape(nil), h.shape...)
}

// Rank returns the number of dimensions (always 3 for images).
func (t Tensor) Rank() int {
	h, err := t.getHandle()
	if err != nil || h.shape == nil {
		return 0
	}
	return len(h.shape)
}

// Size returns the number of elements.
func (t Tensor) Size() int {
	h, err := t.getHandle()
	if err != nil || h.shape == nil {
		return 0
	}
	return h.shape.Size()
}

// Strides returns canonical row-major strides assuming contiguous storage.
func (t Tensor) Strides(dst []int) []int {
	h, err := t.getHandle()
	if err != nil || h.shape == nil {
		return nil
	}
	return h.shape.Strides(dst)
}

// IsContiguous reports whether the tensor memory is contiguous.
func (t Tensor) IsContiguous() bool {
	h, err := t.getHandle()
	if err != nil || h.mat == nil {
		return false
	}
	return h.mat.IsContinuous()
}

// Offset always returns zero because GoCV tensors do not expose sub-views yet.
func (t Tensor) Offset() int {
	return 0
}

// DataWithOffset returns the same result as Data because offset is zero.
func (t Tensor) DataWithOffset() any {
	return t.Data()
}

// Data exposes a read/write slice view of the underlying Mat buffer.
func (t Tensor) Data() any {
	h, err := t.getHandle()
	if err != nil || h.mat == nil || h.shape == nil {
		return nil
	}

	length := h.shape.Size()
	switch h.dtype {
	case types.UINT8:
		data, err := h.dataUint8()
		if err != nil {
			panic(err)
		}
		if len(data) < length {
			return data
		}
		return data[:length]
	case types.FP32:
		data, err := h.dataFloat32()
		if err != nil {
			panic(err)
		}
		if len(data) < length {
			return data
		}
		return data[:length]
	default:
		return nil
	}
}

func validateIndices(shape types.Shape, idx []int) {
	if len(idx) != len(shape) {
		panic("gocv tensor: index dimensionality mismatch")
	}
	for axis, v := range idx {
		if v < 0 || v >= shape[axis] {
			panic(fmt.Sprintf("gocv tensor: index %d out of range for axis %d (size %d)", v, axis, shape[axis]))
		}
	}
}

func (t Tensor) linearIndex(shape types.Shape, indices []int) int {
	switch len(indices) {
	case 1:
		return indices[0]
	case len(shape):
		strides := shape.Strides(nil)
		idx := 0
		for i, v := range indices {
			idx += v * strides[i]
		}
		return idx
	default:
		panic("gocv tensor: invalid index arity")
	}
}

// At returns the element at the given indices. For uint8 tensors the value is
// promoted to float64 preserving the unsigned range.
func (t Tensor) At(indices ...int) float64 {
	h, err := t.getHandle()
	if err != nil {
		panic(err)
	}
	if len(indices) == 0 {
		if h.shape == nil {
			panic("gocv tensor: empty tensor access")
		}
		indices = []int{0}
	}

	if len(indices) != 1 && len(indices) != len(h.shape) {
		panic("gocv tensor: index dimensionality mismatch")
	}

	if len(indices) == len(h.shape) && len(indices) > 1 {
		validateIndices(h.shape, indices)
	}

	size := h.shape.Size()
	idx := t.linearIndex(h.shape, indices)
	if idx < 0 || idx >= size {
		panic("gocv tensor: index out of bounds")
	}

	switch h.dtype {
	case types.UINT8:
		data, err := h.dataUint8()
		if err != nil {
			panic(err)
		}
		val := data[idx]
		return float64(val)
	case types.FP32:
		data, err := h.dataFloat32()
		if err != nil {
			panic(err)
		}
		val := data[idx]
		return float64(val)
	default:
		panic(fmt.Errorf("%w: At for %v", ErrUnsupported, h.dtype))
	}
}

// SetAt writes a value at the given indices. Values are clamped to the valid
// range for uint8 tensors.
func (t Tensor) SetAt(value float64, indices ...int) {
	h, err := t.getHandle()
	if err != nil {
		panic(err)
	}
	if len(indices) == 0 {
		if h.shape == nil {
			panic("gocv tensor: empty tensor write")
		}
		indices = []int{0}
	}

	if len(indices) != 1 && len(indices) != len(h.shape) {
		panic("gocv tensor: index dimensionality mismatch")
	}

	if len(indices) == len(h.shape) && len(indices) > 1 {
		validateIndices(h.shape, indices)
	}

	size := h.shape.Size()
	idx := t.linearIndex(h.shape, indices)
	if idx < 0 || idx >= size {
		panic("gocv tensor: index out of bounds")
	}

	switch h.dtype {
	case types.UINT8:
		data, err := h.dataUint8()
		if err != nil {
			panic(err)
		}
		slice := data
		switch {
		case value < 0:
			slice[idx] = 0
		case value > 255:
			slice[idx] = 255
		default:
			slice[idx] = uint8(value + 0.5)
		}
	case types.FP32:
		data, err := h.dataFloat32()
		if err != nil {
			panic(err)
		}
		data[idx] = float32(value)
	default:
		panic(fmt.Errorf("%w: SetAt for %v", ErrUnsupported, h.dtype))
	}
}

// Elements iterates row-major over tensor elements.
func (t Tensor) Elements(fixedAxisValuePairs ...int) func(func(types.Element) bool) {
	h, err := t.getHandle()
	if err != nil || h.shape == nil {
		return func(yield func(types.Element) bool) {}
	}

	return func(yield func(types.Element) bool) {
		shape := h.shape
		for indices := range shape.Iterator(fixedAxisValuePairs...) {
			validateIndices(shape, indices)
			idx := t.linearIndex(shape, indices)
			elem := element{tensor: t, index: idx}
			if !yield(elem) {
				return
			}
		}
	}
}

type element struct {
	tensor Tensor
	index  int
}

func (e element) Get() float64 {
	return e.tensor.At(e.index)
}

func (e element) Set(value float64) {
	e.tensor.SetAt(value, e.index)
}

// Release closes the underlying Mat if this tensor owns it.
func (t Tensor) Release() {
	if t.handle == nil || t.handle.mat == nil {
		return
	}
	if t.handle.released.Load() {
		return
	}
	if !t.handle.owns {
		t.handle.released.Store(true)
		return
	}
	if t.handle.released.CompareAndSwap(false, true) {
		t.handle.mat.Close()
		t.handle.mat = nil
		t.handle.shape = nil
	}
}

func (h *tensorHandle) dataUint8() ([]uint8, error) {
	if h.mat == nil {
		return nil, ErrNilMat
	}
	data, err := h.mat.DataPtrUint8()
	if err != nil {
		return nil, err
	}
	return data, nil
}

func (h *tensorHandle) dataFloat32() ([]float32, error) {
	if h.mat == nil {
		return nil, ErrNilMat
	}
	data, err := h.mat.DataPtrFloat32()
	if err != nil {
		return nil, err
	}
	return data, nil
}
