package gocv

import (
	"image"

	"github.com/itohio/EasyRobot/x/math/tensor/types"
	cv "gocv.io/x/gocv"
)

// Accessor exposes GoCV-specific capabilities on top of the generic tensor API.
// Callers can type assert types.Tensor to Accessor to retrieve the underlying
// gocv.Mat or convenient image conversions.
type Accessor interface {
	types.Tensor

	// MatRef returns a pointer to the underlying gocv.Mat. The caller MUST NOT
	// call Close on the returned Mat; the tensor retains ownership and will
	// close it via Release. Modifying the Mat mutates the tensor in-place.
	MatRef() (*cv.Mat, error)

	// MatClone returns a deep copy of the underlying gocv.Mat that the caller
	// owns and must Close when finished.
	MatClone() (cv.Mat, error)

	// Image renders the tensor as an image.Image by cloning the underlying Mat.
	Image() (image.Image, error)
}

// MatRef implements Accessor.MatRef.
func (t Tensor) MatRef() (*cv.Mat, error) {
	h, err := t.getHandle()
	if err != nil {
		return nil, err
	}
	return h.mat, nil
}

// MatClone implements Accessor.MatClone.
func (t Tensor) MatClone() (cv.Mat, error) {
	h, err := t.getHandle()
	if err != nil {
		return cv.Mat{}, err
	}
	return h.mat.Clone(), nil
}

// Image implements Accessor.Image.
func (t Tensor) Image() (image.Image, error) {
	mat, err := t.MatClone()
	if err != nil {
		return nil, err
	}
	defer mat.Close()
	return mat.ToImage()
}
