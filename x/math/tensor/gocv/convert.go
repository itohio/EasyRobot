package gocv

import (
	"fmt"
	"image"
	"image/draw"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
	cv "gocv.io/x/gocv"
)

// ToMat returns a cloned Mat representation of the provided tensor.
func ToMat(t types.Tensor) (cv.Mat, error) {
	if accessor, ok := t.(Accessor); ok {
		return accessor.MatClone()
	}
	return cv.NewMat(), fmt.Errorf("%w: ToMat expects gocv tensor", ErrUnsupported)
}

// FromImage converts an image into a GoCV-backed tensor.
func FromImage(img image.Image, opts ...Option) (types.Tensor, error) {
	if img == nil {
		return nil, fmt.Errorf("gocv tensor: nil image")
	}

	rgba := ensureRGBA(img)
	mat, err := cv.ImageToMatRGBA(rgba)
	if err != nil {
		return nil, fmt.Errorf("gocv tensor: convert image -> mat: %w", err)
	}

	tensor, err := FromMat(mat, append(opts, WithAdoptedMat())...)
	if err != nil {
		mat.Close()
		return nil, err
	}
	return tensor, nil
}

// ToImage converts a tensor into an image.Image. The tensor must be backed by a
// GoCV Mat.
func ToImage(t types.Tensor) (image.Image, error) {
	if accessor, ok := t.(Accessor); ok {
		return accessor.Image()
	}
	return nil, fmt.Errorf("%w: ToImage expects gocv tensor", ErrUnsupported)
}

func ensureRGBA(img image.Image) *image.RGBA {
	if rgba, ok := img.(*image.RGBA); ok {
		return rgba
	}
	if nrgba, ok := img.(*image.NRGBA); ok {
		rgba := image.NewRGBA(nrgba.Rect)
		draw.Draw(rgba, rgba.Bounds(), nrgba, nrgba.Rect.Min, draw.Src)
		return rgba
	}

	rect := img.Bounds()
	rgba := image.NewRGBA(rect)
	draw.Draw(rgba, rect, img, rect.Min, draw.Src)
	return rgba
}
