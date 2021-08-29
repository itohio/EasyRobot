package format

import (
	"image"
	"image/color"

	"github.com/foxis/EasyRobot/pkg/core/options"
	"github.com/foxis/EasyRobot/pkg/core/pipeline"
	"github.com/foxis/EasyRobot/pkg/core/pipeline/steps"
	"github.com/foxis/EasyRobot/pkg/core/store"

	"gocv.io/x/gocv"
)

const (
	M2I_NAME = "mat2img"
	I2M_NAME = "mat2img"
)

func init() {
	pipeline.Register(M2I_NAME, NewMatToImage)
	pipeline.Register(I2M_NAME, NewImageToMat)
}

func NewMatToImage(opts ...options.Option) (pipeline.Step, error) {
	o := Options{
		src: store.IMAGE,
		dst: store.IMAGE,
	}
	options.ApplyOptions(&o, opts...)
	newOpts := opts
	newOpts = append(newOpts, WithMatToImage(o.src, o.dst))
	return steps.NewProcessor(newOpts...)
}

func NewImageToMat(opts ...options.Option) (pipeline.Step, error) {
	o := Options{
		src: store.IMAGE,
		dst: store.IMAGE,
	}
	options.ApplyOptions(&o, opts...)
	newOpts := opts
	newOpts = append(newOpts, WithImageToMat(o.src, o.dst))
	return steps.NewProcessor(newOpts...)
}

func WithMatToImage(src, dst store.FQDNType) options.Option {
	return steps.WithNamedProcessorFunc(M2I_NAME, mat2img(src, dst))
}

func WithImageToMat(src, dst store.FQDNType) options.Option {
	return steps.WithNamedProcessorFunc(I2M_NAME, img2mat(src, dst))
}

func mat2img(srcKey, dstKey store.FQDNType) func(src, dst store.Store) error {
	return func(src, dst store.Store) error {
		val, ok := src.Get(srcKey)
		if !ok {
			return nil
		}

		mat, ok := val.(gocv.Mat)
		if !ok {
			return nil
		}
		if mat.Empty() {
			return nil
		}

		img, err := mat.ToImage()
		if err != nil {
			return err
		}

		dst.Set(dstKey, img)

		return nil
	}
}

func img2mat(srcKey, dstKey store.FQDNType) func(src, dst store.Store) error {
	return func(src, dst store.Store) error {
		val, ok := src.Get(dstKey)
		if !ok {
			return nil
		}

		img, ok := val.(image.Image)
		if !ok {
			return nil
		}

		var mat gocv.Mat
		var err error
		switch img.ColorModel() {
		case color.RGBAModel:
			mat, err = gocv.ImageToMatRGB(img)
		default:
			return nil
		}

		if err != nil {
			return err
		}

		dst.Set(dstKey, mat)

		return nil
	}
}
