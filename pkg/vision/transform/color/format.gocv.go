package color

import (
	"github.com/foxis/EasyRobot/pkg/backend"
	"github.com/foxis/EasyRobot/pkg/core/options"
	"github.com/foxis/EasyRobot/pkg/core/pipeline"
	"github.com/foxis/EasyRobot/pkg/core/pipeline/steps"
	"github.com/foxis/EasyRobot/pkg/core/store"

	"gocv.io/x/gocv"
)

func init() {
	pipeline.Register(NAME, NewGoCV)
}

func NewGoCV(opts ...options.Option) (pipeline.Step, error) {
	o := Options{
		src: store.IMAGE,
		dst: store.IMAGE,
	}
	options.ApplyOptions(&o, opts...)
	newOpts := opts
	newOpts = append(newOpts, WithGoCV(o.src, o.dst, gocv.ColorConversionCode(o.code)))
	return steps.NewProcessor(newOpts...)
}

func WithGoCV(src, dst store.FQDNType, code gocv.ColorConversionCode) options.Option {
	return steps.WithNamedProcessorFunc(NAME, colorGoCV(src, dst, code))
}

func colorGoCV(srcKey, dstKey store.FQDNType, code gocv.ColorConversionCode) func(src, dst store.Store) error {
	return func(src, dst store.Store) error {
		val, ok := src.Get(srcKey)
		if !ok {
			return nil
		}

		mat, ok := val.(*gocv.Mat)
		if !ok {
			return nil
		}
		if mat.Empty() {
			return nil
		}

		result := backend.NewGoCVMat()
		gocv.CvtColor(*mat, result, code)

		dst.Set(dstKey, result)

		return nil
	}
}
