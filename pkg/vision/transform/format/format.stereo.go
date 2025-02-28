package format

import (
	"github.com/itohio/EasyRobot/pkg/core/options"
	"github.com/itohio/EasyRobot/pkg/core/pipeline"
	"github.com/itohio/EasyRobot/pkg/core/pipeline/steps"
	"github.com/itohio/EasyRobot/pkg/core/store"
)

const D2S_NAME = "data2stereo"

func init() {
	pipeline.Register(D2S_NAME, NewDataToStereo)
}

func NewDataToStereo(opts ...options.Option) (pipeline.Step, error) {
	newOpts := opts
	newOpts = append(newOpts, WithDataToStereo())
	return steps.NewProcessor(newOpts...)
}

func WithDataToStereo() options.Option {
	return steps.WithNamedProcessorFunc(D2S_NAME, data2stereo)
}

func data2stereo(src, dst store.Store) error {
	val, ok := src.Get(store.SYNC_DATA)
	if !ok {
		return nil
	}

	arr, ok := val.([]interface{})
	if !ok {
		return nil
	}

	if len(arr) > 1 {
		dst.Set(store.STEREO_LEFT, arr[0])
		dst.Set(store.STEREO_RIGHT, arr[1])
	}

	return nil
}
