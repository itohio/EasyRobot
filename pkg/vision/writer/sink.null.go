package writer

import (
	"github.com/foxis/EasyRobot/pkg/pipeline"
	"github.com/foxis/EasyRobot/pkg/pipeline/steps"
	"github.com/foxis/EasyRobot/pkg/plugin"
	"github.com/foxis/EasyRobot/pkg/store"
)

const NULL_NAME = "null"

func init() {
	pipeline.Register(NULL_NAME, NewNull)
}

func NewNull(opts ...plugin.Option) (pipeline.Step, error) {
	o := writerOpts{base: plugin.DefaultOptions()}
	plugin.ApplyOptions(&o, opts...)
	plugin.ApplyOptions(&o.base, opts...)
	newOpts := opts
	newOpts = append(newOpts, WithNullWriter(o.keys))
	return steps.NewSink(newOpts...)
}

func WithNullWriter(keys []store.FQDNType) plugin.Option {
	return steps.WithNamedSinkFunc(NULL_NAME, sink_null(keys))
}

func sink_null(keys []store.FQDNType) steps.SinkFunc {
	return func(data store.Store) error {
		if keys == nil || len(keys) == 0 {
			data.Close(store.ANY)
		} else {
			for _, key := range keys {
				data.Close(key)
			}
		}
		return nil
	}
}
