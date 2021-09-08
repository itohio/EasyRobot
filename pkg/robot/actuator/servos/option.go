package servos

import (
	"github.com/foxis/EasyRobot/pkg/core/options"
)

func WithPin(pin uint32) options.Option {
	return func(o interface{}) {
		o.(*Motor).Pin = pin
	}
}

func WithRange(min, max float32) options.Option {
	return func(o interface{}) {
		o.(*Motor).Min = min
		o.(*Motor).Max = max
	}
}

func WithCalibration(scale, offset float32) options.Option {
	return func(o interface{}) {
		o.(*Motor).Scale = scale
		o.(*Motor).Offset = offset
	}
}

func WithMicroseconds(min, max, middle uint32, angleRange float32) options.Option {
	scale := float32(max-min) / angleRange
	return func(o interface{}) {
		o.(*Motor).Scale = scale
		o.(*Motor).Offset = float32(middle)
	}
}
