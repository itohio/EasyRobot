//go:build tflite
// +build tflite

package marshaller

import (
	"github.com/itohio/EasyRobot/pkg/core/marshaller/tflite"
	"github.com/itohio/EasyRobot/pkg/core/marshaller/types"
)

func init() {
	// TFLite only supports unmarshalling (model loading), not marshalling
	registerUnmarshaller("tflite", func(opts ...types.Option) types.Unmarshaller {
		return tflite.NewUnmarshaller(opts...)
	})
}
