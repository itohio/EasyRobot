//go:build tflite
// +build tflite

package marshaller

import (
	"github.com/itohio/EasyRobot/x/marshaller/tflite"
	"github.com/itohio/EasyRobot/x/marshaller/types"
)

func init() {
	// TFLite only supports unmarshalling (model loading), not marshalling
	registerUnmarshaller("tflite", func(opts ...types.Option) types.Unmarshaller {
		return tflite.NewUnmarshaller(opts...)
	})
}
