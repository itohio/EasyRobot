//go:build !no_json
// +build !no_json

package marshaller

import (
	"github.com/itohio/EasyRobot/x/marshaller/json"
	"github.com/itohio/EasyRobot/x/marshaller/types"
)

func init() {
	registerMarshaller("json", func(opts ...types.Option) types.Marshaller {
		return json.NewMarshaller(opts...)
	})

	registerUnmarshaller("json", func(opts ...types.Option) types.Unmarshaller {
		return json.NewUnmarshaller(opts...)
	})
}
