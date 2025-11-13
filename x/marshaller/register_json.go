// +build !no_json

package marshaller

import (
	"github.com/itohio/EasyRobot/pkg/core/marshaller/json"
	"github.com/itohio/EasyRobot/pkg/core/marshaller/types"
)

func init() {
	registerMarshaller("json", func(opts ...types.Option) types.Marshaller {
		return json.NewMarshaller(opts...)
	})

	registerUnmarshaller("json", func(opts ...types.Option) types.Unmarshaller {
		return json.NewUnmarshaller(opts...)
	})
}

