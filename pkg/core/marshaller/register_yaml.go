// +build !no_yaml

package marshaller

import (
	"github.com/itohio/EasyRobot/pkg/core/marshaller/types"
	"github.com/itohio/EasyRobot/pkg/core/marshaller/yaml"
)

func init() {
	registerMarshaller("yaml", func(opts ...types.Option) types.Marshaller {
		return yaml.NewMarshaller(opts...)
	})

	registerUnmarshaller("yaml", func(opts ...types.Option) types.Unmarshaller {
		return yaml.NewUnmarshaller(opts...)
	})
}

