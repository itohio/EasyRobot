//go:build !no_yaml
// +build !no_yaml

package marshaller

import (
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/marshaller/yaml"
)

func init() {
	registerMarshaller("yaml", func(opts ...types.Option) types.Marshaller {
		return yaml.NewMarshaller(opts...)
	})

	registerUnmarshaller("yaml", func(opts ...types.Option) types.Unmarshaller {
		return yaml.NewUnmarshaller(opts...)
	})
}
