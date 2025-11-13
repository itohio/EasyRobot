package marshaller

import (
	"github.com/itohio/EasyRobot/pkg/core/marshaller/text"
	"github.com/itohio/EasyRobot/pkg/core/marshaller/types"
)

func init() {
	registerMarshaller("text", func(opts ...types.Option) types.Marshaller {
		return text.New(opts...)
	})
}

