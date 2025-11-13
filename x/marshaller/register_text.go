package marshaller

import (
	"github.com/itohio/EasyRobot/x/marshaller/text"
	"github.com/itohio/EasyRobot/x/marshaller/types"
)

func init() {
	registerMarshaller("text", func(opts ...types.Option) types.Marshaller {
		return text.New(opts...)
	})
}
