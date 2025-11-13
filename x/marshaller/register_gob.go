//go:build !no_gob
// +build !no_gob

package marshaller

import (
	"github.com/itohio/EasyRobot/x/marshaller/gob"
	"github.com/itohio/EasyRobot/x/marshaller/types"
)

func init() {
	registerMarshaller("gob", func(opts ...types.Option) types.Marshaller {
		return gob.NewMarshaller(opts...)
	})

	registerUnmarshaller("gob", func(opts ...types.Option) types.Unmarshaller {
		return gob.NewUnmarshaller(opts...)
	})
}
