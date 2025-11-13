//go:build !no_protobuf
// +build !no_protobuf

package marshaller

import (
	marshallerv1 "github.com/itohio/EasyRobot/x/marshaller/proto"
	"github.com/itohio/EasyRobot/x/marshaller/types"
)

func init() {
	// Register protobuf marshaller
	registerMarshaller("protobuf", func(opts ...types.Option) types.Marshaller {
		return marshallerv1.NewMarshaller(opts...)
	})

	// Register protobuf unmarshaller
	registerUnmarshaller("protobuf", func(opts ...types.Option) types.Unmarshaller {
		return marshallerv1.NewUnmarshaller(opts...)
	})
}
