// Package marshallerv1 implements protobuf-based marshalling/unmarshalling
// for EasyRobot domain objects (tensors, layers, models, etc.).
//
// This package provides:
//   - Efficient binary serialization using Protocol Buffers
//   - Cross-language compatibility (via .proto schema)
//   - Support for all core domain types (Tensor, Layer, Model, etc.)
//
// The protobuf schema is centrally located at proto/types/core/types.proto
// and generated code is at types/core/types.pb.go.
// This package provides marshallers/unmarshallers that use those types.
package marshallerv1
