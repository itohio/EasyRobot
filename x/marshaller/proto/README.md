# Protobuf Marshaller

This package implements protobuf-based marshalling/unmarshalling for EasyRobot domain objects.

## Protobuf Schema

The protobuf schema is centrally located at:
- **Proto source**: `proto/types/core/types.proto`
- **Generated Go code**: `types/core/types.pb.go`

This package provides marshallers/unmarshallers that use those centralized protobuf types.

## Regenerating Protobuf Code

The protobuf code is generated using `buf` from the repository root:

```bash
# From repository root
buf generate
```

## Requirements

- `buf` CLI tool (https://buf.build/docs/installation)
- `protoc-gen-go` plugin

Install protoc-gen-go:
```bash
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
```

## Usage

The marshaller automatically registers itself on package initialization. It can be used through the generic marshaller interface:

```go
import (
    "github.com/itohio/EasyRobot/x/marshaller"
    _ "github.com/itohio/EasyRobot/x/marshaller/proto"  // Register protobuf marshaller
)

// Create marshaller directly
m := protobuf.NewMarshaller()

// Marshal data
var buf bytes.Buffer
err := m.Marshal(&buf, tensor)

// Unmarshal data
u := protobuf.NewUnmarshaller()
var restored types.Tensor
err = u.Unmarshal(&buf, &restored)
```
