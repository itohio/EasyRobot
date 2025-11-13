# Marshaller Specification

## Overview

The marshaller subsystem in EasyRobot `pkg/core/marshaller` replaces the previous serializer-focused design. It provides a unified way to encode (`Marshal`) and decode (`Unmarshal`) EasyRobot domain objects such as tensors, matrices, vectors, neural network components, and arbitrary Go data structures. Each backend (e.g., gob, JSON, YAML) supplies concrete implementations that plug into a shared registry while the core package exposes format-agnostic factory.

Key properties:
- **Format Agnostic**: Backends register themselves; callers request formats by name. callers can enumerate registered backend names.
- **Domain Aware**: Known mathematical structures are normalised before encoding and reconstructed after decoding.
- **Streaming Friendly**: APIs use `io.Writer` / `io.Reader` to avoid unnecessary buffering.
- **Extensible**: New backends self-register through `register_<name>.go` without modifying the core. the backend lives in dedicated folder and is imported by `register_<name>.go`. inside this file should be build tag !no_<name>.
- There are two registries - Marshallers and Unmarshallers. The registry is not exposed and factory is not exposed either.
- User only interacts with NewMarshaller, and NewUnmarshaller methods that return Marshaller and Unmarshaller interfaces respectively.

## Core Interfaces (types subpackage)

All exported contracts reside in `pkg/core/marshaller/types`:

```go
type Option interface {
    Apply(*Options)
}

type Options struct {
    FormatVersion string
    Hint          string            // optional value hint (e.g. "tensor", "matrix")
    Metadata      map[string]string // free-form, backend specific
}

type Marshaller interface {
    Format() string
    Marshal(w io.Writer, value any, opts ...Option) error
}

type Unmarshaller interface {
    Format() string
    Unmarshal(r io.Reader, dst any, opts ...Option) error
}
```

Backends may declare additional helper options as long as they satisfy `Option`.

## Registry and Factory

- A global registry keyed by format name stores constructors that produce marshaller/unmarshaller.
- Registration occurs via `registerMarshaller(name string, ctor func(opts ...Option) Marshaller)` within `init()` functions housed in `register_<name>.go` files under each backend directory.
- Access is guarded by `sync.RWMutex` to ensure concurrent reads and writes remain safe.
- The registry stays internal to the `marshaller` package. Public entry points are:
  - `NewMarshaller(name string, opts ...types.Option) (types.Marshaller, error)`
  - `NewUnmarshaller(name string, opts ...types.Option) (types.Unmarshaller, error)`
- Factories apply shared default options, clone instances when necessary, and propagate the supplied options to both marshaller and unmarshaller instances.

## Domain Object Handling

Every backend should recognise and correctly encode/decode the following categories:

- `tensor.Tensor` - Full round-trip support required
- Matrix types from `mat.Matrix` - Full round-trip support required
- Vector types from `vec.Vector` - Full round-trip support required
- Raw numeric arrays (`[]float32`, `[]float64`, etc.) - Full round-trip support required
- Neural network models and layers (`nn.Model`, `nn.Layer`) - Structure and parameters
- Arbitrary Go structs, maps, and primitives (delegated to the underlying encoding library)

Implementations convert recognised types to canonical intermediate representations prior to encoding, ensuring consistent round-tripping across formats. Unknown values fall back to the encoding library, while domain-specific metadata can be carried via `Options.Metadata`.

`Options.Hint` allows callers to disambiguate when static type information is insufficient (e.g., requesting a tensor with a particular dtype).

### Tensor Reconstruction

When unmarshalling tensors:
1. Use `Options.TensorFactory` if provided (signature: `func(DataType, Shape) Tensor`)
2. Default to `tensor.New(dtype, shape)` if no factory provided
3. Do NOT convert arrays to bytes - gob handles type-specific encoding natively
4. Read gob struct with shape, dtype, and data, then create tensor and copy data
5. If `Options.DestinationType` is set, create tensor with that type and convert element-by-element

### Layer and Model Reconstruction

Gob marshaller should:
- Store layer/model structure (type, parameters, configuration)
- NOT attempt full reconstruction (requires type registry)
- Allow parameter extraction and restoration into compatible architectures
- Preserve parameter shapes, types, and values

## Options

- Shared helpers (e.g., `types.WithFormatVersion`, `types.WithHint`, `types.WithMetadata`) live in the `types` package and implement `Option`.
- Options are applied immediately after constructing marshaller/unmarshaller instances.
- Backend-specific options should also implement `types.Option` so they can be passed through the same APIs.
- `types.WithTensorFactory` provides `func(DataType, Shape) Tensor` tensor constructor (defaults to `tensor.New`)
- `types.WithDestinationType` sets target data type for type conversion during unmarshal

## Error Handling

- The `types` package exposes helpers such as `types.NewError(op, format, message, cause error)` to wrap backend failures with consistent context.
- Core factories surface lookup failures as `fmt.Errorf("marshaller: %w", err)` while preserving the original error for `errors.Is` / `errors.As`.
- Backend implementations should wrap domain failures with `types.Error` to retain operation, format, and layer/value context.

## Registration Pattern

```
pkg/core/marshaller/
├── serialize.go           # Registry, factory helpers, option defaults
├── types/
│   └── types.go           # Interfaces, option contracts, error helpers
├── text/
│   └── marshaller.go      # Text marshaller (output only)
├── gob/
│   ├── internal.go        # Internal structs
│   ├── convert.go         # Conversion helpers
│   ├── marshaller.go      # Gob marshaller
│   └── unmarshaller.go    # Gob unmarshaller
├── json/
│   ├── internal.go        # Internal structs
│   ├── marshaller.go      # JSON marshaller
│   └── unmarshaller.go    # JSON unmarshaller
├── yaml/
│   ├── internal.go        # Internal structs
│   ├── marshaller.go      # YAML marshaller
│   └── unmarshaller.go    # YAML unmarshaller
├── proto/
│   ├── marshaller.proto   # Protobuf schema
│   └── README.md          # Protobuf generation instructions
├── register_text.go       # Text registration
├── register_gob.go        # Gob registration (build tag: !no_gob)
├── register_json.go       # JSON registration (build tag: !no_json)
├── register_yaml.go       # YAML registration (build tag: !no_yaml)
└── ...
```

`register_<name>.go` files import their backend implementation and invoke `register*` during package initialisation. Build tags gate optional formats.

There are two registers: for marshallers and unmarshallers.

## Implemented Formats

### Text (output only)
- Human-readable output for debugging/logging
- TensorFlow-style model summaries
- No unmarshaller

### Gob (binary, Go-specific)
- Efficient binary format
- Full round-trip support for tensors, arrays
- Native Go encoding

### JSON (human-readable)
- Standard JSON format
- Full round-trip support for tensors, arrays
- Pretty-printed output
- Type conversion for numeric arrays

### YAML (human-readable)
- YAML format (superset of JSON)
- Full round-trip support for tensors, arrays
- Indented output
- Type conversion for numeric arrays

### Protobuf (binary, cross-language) [Optional]
- Efficient cross-language format
- Requires `buf generate` for code generation
- Build without `-tags no_protobuf` (protobuf is enabled by default)
- See `proto/README.md` for setup
- Directly marshals `proto.Message` types
- Converts domain types (Tensor, Layer, Model) to protobuf messages

### TFLite (model loader) [Optional]
- Loads TensorFlow Lite `.tflite` models
- Converts to EasyRobot `models.Sequential`
- Supports training and fine-tuning
- Requires TensorFlow Lite C library
- Build with `-tags tflite`
- See `tflite/README.md` for setup

### Keras H5 (model loader) [Optional]
- Loads Keras `.h5` models
- Converts to EasyRobot `models.Sequential`
- Supports training and fine-tuning
- Includes CNN layers (Conv2D, Pooling)
- Requires HDF5 C library
- Build with `-tags keras`
- See `keras/README.md` for setup

## Model Format Support

For comprehensive information about loading pre-trained models from various frameworks, see **[MODEL_FORMATS.md](MODEL_FORMATS.md)**.

This document covers:
- TFLite, Keras H5, ONNX, PyTorch, TensorFlow formats
- Capabilities and limitations of each format
- Build instructions and requirements
- Format comparison and recommendations
- Implementation roadmap

## Usage Examples

### Basic Tensor Marshalling

```go
// Create marshaller and unmarshaller
mar, err := marshaller.NewMarshaller("gob")
unmar, err := marshaller.NewUnmarshaller("gob")

// Marshal tensor
var buf bytes.Buffer
t := tensor.New(tensor.DTFP32, tensor.NewShape(2, 3))
err = mar.Marshal(&buf, t)

// Unmarshal tensor
var restored tensor.Tensor
err = unmar.Unmarshal(&buf, &restored)
```

### Type Conversion During Unmarshal

```go
// Marshal FP32 tensor, unmarshal as FP64
mar, _ := marshaller.NewMarshaller("gob")
unmar, _ := marshaller.NewUnmarshaller("gob", 
    types.WithDestinationType(types.FP64))

var buf bytes.Buffer
t := tensor.New(tensor.DTFP32, tensor.NewShape(2, 3))
mar.Marshal(&buf, t)

var restored tensor.Tensor
unmar.Unmarshal(&buf, &restored)
// restored is now FP64 type
```

### Model Parameters

```go
// Marshal model parameters
model := buildModel()
params := model.Parameters()

paramBuffers := make(map[types.ParamIndex]bytes.Buffer)
for idx, param := range params {
    var buf bytes.Buffer
    mar.Marshal(&buf, param.Data)
    paramBuffers[idx] = buf
}

// Unmarshal into new model
newModel := buildModel()
for idx, buf := range paramBuffers {
    var t tensor.Tensor
    unmar.Unmarshal(&buf, &t)
    
    param, _ := newModel.Parameter(idx)
    // Copy restored data to parameter
    for i := 0; i < t.Size(); i++ {
        param.Data.SetAt(t.At(i), i)
    }
}
```

## Testing Strategy

- Registry concurrency tests (concurrent register/lookup).
- Option propagation tests (ensuring options reach implementations).
- Round-trip coverage per backend:
  - Tensor (`tensor.Tensor`)
  - Matrix/vector types
  - Raw numeric arrays
  - Neural models/layers
  - Generic structs with nested fields
- Error propagation tests (malformed payloads, unsupported hints).

## Future Enhancements

1. Typed helper methods (`UnmarshalModel`, `MarshalTensor`) layered on top of the generic API.
2. Streaming chunk encoders for large tensors to reduce memory usage.
3. Cross-format conversion utilities that chain marshaller/unmarshaller pairs.
4. Negotiation helpers that discover supported formats at runtime.
5. Schema/version compatibility checks driven by `Options.FormatVersion`.
# Marshaller Specification

pkg/core/marshaller/SPEC.md