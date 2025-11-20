# Marshaller Specification

## Overview

The marshaller subsystem provides a unified way to encode (`Marshal`) and decode (`Unmarshal`) EasyRobot domain objects such as tensors, matrices, vectors, neural network components, and arbitrary Go data structures. Each backend (e.g., gob, JSON, YAML, GoCV) supplies concrete implementations that users create directly.

Key properties:
- **Format Specific**: Each backend provides its own constructor (e.g., `gob.NewMarshaller()`)
- **Domain Aware**: Known mathematical structures are normalised before encoding and reconstructed after decoding
- **Streaming Friendly**: APIs use `io.Writer` / `io.Reader` to avoid unnecessary buffering
- **Direct Creation**: Users create marshallers directly without registry lookups
- **Consistent Interface**: All marshallers implement the same `Marshaller`/`Unmarshaller` interfaces

## Core Types (types subpackage)

The `marshaller/types` package provides shared types and interfaces:

```go
type Option interface {
    Apply(*Options)
}

type Options struct {
    FormatVersion string
    Hint          string            // optional value hint (e.g. "tensor", "matrix")
    Metadata      map[string]string // free-form, backend specific
}
```

## Direct Construction

Each backend provides direct constructors that return concrete types:

```go
// Standard pattern: separate Marshaller/Unmarshaller constructors
gobMarshaller := gob.NewMarshaller()
gobUnmarshaller := gob.NewUnmarshaller()

// Simplified pattern: single constructor when only one direction is supported
textMarshaller := text.New()           // Text only marshals
v4lUnmarshaller := v4l.New()           // V4L only unmarshals

// Use immediately
err := gobMarshaller.Marshal(writer, myTensor)
err := gobUnmarshaller.Unmarshal(reader, &result)
```

No registry or factory functions - users create marshallers directly.

## Domain Object Handling

Each backend should recognise and correctly encode/decode the following categories:

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

Marshallers should:
- Store layer/model structure (type, parameters, configuration)
- NOT attempt full reconstruction (requires type registry)
- Allow parameter extraction and restoration into compatible architectures
- Preserve parameter shapes, types, and values

## Options

- Shared helpers (e.g., `types.WithFormatVersion`, `types.WithHint`, `types.WithMetadata`) live in the `types` package and implement `Option`.
- Options are passed directly to marshaller/unmarshaller constructors.
- Backend-specific options should also implement `types.Option` so they can be passed through the same APIs.
- `types.WithTensorFactory` provides `func(DataType, Shape) Tensor` tensor constructor (defaults to `tensor.New`)
- `types.WithDestinationType` sets target data type for type conversion during unmarshal
- `types.WithRelease()` enables automatic calling of `Release()` on tensors after processing by sink marshallers

## Error Handling

- The `types` package exposes helpers such as `types.NewError(op, format, message, cause error)` to wrap backend failures with consistent context.
- Backend implementations should wrap domain failures with `types.Error` to retain operation, format, and layer/value context.

## Directory Structure

```
marshaller/
├── types/
│   ├── camera.go          # Camera/display/input interfaces
│   ├── display.go         # Display/input interfaces
│   ├── types.go           # Core interfaces, options, error helpers
│   └── SPEC.md            # Interface documentation
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
├── gocv/
│   ├── codec.go           # Image encoding/decoding
│   ├── config.go          # Configuration structs
│   ├── DESIGN.md          # Architecture documentation
│   ├── FILE_FORMAT.md     # File format specifications
│   ├── loader_*.go        # Data source loaders
│   ├── marshaller.go      # GoCV marshaller
│   ├── options.go         # GoCV-specific options
│   ├── sink.go            # Output sinks
│   ├── SPEC.md            # GoCV marshaller documentation
│   ├── streams.go         # Stream coordination
│   ├── unmarshaller.go    # GoCV unmarshaller
│   └── ...
└── v4l/
    ├── types.go           # V4L types and interfaces
    ├── device.go          # Device management
    ├── stream.go          # Stream handling
    ├── options.go         # V4L-specific options
    └── ...
```

## Implementation Details

### Core Architecture

All marshallers follow a consistent internal structure:

```go
type Marshaller struct {
    opts types.Options    // Base options
    cfg  backendConfig   // Backend-specific configuration
}

// Constructor pattern (standard)
func NewMarshaller(opts ...types.Option) types.Marshaller {
    m := &Marshaller{opts: types.Options{}}
    for _, opt := range opts {
        opt.Apply(&m.opts)
    }
    // Backend-specific initialization
    return m
}
```

**Marshal Flow:**
1. Apply options (merge instance + call options)
2. Type-switch on value to determine handling
3. Convert domain objects to backend-specific format
4. Write to io.Writer

**Unmarshal Flow:**
1. Apply options (merge instance + call options)
2. Read from io.Reader
3. Parse backend-specific format
4. Reconstruct domain objects

### Domain Object Handling Strategy

**Tensor Support:**
- All marshallers handle `types.Tensor` interface
- Convert to backend-specific representation (gobValue, jsonValue, etc.)
- Preserve shape, dtype, and data

**Model/Layer Support:**
- Handle `types.Model` and `types.Layer` interfaces
- Store parameter tensors and metadata
- Support for reconstruction (limited by Go's type system)

**Slice/Array Support:**
- Direct encoding for primitive arrays
- Reflection-based handling for complex types
- Fallback to generic encoding

### Implemented Backends

#### Text Marshaller (Output Only)
- **Constructor:** `text.New(opts ...types.Option) types.Marshaller`
- **Features:** Human-readable summaries, TensorFlow-style model inspection
- **Limitations:** No unmarshaller (by design)
- **Use Case:** Debugging, logging, development

**Implementation Notes:**
- Uses reflection for type inspection
- Generates formatted summaries for tensors, models, layers
- Single-pass writing with no intermediate structures

#### Gob Marshaller (Binary, Go-specific)
- **Constructor:** `gob.NewMarshaller(opts ...types.Option) types.Marshaller`
- **Features:** Native Go encoding, full round-trip support
- **Use Case:** Efficient Go-to-Go data transfer

**Implementation Notes:**
- Wraps domain objects in `gobValue` struct for encoding
- Handles type reconstruction during unmarshal
- Uses Go's built-in gob package

#### JSON Marshaller (Human-readable)
- **Constructor:** `json.NewMarshaller(opts ...types.Option) types.Marshaller`
- **Features:** Standard JSON format, pretty-printed output
- **Use Case:** Configuration files, APIs, debugging

**Implementation Notes:**
- Wraps domain objects in `jsonValue` struct
- Uses `encoding/json` with custom marshaling
- Supports graph structures (unique to JSON backend)

#### YAML Marshaller (Human-readable)
- **Constructor:** `yaml.NewMarshaller(opts ...types.Option) types.Marshaller`
- **Features:** YAML format, indented output, JSON superset
- **Use Case:** Configuration files, complex nested structures

**Implementation Notes:**
- Similar structure to JSON marshaller
- Uses YAML-specific encoding library
- Better for human-edited configuration

#### GoCV Marshaller (Computer Vision)
- **Constructor:** `gocv.NewMarshaller(opts ...types.Option) types.Marshaller`
- **Features:** Image/tensor conversion, video capture, display
- **Use Case:** Computer vision pipelines, camera integration

**Implementation Notes:**
- Delegates to OpenCV for image encoding/decoding
- Handles `gocv.Mat`, `image.Image`, `types.Tensor`, `types.FrameStream`
- Complex configuration with display and stream options

#### V4L Marshaller (Video Devices)
- **Constructor:** `v4l.NewMarshaller(opts ...types.Option) types.Marshaller`
- **Features:** Camera device enumeration, stream configuration
- **Use Case:** Hardware video capture, device management

**Implementation Notes:**
- Marshals device info and stream configurations to JSON
- Unmarshals to `types.FrameStream` for device access
- Hardware-specific with V4L2 integration

#### Graph Marshaller (Persistent Storage)
- **Constructor:** `graph.NewMarshaller(factory types.MappedStorageFactory, opts ...types.Option) (*GraphMarshaller, error)`
- **Features:** Memory-mapped storage, graph persistence
- **Use Case:** Large graph storage, database-like operations

**Implementation Notes:**
- Requires storage factory parameter
- Returns concrete type (not interface) due to factory requirement
- Advanced: append-only updates, defragmentation

#### Protobuf Marshaller (Cross-language) [Optional]
- **Constructor:** `protobuf.NewMarshaller(opts ...types.Option) *Marshaller`
- **Features:** Language-agnostic binary format
- **Use Case:** Multi-language systems, network protocols

**Implementation Notes:**
- Returns concrete type (breaks interface pattern)
- Requires code generation via `buf generate`
- Uses protobuf wire format

#### TFLite Unmarshaller (Model Loading) [Optional]
- **Constructor:** `tflite.NewUnmarshaller(opts ...types.Option) *Unmarshaller`
- **Features:** TensorFlow Lite model loading
- **Use Case:** ML model deployment, inference

**Implementation Notes:**
- Unmarshaller only (no marshal support)
- Returns concrete type
- Loads models into EasyRobot format

## Model Format Support

For comprehensive information about loading pre-trained models from various frameworks, see **[MODEL_FORMATS.md](MODEL_FORMATS.md)**.

This document covers:
- TFLite, Keras H5, ONNX, PyTorch, TensorFlow formats
- Capabilities and limitations of each format
- Build instructions and requirements
- Format comparison and recommendations
- Implementation roadmap

## Usage Recommendations

### Choosing the Right Marshaller

| Use Case | Recommended Marshaller | Rationale |
|----------|------------------------|-----------|
| Go-to-Go data transfer | Gob | Most efficient, native Go types |
| Configuration files | YAML | Human-readable, supports complex structures |
| API responses | JSON | Standard web format, widely supported |
| Debugging/Logging | Text | Human-readable summaries, no unmarshal needed |
| Cross-language | Protobuf | Language-agnostic, efficient |
| Computer vision | GoCV | Native image/tensor support |
| Hardware devices | V4L | Camera enumeration and control |
| Large graphs | Graph | Memory-mapped storage, persistence |

### Best Practices

#### Performance Considerations
- **Gob** for high-performance Go applications
- **Protobuf** for cross-language communication
- **JSON/YAML** for human-edited configurations
- **Text** only for debugging (no unmarshal capability)

#### Memory Usage
- **Graph marshaller** for large persistent datasets
- **Streaming** for large tensors to avoid memory spikes
- **Reference counting** in multi-consumer scenarios

#### Error Handling
- Always check for errors from Marshal/Unmarshal
- Use `types.NewError` for consistent error messages
- Wrap backend errors with context using `types.NewError`

#### Options Usage
- Set options at construction for instance-level defaults
- Pass options to Marshal/Unmarshal for call-specific overrides
- Use `types.WithMetadata` for backend-specific configuration

## Usage Examples

### Basic Tensor Marshalling

```go
// Create marshaller and unmarshaller directly
mar := gob.NewMarshaller()
unmar := gob.NewUnmarshaller()

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
mar := gob.NewMarshaller()
unmar := gob.NewUnmarshaller(types.WithDestinationType(types.FP64))

var buf bytes.Buffer
t := tensor.New(tensor.DTFP32, tensor.NewShape(2, 3))
mar.Marshal(&buf, t)

var restored tensor.Tensor
unmar.Unmarshal(&buf, &restored)
// restored is now FP64 type
```

### Computer Vision Pipeline

```go
// GoCV marshaller for image processing
marshaller := gocv.NewMarshaller()
unmarshaller := gocv.NewUnmarshaller()

// Process camera stream
streamUnmarshaller := gocv.NewUnmarshaller(
    gocv.WithVideoDevice(0, 1920, 1080),
    gocv.WithPixelFormat("MJPEG"),
)

var frameStream types.FrameStream
err := streamUnmarshaller.Unmarshal(nil, &frameStream)
// Camera controller available for runtime adjustments
controller := streamUnmarshaller.CameraController(0)
```

### Automatic Resource Management

```go
// Display marshaller with automatic tensor cleanup
displayMarshaller := gocv.NewMarshaller(
    gocv.WithDisplay(ctx),
    types.WithRelease(), // Automatically release tensors after display
)

// Tensors will be released after being displayed, preventing memory leaks
var frameStream types.FrameStream
err := unmarshaller.Unmarshal(nil, &frameStream)
```

### Device Management

```go
// V4L marshaller for camera enumeration
unmarshaller := v4l.NewUnmarshaller()

// List available cameras
var devices []types.CameraInfo
err := unmarshaller.Unmarshal(strings.NewReader("list"), &devices)

// Configure specific camera
streamUnmarshaller := v4l.NewUnmarshaller(
    v4l.WithVideoDeviceEx(0, 1920, 1080, 30, "YUYV"),
    v4l.WithCameraControls(map[string]int32{
        "brightness": 128,
        "exposure": 500,
    }),
)
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

- Option propagation tests (ensuring options reach implementations).
- Round-trip coverage per backend:
  - Tensor (`tensor.Tensor`)
  - Matrix/vector types
  - Raw numeric arrays
  - Neural models/layers
  - Generic structs with nested fields
- Error propagation tests (malformed payloads, unsupported hints).

## Device/Camera Marshallers (New Category)

**Latest Addition**: Based on V4L and GoCV implementations, a new category of marshallers handles real-time hardware devices (cameras, sensors, etc.). These follow different patterns than traditional data marshallers.

### Key Characteristics

- **Real-time streaming**: Handle continuous data streams rather than static data
- **Hardware management**: Control physical devices with runtime parameters
- **Synchronization**: Coordinate multiple devices for synchronized capture
- **Buffer pooling**: Efficient memory reuse for high-throughput data
- **Controller access**: Runtime device control during operation

### Implementation Patterns

#### 1. Direct Creation
```go
// Device marshallers are created directly
unmarshaller := v4l.NewUnmarshaller(
    // Configure sources via options
    v4l.WithVideoDeviceEx(0, 1920, 1080, 30, "MJPEG"),
    v4l.WithCameraControls(map[string]int32{...}),
)

// Unmarshal to FrameStream for automatic device management
var stream types.FrameStream
err := unmarshaller.Unmarshal(nil, &stream)
```

#### 2. Controller Management
```go
// Provide runtime access to device controls
type UnmarshallerWithControllers interface {
    types.Unmarshaller
    CameraController(devicePath string) types.CameraController
    Cameras() []CameraDevice // Optional: device enumeration
}
```

#### 3. Shared Types Integration
```go
// Use shared types from marshaller/types
type CameraInfo = types.CameraInfo
type CameraController = types.CameraController
type VideoFormat = types.VideoFormat
type ControlInfo = types.ControlInfo

// Standard control names
types.CameraControlBrightness
types.CameraControlExposure
types.CameraControlContrast
```

#### 4. Controller Registration Pattern
```go
// Unmarshaller manages active controllers
type Unmarshaller struct {
    activeControllers map[string]types.CameraController // device path -> controller
    activeStreams     map[string]types.CameraStream     // device path -> stream
}

// Register controllers when streams are created
func (u *Unmarshaller) registerController(devicePath string, controller types.CameraController) {
    u.activeControllers[devicePath] = controller
}

// Provide access to controllers
func (u *Unmarshaller) CameraController(devicePath string) types.CameraController {
    return u.activeControllers[devicePath]
}
```

#### 4. Option Patterns
```go
// Device-specific options
WithVideoDeviceEx(id, width, height, fps, format) // Extended config
WithCameraControls(controls map[string]int32)     // Initial settings
WithDevicePath(path)                              // Device selection
```

### Best Practices from V4L Implementation

**Key Lesson**: Device marshallers should follow the **gocv pattern** where the unmarshaller is the entry point and configuration happens via options.

#### API Design Evolution
1. **Initial Approach**: Manual device/stream creation (`v4l.NewDevice()`, `device.Open()`)
2. **Learned from GoCV**: Unmarshaller-centric design with option-based configuration
3. **Final Result**: Simple, consistent API matching existing marshallers

#### Memory Management
- **Buffer pooling**: Use `helpers.Pool[T]` for efficient memory reuse
- **Tensor wrappers**: Create pooled tensor implementations with automatic cleanup
- **Reference counting**: Use smart tensors for multi-consumer scenarios

#### Error Handling
- **Device errors**: Wrap hardware failures with context
- **Stream errors**: Handle real-time streaming failures gracefully
- **Recovery**: Implement best-effort synchronization for robustness

#### Performance Optimization
- **Zero-copy**: Memory-map buffers when possible
- **Synchronization**: Efficient multi-device coordination
- **Resource cleanup**: Proper device/stream lifecycle management

#### Code Organization
```
v4l/
├── types.go           # Shared type definitions & conversions
├── tensor.go          # Pooled tensor implementation
├── device.go          # Device enumeration & management
├── stream.go          # Stream creation & synchronization
├── options.go         # Option handling & configuration
├── marshaller.go      # Basic marshal/unmarshal (minimal)
├── unmarshaller.go    # Main entry point with controllers
├── SPEC.md           # Implementation details
├── README.md         # User documentation
└── example_test.go   # Usage examples
```

## Implementation Guidelines for New Marshallers

### 1. Study Existing Implementations First

**Critical Lesson from V4L**: Always examine successful implementations (like GoCV) before starting your own. The V4L marshaller went through multiple iterations:

1. **Initial**: Complex manual API (`NewDevice()`, `Open()`, `Start()`)
2. **After studying GoCV**: Simple unmarshaller-centric API with options
3. **Final**: Consistent with existing marshaller patterns

**Recommendation**: Read GoCV's implementation thoroughly before implementing device marshallers.

### 2. Choose the Right Pattern

**For Data Marshallers** (traditional):
- Focus on `Marshal`/`Unmarshal` of static data
- Handle domain objects (tensors, models, etc.)
- Use standard options pattern

**For Device Marshallers** (new pattern):
- Use unmarshaller as entry point for device management
- Implement streaming via `types.FrameStream`
- Provide controller access for runtime control
- Use shared camera types from `marshaller/types`

### 2. Integration with Shared Types

```go
// Always use shared interfaces when available
type MyMarshaller struct {
    opts types.Options
    cfg  MyConfig
}

// Implement shared interfaces
func (m *MyMarshaller) Format() string { return "myformat" }
func (m *MyMarshaller) Marshal(w io.Writer, value any, opts ...types.Option) error { /*...*/ }
func (m *MyMarshaller) Unmarshal(r io.Reader, dst any, opts ...types.Option) error { /*...*/ }

// For device marshallers, also implement:
func (u *MyUnmarshaller) CameraController(id string) types.CameraController { /*...*/ }
```

### 3. Option Handling Best Practices

```go
// Create focused option structs
type myConfig struct {
    codec   codecConfig
    stream  streamConfig
    display displayConfig
}

// Apply options through focused configs
func applyOptions(baseOpts types.Options, baseCfg myConfig, opts ...types.Option) (types.Options, myConfig) {
    finalOpts := baseOpts
    finalCfg := baseCfg

    for _, opt := range opts {
        opt.Apply(&finalOpts)
        // Apply to focused configs if applicable
    }

    return finalOpts, finalCfg
}
```

### 4. Error Handling Patterns

```go
// Use types.NewError for consistent error messages
return types.NewError("unmarshal", "v4l", "failed to open device", err)

// Wrap hardware-specific errors
if err := device.Open(); err != nil {
    return types.NewError("unmarshal", "v4l", fmt.Sprintf("device %s open failed", devicePath), err)
}
```

### 5. Testing Strategy

```go
// Test device enumeration
func TestDeviceEnumeration(t *testing.T) {
    unmarshaller := v4l.NewUnmarshaller()
    var devices []types.CameraInfo
    err := unmarshaller.Unmarshal(strings.NewReader("list"), &devices)
    // Test device discovery logic
}

// Test stream creation
func TestStreamCreation(t *testing.T) {
    unmarshaller := v4l.NewUnmarshaller(v4l.WithVideoDeviceEx(0, 640, 480, 30, "MJPEG"))
    var stream types.FrameStream
    err := unmarshaller.Unmarshal(nil, &stream)
    // Test stream initialization
}

// Test controller access
func TestControllerAccess(t *testing.T) {
    // Setup stream first
    controller := unmarshaller.CameraController("/dev/video0")
    assert.NotNil(t, controller)
    // Test control operations
}
```

### 6. Documentation Requirements

- **README.md**: Include usage examples showing both simple and advanced usage
- **SPEC.md**: Document any device-specific features and limitations
- **Example tests**: Provide comprehensive examples in `*_test.go` files
- **API compatibility**: Ensure examples work across different implementations

## API Discrepancies & Unification Recommendations

### Current API Inconsistencies

Analysis of the implemented marshallers reveals several API inconsistencies that should be addressed for better uniformity:

#### 1. Constructor Function Names
**Standard Pattern:** `NewMarshaller(opts ...types.Option) types.Marshaller`
- ✅ **Used by:** gob, json, yaml, gocv, v4l
- ❌ **Text marshaller:** `New(opts ...types.Option) types.Marshaller` (inconsistent naming)

**Recommendation:** Standardize on `NewMarshaller` and `NewUnmarshaller` for all backends.

#### 2. Return Types
**Interface Pattern:** Return `types.Marshaller` / `types.Unmarshaller` interfaces
- ✅ **Used by:** gob, json, yaml, gocv, v4l
- ❌ **Graph marshaller:** Returns `(*GraphMarshaller, error)` - concrete type + error
- ❌ **Protobuf marshaller:** Returns `*Marshaller` - concrete type
- ❌ **TFLite unmarshaller:** Returns `*Unmarshaller` - concrete type

**Recommendation:** All constructors should return interface types. Special cases requiring additional parameters should use option pattern.

#### 3. Error Handling in Constructors
**No Error Pattern:** Constructors don't return errors
- ✅ **Used by:** gob, json, yaml, gocv, v4l, text
- ❌ **Graph marshaller:** Returns `(*GraphMarshaller, error)` - includes error

**Recommendation:** Use options pattern for required parameters instead of constructor errors.

#### 4. Required Parameters
**No Parameters Pattern:** Standard constructors take only options
- ✅ **Used by:** gob, json, yaml, gocv, v4l, text
- ❌ **Graph marshaller:** Requires `storageFactory types.MappedStorageFactory` parameter

**Recommendation:** Graph marshaller should accept storage factory via options pattern.

### Unification Strategy

#### Phase 1: Immediate Fixes
1. **Rename text.New to text.NewMarshaller** for consistency
2. **Add options for required parameters:**
   - Graph marshaller: `WithStorageFactory(factory types.MappedStorageFactory)`
   - Any other required parameters should use options

#### Phase 2: Interface Compliance
1. **Update concrete type returns to interfaces:**
   - Protobuf: Change to return `types.Marshaller` interface
   - TFLite: Change to return `types.Unmarshaller` interface
   - Graph: Change to return `types.Marshaller` interface

#### Phase 3: Constructor Standardization
1. **Ensure all constructors follow the pattern:**
   ```go
   func NewMarshaller(opts ...types.Option) types.Marshaller
   func NewUnmarshaller(opts ...types.Option) types.Unmarshaller
   ```

#### Phase 4: Testing & Documentation
1. **Update all examples and documentation**
2. **Add tests ensuring interface compliance**
3. **Create migration guide for API changes**

### Benefits of Unification

- **Consistent Developer Experience:** Same constructor patterns across all backends
- **Easier Testing:** Interface-based testing works uniformly
- **Better Documentation:** Single pattern to document and teach
- **Future Maintenance:** Easier to add new marshallers following established patterns
- **Type Safety:** Interface compliance ensures proper implementation

### Backward Compatibility

API changes should maintain backward compatibility where possible:
- Keep existing constructors as deprecated but functional
- Add new standardized constructors alongside old ones
- Update documentation to recommend new patterns
- Provide clear migration timeline

## Future Enhancements

1. Typed helper methods (`UnmarshalModel`, `MarshalTensor`) layered on top of the generic API.
2. Streaming chunk encoders for large tensors to reduce memory usage.
3. Cross-format conversion utilities that chain marshaller/unmarshaller pairs.
4. Schema/version compatibility checks driven by `Options.FormatVersion`.
5. **Device marshaller standardization**: Common patterns for camera/sensor marshallers.
6. **Hardware abstraction**: Unified interfaces for different device types.
7. **Performance monitoring**: Built-in metrics for marshaller performance.
8. **API unification**: Complete standardization of constructor patterns across all marshallers.

pkg/core/marshaller/SPEC.md