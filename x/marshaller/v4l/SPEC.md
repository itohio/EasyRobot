# V4L Video Device Unmarshaller Specification

## Overview

The V4L (Video for Linux) unmarshaller provides an interface for capturing video frames from V4L2-compatible devices (cameras, capture cards, etc.) and converting them into EasyRobot tensor streams. It leverages the `go4vl` library for low-level V4L2 interaction and integrates with the marshaller ecosystem for seamless tensor production.

Key features:
- **Device Discovery**: Enumerate and query V4L2 devices with capabilities
- **Multi-Device Support**: Open and manage multiple video devices simultaneously
- **Runtime Controls**: Adjust camera parameters (exposure, brightness, etc.) while streaming
- **Format Flexibility**: Support multiple pixel formats with automatic conversion
- **Buffer Pooling**: Efficient memory reuse using pooled tensor buffers
- **Zero-Copy Streaming**: Memory-mapped buffers for high-throughput capture

## Core Concepts

### Device Management

Devices are represented by paths (`/dev/video0`, `/dev/video1`, etc.) and expose capabilities through V4L2 ioctl interfaces. The unmarshaller provides:

- Device enumeration with capability reporting
- Device opening with configurable options
- Runtime parameter adjustment
- Stream lifecycle management

### Tensor Integration

Video frames are converted to tensors with:
- **Data Type**: `UINT8` for raw pixel data
- **Shape**: `[height, width, channels]` (HWC format)
- **Buffer Reuse**: Pooled buffers for zero-allocation streaming

### Buffer Pooling

The unmarshaller uses `helpers.Pool[uint8]` for efficient buffer reuse:
- Tiered buffer pools based on frame size
- Automatic buffer lifecycle management
- Integration with tensor `Release()` method

## Architecture

### Core Interfaces

The V4L marshaller implements the shared camera interfaces from `marshaller/types`:

```go
// CameraDevice implements the shared CameraDevice interface
type CameraDevice interface {
    // Info returns device information and capabilities
    Info() types.CameraInfo

    // Open opens the device with specified options
    Open(opts ...types.CameraOption) (types.CameraStream, error)

    // Close closes the device
    Close() error
}

// CameraStream implements the shared CameraStream interface
type CameraStream interface {
    // Start begins frame capture
    Start(ctx context.Context) error

    // Stop halts frame capture
    Stop() error

    // Controller returns the camera controller for runtime control
    Controller() types.CameraController

    // Close closes the stream
    Close() error
}

// CameraController implements the shared CameraController interface
type CameraController interface {
    // Controls returns available device controls
    Controls() []types.ControlInfo

    // SetControl sets a control value by name
    SetControl(name string, value int32) error

    // GetControl gets a control value by name
    GetControl(name string) (int32, error)

    // GetControls gets all control values
    GetControls() (map[string]int32, error)

    // SetControls sets multiple control values
    SetControls(map[string]int32) error
}

// Frame represents a captured video frame
type Frame struct {
    // Index is the frame sequence number
    Index int

    // Timestamp is capture timestamp (nanoseconds since epoch)
    Timestamp int64

    // Tensor contains the frame data as uint8 tensor
    Tensor types.Tensor

    // Metadata contains frame-specific information
    Metadata map[string]any
}
```

### Device Information

```go
// DeviceInfo contains device capabilities and metadata
type DeviceInfo struct {
    // Path is the device node path
    Path string

    // Name is the device name
    Name string

    // Driver is the kernel driver name
    Driver string

    // BusInfo is the bus information
    BusInfo string

    // Version is the driver version
    Version string

    // Capabilities bitmap
    Capabilities CapabilityFlags

    // SupportedFormats lists available pixel formats
    SupportedFormats []Format

    // Controls lists available device controls
    Controls []ControlInfo
}

// CapabilityFlags represents device capabilities
type CapabilityFlags uint32

const (
    CapVideoCapture CapabilityFlags = 1 << iota
    CapVideoOutput
    CapVideoOverlay
    CapVBI_CAPTURE
    CapVBI_OUTPUT
    CapSLICED_VBI_CAPTURE
    CapSLICED_VBI_OUTPUT
    CapRDS_CAPTURE
    CapVideoOutputOverlay
    CapHW_FREQ_SEEK
    CapRDS_OUTPUT
    CapVideoCaptureMplane
    CapVideoOutputMplane
    CapVideoM2M
    CapVideoM2MMplane
    CapTuner
    CapAudio
    CapRadio
    CapModulator
    CapReadWrite
    CapAsyncIO
    CapStreaming
    CapDeviceCaps
)
```

### Format and Controls

```go
// Format describes video stream format
type Format struct {
    // Width in pixels
    Width int

    // Height in pixels
    Height int

    // PixelFormat is the fourcc pixel format
    PixelFormat PixelFormat

    // FrameRate is frames per second
    FrameRate Fraction

    // Field order (interlaced/progressive)
    Field Field
}

// ControlInfo describes a device control
type ControlInfo struct {
    // ID is the control identifier
    ID ControlID

    // Name is the human-readable name
    Name string

    // Type indicates the control type
    Type ControlType

    // Min/Max/Default values
    Min, Max, Default int32

    // Step size for integer controls
    Step int32

    // Menu items for menu controls
    MenuItems []string
}

// ControlID represents V4L2 control identifiers
type ControlID uint32

// Common control IDs
const (
    CtrlBrightness ControlID = iota
    CtrlContrast
    CtrlSaturation
    CtrlHue
    CtrlAutoWhiteBalance
    CtrlDoWhiteBalance
    CtrlRedBalance
    CtrlBlueBalance
    CtrlGamma
    CtrlExposure
    CtrlAutogain
    CtrlGain
    CtrlHFlip
    CtrlVFlip
    CtrlPowerLineFrequency
    CtrlHueAuto
    CtrlWhiteBalanceTemperature
    CtrlSharpness
    CtrlBacklightCompensation
    CtrlChromaAGC
    CtrlColorKiller
    CtrlColorEffects
    CtrlAutobrightness
    CtrlBandStopFilter
    CtrlRot
    CtrlBgColor
    CtrlChromaGain
    CtrlIlluminator1
    CtrlIlluminator2
)
```

### Options Pattern

```go
// DeviceOption configures device opening
type Option interface {
    Apply(*Options)
}

// Options holds device configuration
type Options struct {
    // BufferCount is number of capture buffers (default: 4)
    BufferCount int

    // PixelFormat specifies desired pixel format
    PixelFormat PixelFormat

    // Width/Height specify desired resolution
    Width, Height int

    // FrameRate specifies desired frame rate
    FrameRate Fraction

    // Controls specifies initial control values
    Controls map[ControlID]int32

    // BufferPool provides custom buffer pool (default: internal pool)
    BufferPool *helpers.Pool[uint8]

    // TensorFactory creates tensors from buffer data
    TensorFactory func([]uint8, int, int, int) types.Tensor
}
```

## Tensor Integration

### Pooled Tensor Implementation

The unmarshaller provides a pooled tensor implementation that integrates with the buffer pool:

```go
// PooledTensor wraps uint8 data with buffer pool integration
type PooledTensor struct {
    data   []uint8
    pool   *helpers.Pool[uint8]
    width  int
    height int
    channels int
}

// NewPooledTensor creates a tensor from pooled buffer
func NewPooledTensor(pool *helpers.Pool[uint8], width, height, channels int) *PooledTensor {
    size := width * height * channels
    data := pool.Get(size)

    return &PooledTensor{
        data:     data,
        pool:     pool,
        width:    width,
        height:  height,
        channels: channels,
    }
}

// Release returns the buffer to the pool
func (t *PooledTensor) Release() {
    if t.data != nil && t.pool != nil {
        t.pool.Put(t.data)
        t.data = nil
    }
}

// Data returns the underlying uint8 data
func (t *PooledTensor) Data() any {
    return t.data[:t.width*t.height*t.channels]
}

// Shape returns [height, width, channels]
func (t *PooledTensor) Shape() types.Shape {
    return types.NewShape(t.height, t.width, t.channels)
}

// DataType returns UINT8
func (t *PooledTensor) DataType() types.DataType {
    return types.UINT8
}
```

### Tensor Factory

The default tensor factory creates pooled tensors:

```go
func defaultTensorFactory(pool *helpers.Pool[uint8]) func([]uint8, int, int, int) types.Tensor {
    return func(data []uint8, width, height, channels int) types.Tensor {
        // For zero-copy, wrap the existing buffer
        return &PooledTensor{
            data:     data,
            pool:     pool,
            width:    width,
            height:   height,
            channels: channels,
        }
    }
}
```

## Marshaller Integration

### Unmarshaller Interface

The V4L unmarshaller implements the standard marshaller interfaces:

```go
type Unmarshaller struct {
    opts Options
    cfg  config
}

func NewUnmarshaller(opts ...types.Option) types.Unmarshaller {
    baseOpts, baseCfg := applyOptions(types.Options{}, defaultConfig(), opts)
    return &Unmarshaller{
        opts: baseOpts,
        cfg:  baseCfg,
    }
}

func (u *Unmarshaller) Format() string {
    return "v4l"
}

func (u *Unmarshaller) Unmarshal(r io.Reader, dst any, opts ...types.Option) error {
    // Implementation handles:
    // - Device enumeration (*[]DeviceInfo)
    // - Device opening (*Stream)
    // - Stream unmarshalling (*types.FrameStream)
}
```

### Supported Destinations

The unmarshaller supports unmarshalling to:

1. **Device enumeration**:
   ```go
   var devices []DeviceInfo
   err := unmarshaller.Unmarshal(reader, &devices)
   ```

2. **Device opening**:
   ```go
   var stream Stream
   err := unmarshaller.Unmarshal(strings.NewReader("/dev/video0"), &stream)
   ```

3. **Frame stream creation**:
   ```go
   var frameStream types.FrameStream
   err := unmarshaller.Unmarshal(configReader, &frameStream)
   ```

## Usage Examples

### Device Discovery

```go
// Create unmarshaller
unmarshaller := v4l.NewUnmarshaller()
if err != nil {
    log.Fatal(err)
}

// Enumerate devices
var devices []types.CameraInfo
err = unmar.Unmarshal(strings.NewReader("list"), &devices)
for _, dev := range devices {
    fmt.Printf("Camera: %s (%s)\n", dev.Name, dev.Path)
    fmt.Printf("  Formats: %d available\n", len(dev.SupportedFormats))
    fmt.Printf("  Controls: %d available\n", len(dev.Controls))
}
```

### Single Camera Capture

```go
// Configure camera via unmarshaller options
unmarshaller := v4l.NewUnmarshaller(
    // Camera 0: 1024x768 MJPEG at 30fps
    v4l.WithVideoDeviceEx(0, 1024, 768, 30, "MJPEG"),

    // Optional: Configure camera controls
    v4l.WithCameraControls(map[string]int32{
        types.CameraControlBrightness: 128,
        types.CameraControlExposure: 500,
    }),
)

// Create frame stream
var frameStream types.FrameStream
err := unmarshaller.Unmarshal(nil, &frameStream)
if err != nil {
    log.Fatal(err)
}
defer frameStream.Close()

// Process frames
for frame := range frameStream.C {
    tensor := frame.Tensors[0] // Single camera
    fmt.Printf("Frame: %v\n", tensor.Shape())

    // Access camera controls during streaming
    if controller := unmarshaller.CameraController("/dev/video0"); controller != nil {
        brightness, _ := controller.GetControl(types.CameraControlBrightness)
        controller.SetControl(types.CameraControlContrast, 32)
    }

    tensor.Release()
}
```

### Multi-Camera Synchronized Capture

```go
// Configure dual camera capture
unmarshaller := v4l.NewUnmarshaller(
    // Camera 0: 1024x768 YUV at 30fps
    v4l.WithVideoDeviceEx(0, 1024, 768, 30, "YUYV"),

    // Camera 1: 1024x768 YUV at 30fps
    v4l.WithVideoDeviceEx(1, 1024, 768, 30, "YUYV"),

    // Configure controls for both cameras
    v4l.WithCameraControls(map[string]int32{
        types.CameraControlBrightness: 128,
        types.CameraControlContrast:   32,
        types.CameraControlExposure:   500,
    }),
)

// Create synchronized frame stream
var frameStream types.FrameStream
err := unmarshaller.Unmarshal(nil, &frameStream)
if err != nil {
    log.Fatal(err)
}
defer frameStream.Close()

// Process synchronized frames
frameCount := 0
for frame := range frameStream.C {
    frameCount++
    fmt.Printf("Frame %d: %d tensors\n", frameCount, len(frame.Tensors))

    // Process each camera's frame
    for i, tensor := range frame.Tensors {
        fmt.Printf("  Camera %d: %v\n", i, tensor.Shape())

        // Access individual camera controls
        devicePath := fmt.Sprintf("/dev/video%d", i)
        if controller := unmarshaller.CameraController(devicePath); controller != nil {
            exposure, _ := controller.GetControl(types.CameraControlExposure)
            fmt.Printf("    Exposure: %d\n", exposure)
        }

        tensor.Release()
    }

    if frameCount >= 100 { // Demo limit
        break
    }
}
```

### Runtime Camera Control

```go
// Get camera controller from unmarshaller
controller := unmarshaller.CameraController("/dev/video0")

// Adjust controls while streaming
err := controller.SetControl(types.CameraControlExposure, 750)
brightness, err := controller.GetControl(types.CameraControlBrightness)

// Set multiple controls
err = controller.SetControls(map[string]int32{
    types.CameraControlContrast: 32,
    types.CameraControlSaturation: 64,
})
```

### Runtime Control

```go
// Get stream from unmarshalled FrameStream
// (implementation detail - access underlying Stream)

// Adjust exposure while streaming
err := stream.SetControl(CtrlExposure, 500)
if err != nil {
    log.Printf("Failed to set exposure: %v", err)
}
```

## Configuration Format

The unmarshaller accepts JSON configuration for device setup:

```json
{
    "device": "/dev/video0",
    "buffer_count": 4,
    "width": 1920,
    "height": 1080,
    "pixel_format": "MJPEG",
    "frame_rate": {
        "numerator": 30,
        "denominator": 1
    },
    "controls": {
        "brightness": 128,
        "contrast": 32,
        "exposure": 500
    },
    "options": {
        "buffer_pool_tiers": [65536, 262144, 1048576]
    }
}
```

## Error Handling

The unmarshaller provides specific error types:

```go
// Device errors
type DeviceError struct {
    Device string
    Op     string
    Err    error
}

// Stream errors
type StreamError struct {
    Device string
    Op     string
    Err    error
}

// Format errors
type FormatError struct {
    Requested Format
    Available []Format
    Err       error
}
```

## Performance Considerations

### Buffer Pool Sizing

- **Default tiers**: 64KB, 256KB, 1MB, 4MB (configurable)
- **Tier calculation**: Based on frame size (width × height × channels)
- **Pool growth**: Automatic expansion for larger frames

### Zero-Copy Operation

- Memory-mapped V4L2 buffers when possible
- Direct tensor wrapping of capture buffers
- Reference counting for multi-consumer scenarios

### Memory Management

- Automatic buffer return on `Tensor.Release()`
- Pool prevents garbage collection pressure
- Configurable pool limits prevent memory exhaustion

## Implementation Roadmap

1. **Phase 1**: Basic device enumeration and single-device streaming
2. **Phase 2**: Multi-device support and runtime controls
3. **Phase 3**: Advanced format support and buffer optimization
4. **Phase 4**: Hardware acceleration integration (V4L2 M2M)

## Dependencies

- `github.com/vladimirvivien/go4vl/v4l2` - V4L2 API bindings
- `github.com/vladimirvivien/go4vl/device` - Device management
- `x/math/primitive/generics/helpers.Pool` - Buffer pooling
- `x/math/tensor/types` - Tensor interfaces
