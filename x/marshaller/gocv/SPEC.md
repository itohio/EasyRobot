# GoCV Marshaller Specification

## Overview

The GoCV marshaller provides a unified interface for computer vision operations using the GoCV library. It supports marshalling and unmarshalling of various data types including images, videos, tensors, DNN models, and real-time camera streams. The marshaller follows GoCV-first principles, delegating image/video/DNN handling to GoCV while using protobuf for metadata and control structures.

Key features:
- **Multi-format Support**: Images (PNG/JPEG/BMP/TIFF), videos (AVI/MP4/MOV/MKV), DNN models (ONNX/Caffe/TensorFlow)
- **Real-time Streaming**: Video capture from cameras with configurable parameters
- **Tensor Integration**: Seamless conversion between GoCV Mats and EasyRobot tensors
- **Display Support**: Real-time visualization with customizable event handling
- **Device Management**: Camera enumeration, selection, and runtime parameter control

## Core Concepts

### Data Types

The marshaller handles several core data types:

1. **Mat/Image/Tensor**: Single images converted to GoCV-managed formats (PNG/JPEG/BMP)
2. **FrameStream**: Real-time or file-based video streams from multiple sources
3. **DNN Models**: Neural network models in various formats (ONNX primary support)
4. **Camera Devices**: Video capture devices with configurable parameters

### Marshalling vs Unmarshalling

- **Marshalling**: Converts Go types to serialized bytes or streams
- **Unmarshalling**: Creates Go types from serialized data or device sources

### Configuration Architecture

The marshaller uses focused configuration structs:

```go
type config struct {
    codec   codecConfig    // Image encoding, tensor options
    stream  streamConfig   // Source specifications, sync modes
    display displayConfig  // Window settings, event handlers
    dnn     dnnConfig      // DNN backend/target preferences
}
```

## Use Cases and Examples

### 1. Single Image Processing

**Reading an image file:**
```go
unmarshaller := gocv.NewUnmarshaller()
var mat gocv.Mat
err := unmarshaller.Unmarshal(strings.NewReader("image.png"), &mat)
```

**Marshalling an image:**
```go
marshaller := gocv.NewMarshaller()
var buf bytes.Buffer
err := marshaller.Marshal(&buf, mat, gocv.WithImageEncoding("jpeg"))
```

### 2. Folder of Images

**Processing all images in a directory:**
```go
unmarshaller := gocv.NewUnmarshaller(gocv.WithPath("/path/to/images/*.png"))
var stream types.FrameStream
err := unmarshaller.Unmarshal(nil, &stream)
defer stream.Close()

for frame := range stream.C {
    // Process each frame
    tensor := frame.Tensor
    // ... processing logic ...
    tensor.Release()
}
```

**Saving processed images:**
```go
marshaller := gocv.NewMarshaller(gocv.WithPath("/output/directory"))
err := marshaller.Marshal(nil, stream) // Writes all frames to disk
```

### 3. Stereo Camera Capture

**Dual camera setup:**
```go
unmarshaller := gocv.NewUnmarshaller(
    gocv.WithVideoDevice(0, 1920, 1080), // Left camera
    gocv.WithVideoDevice(1, 1920, 1080), // Right camera
)
var stream types.FrameStream
err := unmarshaller.Unmarshal(nil, &stream)
defer stream.Close()

for frame := range stream.C {
    // frame contains tensors from both cameras
    left := frame.Tensor[0]   // Camera 0
    right := frame.Tensor[1]  // Camera 1
    // ... stereo processing ...
}
```

### 4. Single Camera Capture

**Basic camera capture:**
```go
unmarshaller := gocv.NewUnmarshaller(gocv.WithVideoDevice(0, 640, 480))
var stream types.FrameStream
err := unmarshaller.Unmarshal(nil, &stream)
defer stream.Close()

for frame := range stream.C {
    tensor := frame.Tensor[0]
    // Process single camera frame
    tensor.Release()
}
```

### 5. Video File Processing

**Reading from video file:**
```go
unmarshaller := gocv.NewUnmarshaller(gocv.WithPath("input.mp4"))
var stream types.FrameStream
err := unmarshaller.Unmarshal(nil, &stream)
defer stream.Close()

for frame := range stream.C {
    // Process each frame from video
    tensor := frame.Tensor[0]
    tensor.Release()
}
```

### 6. Video File Writing

**Recording to video file:**
```go
// Note: Video encoding not yet implemented in current version
// Planned feature for future release
marshaller := gocv.NewMarshaller(gocv.WithVideoOutput("output.mp4", 30))
err := marshaller.Marshal(nil, stream)
```

### 7. DNN Model Loading

**Loading ONNX model:**
```go
unmarshaller := gocv.NewUnmarshaller(gocv.WithDNNFormat("onnx"))
var net gocv.Net
modelBytes, _ := os.ReadFile("model.onnx")
err := unmarshaller.Unmarshal(bytes.NewReader(modelBytes), &net,
    gocv.WithNetBackend(cv.NetBackendCUDA),
    gocv.WithNetTarget(cv.NetTargetGPU))
```

### 8. Real-time Display

**Displaying camera feed:**
```go
ctx, cancel := context.WithCancel(context.Background())
defer cancel()

unmarshaller := gocv.NewUnmarshaller(
    gocv.WithVideoDevice(0, 640, 480),
    gocv.WithDisplay(ctx),
    gocv.WithTitle("Camera Feed"),
    gocv.WithOnKey(func(key int) bool {
        if key == 27 { // ESC
            cancel()
            return false
        }
        return true
    }),
)

var stream types.FrameStream
err := unmarshaller.Unmarshal(nil, &stream)
defer stream.Close()

// Stream automatically displays frames
stream.Wait() // Blocks until display closes
```

### 9. Camera Device Enumeration

**Listing available cameras:**
```go
unmarshaller := gocv.NewUnmarshaller()
var devices []types.CameraInfo // or []gocv.CameraInfo (re-exported)
err := unmarshaller.Unmarshal(strings.NewReader("list"), &devices)
for _, dev := range devices {
    fmt.Printf("Camera %d: %s (%s)\n", dev.ID, dev.Name, dev.Path)
}
```

### 10. Configurable Camera Capture

**Camera with specific settings:**
```go
unmarshaller := gocv.NewUnmarshaller(
    gocv.WithVideoDevice(0, 1920, 1080),
    gocv.WithFrameRate(30),
    gocv.WithPixelFormat("MJPEG"),
    gocv.WithCameraControls(map[string]int32{
        "brightness": 128,
        "contrast": 32,
        "exposure": 500,
    }),
)
var stream types.FrameStream
err := unmarshaller.Unmarshal(nil, &stream)
// Camera starts with specified parameters
```

### 11. Runtime Camera Control

**Adjusting camera parameters while streaming:**
```go
// Get the camera controller from the unmarshaller
controller := unmarshaller.CameraController(0) // Camera 0
err := controller.SetControl("exposure", 750)
current := controller.GetControl("brightness")
```

### 12. Mixed Source Streaming

**Combining multiple input types:**
```go
unmarshaller := gocv.NewUnmarshaller(
    gocv.WithPath("/images/frame_*.png"),     // Image sequence
    gocv.WithPath("background.mp4"),          // Background video
    gocv.WithVideoDevice(0, 640, 480),       // Live camera
)
var stream types.FrameStream
err := unmarshaller.Unmarshal(nil, &stream)
// Stream synchronizes across all sources
```

## Configuration Options

### Camera Configuration

```go
// Device selection
gocv.WithVideoDevice(id int, width, height int)

// Frame format and rate
gocv.WithPixelFormat(format string)     // "MJPEG", "YUYV", etc.
gocv.WithFrameRate(fps int)            // Target frame rate

// Initial camera controls
gocv.WithCameraControls(map[string]int32{
    "brightness": 128,
    "contrast": 32,
    "saturation": 64,
    "hue": 0,
    "gamma": 100,
    "exposure": 500,
    "gain": 0,
    "sharpness": 3,
})
```

### Stream Configuration

```go
// Source synchronization
gocv.WithSequential(true)              // Process sources sequentially vs parallel
gocv.WithBestEffortDevices(true)       // Allow dropped frames for devices

// File ordering
gocv.WithSorter(func(files []string) []string {
    // Custom file sorting logic
    return files
})
```

### Codec Configuration

```go
// Image encoding
gocv.WithImageEncoding("png")          // "png", "jpeg", "bmp", "tiff"

// Tensor conversion options
gocv.WithTensorOptions(tensorgocv.WithDevice("cuda"))
```

### Display Configuration

The display system uses shared interfaces from `marshaller/types` for cross-marshaller compatibility:

```go
// Window setup
gocv.WithDisplay(ctx)
gocv.WithTitle("Window Title")
gocv.WithWindowSize(800, 600)

// Event handling (legacy signatures - still supported)
gocv.WithOnKey(func(key int) bool { /* handle key */ })
gocv.WithOnMouse(func(event, x, y, flags int) bool { /* handle mouse */ })

// Event handling (new shared interfaces - recommended)
gocv.WithOnKey(func(event types.KeyEvent) bool { /* handle key */ })
gocv.WithOnMouse(func(event types.MouseEvent) bool { /* handle mouse */ })
gocv.WithEventLoop(func(ctx context.Context, shouldContinue func() bool) {
    // Custom event loop
})
```

### DNN Configuration

```go
// Model loading preferences
gocv.WithDNNFormat("onnx")             // "onnx", "caffe", "tensorflow"
gocv.WithNetBackend(cv.NetBackendCUDA)
gocv.WithNetTarget(cv.NetTargetGPU)
```

## Camera Control Interface

### Runtime Controls

When streaming from cameras, the marshaller provides runtime access to camera controls:

```go
type CameraController interface {
    // Get available controls
    Controls() []ControlInfo

    // Get/set control values
    GetControl(name string) (int32, error)
    SetControl(name string, value int32) error

    // Get/set multiple controls atomically
    GetControls() (map[string]int32, error)
    SetControls(map[string]int32) error
}
```

### Control Types

Common camera controls include:

- **brightness**: Image brightness (0-255)
- **contrast**: Image contrast (0-255)
- **saturation**: Color saturation (0-255)
- **hue**: Color hue (-180 to 180)
- **gamma**: Gamma correction (1-300)
- **exposure**: Exposure time (microseconds)
- **gain**: Analog gain (0-255)
- **sharpness**: Image sharpness (0-255)
- **white_balance_temperature**: White balance temperature (2000-8000K)

## Device Information

### CameraInfo Structure

The `CameraInfo` structure is defined in the shared `marshaller/types` package:

```go
type CameraInfo struct {
    ID          int           // Device ID (0, 1, 2, ...)
    Path        string        // Device path (/dev/video0)
    Name        string        // Device name
    Driver      string        // Kernel driver
    Card        string        // Device description
    BusInfo     string        // Bus information
    Capabilities any          // Device capabilities (implementation-specific)
    SupportedFormats []VideoFormat // Supported video formats
    Controls    []ControlInfo // Available camera controls
    Metadata    map[string]any // Additional implementation-specific info
}
```

### VideoFormat Structure

The `VideoFormat` structure is defined in the shared `marshaller/types` package:

```go
type VideoFormat struct {
    PixelFormat any          // Pixel format identifier (implementation-specific)
    Description string       // Human-readable description
    Width       int          // Frame width
    Height      int          // Frame height
    Metadata    map[string]any // Additional format-specific metadata
}
```

## Error Handling

The marshaller provides specific error types:

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
    Requested string
    Available []string
    Err       error
}
```

## Performance Considerations

### Memory Management

- **Zero-copy tensor wrapping**: Camera buffers are wrapped directly in tensors when possible
- **Buffer pooling**: Reuse of tensor buffers across frames
- **Automatic cleanup**: Tensor.Release() returns buffers to pools

### Synchronization

- **Parallel processing**: Multiple sources processed in lockstep by default
- **Sequential mode**: Process sources one after another
- **Best-effort devices**: Allow dropped frames to prevent blocking

### Hardware Acceleration

- **GPU support**: Tensor operations can use CUDA/Vulkan backends
- **DNN acceleration**: Models can use GPU acceleration when available
- **Camera acceleration**: Hardware-accelerated camera interfaces when supported

## Implementation Roadmap

### Phase 1 (Current)
- ✅ Basic image marshalling/unmarshalling
- ✅ Directory/folder processing
- ✅ Video file reading
- ✅ DNN model loading
- ✅ Display support
- ✅ Basic camera capture

### Phase 2 (In Progress)
- 🔄 Camera device enumeration ("list" command)
- 🔄 Camera selection and configuration
- 🔄 Runtime camera parameter control

### Phase 3 (Future)
- 📋 Video encoding support
- 📋 Advanced pixel format support
- 📋 Multi-camera synchronization
- 📋 Hardware-accelerated processing
- 📋 Stream recording and playback

## Dependencies

- `gocv.io/x/gocv` - Core GoCV library
- `github.com/itohio/EasyRobot/x/math/tensor/gocv` - Tensor integration
- `github.com/itohio/EasyRobot/x/math/primitive/generics/helpers.Pool` - Buffer pooling
- `google.golang.org/protobuf` - Protocol buffer support
