# V4L Video Device Marshaller

The V4L marshaller provides an interface for capturing video frames from V4L2-compatible devices (cameras, capture cards, etc.) and converting them into EasyRobot tensor streams.

## Features

- **Device Discovery**: Enumerate and query V4L2 devices with capabilities
- **Multi-Device Support**: Open and manage multiple video devices simultaneously
- **Runtime Controls**: Adjust camera parameters (exposure, brightness, etc.) while streaming
- **Format Flexibility**: Support multiple pixel formats with automatic conversion
- **Buffer Pooling**: Efficient memory reuse using pooled tensor buffers
- **Zero-Copy Streaming**: Memory-mapped buffers for high-throughput capture
- **Synchronization**: Synchronized frame capture across multiple cameras

## Quick Start

### Device Discovery

```go
// Create unmarshaller (entry point for all V4L operations)
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
unmarshaller := v4l.NewUnmarshaller(,
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
unmarshaller := v4l.NewUnmarshaller(,
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
// Get camera controller
controller := stream.Controller()

// Adjust exposure while streaming
err := controller.SetControl(types.CameraControlExposure, 500)
if err != nil {
    log.Printf("Failed to set exposure: %v", err)
}

// Get current brightness
brightness, err := controller.GetControl(types.CameraControlBrightness)
if err != nil {
    log.Printf("Failed to get brightness: %v", err)
} else {
    fmt.Printf("Brightness: %d\n", brightness)
}

// Set multiple controls at once
err = controller.SetControls(map[string]int32{
    types.CameraControlBrightness: 128,
    types.CameraControlContrast: 32,
    types.CameraControlSaturation: 64,
})
```

## Configuration Options

### Device Options

- `buffer_count`: Number of capture buffers (default: 4)
- `width`, `height`: Resolution (default: 640x480)
- `pixel_format`: Pixel format (default: "MJPEG")
- `frame_rate`: Frame rate as fraction (default: 30/1 fps)
- `controls`: Initial control values (map of control ID to value)
- `allow_best_effort`: Enable best-effort synchronization (default: false)
- `sequential`: Process devices sequentially instead of synchronized (default: false)

### Supported Pixel Formats

- `MJPEG`: Motion JPEG (compressed)
- `YUYV`: YUV 4:2:2 packed
- `RGB24`: RGB 24-bit
- `BGR24`: BGR 24-bit
- `NV12`: YUV 4:2:0 planar
- `GREY`: Grayscale

### Camera Controls

Common controls include:
- `CtrlBrightness`: Image brightness
- `CtrlContrast`: Image contrast
- `CtrlSaturation`: Color saturation
- `CtrlHue`: Color hue
- `CtrlExposure`: Exposure time
- `CtrlGain`: Camera gain
- `CtrlWhiteBalanceTemperature`: White balance temperature

## Buffer Pooling

The marshaller uses pooled buffers for efficient memory management:

```go
// Custom buffer pool with specific tiers
pool := &helpers.Pool[uint8]{}
pool.Reconfigure(65536, 262144, 1048576) // 64KB, 256KB, 1MB tiers

unmar, _ := v4l.NewUnmarshaller(, v4l.WithBufferPool(pool))
```

## Integration with FrameStream

The V4L marshaller integrates with the EasyRobot FrameStream ecosystem:

```go
var frameStream types.FrameStream
err := unmar.Unmarshal(configReader, &frameStream)
if err != nil {
    log.Fatal(err)
}
defer frameStream.Close()

// Use with pipeline processing
for frame := range frameStream.C {
    // Process tensors
    for _, tensor := range frame.Tensors {
        // Apply computer vision algorithms
        result := processTensor(tensor)
        tensor.Release()
    }
}
```

## Error Handling

The marshaller provides specific error types:

```go
switch err := err.(type) {
case *v4l.DeviceError:
    log.Printf("Device error on %s: %v", err.Device, err.Err)
case *v4l.StreamError:
    log.Printf("Stream error on %s: %v", err.Device, err.Err)
default:
    log.Printf("Unknown error: %v", err)
}
```

## Performance Considerations

### Buffer Pool Sizing

Choose buffer pool tiers based on your frame sizes:
- Small frames (640x480): 64KB, 256KB tiers
- HD frames (1920x1080): 256KB, 1MB, 4MB tiers
- 4K frames: 1MB, 4MB, 16MB tiers

### Synchronization Modes

- **Synchronized**: All cameras capture frames with the same index (default)
- **Best Effort**: Cameras capture independently, frames combined when available
- **Sequential**: Cameras processed one after another

### Memory Management

- Always call `tensor.Release()` after processing frames
- Use appropriate buffer pool sizes to prevent GC pressure
- Monitor frame drop rates when using synchronization

## Dependencies

- `github.com/vladimirvivien/go4vl/v4l2`: V4L2 API bindings
- `github.com/vladimirvivien/go4vl/device`: Device management
- `x/math/primitive/generics/helpers.Pool`: Buffer pooling
- `x/math/tensor/types`: Tensor interfaces

## Build Tags

The V4L marshaller is enabled by default. To disable:

```bash
go build -tags no_v4l
```

## Troubleshooting

### Device Not Found

```bash
# List available video devices
ls -l /dev/video*

# Check device capabilities
v4l2-ctl --list-devices
```

### Permission Denied

```bash
# Add user to video group
sudo usermod -a -G video $USER
# Log out and back in
```

### No Frames Received

```bash
# Check device capabilities
v4l2-ctl -d /dev/video0 --all

# Test basic capture
v4l2-ctl -d /dev/video0 --stream-mmap --stream-count=1
```
