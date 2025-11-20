# Marshaller Types Specification

## Overview

The `marshaller/types` package provides shared interfaces and types for computer vision marshallers. These common types enable consistent APIs across different marshaller implementations (GoCV, V4L, etc.) and promote code reuse.

### Key Principles

- **Interface-based design** - Types define contracts, not implementations
- **Cross-marshaller compatibility** - Same interfaces work across backends
- **Extensibility** - New marshallers can implement existing interfaces
- **Backward compatibility** - Legacy types are supported alongside new ones
- **Generic enough** - Types work with different hardware and software backends

## Core Interfaces

### Marshaller/Unmarshaller

```go
type Marshaller interface {
    Format() string
    Marshal(w io.Writer, value any, opts ...Option) error
}

type Unmarshaller interface {
    Format() string
    Unmarshal(r io.Reader, dst any, opts ...Option) error
}
```

**For Implementers:**
- Return your format identifier (e.g., "gocv", "v4l")
- Handle marshalling/unmarshalling of supported types
- Use `Option` interface for configuration

## Camera Types

Camera-related types provide a unified interface for video capture devices across different backends.

### CameraInfo

```go
type CameraInfo struct {
    ID          int           // Device identifier
    Path        string        // Device path (e.g., "/dev/video0")
    Name        string        // Human-readable name
    Driver      string        // Driver name
    Card        string        // Device description
    BusInfo     string        // Bus information
    Capabilities any          // Backend-specific capabilities
    SupportedFormats []VideoFormat // Available video formats
    Controls    []ControlInfo // Available controls
    Metadata    map[string]any // Additional backend data
}
```

**For Users:**
- Enumerate devices to get `CameraInfo` instances
- Check `SupportedFormats` and `Controls` for capabilities
- Use `ID` or `Path` to identify specific devices

**For Implementers:**
- Populate all fields with backend-specific data
- Convert backend capabilities to common format
- Include backend-specific metadata when needed

### VideoFormat

```go
type VideoFormat struct {
    PixelFormat any             // Backend-specific format identifier
    Description string          // Human-readable description
    Width       int             // Frame width
    Height      int             // Frame height
    Metadata    map[string]any  // Additional format data
}
```

**For Users:**
- Check available formats from `CameraInfo.SupportedFormats`
- Select format by `PixelFormat` or description

**For Implementers:**
- Map backend format identifiers to this structure
- Provide meaningful descriptions for users

### ControlInfo

```go
type ControlInfo struct {
    ID          any            // Backend-specific control identifier
    Name        string         // Control name for programmatic access
    Description string         // Human-readable description
    Type        string         // Control type ("integer", "boolean", "menu")
    Min, Max, Default int32    // Value range and default
    Step        int32          // Step size for integer controls
    MenuItems   []string       // Menu options (for menu controls)
    Metadata    map[string]any // Additional control data
}
```

**For Users:**
- Access controls by `Name` (e.g., "brightness", "exposure")
- Check `Type`, `Min`, `Max` for valid operations
- Use `Default` for resetting values

**For Implementers:**
- Map backend control IDs to common names when possible
- Provide accurate range and type information
- Include menu options for enumerated controls

### CameraController

```go
type CameraController interface {
    Controls() []ControlInfo
    GetControl(name string) (int32, error)
    SetControl(name string, value int32) error
    GetControls() (map[string]int32, error)
    SetControls(controls map[string]int32) error
}
```

**For Users:**
- Get available controls with `Controls()`
- Read/write individual controls by name
- Batch operations with `GetControls()`/`SetControls()`

**For Implementers:**
- Implement control mapping to backend-specific APIs
- Handle type conversions and validation
- Return appropriate errors for unsupported operations

### Camera Device Interfaces

```go
type CameraDevice interface {
    Info() CameraInfo
    Open(opts ...CameraOption) (CameraStream, error)
    Close() error
}

type CameraStream interface {
    Start(ctx Context) error
    Stop() error
    Controller() CameraController
    Close() error
}
```

**For Users:**
- Use `CameraDevice` for device enumeration and opening
- Use `CameraStream` for active capture sessions
- Access runtime controls via `Controller()`

**For Implementers:**
- `CameraDevice` manages device lifecycle
- `CameraStream` handles active capture and control
- Implement proper resource cleanup

## Display Types

Display types provide unified window management and rendering across different backends.

### DisplayWindow

```go
type DisplayWindow interface {
    ID() string
    Title() string
    Size() (width, height int)
    Show()
    Hide()
    Close() error
    IsOpen() bool
}
```

**For Users:**
- Basic window operations (show, hide, close)
- Query window properties

**For Implementers:**
- Map to backend-specific window handles
- Handle window lifecycle properly

### DisplayManager

```go
type DisplayManager interface {
    CreateWindow(title string, width, height int) (DisplayWindow, error)
    GetWindow(id string) (DisplayWindow, bool)
    CloseAll() error
    WaitForEvents(timeout int) bool
    PollEvents()
}
```

**For Users:**
- Create and manage multiple windows
- Handle event processing

**For Implementers:**
- Manage window registry
- Implement event loop integration
- Handle cross-platform differences

### Input Event Types

#### KeyEvent

```go
type KeyEvent struct {
    Key       int          // Key code
    Action    KeyAction    // Press/Release/Repeat
    Modifiers KeyModifier  // Modifier key states
    Timestamp int64        // Event timestamp (nanoseconds)
}
```

#### MouseEvent

```go
type MouseEvent struct {
    X, Y       int          // Mouse coordinates
    Button     MouseButton  // Mouse button
    Action     MouseAction  // Press/Release/Move/Scroll
    Modifiers  KeyModifier  // Modifier keys
    ScrollDeltaX, ScrollDeltaY int // Scroll amounts
    Timestamp int64         // Event timestamp
}
```

#### WindowEvent

```go
type WindowEvent struct {
    Action    WindowAction // What happened to window
    Width, Height int      // New dimensions (for resize)
    Timestamp int64        // Event timestamp
}
```

#### InputEvent (Generic)

```go
type InputEvent struct {
    Type      InputEventType // Event category
    Key       KeyEvent       // Key event data
    Mouse     MouseEvent     // Mouse event data
    WindowID  string         // Target window
    Timestamp int64          // Event timestamp
}
```

**For Users:**
- Handle input in event callbacks
- Check event types and extract relevant data
- Use timestamps for timing-sensitive operations

**For Implementers:**
- Convert backend events to these structures
- Maintain accurate timestamps
- Map backend key/button codes appropriately

### InputHandler

```go
type InputHandler interface {
    HandleKey(event KeyEvent) bool
    HandleMouse(event MouseEvent) bool
    HandleWindow(event WindowEvent) bool
}
```

**For Users:**
- Implement interface for custom event handling
- Return `false` to stop event processing

**For Implementers:**
- Call appropriate handler methods
- Respect return values for event filtering

### EventLoop

```go
type EventLoop func(ctx context.Context, shouldContinue func() bool)
```

**For Users:**
- Provide custom event processing logic
- Integrate with existing event loops

**For Implementers:**
- Support custom event loops when provided
- Fall back to default event processing

## Configuration Options

### Camera Options

```go
// Device configuration
WithCameraResolution(width, height int) CameraOption
WithCameraFrameRate(fps int) CameraOption
WithCameraPixelFormat(format any) CameraOption
WithCameraControls(controls map[string]int32) CameraOption
WithCameraBufferCount(count int) CameraOption
```

**For Users:**
- Configure camera parameters before opening
- Set initial control values

**For Implementers:**
- Apply options to backend-specific settings
- Validate option compatibility

### Display Options

```go
// Window configuration
WithDisplayEnabled() DisplayOption
WithDisplayTitle(title string) DisplayOption
WithDisplaySize(width, height int) DisplayOption

// Event handling
WithKeyHandler(func(KeyEvent) bool) DisplayOption
WithMouseHandler(func(MouseEvent) bool) DisplayOption
WithWindowHandler(func(WindowEvent) bool) DisplayOption
WithEventLoop(EventLoop) DisplayOption
WithDisplayContext(ctx context.Context) DisplayOption
```

**For Users:**
- Configure window properties
- Set up event handlers
- Provide custom event processing

**For Implementers:**
- Apply display options to window creation
- Set up event routing to handlers

## Common Constants

### Control Names

```go
const (
    CameraControlBrightness            = "brightness"
    CameraControlContrast              = "contrast"
    CameraControlSaturation            = "saturation"
    CameraControlHue                   = "hue"
    CameraControlGamma                 = "gamma"
    CameraControlExposure              = "exposure"
    CameraControlGain                  = "gain"
    CameraControlSharpness             = "sharpness"
    CameraControlWhiteBalanceTemp      = "white_balance_temperature"
    CameraControlAutoWhiteBalance      = "auto_white_balance"
    CameraControlAutogain              = "autogain"
    CameraControlBacklightCompensation = "backlight_compensation"
    CameraControlPowerLineFrequency     = "power_line_frequency"
    CameraControlHFlip                 = "horizontal_flip"
    CameraControlVFlip                 = "vertical_flip"
)
```

### Key Codes

```go
const (
    KeyUnknown = 0
    KeySpace   = 32
    KeyEscape  = 27
    KeyEnter   = 13
    KeyTab     = 9
    KeyBackspace = 8
    KeyDelete  = 127
    KeyArrowLeft  = 81
    KeyArrowUp    = 82
    KeyArrowRight = 83
    KeyArrowDown  = 84
    // ... function keys, numbers, letters
)
```

## Usage Examples

### For Users

#### Camera Device Enumeration
```go
// Using a marshaller that supports camera enumeration
var devices []types.CameraInfo
err := marshaller.Unmarshal(strings.NewReader("list"), &devices)

for _, dev := range devices {
    fmt.Printf("Camera: %s (%s)\n", dev.Name, dev.Path)
    fmt.Printf("  Formats: %d available\n", len(dev.SupportedFormats))
    fmt.Printf("  Controls: %d available\n", len(dev.Controls))
}
```

#### Camera Streaming with Controls
```go
unmarshaller := v4l.NewUnmarshaller(
    myMarshaller.WithCameraDevice(0, 1920, 1080),
    types.WithCameraControls(map[string]int32{
        types.CameraControlBrightness: 128,
        types.CameraControlExposure: 500,
    }),
)

var stream types.CameraStream
err := unmarshaller.Unmarshal(nil, &stream)
defer stream.Close()

// Runtime control adjustment
controller := stream.Controller()
err = controller.SetControl(types.CameraControlExposure, 750)
```

#### Display with Event Handling
```go
marshaller := myMarshaller.NewUnmarshaller(
    types.WithDisplayEnabled(),
    types.WithDisplayTitle("Video Feed"),
    types.WithDisplaySize(800, 600),
    types.WithKeyHandler(func(event types.KeyEvent) bool {
        if event.Key == types.KeyEscape {
            return false // Stop processing
        }
        return true
    }),
)

var frameStream types.FrameStream
err := marshaller.Unmarshal(source, &frameStream)
// Display automatically shows frames with event handling
```

### For Implementers

#### Implementing CameraController
```go
type myCameraController struct {
    deviceHandle *backend.Device
}

func (c *myCameraController) Controls() []types.ControlInfo {
    // Map backend controls to shared format
    return []types.ControlInfo{
        {
            ID:          backend.ControlBrightness,
            Name:        types.CameraControlBrightness,
            Description: "Image brightness",
            Type:        "integer",
            Min:         0,
            Max:         255,
            Default:     128,
            Step:        1,
        },
        // ... more controls
    }
}

func (c *myCameraController) GetControl(name string) (int32, error) {
    // Map name to backend control ID
    controlID, err := c.mapControlName(name)
    if err != nil {
        return 0, err
    }

    // Get value from backend
    value, err := c.deviceHandle.GetControl(controlID)
    return int32(value), err
}

func (c *myCameraController) SetControl(name string, value int32) error {
    controlID, err := c.mapControlName(name)
    if err != nil {
        return err
    }

    return c.deviceHandle.SetControl(controlID, backend.Value(value))
}
```

#### Implementing DisplayWindow
```go
type myDisplayWindow struct {
    id       string
    backendWindow *backend.Window
}

func (w *myDisplayWindow) ID() string {
    return w.id
}

func (w *myDisplayWindow) Title() string {
    return w.backendWindow.GetTitle()
}

func (w *myDisplayWindow) Size() (width, height int) {
    return w.backendWindow.GetSize()
}

func (w *myDisplayWindow) Show() {
    w.backendWindow.Show()
}

func (w *myDisplayWindow) Hide() {
    w.backendWindow.Hide()
}

func (w *myDisplayWindow) Close() error {
    return w.backendWindow.Destroy()
}

func (w *myDisplayWindow) IsOpen() bool {
    return w.backendWindow.IsValid()
}
```

## Best Practices

### For Users

1. **Check capabilities first** - Use `CameraInfo` to understand device capabilities before use
2. **Handle errors gracefully** - Camera/display operations can fail due to hardware issues
3. **Use appropriate event handlers** - Choose between simple callbacks and full event objects
4. **Clean up resources** - Always close streams and windows when done
5. **Prefer shared interfaces** - Use `types.*` interfaces for maximum compatibility

### For Implementers

1. **Map backend specifics appropriately** - Convert backend types to shared interfaces
2. **Provide comprehensive metadata** - Include all relevant information in `*Info` structures
3. **Handle edge cases** - Account for devices that don't support certain features
4. **Maintain thread safety** - Ensure interface implementations are thread-safe
5. **Document limitations** - Clearly indicate what backend features aren't supported
6. **Use common control names** - Map backend controls to standard names when possible

## Extension Points

### Adding New Control Types
- Extend `ControlInfo.Type` with new string values
- Add corresponding constants for common names
- Implement appropriate validation in marshallers

### Adding New Event Types
- Extend `*Event` structures with new fields
- Add new event handler methods to `InputHandler`
- Update `InputEvent` union type

### Adding New Option Types
- Create new `Option` implementations
- Follow the existing pattern of `with*` functions
- Document option behavior clearly

## Migration Guide

### From Legacy Marshaller Types

If you're migrating from marshaller-specific types to shared types:

1. **Update imports** - Use `types.*` instead of marshaller-specific types
2. **Update event handlers** - Change function signatures to use `types.KeyEvent`, etc.
3. **Update option calls** - Use `types.With*` functions where available
4. **Test thoroughly** - Ensure all functionality works with new types

### Backward Compatibility

The shared types are designed to be backward compatible:
- Legacy event handlers are still supported alongside new ones
- Existing option patterns continue to work
- Type aliases provide smooth migration paths

This specification provides a foundation for consistent, interoperable computer vision marshallers across different backends and implementations.
