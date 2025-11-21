# Obtainer Framework - Specification

## Overview

The obtainer framework provides a unified interface for spectrometer devices that can directly obtain spectra. This framework allows the spectrometer application to support multiple device types (CR30, AS734x, etc.) through a common interface.

## Responsibilities

1. **Device Interface**: Define `Obtainer` interface for devices that can obtain spectra
2. **Device Registry**: Register and discover available device implementations
3. **Device Factory**: Create obtainers from device type strings
4. **Device Implementations**: Wrap device-specific packages (e.g., `x/devices/cr30`)

## Interface

```go
// Obtainer represents a device that can obtain spectra directly
type Obtainer interface {
    // Connect establishes connection with the device
    Connect(ctx context.Context) error
    
    // Disconnect closes the connection
    Disconnect() error
    
    // Measure obtains a spectrum measurement from the device (PC-initiated).
    // This triggers a measurement programmatically and returns the result.
    Measure(ctx context.Context) (colorscience.SPD, error)

    // Start starts a background goroutine that waits for user-initiated measurements
    // (e.g., button presses) and calls the callback when measurements are available.
    // The listener runs until the context is cancelled or Stop is called.
    // Multiple measurements can be received while the listener is active.
    // For devices without button-initiated measurements, this may not be supported.
    Start(ctx context.Context, callback MeasurementCallback) error

    // Stop stops the background measurement listener.
    // This gracefully shuts down the listener goroutine.
    Stop(ctx context.Context) error

    // Wavelengths returns the wavelength vector for this device
    Wavelengths() vec.Vector
    
    // DeviceInfo returns device identification and version information
    DeviceInfo() DeviceInfo
    
    // NumWavelengths returns the number of wavelengths in the spectrum
    NumWavelengths() int
}

// DeviceInfo contains device identification and version information
type DeviceInfo struct {
    Name     string
    Model    string
    Serial   string
    Firmware string
    Build    string
}
```

## Device Registry

```go
// RegisterObtainer registers a device obtainer factory
func RegisterObtainer(deviceType string, factory ObtainerFactory)

// NewObtainer creates a new obtainer for the specified device type
func NewObtainer(ctx context.Context, deviceType string, config map[string]interface{}) (Obtainer, error)

// AvailableDevices returns list of available device types
func AvailableDevices() []string
```

## CR30 Implementation

**Device Type**: `cr30`

**Configuration**:
- `port` (string): Serial port device path (e.g., `/dev/ttyUSB0` or `COM3`)
- `baud` (int): Serial port baud rate (default: 19200)

**Implementation**:
- Wraps `x/devices/cr30` package
- Uses `cr30.New(serial)` and `cr30.Device` methods
- `Measure(ctx, dst)` uses `cr30.Device.Measure()` - PC-initiated measurement (for averaging)
  - Writes directly to caller's destination matrix (destination pattern, no allocation)
  - Destination must be 2 rows x NumWavelengths() columns
- `Start()` runs `cr30.Device.WaitMeasurement()` in a goroutine loop
  - Uses preallocated internal matrix for listener (reused for all measurements)
  - Allocates SPD only when passing to callback (acceptable - callback allocation)
- `Stop()` gracefully shuts down listener goroutine
- `Wavelengths()` allocates and returns vector (acceptable - called once)
- Provides device info from CR30 handshake
- Measurement pattern:
  - Subscribe to button-initiated measurements via `Start()`
  - For averaging: use `Measure()` for PC-initiated samples after first button measurement

**Example Usage**:
```go
config := map[string]interface{}{
    "port": "/dev/ttyUSB0",
    "baud": 19200,
}
obtainer, err := NewObtainer(ctx, "cr30", config)
if err != nil {
    return err
}
defer obtainer.Disconnect()

spd, err := obtainer.Measure(ctx)
// spd is colorscience.SPD ready for colorimetry calculations
```

## Future Device Support

**AS734x** (future):
- Device Type: `as734x`
- Configuration: I2C device path, address
- Wraps `x/devices/as734x` package
- Uses spectral reconstruction from sensor bands

## Extensibility

The framework is designed to be easily extensible:
1. Implement `Obtainer` interface for new device
2. Register device factory via `RegisterObtainer()`
3. Device is immediately available via `-device=<type>` flag

