# Acquirer Framework - Specification

## Overview

The acquirer framework provides a unified interface for spectrometer devices that can directly acquire spectra. This framework allows the spectrometer application to support multiple device types (CR30, AS734x, etc.) through a common interface.

## Responsibilities

1. **Device Interface**: Define `Acquirer` interface for devices that can acquire spectra
2. **Device Registry**: Register and discover available device implementations
3. **Device Factory**: Create acquirers from device type strings
4. **Device Implementations**: Wrap device-specific packages (e.g., `x/devices/cr30`)

## Interface

```go
// Acquirer represents a device that can acquire spectra directly
type Acquirer interface {
    // Connect establishes connection with the device
    Connect(ctx context.Context) error
    
    // Disconnect closes the connection
    Disconnect() error
    
    // Measure acquires a spectrum measurement from the device (PC-initiated).
    // This triggers a measurement programmatically and writes to the destination matrix.
    // The destination matrix must be 2 rows x NumWavelengths() columns:
    //   - Row 0: wavelengths (nm)
    //   - Row 1: spectral power distribution values
    // Use NumWavelengths() to determine the required matrix size.
    Measure(ctx context.Context, dst matTypes.Matrix) error

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
// RegisterAcquirer registers a device acquirer factory
func RegisterAcquirer(deviceType string, factory AcquirerFactory)

// NewAcquirer creates a new acquirer for the specified device type
func NewAcquirer(ctx context.Context, deviceType string, config map[string]interface{}) (Acquirer, error)

// AvailableDevices returns list of available device types
func AvailableDevices() []string
```

## CR30 Implementation

**Device Type**: `cr30`

**Status**: ✅ **COMPLETE** - Fully implemented and integrated

**Configuration**:
- `port` (string, required): Serial port device path (e.g., `/dev/ttyUSB0` or `COM3`)
- `baud` (int, optional): Serial port baud rate (default: 19200)

**Features**:
- ✅ PC-initiated measurements via `Measure(ctx, dst)`
- ✅ Button-initiated measurements via `Start()` / `Stop()` with callbacks
- ✅ Destination pattern for zero-allocation measurements
- ✅ Preallocated buffers for efficient memory usage
- ✅ Graceful shutdown and cleanup
- ✅ Device information retrieval
- ✅ Thread-safe listener management

**Implementation Details**:
- Wraps `x/devices/cr30` package
- Uses `cr30.New(serial)` and `cr30.Device` methods
- `Measure(ctx, dst)` uses `cr30.Device.Measure()` - PC-initiated measurement
  - Writes directly to caller's destination matrix (destination pattern, no allocation)
  - Destination must be 2 rows x NumWavelengths() columns
  - Supports averaging multiple measurements
- `Start()` runs `cr30.Device.WaitMeasurement()` in a goroutine loop
  - Uses preallocated internal matrix for listener (reused for all measurements)
  - Allocates SPD only when passing to callback (acceptable - callback allocation)
  - Supports multiple button presses while listener is active
  - Callback can return error to stop listening
- `Stop()` gracefully shuts down listener goroutine
  - Cancels listener context
  - Waits for goroutine to finish (with timeout)
  - Thread-safe with mutex protection
- `Wavelengths()` allocates and returns vector (acceptable - called once)
- `DeviceInfo()` provides device identification from CR30 handshake
  - Name, Model, Serial, Firmware, Build
- `NumWavelengths()` returns 31 (CR30 fixed resolution)

**Measurement Patterns**:
1. **Single PC-initiated**: Call `Measure(ctx, dst)` once
2. **Averaged PC-initiated**: Call `Measure(ctx, dst)` multiple times, average results
3. **Button-initiated**: Use `Start()` with callback, wait for button presses
4. **Mixed**: Use `Start()` for first measurement, then `Measure()` for subsequent samples

**Integration**:
- ✅ Integrated with `measure` command
- ✅ Supports `--device=cr30` flag
- ✅ Supports `--port` and `--baud` flags
- ✅ Works with patch-based measurements
- ✅ Device-agnostic validation (via `AvailableDevices()`)

**Example Usage**:
```go
config := map[string]interface{}{
    "port": "/dev/ttyUSB0",
    "baud": 19200,
}
acquirer, err := NewAcquirer(ctx, "cr30", config)
if err != nil {
    return err
}
defer acquirer.Disconnect()

if err := acquirer.Connect(ctx); err != nil {
    return err
}

// PC-initiated measurement
numWl := acquirer.NumWavelengths()
dst := mat.New(2, numWl)
if err := acquirer.Measure(ctx, dst); err != nil {
    return err
}

// Extract wavelengths and values
wavelengths := dst.Row(0).(vec.Vector)
values := dst.Row(1).(vec.Vector)
spd := colorscience.NewSPD(wavelengths, values)
```

## Future Device Support

**AS734x** (future):
- Device Type: `as734x`
- Configuration: I2C device path, address
- Wraps `x/devices/as734x` package
- Uses spectral reconstruction from sensor bands

## Extensibility

The framework is designed to be easily extensible:
1. Implement `Acquirer` interface for new device
2. Register device factory via `RegisterAcquirer()`
3. Device is immediately available via `-device=<type>` flag

