# Acquirer Framework - Implementation Plan

## Tasks

### 1. Acquirer Interface
- [x] Define `Acquirer` interface (Connect, Disconnect, Measure, Start, Stop, Wavelengths, DeviceInfo, NumWavelengths) ✅
- [x] Define `DeviceInfo` struct ✅
- [x] Define `MeasurementCallback` type ✅
- [x] Document interface contract ✅

### 2. Device Registry
- [x] Implement device registry (map[string]AcquirerFactory) ✅
- [x] Implement `RegisterAcquirer()` function ✅
- [x] Implement `NewAcquirer()` factory function ✅
- [x] Implement `AvailableDevices()` function ✅
- [x] Thread-safe registry access ✅

### 3. CR30 Acquirer Implementation
- [x] Implement `CR30Acquirer` struct wrapping `x/devices/cr30` ✅
- [x] Implement `Connect()` (uses cr30.Connect()) ✅
- [x] Implement `Disconnect()` (uses cr30.Disconnect()) ✅
- [x] Implement `Measure(ctx, dst)` (uses cr30.Device.Measure()) ✅
  - Uses destination pattern (writes to caller's matrix, no allocation)
  - Destination must be 2 rows x NumWavelengths() columns
- [x] Implement `Start()` (runs cr30.Device.WaitMeasurement() in goroutine loop) ✅
  - Uses preallocated internal matrix for listener
  - Allocates SPD only when passing to callback
- [x] Implement `Stop()` (gracefully shuts down listener goroutine) ✅
- [x] Implement `Wavelengths()` (uses cr30.Device.Wavelengths()) ✅
- [x] Implement `DeviceInfo()` (uses cr30.DeviceInfo()) ✅
- [x] Implement `NumWavelengths()` (uses cr30.Device.NumWavelengths()) ✅
- [x] Register CR30 acquirer in registry ✅

### 4. Factory Function
- [x] Implement factory function for CR30 (reads port/baud from config) ✅
- [x] Error handling for missing/invalid configuration ✅
- [x] Device-specific validation ✅

### 5. Integration
- [x] Integrate with `measure` command ✅
- [x] Support `-device` flag ✅
- [x] Device-specific flag handling (--port, --baud for CR30) ✅
- [x] Device-agnostic validation (checks AvailableDevices() instead of hardcoding CR30) ✅
- [x] Patch measurement works with any acquirer (via interface) ✅

### 6. Testing
- [ ] Unit tests for Acquirer interface
- [ ] Unit tests for device registry
- [ ] Integration tests with CR30 device (mock or real)
- [ ] Error handling tests

## Implementation Order

1. ✅ Acquirer interface definition
2. ✅ Device registry (simple map-based implementation)
3. ✅ CR30 acquirer implementation (wraps existing cr30 package)
4. ✅ Factory function and registration
5. ✅ Integration with measure command
6. [ ] Testing

## Dependencies

- `x/devices/cr30` - CR30 device package ✅ (already implemented)
- `x/math/colorscience` - SPD type ✅ (already implemented)
- `x/math/vec` - Vector operations ✅ (already implemented)
- `x/math/mat` - Matrix operations ✅ (already implemented)

## Completed Features Summary

### CR30 Acquirer ✅
- ✅ Full interface implementation
- ✅ PC-initiated measurements with destination pattern
- ✅ Button-initiated measurements with callback pattern
- ✅ Thread-safe listener management
- ✅ Graceful shutdown and cleanup
- ✅ Device information retrieval
- ✅ Factory pattern with configuration
- ✅ Integration with measure command
- ✅ Patch measurement support
- ✅ Device-agnostic validation

## Future Enhancements

- Camera-based acquirer implementation (DIY spectrometer with DVD/prism grating)
- AS734x acquirer implementation (when needed)
- Device capability querying (e.g., supports calibration, supports multiple modes)
- Device-specific configuration validation
- Device discovery (automatic device detection)

