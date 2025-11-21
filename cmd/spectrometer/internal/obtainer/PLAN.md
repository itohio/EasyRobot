# Obtainer Framework - Implementation Plan

## Tasks

### 1. Obtainer Interface
- [ ] Define `Obtainer` interface (Connect, Disconnect, Measure, Wavelengths, DeviceInfo, NumWavelengths)
- [ ] Define `DeviceInfo` struct
- [ ] Document interface contract

### 2. Device Registry
- [ ] Implement device registry (map[string]ObtainerFactory)
- [ ] Implement `RegisterObtainer()` function
- [ ] Implement `NewObtainer()` factory function
- [ ] Implement `AvailableDevices()` function
- [ ] Thread-safe registry access

### 3. CR30 Obtainer Implementation
- [ ] Implement `CR30Obtainer` struct wrapping `x/devices/cr30`
- [ ] Implement `Connect()` (uses cr30.Connect())
- [ ] Implement `Disconnect()` (uses cr30.Disconnect())
- [ ] Implement `Measure()` (uses cr30.WaitMeasurement() or cr30.Measure())
  - Convert CR30 measurement format to colorscience.SPD
- [ ] Implement `Wavelengths()` (uses cr30.Wavelengths())
- [ ] Implement `DeviceInfo()` (uses cr30.DeviceInfo())
- [ ] Implement `NumWavelengths()` (uses cr30.NumWavelengths())
- [ ] Register CR30 obtainer in registry

### 4. Factory Function
- [ ] Implement factory function for CR30 (reads port/baud from config)
- [ ] Error handling for missing/invalid configuration
- [ ] Device-specific validation

### 5. Integration
- [ ] Integrate with `measure` command
- [ ] Support `-device` flag
- [ ] Device-specific flag handling (--port, --baud for CR30)

### 6. Testing
- [ ] Unit tests for Obtainer interface
- [ ] Unit tests for device registry
- [ ] Integration tests with CR30 device (mock or real)
- [ ] Error handling tests

## Implementation Order

1. Obtainer interface definition
2. Device registry (simple map-based implementation)
3. CR30 obtainer implementation (wraps existing cr30 package)
4. Factory function and registration
5. Integration with measure command
6. Testing

## Dependencies

- `x/devices/cr30` - CR30 device package ✅ (already implemented)
- `x/math/colorscience` - SPD type ✅ (already implemented)
- `x/math/vec` - Vector operations ✅ (already implemented)

## Future Enhancements

- AS734x obtainer implementation (when needed)
- Device capability querying (e.g., supports calibration, supports multiple modes)
- Device-specific configuration validation
- Device discovery (automatic device detection)

