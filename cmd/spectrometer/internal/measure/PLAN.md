# Measure Module - Implementation Plan

## Tasks

### 1. Single Measurement
- [ ] Implement single measurement workflow
- [ ] Integrate with extract, correction, calibrate modules
- [ ] Integrate with colorimetry module (optional)
- [ ] Error handling and logging

### 2. Averaged Measurement
- [ ] Implement frame averaging
- [ ] Support different averaging methods (mean, median)
- [ ] Integrate with single measurement workflow
- [ ] Handle frame capture timing

### 3. Transient Measurement
- [ ] Implement time-series capture
- [ ] Handle frame rate control
- [ ] Store measurements with timestamps
- [ ] Handle duration limits
- [ ] Memory management (for long captures)

### 4. Continuous Measurement
- [ ] Implement streaming workflow
- [ ] Real-time display integration
- [ ] No export (streaming only)

### 5. Export - CSV
- [ ] Implement CSV export via `x/marshaller/csv`
- [ ] Format: wavelength, intensity columns
- [ ] Support header option
- [ ] Handle multiple measurements (transient)

### 6. Export - JSON
- [ ] Implement JSON export via `x/marshaller/json`
- [ ] Export full `SpectrumMeasurement` message
- [ ] Pretty-print JSON for readability

### 7. Export - Proto
- [ ] Implement proto export via `x/marshaller/proto`
- [ ] Export binary `SpectrumMeasurement` message

### 8. Export - PNG
- [ ] Integrate with render module
- [ ] Render spectrum graph
- [ ] Save as PNG via gocv
- [ ] Handle graph rendering errors

### 9. Export - Video
- [ ] Integrate with display/destination video writer
- [ ] Record spectrum display to video file
- [ ] Handle video encoding

### 10. Exporter Integration
- [ ] Implement Exporter interface
- [ ] Auto-detect format from file extension
- [ ] Support multiple formats (export all)
- [ ] Error handling and logging

### 11. Testing
- [ ] Unit tests for each measurement mode
- [ ] Unit tests for export (each format)
- [ ] Integration tests with real frames
- [ ] Test data generation

## Implementation Order

1. Single measurement (foundation)
2. Averaged measurement (extends single)
3. Export - CSV (simplest)
4. Export - JSON (text format)
5. Export - Proto (binary format)
6. Export - PNG (image format)
7. Export - Video (complex, depends on display/destination)
8. Transient measurement (time-series)
9. Continuous measurement (streaming)
10. Exporter integration
11. Testing

## Dependencies

- All other modules must be available (extract, correction, calibrate, colorimetry, render)
- CSV marshaller must be implemented (Phase 9)
- Config module for measurement settings

