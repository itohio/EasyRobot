# Controls Module - Specification

## Overview

The controls module provides GUI controls (trackbars, buttons) for camera settings, processing parameters, and display toggles using gocv. These controls allow real-time adjustment of spectrometer parameters during operation.

## Responsibilities

1. **Camera Controls**: Trackbars for exposure (seconds), gain, brightness, contrast
2. **Processing Controls**: Trackbars for filter parameters, peak detection thresholds
3. **Display Controls**: Toggles/buttons for CCT, XYZ/LAB, overlays, peak labels
4. **Event Handling**: Keyboard and mouse event handling for interactive controls
5. **Control State Management**: Track current values and update camera/processing accordingly

## Interfaces

```go
// CameraControlManager manages camera control trackbars
type CameraControlManager interface {
    CreateTrackbars(ctx context.Context, windowName string, source *source.Source) error
    UpdateFromTrackbars(ctx context.Context) error
    Destroy() error
}

// ProcessingControlManager manages processing control trackbars
type ProcessingControlManager interface {
    CreateTrackbars(ctx context.Context, windowName string, config *types.spectrometer.SpectrometerConfiguration) error
    UpdateFromTrackbars(ctx context.Context) (*types.spectrometer.SpectrometerConfiguration, error)
    Destroy() error
}

// DisplayControlManager manages display control toggles
type DisplayControlManager interface {
    CreateButtons(ctx context.Context, windowName string) error
    HandleEvents(ctx context.Context) error
    State() *DisplayControlState
    Destroy() error
}

// DisplayControlState represents current display control state
type DisplayControlState struct {
    ShowCCT      bool
    ShowXYZLAB   bool
    ShowOverlays bool
    ShowPeakLabels bool
    PeakHold     bool
}
```

## Camera Controls

**Trackbars**:
- **Exposure** (seconds): 0.0001 to 10 seconds (logarithmic or linear scale)
- **Gain**: 0.0 to 10.0 (linear)
- **Brightness**: -100 to 100 (camera control)
- **Contrast**: -100 to 100 (camera control)
- **Saturation**: 0 to 100 (if color camera)

**Implementation**:
- Use gocv trackbars (createTrackbar)
- Update camera settings on trackbar change
- Persist values to config (optional, via callbacks)

## Processing Controls

**Trackbars**:
- **Savitzky-Golay Window Size**: 3 to 31 (odd numbers only)
- **Savitzky-Golay Polynomial Order**: 1 to 15 (must be < window_size - 1)
- **Peak Detection Threshold**: 0 to 100 (relative percentage)
- **Peak Minimum Distance**: 1 to 100 pixels

**Implementation**:
- Validate parameter ranges
- Update processing config in real-time
- Apply changes to next frame

## Display Controls

**Toggles/Buttons**:
- **Show CCT**: Toggle CCT display overlay
- **Show XYZ/LAB**: Toggle XYZ/LAB display overlay
- **Show Overlays**: Toggle illuminant/calibration overlays
- **Show Peak Labels**: Toggle peak wavelength labels
- **Peak Hold**: Toggle peak hold mode
- **Capture Dark Frame**: Button to capture dark frame
- **Capture Reference**: Button to capture reference spectrum

**Implementation**:
- Use gocv buttons (createButton) or keyboard shortcuts
- Update display state immediately
- Trigger actions (capture dark/reference) via callbacks

## Event Handling

**Keyboard Events**:
- `q`: Quit
- `h`: Toggle peak hold
- `m`: Toggle measurement cursor
- `p`: Toggle peak labels
- `s`: Save current frame
- `d`: Capture dark frame
- `r`: Capture reference spectrum
- `c`: Toggle CCT display
- `x`: Toggle XYZ/LAB display
- `o`: Toggle overlays

**Mouse Events**:
- Mouse move: Update cursor position, display wavelength at cursor
- Click: Measure wavelength/intensity at clicked position

**Integration**:
- Handle events in main loop
- Update control state
- Trigger callbacks for actions

## Dependencies

- `gocv` - Trackbars, buttons, event handling
- `cmd/display/source` - Camera source (for camera controls)
- `types/spectrometer` - Configuration types
- `log/slog` - Structured logging

## Testing

- Unit tests for control state management
- Unit tests for parameter validation
- Integration tests with gocv windows (headless mode if available)
- Event handling tests

