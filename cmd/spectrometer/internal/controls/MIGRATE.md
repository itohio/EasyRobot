# Controls Module Migration

## Status: ⚠️ TO BE MOVED

This module should **NOT** be in `spectrometer/internal/controls/`.

**Target Location**: `x/marshaller/gocv/controls/`

## Rationale

GUI controls (trackbars, buttons) are application-agnostic and should be reusable across all applications using gocv, not spectrometer-specific.

## Migration Plan

### Current Structure

```
cmd/spectrometer/internal/controls/
├── camera.go         # Camera control trackbars
├── processing.go     # Processing control trackbars
└── display.go        # Display control toggles
```

### Target Structure

```
x/marshaller/gocv/controls/
├── trackbar.go       # Generic trackbar implementation
├── button.go         # Generic button implementation
└── window.go         # Window management utilities
```

### Usage in Spectrometer

After migration, spectrometer commands will use:

```go
import "github.com/itohio/EasyRobot/x/marshaller/gocv/controls"

// Create exposure trackbar
trackbar := controls.NewTrackbar("Exposure", "WindowName", 0, 100, 
    func(value int) {
        // Update camera exposure
        camera.SetExposure(value)
    })

// Create button for dark frame capture
button := controls.NewButton("Capture Dark", "WindowName",
    func() {
        // Capture dark frame
        darkFrame := captureDarkFrame()
    })
```

## Implementation Requirements

See `x/marshaller/gocv/SPEC.md` for GUI controls implementation requirements.

**Priority**: HIGH (needed for spectrometer Phase 4)

