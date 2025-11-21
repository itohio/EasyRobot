# Controls Module - Implementation Plan

## ⚠️ STATUS: TO BE MOVED TO `x/marshaller/gocv/controls/`

**This module should NOT be implemented in `spectrometer/internal/controls/`.**

**Target Location**: `x/marshaller/gocv/controls/`

See `MIGRATE.md` for migration details.

---

## Tasks (To be implemented in `x/marshaller/gocv/controls/`)

### 1. Camera Control Manager
- [ ] Research gocv trackbar API
- [ ] Implement exposure trackbar (critical for spectroscopy)
- [ ] Implement gain trackbar
- [ ] Implement brightness/contrast trackbars
- [ ] Implement saturation trackbar (if applicable)
- [ ] Update camera settings on trackbar change
- [ ] Handle value updates via callbacks

### 2. Processing Control Manager
- [ ] Implement Savitzky-Golay window size trackbar
- [ ] Implement Savitzky-Golay polynomial order trackbar
- [ ] Implement peak detection threshold trackbar
- [ ] Implement peak minimum distance trackbar
- [ ] Validate parameter ranges and relationships
- [ ] Update processing config in real-time

### 3. Display Control Manager
- [ ] Research gocv button API
- [ ] Implement toggle buttons for display options
- [ ] Implement action buttons (capture dark/reference)
- [ ] Manage display control state
- [ ] Handle button clicks via callbacks

### 4. Event Handling
- [ ] Implement keyboard event handling
- [ ] Implement mouse event handling
- [ ] Map keyboard shortcuts to actions
- [ ] Update cursor position tracking
- [ ] Integrate with main loop

### 5. Integration
- [ ] Integrate with camera source (update settings)
- [ ] Integrate with processing (update config)
- [ ] Integrate with display (update state)
- [ ] Error handling and logging

### 6. Testing
- [ ] Unit tests for control state management
- [ ] Unit tests for parameter validation
- [ ] Integration tests (headless mode if available)
- [ ] Manual testing with GUI

## Implementation Order

1. Camera control manager (exposure critical)
2. Event handling (keyboard/mouse)
3. Processing control manager
4. Display control manager
5. Integration
6. Testing

## Research Needed

- gocv trackbar API and best practices
- gocv button API and creation
- Event handling patterns in gocv
- Headless testing strategies

## Dependencies

- gocv must be available
- `cmd/display/source` for camera control
- Config module for processing settings
- Render module for display state

