# Render Module - Implementation Plan

## Tasks

### 1. Spectrum Graph Rendering
- [ ] Implement wavelength vs. intensity plot
- [ ] Implement grid and axis labels
- [ ] Implement color-coded spectrum (wavelength to RGB)
- [ ] Implement configurable scaling (auto-scale or fixed range)
- [ ] Use `x/math/colorscience` for wavelength-to-RGB conversion

### 2. Colorimetry Overlay Rendering
- [ ] Implement XYZ value display formatting
- [ ] Implement LAB value display formatting
- [ ] Implement RGB HEX value display formatting
- [ ] Implement rotation on spacebar (XYZ → LAB → RGB HEX → repeat)
- [ ] Implement semi-transparent overlay background
- [ ] Integrate with display/destination event handling for keyboard input

### 3. Color Swatch Rendering
- [ ] Implement RGB color swatch drawing
- [ ] Implement border for visibility
- [ ] Implement configurable size and position
- [ ] Use gocv rectangle drawing

### 4. Patch Comparison Rendering
- [ ] Implement half reference, half measured patch display
- [ ] Implement patch name display
- [ ] Implement delta E (ΔE) display
- [ ] Implement colorimetry values display (with rotation support)
- [ ] Implement visual comparison layout

### 5. Delta E Calculation
- [ ] Implement CIE76 delta E calculation (already in colorscience for CRI)
- [ ] Create standalone function if needed: `DeltaE76(lab1, lab2) float32`
- [ ] Format delta E display (2 decimal places)

### 6. Integration
- [ ] Integrate with measure command
- [ ] Handle keyboard events (spacebar for rotation)
- [ ] Support display settings (width, height, fullscreen)

### 7. Testing
- [ ] Unit tests for delta E calculation
- [ ] Unit tests for colorimetry overlay formatting
- [ ] Integration tests with display window
- [ ] Visual validation tests

## Implementation Order

1. Delta E calculation (simple math, can test independently)
2. Color swatch rendering (simple drawing)
3. Spectrum graph rendering (core functionality)
4. Colorimetry overlay rendering (depends on swatch and spectrum)
5. Patch comparison rendering (depends on colorimetry and swatch)
6. Integration with measure command
7. Testing

## Dependencies

- `x/math/colorscience` - Colorimetry calculations ✅ (already implemented)
- `gocv` - Image drawing ✅ (already available)
- `cmd/display/destination` - Display window and events ✅ (already implemented)

## Notes

- Delta E calculation uses CIE76 formula (already used in colorscience.ComputeCRI())
- Wavelength-to-RGB conversion uses colorscience functions
- Keyboard event handling via display/destination event callbacks
