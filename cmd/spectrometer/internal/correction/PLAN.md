# Correction Module - Implementation Plan

## Tasks

### 1. Dark Frame Correction
- [ ] Implement dark frame capture
- [ ] Implement dark frame subtraction
- [ ] Implement dark frame loading/saving
- [ ] Handle edge cases (negative values, zero dark)
- [ ] Average multiple frames for noise reduction

### 2. Reference Spectrum Correction
- [ ] Implement reference spectrum capture
- [ ] Implement subtract mode
- [ ] Implement normalize mode
- [ ] Implement ratio mode
- [ ] Implement reference spectrum loading/saving
- [ ] Handle division by zero

### 3. Spectrum Masking
- [ ] Implement wavelength region filtering
- [ ] Support include/exclude regions
- [ ] Handle multiple regions
- [ ] Apply masking to SPD

### 4. Normalization
- [ ] Implement normalize to reference
- [ ] Implement normalize to illuminant
- [ ] Load standard illuminant SPDs (from colorscience)
- [ ] Handle edge cases (zero reference)

### 5. Linearization
- [ ] Implement polynomial linearization
- [ ] Support arbitrary polynomial order
- [ ] Validate coefficients
- [ ] Apply to intensity vectors

### 6. Integration
- [ ] Integrate all correction steps
- [ ] Apply corrections in correct order
- [ ] Error handling and logging

### 7. Testing
- [ ] Unit tests for each correction method
- [ ] Integration tests with real spectra
- [ ] Test data generation (known spectra)

## Implementation Order

1. Linearization (simplest, applies to vectors)
2. Dark frame correction (independent)
3. Reference spectrum correction (similar to dark frame)
4. Masking (applies to SPD)
5. Normalization (uses illuminant SPDs)
6. Integration
7. Testing

## Dependencies

- `x/math/colorscience` must be available
- Extract module for spectrum extraction (for capture)
- Config module for correction settings
- Marshallers for loading/saving spectra

