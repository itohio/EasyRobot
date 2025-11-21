# Config Module - Implementation Plan

## Tasks

### 1. Core Structure
- [ ] Define `Config` struct wrapping `types.spectrometer.SpectrometerConfiguration`
- [ ] Define `Loader` interface and implementation
- [ ] Define `Saver` interface and implementation
- [ ] Define `Validator` interface and implementation
- [ ] Define `ReferenceResolver` interface and implementation

### 2. Loading Implementation
- [ ] Implement YAML loading via `x/marshaller/yaml`
- [ ] Implement JSON loading via `x/marshaller/json`
- [ ] Implement proto loading via `x/marshaller/proto`
- [ ] Auto-detect format from file extension
- [ ] Handle format errors gracefully

### 3. Saving Implementation
- [ ] Implement YAML saving via `x/marshaller/yaml`
- [ ] Implement JSON saving via `x/marshaller/json`
- [ ] Implement proto saving via `x/marshaller/proto`
- [ ] Auto-detect format from file extension
- [ ] Pretty-print YAML/JSON for readability

### 4. Validation
- [ ] Camera ID validation (delegate to camera enumeration)
- [ ] Resolution validation
- [ ] Window bounds validation
- [ ] Calibration polynomial validation
- [ ] Wavelength range validation
- [ ] Signal processing parameter validation
- [ ] Reference spectrum path validation (optional, warn only)

### 5. Reference Spectrum Resolution
- [ ] Implement path resolution (absolute/relative)
- [ ] Load CSV format reference spectra
- [ ] Load JSON format reference spectra
- [ ] Load proto format reference spectra
- [ ] Load YAML format reference spectra
- [ ] Handle inline matrices (no loading needed)
- [ ] Cache loaded spectra
- [ ] Error handling for missing files

### 6. Default Configuration
- [ ] Create default configuration factory
- [ ] Merge defaults with loaded config
- [ ] Validate final merged config

### 7. Testing
- [ ] Unit tests for Loader
- [ ] Unit tests for Saver
- [ ] Unit tests for Validator
- [ ] Unit tests for ReferenceResolver
- [ ] Integration tests with real files
- [ ] Test data files (sample configs)

## Implementation Order

1. Core structure and interfaces
2. Loading (YAML first, then JSON, proto)
3. Saving (YAML first, then JSON, proto)
4. Validation
5. Reference spectrum resolution
6. Default configuration
7. Testing

## Dependencies

- Proto definitions must be generated first (`make proto`)
- Marshallers (proto, yaml, json) must be available
- CSV marshaller needed for reference spectrum loading (can be implemented in parallel)

