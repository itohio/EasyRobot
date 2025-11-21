# Extract Module - Implementation Plan

## Tasks

### 1. Window Detection
- [ ] Implement variance-based window detection
- [ ] Implement Hough transform for orientation detection (gocv)
- [ ] Implement OBB creation from detected strip
- [ ] Add configurable parameters (variance threshold, smoothing)
- [ ] Test with synthetic and real images

### 2. Window Extraction
- [ ] Extract rectangular region from frame using OBB
- [ ] Handle rotation if window is rotated
- [ ] Convert to grayscale if needed
- [ ] Optimize for real-time performance

### 3. Row Averaging
- [ ] Implement mean averaging
- [ ] Implement median averaging
- [ ] Implement weighted mean (Gaussian weights)
- [ ] Implement weighted mean (linear weights)
- [ ] Support center row selection ("middle" or index)

### 4. Filtering
- [ ] **USE `savgol.SavitzkyGolay()` from `x/math/filter/savgol`** (when implemented)
- [ ] **USE `gaussian.Gaussian()` from `x/math/filter/gaussian`** (when implemented)
- [ ] **USE `median.Median()` from `x/math/filter/median`** (when implemented)
- [ ] **USE `ma.MovingAverage` from `x/math/filter/ma`** (already exists ✅)
- [ ] Create wrapper functions for spectrometer configuration
- [ ] Add parameter validation (window_size odd where applicable, valid ranges)
- [ ] **DO NOT reimplement** - wait for implementation in `x/math/filter`

**Dependencies**: `x/math/filter` must implement Savitzky-Golay, Gaussian, and Median filters first (see `cmd/spectrometer/PLAN.md` - Required Implementations)

### 5. Intensity Linearization
- [ ] Apply polynomial linearization if coefficients provided
- [ ] Support arbitrary polynomial order

### 6. Extractor Integration
- [ ] Implement Extractor interface
- [ ] Combine window extraction, averaging, linearization, filtering
- [ ] Error handling and logging

### 7. Testing
- [ ] Unit tests for window detection algorithms
- [ ] Unit tests for row averaging methods
- [ ] Unit tests for Savitzky-Golay filter
- [ ] Integration tests with real frames
- [ ] Test data generation (synthetic spectrum images)

## Implementation Order

1. Row averaging (simplest, no dependencies)
2. Filtering (Savitzky-Golay)
3. Window extraction
4. Window detection
5. Intensity linearization
6. Extractor integration
7. Testing

## Research Needed

- Industry-standard Savitzky-Golay implementations
- Best practices for spectrum window detection
- Optimal row averaging methods for spectroscopy
- gocv capabilities for Hough transform and line detection

## Dependencies

- gocv must be available
- `x/math/mat` and `x/math/vec` must be available
- Config module for window and processing settings

