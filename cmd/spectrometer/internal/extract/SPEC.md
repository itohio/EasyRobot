# Extract Module - Specification

## Overview

The extract module extracts spectrum intensity data from camera frames by detecting the spectrum window, averaging rows, and applying signal processing filters (using `x/math/filter`). The module does NOT reimplement filtering algorithms - it uses existing implementations from `x/math/filter` when available.

## Responsibilities

1. **Window Detection**: Automatically detect spectrum window (OBB) from image variance analysis (spectrometer-specific)
2. **Window Extraction**: Extract the spectrum region from frames using detected/specified window (spectrometer-specific)
3. **Row Averaging**: Average multiple rows to reduce noise (mean, median, weighted) (spectrometer-specific)
4. **Filtering**: Use filters from `x/math/filter` (`savgol.SavitzkyGolay()`, `gaussian.Gaussian()`, `median.Median()`, `ma.MovingAverage`) when implemented
5. **Intensity Extraction**: Extract intensity vector from processed window

## Interfaces

```go
// Extractor extracts spectrum intensity from frames
type Extractor interface {
    Extract(ctx context.Context, frame gocv.Mat) (vec.Vector, error)
    SetWindow(window *types.spectrometer.SpectrumWindow)
    Window() *types.spectrometer.SpectrumWindow
}

// WindowDetector automatically detects spectrum window from frames
type WindowDetector interface {
    Detect(ctx context.Context, frames []gocv.Mat) (*types.spectrometer.SpectrumWindow, error)
}

// Averager averages rows within the spectrum window
type Averager interface {
    Average(ctx context.Context, window gocv.Mat, method string, rows int, centerRow string) (vec.Vector, error)
}

// Filter wraps filter functions for spectrometer use
// Uses filter/savgol.SavitzkyGolay(), filter/gaussian.Gaussian(), filter/median.Median(), filter/ma.MovingAverage internally
type Filter interface {
    Apply(ctx context.Context, intensity vec.Vector, config *types.spectrometer.SignalProcessing) (vec.Vector, error)
}

// Note: Filters should be implemented in x/math/filter, not reimplemented here
```

## Window Detection

**Algorithm** (industry standard approach):
1. Capture N reference frames (default: 10) and average
2. Convert to grayscale if needed
3. Compute vertical variance for each column: `variance[i] = var(frame[:, i])`
4. Apply smoothing (Gaussian blur or moving average) to variance signal
5. Find horizontal strip with highest average variance
6. Detect orientation using Hough transform or line fitting (gocv)
7. Create OBB from detected strip

**Alternative Approaches** (can be configurable):
- Template matching (if reference spectrum image available)
- Edge detection + line fitting
- Intensity-based region growing

## Row Averaging

**Methods**:
- **Mean**: Simple arithmetic mean (default)
- **Median**: Median (robust to outliers)
- **Weighted Mean**: Center row weighted higher (Gaussian or linear weights)

**Implementation**:
```go
// Extract window region
window := frame.Region(windowRect)

// Average rows
intensity := averager.Average(ctx, window, "mean", 3, "middle")
```

## Filtering

**Uses filters from `x/math/filter`**:

**Savitzky-Golay Filter** (when implemented in `x/math/filter/savgol`):
```go
// Apply Savitzky-Golay filter to intensity vector
import "github.com/itohio/EasyRobot/x/math/filter/savgol"
filtered, err := savgol.SavitzkyGolay(intensity, windowSize, polynomialOrder)
// Parameters:
//   - windowSize: Must be odd (e.g., 17, 21, 31)
//   - polynomialOrder: Must be < windowSize - 1 (e.g., 7 for windowSize=17)
```

**Gaussian Filter** (when implemented in `x/math/filter/gaussian`):
```go
// Apply Gaussian smoothing
import "github.com/itohio/EasyRobot/x/math/filter/gaussian"
filtered, err := gaussian.Gaussian(intensity, sigma, windowSize)
```

**Median Filter** (when implemented in `x/math/filter/median`):
```go
// Apply median filter
import "github.com/itohio/EasyRobot/x/math/filter/median"
filtered, err := median.Median(intensity, windowSize)
```

**Moving Average** (already exists in `x/math/filter/ma` ✅):
```go
// Apply moving average
import "github.com/itohio/EasyRobot/x/math/filter/ma"
maFilter := ma.New(windowSize)
filtered := maFilter.ProcessBuffer(intensity, output)
```

**Implementation**: Filter implementations should provide:
- Industry-standard algorithms
- Smoothing while preserving peak shape (for Savitzky-Golay)
- Edge case handling (beginning/end of signal)
- Streaming support via `Process()` and `ProcessBuffer()` methods

**Spectrometer-Specific**: The extract module wraps these functions for spectrometer configuration parameters (from `types.spectrometer.SignalProcessing` config).

**Note**: Filtering is a general signal processing operation, not spectrometer-specific. Filters belong in `x/math/filter` package, not reimplemented in spectrometer.

**Status**: 
- ✅ Moving Average - Already implemented
- ⚠️ Savitzky-Golay, Gaussian, Median - **NOT YET IMPLEMENTED** - Must be added to `x/math/filter` before spectrometer extract module implementation.

**Alternative Filters** (future):
- Gaussian smoothing
- Median filter
- Moving average

## Intensity Linearization

If camera linearization coefficients provided in config:
```go
intensity_linear = c0 + c1*intensity + c2*intensity² + c3*intensity³ + ...
```

## Implementation Details

### Extraction Pipeline

```
Frame (gocv.Mat)
  ↓
Window Extraction (using OBB)
  ↓
Row Averaging (mean/median/weighted)
  ↓
Intensity Linearization (if configured)
  ↓
Filtering (Savitzky-Golay if enabled)
  ↓
Intensity Vector (vec.Vector)
```

## Dependencies

- `gocv` - Image processing
- `x/math/vec` - Vector operations
- `x/math/mat` - Matrix operations
- `types/spectrometer` - Configuration types
- `log/slog` - Structured logging

## Testing

- Unit tests for window detection (synthetic test images)
- Unit tests for row averaging
- Unit tests for filtering
- Integration tests with real camera frames
- Test data generation (synthetic spectrum images)

