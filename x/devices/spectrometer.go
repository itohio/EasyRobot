package devices

import (
	"context"

	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// Spectrometer represents a spectrometer device that can measure spectral power distributions.
// The interface supports destination-based measurement where the caller provides
// the destination matrix for storing results.
//
// Matrix format:
//   - If dst has 1 row: writes only SPD values to row 0
//   - If dst has 2 rows: writes row0 = wavelengths, row1 = SPD values
type Spectrometer interface {
	// NumWavelengths returns the number of wavelength bands measured by the device.
	NumWavelengths() int

	// Wavelengths writes the wavelength values to the destination vector.
	// The wavelengths are written in nanometers (nm) in ascending order.
	// Returns the destination vector for method chaining.
	Wavelengths(dst vecTypes.Vector) vecTypes.Vector

	// Measure triggers a PC-initiated measurement and reads the data into the destination matrix.
	// The context is used for cancellation and timeout control.
	// If dst has 1 row, writes only SPD values to row 0.
	// If dst has 2 rows, writes row0 = wavelengths, row1 = SPD values.
	Measure(ctx context.Context, dst matTypes.Matrix) error

	// WaitMeasurement waits for a user-initiated measurement (e.g., button press)
	// and reads the data into the destination matrix.
	// The context is used for cancellation and timeout control.
	// If dst has 1 row, writes only SPD values to row 0.
	// If dst has 2 rows, writes row0 = wavelengths, row1 = SPD values.
	WaitMeasurement(ctx context.Context, dst matTypes.Matrix) error
}

