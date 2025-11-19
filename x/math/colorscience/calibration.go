package colorscience

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// FindWavelengthIndex finds the index in a calibrated SPD vector that corresponds to
// the given wavelength (or nearest wavelength).
// Returns the index and true if exact match, false if nearest match.
func FindWavelengthIndex(wavelengths vecTypes.Vector, targetWavelength float32) (int, bool) {
	wl := wavelengths.View().(vec.Vector)
	if wl.Len() == 0 {
		return -1, false
	}

	// Binary search for exact or nearest match
	left := 0
	right := wl.Len() - 1

	for left <= right {
		mid := (left + right) / 2
		if wl[mid] == targetWavelength {
			return mid, true
		} else if wl[mid] < targetWavelength {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}

	// Find nearest
	if right < 0 {
		return 0, false
	}
	if left >= wl.Len() {
		return wl.Len() - 1, false
	}

	// Compare left and right neighbors
	if left < wl.Len() && right >= 0 {
		distLeft := math32.Abs(targetWavelength - wl[left])
		distRight := math32.Abs(wl[right] - targetWavelength)
		if distLeft < distRight {
			return left, false
		}
		return right, false
	}

	if left < wl.Len() {
		return left, false
	}
	return right, false
}
