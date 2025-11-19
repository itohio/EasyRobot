package colorscience

import (
	"github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
)

// Option configures a ColorScience instance.
type Option func(*ColorScience) error

// WithObserver sets the observer type (default: Observer10Deg).
func WithObserver(observer ObserverType) Option {
	return func(cs *ColorScience) error {
		cs.observer = observer
		return nil
	}
}

// WithIlluminant sets the illuminant by name (e.g., "D65", "D50", "A").
// Default: "D65"
// Automatically sets appropriate white point based on illuminant name and observer.
func WithIlluminant(name string) Option {
	return func(cs *ColorScience) error {
		cs.illuminantName = name
		// White point will be set automatically in New() if not already set
		return nil
	}
}

// WithIlluminantSPD sets a custom illuminant SPD.
func WithIlluminantSPD(illuminant matTypes.Matrix) Option {
	return func(cs *ColorScience) error {
		// Clone the matrix to ensure we have our own copy
		spd := SPD{Matrix: illuminant}
		cloned := mat.New(2, spd.Len())
		cloned.CopyFrom(illuminant)
		cs.illuminant = SPD{Matrix: cloned}
		cs.illuminantName = "" // Clear name when using custom SPD
		return nil
	}
}

// WithWhitePoint sets the white point (default: WhitePointD65_10).
// Overrides any white point that would be auto-set by WithIlluminant.
func WithWhitePoint(wp WhitePoint) Option {
	return func(cs *ColorScience) error {
		cs.whitePoint = wp
		cs.whitePointSet = true
		return nil
	}
}

// WithDark sets the dark calibration SPD (sensor dark reading).
// This is used to subtract dark current from measurements.
func WithDark(dark matTypes.Matrix) Option {
	return func(cs *ColorScience) error {
		// Clone the matrix to ensure we have our own copy
		spd := SPD{Matrix: dark}
		cloned := mat.New(2, spd.Len())
		cloned.CopyFrom(dark)
		cs.dark = SPD{Matrix: cloned}
		return nil
	}
}

// WithLight sets the light calibration SPD (reference white/light reading).
// This is used to normalize measurements to the reference illuminant.
func WithLight(light matTypes.Matrix) Option {
	return func(cs *ColorScience) error {
		// Clone the matrix to ensure we have our own copy
		spd := SPD{Matrix: light}
		cloned := mat.New(2, spd.Len())
		cloned.CopyFrom(light)
		cs.light = SPD{Matrix: cloned}
		return nil
	}
}
