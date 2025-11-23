package calibrate

// CalibrationTarget represents a set of emission lines for wavelength calibration
type CalibrationTarget struct {
	Name        string
	Description string
	Wavelengths []float32 // Emission line wavelengths in nanometers
}

// GetCalibrationTarget returns a calibration target by name
// Supported targets: "hg" (Mercury), "ar" (Argon), "ne" (Neon)
func GetCalibrationTarget(name string) *CalibrationTarget {
	switch name {
	case "hg", "mercury":
		return &CalibrationTarget{
			Name:        "Hg",
			Description: "Mercury emission lines",
			Wavelengths: []float32{405.4, 436.6, 546.1, 577.0, 579.1, 623.4},
		}
	case "ar", "argon":
		return &CalibrationTarget{
			Name:        "Ar",
			Description: "Argon emission lines",
			Wavelengths: []float32{415.9, 427.2, 451.1, 459.0, 514.5},
		}
	case "ne", "neon":
		return &CalibrationTarget{
			Name:        "Ne",
			Description: "Neon emission lines",
			Wavelengths: []float32{540.1, 585.2, 588.2, 594.5, 603.0, 616.4},
		}
	default:
		// Default to Mercury
		return GetCalibrationTarget("hg")
	}
}

// ListCalibrationTargets returns all available calibration targets
func ListCalibrationTargets() []*CalibrationTarget {
	return []*CalibrationTarget{
		GetCalibrationTarget("hg"),
		GetCalibrationTarget("ar"),
		GetCalibrationTarget("ne"),
	}
}

