package colorscience

import "github.com/itohio/EasyRobot/x/math/vec"

// WhitePoint represents a CIE XYZ white point.
type WhitePoint vec.Vector3D

// Standard white point constants.
var (
	// D50/10 degree observer
	WhitePointD50_10 = WhitePoint{96.72, 100.000, 81.43}
	// D55/10 degree observer
	WhitePointD55_10 = WhitePoint{95.682, 100.000, 92.149}
	// D65/10 degree observer (standard for sRGB)
	WhitePointD65_10 = WhitePoint{94.81, 100.000, 107.32}
	// D75/10 degree observer
	WhitePointD75_10 = WhitePoint{94.972, 100.000, 122.638}
	// D50/2 degree observer
	WhitePointD50_2 = WhitePoint{96.422, 100.000, 82.521}
	// D65/2 degree observer
	WhitePointD65_2 = WhitePoint{95.047, 100.000, 108.883}
	// D75/2 degree observer
	WhitePointD75_2 = WhitePoint{94.972, 100.000, 122.638}
	// Illuminant A
	WhitePointA = WhitePoint{109.850, 100.000, 35.585}
	// Illuminant B
	WhitePointB = WhitePoint{99.092, 100.000, 85.313}
	// Illuminant C
	WhitePointC = WhitePoint{98.074, 100.000, 118.232}
	// Illuminant E
	WhitePointE = WhitePoint{100.000, 100.000, 100.000}
	// F1
	WhitePointF1 = WhitePoint{92.834, 100.000, 103.665}
	// F2
	WhitePointF2 = WhitePoint{99.187, 100.000, 67.395}
	// F3
	WhitePointF3 = WhitePoint{103.754, 100.000, 49.861}
	// F4
	WhitePointF4 = WhitePoint{109.147, 100.000, 38.813}
	// F5
	WhitePointF5 = WhitePoint{90.872, 100.000, 98.723}
	// F6
	WhitePointF6 = WhitePoint{97.309, 100.000, 60.188}
	// F7
	WhitePointF7 = WhitePoint{95.044, 100.000, 108.755}
	// F8
	WhitePointF8 = WhitePoint{96.413, 100.000, 82.333}
	// F9
	WhitePointF9 = WhitePoint{100.365, 100.000, 67.868}
	// F10
	WhitePointF10 = WhitePoint{96.174, 100.000, 108.882}
	// F11
	WhitePointF11 = WhitePoint{100.966, 100.000, 64.370}
	// F12
	WhitePointF12 = WhitePoint{108.046, 100.000, 39.228}
	// Common white LEDs
	WhitePointLED_CW_6500K  = WhitePoint{95.04, 100.0, 108.88}
	WhitePointLED_NW_4300K  = WhitePoint{97.0, 100.0, 92.0}
	WhitePointLED_WW_3000K  = WhitePoint{98.5, 100.0, 67.0}
	WhitePointLED_VWW_2200K = WhitePoint{103.0, 100.0, 50.0}
)

// XYZ returns the XYZ components of the white point.
func (wp WhitePoint) XYZ() (float32, float32, float32) {
	return float32(wp[0]), float32(wp[1]), float32(wp[2])
}

// getWhitePointForIlluminant returns the appropriate white point for the given illuminant name and observer.
// Returns WhitePointD65_10 as default if no match is found.
func getWhitePointForIlluminant(illuminantName string, observer ObserverType) WhitePoint {
	switch illuminantName {
	case "D50":
		if observer == Observer2Deg {
			return WhitePointD50_2
		}
		return WhitePointD50_10
	case "D55":
		if observer == Observer2Deg {
			return WhitePointD55_10 // D55_2 not defined, use 10-deg
		}
		return WhitePointD55_10
	case "D65":
		if observer == Observer2Deg {
			return WhitePointD65_2
		}
		return WhitePointD65_10
	case "D75":
		if observer == Observer2Deg {
			return WhitePointD75_2
		}
		return WhitePointD75_10
	case "A":
		return WhitePointA
	case "B":
		return WhitePointB
	case "C":
		return WhitePointC
	case "E":
		return WhitePointE
	case "F1":
		return WhitePointF1
	case "F2":
		return WhitePointF2
	case "F3":
		return WhitePointF3
	case "F4":
		return WhitePointF4
	case "F5":
		return WhitePointF5
	case "F6":
		return WhitePointF6
	case "F7":
		return WhitePointF7
	case "F8":
		return WhitePointF8
	case "F9":
		return WhitePointF9
	case "F10":
		return WhitePointF10
	case "F11":
		return WhitePointF11
	case "F12":
		return WhitePointF12
	case "LED_CW_6500K":
		return WhitePointLED_CW_6500K
	case "LED_NW_4300K":
		return WhitePointLED_NW_4300K
	case "LED_WW_3000K":
		return WhitePointLED_WW_3000K
	case "LED_VWW_2200K":
		return WhitePointLED_VWW_2200K
	default:
		// Default to D65/10 if unknown
		if observer == Observer2Deg {
			return WhitePointD65_2
		}
		return WhitePointD65_10
	}
}
