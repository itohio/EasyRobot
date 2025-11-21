package render

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/math"
	"github.com/itohio/EasyRobot/x/math/colorscience"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// ColorimetryDisplayMode specifies what colorimetry values to display
type ColorimetryDisplayMode int

const (
	// ColorimetryDisplayXYZ displays XYZ values
	ColorimetryDisplayXYZ ColorimetryDisplayMode = iota
	// ColorimetryDisplayLAB displays LAB values
	ColorimetryDisplayLAB
	// ColorimetryDisplayRGBHEX displays RGB HEX values
	ColorimetryDisplayRGBHEX
)

// FormatXYZ formats XYZ values as a string
func FormatXYZ(xyz colorscience.XYZ) string {
	X, Y, Z := vec.Vector3D(xyz).XYZ()
	return fmt.Sprintf("XYZ: %.2f %.2f %.2f", X, Y, Z)
}

// FormatLAB formats LAB values as a string
func FormatLAB(lab colorscience.LAB) string {
	L, a, b := vec.Vector3D(lab).XYZ()
	return fmt.Sprintf("LAB: %.2f %.2f %.2f", L, a, b)
}

// FormatRGBHEX formats RGB values as HEX string
func FormatRGBHEX(rgb colorscience.RGB) string {
	R, G, B := vec.Vector3D(rgb).XYZ()
	// Convert to 0-255 range and clamp
	R = math.Clamp(R, 0, 255)
	G = math.Clamp(G, 0, 255)
	B = math.Clamp(B, 0, 255)
	return fmt.Sprintf("#%02X%02X%02X", uint8(R), uint8(G), uint8(B))
}

// FormatColorimetry formats colorimetry values based on display mode
func FormatColorimetry(xyz colorscience.XYZ, lab colorscience.LAB, rgb colorscience.RGB, mode ColorimetryDisplayMode) string {
	switch mode {
	case ColorimetryDisplayXYZ:
		return FormatXYZ(xyz)
	case ColorimetryDisplayLAB:
		return FormatLAB(lab)
	case ColorimetryDisplayRGBHEX:
		return FormatRGBHEX(rgb)
	default:
		return FormatLAB(lab)
	}
}
