package colorscience

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// AdaptationMethod specifies the chromatic adaptation transform method.
type AdaptationMethod string

const (
	// AdaptationBradford uses the Bradford transform matrix.
	AdaptationBradford AdaptationMethod = "bradford"
	// AdaptationVonKries uses the Von Kries transform matrix.
	AdaptationVonKries AdaptationMethod = "von_kries"
	// AdaptationCAT02 uses the CAT02 transform matrix.
	AdaptationCAT02 AdaptationMethod = "cat02"
)

// AdaptXYZ adapts XYZ values from source white point Ws to destination white point Wd.
func AdaptXYZ(X, Y, Z float32, Ws, Wd WhitePoint, method AdaptationMethod) (float32, float32, float32, error) {
	// Get adaptation matrix
	adaptMatrix, err := getAdaptationMatrix(method)
	if err != nil {
		return 0, 0, 0, err
	}

	// Get white point XYZ values
	Xs, Ys, Zs := Ws.XYZ()
	Xd, Yd, Zd := Wd.XYZ()

	// Convert white points to RGB using adaptation matrix
	wsVec := vec.NewFrom(Xs, Ys, Zs)
	wdVec := vec.NewFrom(Xd, Yd, Zd)
	wsRGB := adaptMatrix.MulVec(wsVec, vec.New(3)).(vec.Vector)
	wdRGB := adaptMatrix.MulVec(wdVec, vec.New(3)).(vec.Vector)

	// Calculate scaling factors
	scale := vec.New(3)
	scale[0] = wdRGB[0] / wsRGB[0]
	scale[1] = wdRGB[1] / wsRGB[1]
	scale[2] = wdRGB[2] / wsRGB[2]

	// Apply scaling in RGB space
	xyzSrc := vec.NewFrom(X, Y, Z)
	xyzSrcRGB := adaptMatrix.MulVec(xyzSrc, vec.New(3)).(vec.Vector)
	xyzSrcRGB[0] *= scale[0]
	xyzSrcRGB[1] *= scale[1]
	xyzSrcRGB[2] *= scale[2]

	// Convert back to XYZ
	invMatrix, err := getAdaptationMatrixInverse(method)
	if err != nil {
		return 0, 0, 0, err
	}

	xyzDst := invMatrix.MulVec(xyzSrcRGB, vec.New(3)).(vec.Vector)

	return xyzDst[0], xyzDst[1], xyzDst[2], nil
}

// getAdaptationMatrix returns the adaptation transform matrix for the specified method.
func getAdaptationMatrix(method AdaptationMethod) (matTypes.Matrix, error) {
	var m mat.Matrix

	switch method {
	case AdaptationBradford:
		// Bradford transform matrix
		m = mat.New(3, 3,
			0.8951, 0.2664, -0.1614,
			-0.7502, 1.7135, 0.0367,
			0.0389, -0.0685, 1.0296,
		)
	case AdaptationVonKries:
		// Von Kries transform matrix (Hunt-Pointer-Estevez)
		m = mat.New(3, 3,
			0.3897, 0.6890, -0.0787,
			-0.2298, 1.1834, 0.0464,
			0.0000, 0.0000, 1.0000,
		)
	case AdaptationCAT02:
		// CAT02 transform matrix
		m = mat.New(3, 3,
			0.7328, 0.4296, -0.1624,
			-0.7036, 1.6975, 0.0061,
			0.0030, 0.0136, 0.9834,
		)
	default:
		return nil, fmt.Errorf("unknown adaptation method: %s (use 'bradford', 'von_kries', or 'cat02')", method)
	}

	return m, nil
}

// getAdaptationMatrixInverse returns the inverse adaptation transform matrix.
func getAdaptationMatrixInverse(method AdaptationMethod) (matTypes.Matrix, error) {
	var m mat.Matrix

	switch method {
	case AdaptationBradford:
		// Inverse Bradford transform matrix
		m = mat.New(3, 3,
			0.9869929, -0.1470543, 0.1599627,
			0.4323053, 0.5183603, 0.0492912,
			-0.0085287, 0.0400428, 0.9684867,
		)
	case AdaptationVonKries:
		// Inverse Von Kries transform matrix
		m = mat.New(3, 3,
			1.910197, -1.112124, 0.201908,
			0.370950, 0.629054, 0.000000,
			0.000000, 0.000000, 1.000000,
		)
	case AdaptationCAT02:
		// Inverse CAT02 transform matrix
		m = mat.New(3, 3,
			1.096124, -0.278869, 0.182745,
			0.454369, 0.473533, 0.072098,
			-0.009628, -0.005698, 1.015326,
		)
	default:
		return nil, fmt.Errorf("unknown adaptation method: %s", method)
	}

	return m, nil
}
