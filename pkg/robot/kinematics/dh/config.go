package dh

import (
	"github.com/chewxy/math32"
	"github.com/foxis/EasyRobot/pkg/core/math/mat"
)

///
/// FK and IK using Denavit-Hartenberg parameter table for homogenious transform matrix calculation.
/// How to construct DH parameter table:
/// https://www.youtube.com/watch?v=D3w3ZANOy3s (one prizmatic joint)
/// https://www.youtube.com/watch?v=dASdcqgBlqw (only rotational joints)
/// https://en.wikipedia.org/wiki/Denavit%E2%80%93Hartenberg_parameters
///
/// DH table format:
///
///  n | theta | alpha | r | d | parameter index |
///  1 |
///
/// n - frame index
/// theta - rotation around Z previous axis
/// alpha - rotation around X previous axis
/// r - displacement along X axis
/// d - displacement along Z axis
/// parameter index - index of the column that joint parameter relates to (either 0 or 3 for rotational and prizmatic respectively)

type Config struct {
	Min   float32
	Max   float32
	Theta float32
	Alpha float32
	R     float32
	D     float32
	Index int
}

func (c Config) Limit(a float32) float32 {
	switch {
	case a < c.Min:
		return c.Min
	case a > c.Max:
		return c.Max
	default:
		return a
	}
}

func (c Config) CalculateTransform(parameter float32, m mat.Matrix) bool {
	parameter = c.Limit(parameter)
	alpha := c.Alpha
	theta := c.Theta
	r := c.R
	d := c.D
	switch c.Index {
	case 0:
		theta += parameter
		break
	case 1:
		alpha += parameter
		break
	case 2:
		r += parameter
		break
	case 3:
		d += parameter
		break
	default:
		return false
	}
	ct := math32.Cos(theta)
	st := math32.Sin(theta)
	ca := math32.Cos(alpha)
	sa := math32.Sin(alpha)

	m.Data[0] = ct
	m.Data[1] = -st * ca
	m.Data[2] = st * sa
	m.Data[3] = r * ct
	m.Data[4] = st
	m.Data[5] = ct * ca
	m.Data[6] = -ct * sa
	m.Data[7] = r * st
	m.Data[8] = 0
	m.Data[9] = sa
	m.Data[10] = ca
	m.Data[11] = d
	m.Data[12] = 0
	m.Data[13] = 0
	m.Data[14] = 0
	m.Data[15] = 1
	return true
}
