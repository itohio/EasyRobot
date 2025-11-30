// Package dsp provides digital signal processing functions for EasyRobot.
// It includes FFT transforms, convolution, windowing, signal generation,
// and measurement functions optimized for fp32 operations.
package dsp

import "github.com/itohio/EasyRobot/x/types"

type Sample interface {
	types.Number | types.Unsigned
}
