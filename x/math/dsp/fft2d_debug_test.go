package dsp

import (
    "testing"

    "github.com/itohio/EasyRobot/x/math/mat"
)

func TestFFT2D_BackwardDump(t *testing.T) {
    fftInstance := NewFFT2D(2, 3)
    fft := &fftInstance

    srcFreq := []float32{
        21, -4, 7, -4, 0, -7, 0, 7,
        -9, 0, -3, 0, 0, 3, 0, -3,
    }
    src := mat.New(fft.rows, 2*fft.cols, srcFreq...)

    dst := mat.New(fft.rows, fft.cols, make([]float32, fft.rows*fft.cols)...)
    fft.Backward(dst, src)

    t.Logf("dst: %v", dst.Flat())
}