package dsp

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
)

func TestFFT2D_ForwardMatchesNumpy(t *testing.T) {
	fftInstance := NewFFT2D(2, 3)
	fft := &fftInstance

	src := mat.New(2, 3,
		1, 2, 3,
		4, 5, 6,
	)

	dst := mat.New(fft.rows, 2*fft.cols, make([]float32, fft.rows*2*fft.cols)...)
	fft.Forward(dst, src)

	// Reference generated via:
	// python3 -c "import numpy as np; mat=np.array([[1,2,3,0],[4,5,6,0]],dtype=np.float32); print(np.fft.fft2(mat))"
	expected := []float32{
		21, -4, 7, -4, 0, -7, 0, 7,
		-9, 0, -3, 0, 0, 3, 0, -3,
	}
	assertFloatSliceClose(t, dst.Flat(), expected, 1e-4)
}

func TestFFT2D_BackwardMatchesNumpy(t *testing.T) {
	fftInstance := NewFFT2D(2, 3)
	fft := &fftInstance

	srcFreq := []float32{
		21, -4, 7, -4, 0, -7, 0, 7,
		-9, 0, -3, 0, 0, 3, 0, -3,
	}
	src := mat.New(fft.rows, 2*fft.cols, srcFreq...)

	dst := mat.New(fft.rows, fft.cols, make([]float32, fft.rows*fft.cols)...)
	fft.Backward(dst, src)

	// Reference generated via:
	// python3 -c "import numpy as np; mat=np.array([[1,2,3,0],[4,5,6,0]],dtype=np.float32); fft=np.fft.fft2(mat); print(np.fft.ifft2(fft))"
	expected := []float32{
		1, 2, 3, 0,
		4, 5, 6, 0,
	}
	assertFloatSliceClose(t, dst.Flat(), expected, 1e-4)
}
