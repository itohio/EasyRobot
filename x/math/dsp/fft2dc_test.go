package dsp

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/shape"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
)

func TestFFT2DC_ForwardBackwardMatchesNumpy(t *testing.T) {
	fftInstance := NewFFT2DC(2, 2)
	fft := &fftInstance

	src := tensor.New(types.C64, shape.NewShape(2, 2))
	srcData := types.GetTensorData[[]complex64](src)
	copy(srcData, []complex64{
		1 + 0i, 0.5 - 0.5i,
		-1 + 2i, 0 + 0i,
	})

	dst := tensor.New(types.C64, shape.NewShape(fft.rows, fft.cols))
	fft.Forward(dst, src)

	// Reference generated via:
	// python3 -c "import numpy as np; mat=np.array([[1+0j,0.5-0.5j],[-1+2j,0+0j]],dtype=np.complex64); print(np.fft.fft2(mat))"
	expectedFreq := []complex64{
		0.5 + 1.5i, -0.5 + 2.5i,
		2.5 - 2.5i, 1.5 - 1.5i,
	}
	assertComplexSliceClose(t, types.GetTensorData[[]complex64](dst), expectedFreq, 1e-4)

	reconstructed := tensor.New(types.C64, shape.NewShape(fft.rows, fft.cols))
	fft.Backward(reconstructed, dst)

	// Reference generated via:
	// python3 -c "import numpy as np; mat=np.array([[1+0j,0.5-0.5j],[-1+2j,0+0j]],dtype=np.complex64); fft=np.fft.fft2(mat); print(np.fft.ifft2(fft))"
	expectedTime := []complex64{
		1 + 0i, 0.5 - 0.5i,
		-1 + 2i, 0 + 0i,
	}
	assertComplexSliceClose(t, types.GetTensorData[[]complex64](reconstructed), expectedTime, 1e-4)
}
