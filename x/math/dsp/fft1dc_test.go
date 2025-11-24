package dsp

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/shape"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
)

func TestFFT1DC_ForwardBackward(t *testing.T) {
	fft := NewFFT1DC(4)
	fftPtr := &fft

	src := newComplexTensor([]complex64{
		1 + 0i,
		-1 + 2i,
		0.5 - 0.25i,
		0 + 0i,
	})
	defer src.Release()

	dst := tensor.New(types.C64, shape.NewShape(fftPtr.length))
	defer dst.Release()

	fftPtr.Forward(dst, src)

	// Reference generated via:
	// python3 -c "import numpy as np; signal=np.array([1+0j,-1+2j,0.5-0.25j,0],dtype=np.complex64); print(np.fft.fft(signal))"
	expectedFreq := []complex64{
		0.5 + 1.75i,
		2.5 + 1.25i,
		2.5 - 2.25i,
		-1.5 - 0.75i,
	}
	assertComplexSliceClose(t, types.GetTensorData[[]complex64](dst), expectedFreq, 1e-4)

	reconstructed := tensor.New(types.C64, shape.NewShape(fftPtr.length))
	defer reconstructed.Release()

	fftPtr.Backward(reconstructed, dst)
	// Reference generated via:
	// python3 -c "import numpy as np; vals=[1+0j,-1+2j,0.5-0.25j,0]; fft=np.fft.fft(vals); print(np.fft.ifft(fft))"
	expectedTime := []complex64{
		1 + 0i,
		-1 + 2i,
		0.5 - 0.25i,
		0 + 0i,
	}
	assertComplexSliceClose(t, types.GetTensorData[[]complex64](reconstructed), expectedTime, 1e-4)
}

func TestFFT1DC_ConvolveMatchesNaive(t *testing.T) {
	signal := newComplexTensor([]complex64{
		1 + 0i,
		2 - 1i,
		-0.5 + 0.5i,
	})
	defer signal.Release()

	kernel := newComplexTensor([]complex64{
		0.5 + 0.5i,
		-1 + 0i,
	})
	defer kernel.Release()

	resultLen := signal.Size() + kernel.Size() - 1
	dst := tensor.New(types.C64, shape.NewShape(resultLen))
	defer dst.Release()

	fft := NewFFT1DC(resultLen)
	fftPtr := &fft
	fftPtr.Convolve(dst, signal, kernel)

	// Reference generated via:
	// python3 -c "import numpy as np; s=np.array([1+0j,2-1j,-0.5+0.5j],dtype=np.complex64); k=np.array([0.5+0.5j,-1+0j],dtype=np.complex64); pad_s=np.append(s,[0]); pad_k=np.append(k,[0,0]); print(np.fft.ifft(np.fft.fft(pad_s)*np.fft.fft(pad_k)))"
	expected := []complex64{
		0.5 + 0.5i,
		0.5 + 0.5i,
		-2.5 + 1i,
		0.5 - 0.5i,
	}
	assertComplexSliceClose(t, types.GetTensorData[[]complex64](dst), expected, 1e-3)
}

func TestFFT1DC_CorrelateMatchesNaive(t *testing.T) {
	sig1 := newComplexTensor([]complex64{
		1 + 0i,
		-0.5 + 0.5i,
		2 - 1i,
	})
	defer sig1.Release()

	sig2 := newComplexTensor([]complex64{
		0.5 + 0i,
		-1 + 2i,
	})
	defer sig2.Release()

	resultLen := sig1.Size() + sig2.Size() - 1
	dst := tensor.New(types.C64, shape.NewShape(resultLen))
	defer dst.Release()

	fft := NewFFT1DC(resultLen)
	fft.Correlate(dst, sig1, sig2)

	// Reference generated via:
	// python3 -c "import numpy as np; a=np.array([1+0j,-0.5+0.5j,2-1j],dtype=np.complex64); b=np.array([0.5+0j,-1+2j],dtype=np.complex64); L=len(a)+len(b)-1; N=1; while N<L: N*=2; pa=np.zeros(N,dtype=np.complex64); pb=np.zeros(N,dtype=np.complex64); pa[:len(a)]=a; pb[:len(b)]=b; res=np.fft.ifft(np.fft.fft(pa)*np.conjugate(np.fft.fft(pb))); print(res[:L])"
	expected := []complex64{
		2 + 0.5i,
		-4.25 - 2.75i,
		1 - 0.5i,
		-1 - 2i,
	}
	assertComplexSliceClose(t, types.GetTensorData[[]complex64](dst), expected, 1e-3)
}

func newComplexTensor(values []complex64) types.Tensor {
	t := tensor.New(types.C64, shape.NewShape(len(values)))
	data := types.GetTensorData[[]complex64](t)
	copy(data, values)
	return t
}
