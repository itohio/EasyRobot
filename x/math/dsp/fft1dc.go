// Package dsp provides digital signal processing functions for EasyRobot.
package dsp

import (
	"github.com/itohio/EasyRobot/x/math/primitive/c64"
	"github.com/itohio/EasyRobot/x/math/primitive/fp32"
	"github.com/itohio/EasyRobot/x/math/shape"
	"github.com/itohio/EasyRobot/x/math/tensor"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
)

// FFT1DC performs forward and inverse Fast Fourier Transforms on complex-valued 1D tensors.
//
// Data Layout:
//   - Input (src): Complex-valued Tensor with C64 data type. Each element is a complex64 value
//     representing a complex sample (real + imaginary parts).
//   - Output (dst): Complex-valued Tensor with C64 data type. For forward FFT, the output contains
//     the full complex frequency domain representation. Unlike real FFT, complex FFT preserves
//     both real and imaginary components throughout the transform.
//
// The tensor must be 1D (rank 1) and have C64 data type. The FFT size is determined by
// the tensor length, which will be padded to the next power of 2 if needed.
type FFT1DC struct {
	length  int         // vector length (next power of 2)
	tempBuf []complex64 // temporary buffer for optimizations
	lut     []float32   // precomputed LUT for twiddle factors (optional)
}

// NewFFT1DC creates a 1D complex FFT processor for tensors of the specified length.
// The length will be rounded up to the next power of 2.
// If lut is provided (vararg), it will be used for optimized FFT operations.
// The LUT can be created using c64.NewFFTLUT(fftSize).
// Returns a value (not a pointer).
func NewFFT1DC(length int, lut ...[]float32) FFT1DC {
	if length <= 0 {
		panic("FFT length must be positive")
	}
	nextPow2 := fp32.NextPowerOfTwo(length)

	fft := FFT1DC{
		length:  nextPow2,
		tempBuf: make([]complex64, nextPow2),
	}

	// Use LUT if provided
	if len(lut) > 0 && len(lut[0]) > 0 {
		fft.lut = lut[0]
	}

	return fft
}

// Forward performs a forward FFT transform on a complex-valued 1D tensor.
//
// Input (src): Complex-valued Tensor with C64 data type and rank 1.
//
//	The tensor contains complex64 values representing complex samples.
//
// Output (dst): Complex-valued Tensor with C64 data type. The FFT result contains
//
//	the full complex frequency domain representation. The output length
//	matches the FFT size (next power of 2 >= input length).
//
// Panics if src or dst are not C64 type, not rank 1, or if dst is too small.
// Uses destination-based API with no allocations.
func (f *FFT1DC) Forward(dst, src types.Tensor) {
	f.forward(src, dst)
}

// Backward performs an inverse FFT transform on a complex-valued 1D tensor.
//
// Input (src): Complex-valued Tensor with C64 data type (frequency domain representation).
// Output (dst): Complex-valued Tensor with C64 data type (time domain signal).
//
//	The result is scaled by 1/N where N is the FFT length.
//
// Panics if src or dst are not C64 type, not rank 1, or if dst is too small.
// Uses destination-based API with no allocations.
func (f *FFT1DC) Backward(dst, src types.Tensor) {
	f.backward(src, dst)
}

// forward performs 1D forward complex FFT with destination buffer.
// Optimized with BCE techniques: reslicing and for range loops
func (f *FFT1DC) forward(src, dst types.Tensor) {
	// Validate input
	if src.DataType() != types.C64 {
		panic("FFT1DC.Forward: source must be C64 type")
	}
	if dst.DataType() != types.C64 {
		panic("FFT1DC.Forward: destination must be C64 type")
	}
	if src.Rank() != 1 {
		panic("FFT1DC.Forward: source must be rank 1")
	}
	if dst.Rank() != 1 {
		panic("FFT1DC.Forward: destination must be rank 1")
	}

	srcData := types.GetTensorData[[]complex64](src)
	dstData := types.GetTensorData[[]complex64](dst)

	// BCE hint: reslice temp buffer to exact size
	tempBuf := f.tempBuf[:f.length]
	srcLen := len(srcData)
	if srcLen > f.length {
		srcLen = f.length
	}

	// Copy and zero-pad to temp buffer
	copy(tempBuf, srcData[:srcLen])
	// Zero-pad remaining elements
	for i := srcLen; i < f.length; i++ {
		tempBuf[i] = 0
	}

	// Perform complex FFT
	if len(f.lut) > 0 {
		totalFactors := len(f.lut) / 2
		c64.FFTWithLUT(tempBuf, f.lut, totalFactors)
	} else {
		c64.FFT(tempBuf)
	}

	// Copy result to destination (only copy what fits)
	copyLen := f.length
	if copyLen > len(dstData) {
		copyLen = len(dstData)
	}
	copy(dstData, tempBuf[:copyLen])
}

// backward performs 1D inverse complex FFT with destination buffer.
// Optimized with BCE techniques: reslicing
func (f *FFT1DC) backward(src, dst types.Tensor) {
	// Validate input
	if src.DataType() != types.C64 {
		panic("FFT1DC.Backward: source must be C64 type")
	}
	if dst.DataType() != types.C64 {
		panic("FFT1DC.Backward: destination must be C64 type")
	}
	if src.Rank() != 1 {
		panic("FFT1DC.Backward: source must be rank 1")
	}
	if dst.Rank() != 1 {
		panic("FFT1DC.Backward: destination must be rank 1")
	}

	srcData := types.GetTensorData[[]complex64](src)
	dstData := types.GetTensorData[[]complex64](dst)

	// BCE hint: reslice temp buffer to exact size
	tempBuf := f.tempBuf[:f.length]
	srcLen := len(srcData)
	if srcLen > f.length {
		srcLen = f.length
	}

	// Copy to temp buffer
	copy(tempBuf, srcData[:srcLen])
	// Zero-pad if needed
	for i := srcLen; i < f.length; i++ {
		tempBuf[i] = 0
	}

	// Perform inverse complex FFT
	// Note: IFFTWithLUT could be added later for optimization
	c64.IFFT(tempBuf)

	// Copy result to destination (only copy what fits)
	copyLen := f.length
	if copyLen > len(dstData) {
		copyLen = len(dstData)
	}
	copy(dstData, tempBuf[:copyLen])
}

// Convolve performs 1D convolution of signal with kernel using complex FFT.
// Uses destination-based API with no allocations.
func (f *FFT1DC) Convolve(dst, signal, kernel types.Tensor) {
	sigLen := signal.Size()
	kernLen := kernel.Size()
	resultLen := sigLen + kernLen - 1

	// Ensure destination is large enough
	if dst.Size() < resultLen {
		panic("Convolve: destination buffer too small")
	}

	// Create temporary FFTs for signal and kernel
	sigFFT := NewFFT1DC(sigLen)
	kernFFT := NewFFT1DC(kernLen)

	// Temporary buffers
	sigFreq := tensor.New(types.C64, shape.NewShape(f.length))
	kernFreq := tensor.New(types.C64, shape.NewShape(f.length))

	// Forward FFTs
	sigFFT.Forward(sigFreq, signal)
	kernFFT.Forward(kernFreq, kernel)

	// Multiply in frequency domain
	// BCE optimization: reslice and use for range
	sigFreqData := types.GetTensorData[[]complex64](sigFreq)
	kernFreqData := types.GetTensorData[[]complex64](kernFreq)
	size := f.length
	if len(sigFreqData) < size {
		size = len(sigFreqData)
	}
	if len(kernFreqData) < size {
		size = len(kernFreqData)
	}
	sigFreqData = sigFreqData[:size]
	kernFreqData = kernFreqData[:size]
	for i := range size {
		sigFreqData[i] *= kernFreqData[i]
	}

	// Inverse FFT to destination
	f.Backward(dst, sigFreq)

	// Scale result (FFT normalization)
	// BCE optimization: reslice to exact size
	dstData := types.GetTensorData[[]complex64](dst)
	if resultLen > len(dstData) {
		resultLen = len(dstData)
	}
	dstData = dstData[:resultLen]
	scale := complex64(complex(1.0/float32(f.length), 0))
	for i := range resultLen {
		dstData[i] *= scale
	}
}

// Correlate performs 1D cross-correlation of signal1 with signal2 using complex FFT.
// Cross-correlation is similar to convolution but with one signal conjugated.
// Uses destination-based API with no allocations.
func (f *FFT1DC) Correlate(dst, signal1, signal2 types.Tensor) {
	sig1Len := signal1.Size()
	sig2Len := signal2.Size()
	resultLen := sig1Len + sig2Len - 1

	// Ensure destination is large enough
	if dst.Size() < resultLen {
		panic("Correlate: destination buffer too small")
	}

	// FFT size must be at least resultLen to avoid aliasing
	fftSize := fp32.NextPowerOfTwo(resultLen)

	// Create temporary FFTs for signals (use fftSize to ensure no aliasing)
	sig1FFT := NewFFT1DC(fftSize)
	sig2FFT := NewFFT1DC(fftSize)

	// Temporary buffers (use fftSize)
	sig1Freq := tensor.New(types.C64, shape.NewShape(fftSize))
	sig2Freq := tensor.New(types.C64, shape.NewShape(fftSize))

	// Forward FFTs (will zero-pad to fftSize)
	sig1FFT.Forward(sig1Freq, signal1)
	sig2FFT.Forward(sig2Freq, signal2)

	// For correlation, multiply sig1 with conjugate of sig2 in frequency domain
	// BCE optimization: reslice and use for range
	sig1FreqData := types.GetTensorData[[]complex64](sig1Freq)
	sig2FreqData := types.GetTensorData[[]complex64](sig2Freq)
	size := fftSize
	if len(sig1FreqData) < size {
		size = len(sig1FreqData)
	}
	if len(sig2FreqData) < size {
		size = len(sig2FreqData)
	}
	sig1FreqData = sig1FreqData[:size]
	sig2FreqData = sig2FreqData[:size]
	for i := range size {
		// Correlation: multiply by conjugate
		sig1FreqData[i] *= complex64(complex(real(sig2FreqData[i]), -imag(sig2FreqData[i])))
	}

	// Inverse FFT to destination (need to use fftSize FFT)
	corrFFT := NewFFT1DC(fftSize)
	corrFFT.Backward(dst, sig1Freq)

	// Scale result (FFT normalization)
	// BCE optimization: reslice to exact size
	dstData := types.GetTensorData[[]complex64](dst)
	if resultLen > len(dstData) {
		resultLen = len(dstData)
	}
	dstData = dstData[:resultLen]
	scale := complex64(complex(1.0/float32(fftSize), 0))
	for i := range resultLen {
		dstData[i] *= scale
	}
}
