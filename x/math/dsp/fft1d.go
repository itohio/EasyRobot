// Package dsp provides digital signal processing functions for EasyRobot.
package dsp

import (
	"github.com/itohio/EasyRobot/x/math/primitive/fp32"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// FFT1D performs forward and inverse Fast Fourier Transforms on real-valued vectors.
//
// Data Layout:
//   - Input (src): Real-valued float32 vector. Each element represents a real sample.
//   - Output (dst): float32 vector of length 2*N where N is the FFT size.
//     The output stores complex values in packed format:
//   - Real parts: dst[0:N]
//   - Imaginary parts: dst[N:2*N]
//     For a real input signal, the frequency domain has Hermitian symmetry.
//
// Note: The destination buffer must be at least 2*N in length to store both real and imaginary parts.
// For complex input/output, use FFT1DC.
type FFT1D struct {
	length  int       // FFT size (number of complex samples)
	tempBuf []float32 // temporary buffer for optimizations (size 2*length for packed format)
	lut     []float32 // precomputed LUT for twiddle factors (optional)
}

// NewFFT1D creates a 1D FFT processor for real-valued vectors of the specified length.
// The FFT size will be rounded up to the next power of 2.
// If lut is provided (vararg), it will be used for optimized FFT operations.
// The LUT can be created using fp32.NewFFTLUT(fftSize).
// Returns a value (not a pointer).
func NewFFT1D(length int, lut ...[]float32) FFT1D {
	if length <= 0 {
		panic("FFT length must be positive")
	}
	nextPow2 := fp32.NextPowerOfTwo(length)
	tempBuf := make([]float32, 2*nextPow2)

	fft := FFT1D{
		length:  nextPow2,
		tempBuf: tempBuf, // 2*N for packed format (real + imag)
	}

	// Use LUT if provided
	if len(lut) > 0 && len(lut[0]) > 0 {
		fft.lut = lut[0]
	}

	return fft
}

// Forward performs a forward FFT transform on a real-valued vector.
//
// Input (src): Real-valued float32 vector of length N.
// Output (dst): float32 vector of length 2*N where N is the FFT size.
//
//	The output stores complex values in packed format:
//	- Real parts: dst[0:N]
//	- Imaginary parts: dst[N:2*N]
//	For a real input signal, the frequency domain has Hermitian symmetry.
//
// The destination buffer must be at least 2*N in length.
// Uses destination-based API with no allocations.
func (f *FFT1D) Forward(dst, src vec.Vector) {
	f.forward(src, dst)
}

// Backward performs an inverse FFT transform on packed complex frequency domain data.
//
// Input (src): float32 vector of length 2*N in packed format:
//   - Real parts: src[0:N]
//   - Imaginary parts: src[N:2*N]
//
// Output (dst): Real-valued float32 vector (time domain signal) of length N.
// Uses destination-based API with no allocations.
func (f *FFT1D) Backward(dst, src vec.Vector) {
	f.backward(src, dst)
}

// forward performs 1D forward FFT with destination buffer.
// Optimized with BCE techniques: reslicing and for range loops
func (f *FFT1D) forward(src, dst vec.Vector) {
	srcData := []float32(src)
	dstData := []float32(dst)

	// Ensure destination is large enough for packed format (2*N)
	if len(dstData) < 2*f.length {
		panic("FFT1D.Forward: destination buffer too small, need 2*N for packed format")
	}

	// BCE hint: reslice temp buffer to exact size (2*N for packed format)
	tempBuf := f.tempBuf[:2*f.length]
	srcLen := len(srcData)
	if srcLen > f.length {
		srcLen = f.length
	}

	// Copy real input to real part of packed format, zero imaginary parts
	// Real parts: tempBuf[0:N]
	// Imaginary parts: tempBuf[N:2*N]
	copy(tempBuf[:srcLen], srcData[:srcLen])
	// Zero-pad remaining real parts
	for i := srcLen; i < f.length; i++ {
		tempBuf[i] = 0
	}
	// Zero all imaginary parts
	for i := range f.length {
		tempBuf[i+f.length] = 0
	}

	// Perform FFT on packed format
	if len(f.lut) > 0 {
		totalFactors := len(f.lut) / 2
		fp32.FFTWithLUT(tempBuf, f.lut, totalFactors)
	} else {
		fp32.FFT(tempBuf)
	}

	// Copy result to destination (copy all 2*N elements)
	copyLen := 2 * f.length
	if copyLen > len(dstData) {
		copyLen = len(dstData)
	}
	copy(dstData, tempBuf[:copyLen])
}

// backward performs 1D inverse FFT with destination buffer.
// Optimized with BCE techniques: reslicing
func (f *FFT1D) backward(src, dst vec.Vector) {
	srcData := []float32(src)
	dstData := []float32(dst)

	// Ensure source is large enough for packed format (2*N)
	if len(srcData) < 2*f.length {
		panic("FFT1D.Backward: source buffer too small, need 2*N for packed format")
	}

	// BCE hint: reslice temp buffer to exact size (2*N for packed format)
	tempBuf := f.tempBuf[:2*f.length]

	// Copy packed complex data from source
	copyLen := 2 * f.length
	if copyLen > len(srcData) {
		copyLen = len(srcData)
	}
	copy(tempBuf, srcData[:copyLen])
	// Zero-pad if needed
	for i := copyLen; i < 2*f.length; i++ {
		tempBuf[i] = 0
	}

	// Perform inverse FFT on packed format
	fp32.IFFT(tempBuf)

	// Copy real part to destination (imaginary parts should be ~0 for real output)
	copyLen = f.length
	if copyLen > len(dstData) {
		copyLen = len(dstData)
	}
	copy(dstData, tempBuf[:copyLen])
}

// Convolve performs 1D convolution of signal with kernel using FFT.
// Uses destination-based API with no allocations.
func (f *FFT1D) Convolve(dst, signal, kernel vec.Vector) {
	sigLen := len(signal)
	kernLen := len(kernel)
	resultLen := sigLen + kernLen - 1

	// Ensure destination is large enough
	if len(dst) < resultLen {
		panic("Convolve: destination buffer too small")
	}

	requiredFFTSize := fp32.NextPowerOfTwo(resultLen)
	if f.length != requiredFFTSize {
		panic("Convolve: FFT length mismatch, pad inputs or recreate FFT with proper size")
	}
	if sigLen > f.length || kernLen > f.length {
		panic("Convolve: input length exceeds FFT length")
	}

	// Temporary buffers (need 2*N for packed format)
	sigFreqBuf := fp32.Pool.Get(2 * f.length)
	defer fp32.Pool.Put(sigFreqBuf[:0])
	sigFreq := vec.Vector(sigFreqBuf)

	kernFreqBuf := fp32.Pool.Get(2 * f.length)
	defer fp32.Pool.Put(kernFreqBuf[:0])
	kernFreq := vec.Vector(kernFreqBuf)

	// Forward FFTs
	f.Forward(sigFreq, signal)
	f.Forward(kernFreq, kernel)

	// Multiply in frequency domain (complex multiplication on packed format)
	// BCE optimization: reslice and use for range
	sigFreqData := []float32(sigFreq)
	kernFreqData := []float32(kernFreq)
	size := f.length
	if len(sigFreqData) < 2*size {
		size = len(sigFreqData) / 2
	}
	if len(kernFreqData) < 2*size {
		size = len(kernFreqData) / 2
	}
	sigFreqData = sigFreqData[:2*size]
	kernFreqData = kernFreqData[:2*size]
	for i := range size {
		// Complex multiplication: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
		real1, imag1 := sigFreqData[i], sigFreqData[i+size]
		real2, imag2 := kernFreqData[i], kernFreqData[i+size]
		sigFreqData[i] = real1*real2 - imag1*imag2
		sigFreqData[i+size] = real1*imag2 + imag1*real2
	}

	// Inverse FFT to destination
	f.Backward(dst, sigFreq)

	dstData := []float32(dst)
	if resultLen > len(dstData) {
		resultLen = len(dstData)
	}
	for i := resultLen; i < len(dstData); i++ {
		dstData[i] = 0
	}
}

// Correlate performs 1D cross-correlation of signal1 with signal2 using FFT.
// Cross-correlation is similar to convolution but with one signal time-reversed.
// Uses destination-based API with no allocations.
func (f *FFT1D) Correlate(dst, signal1, signal2 vec.Vector) {
	sig1Len := len(signal1)
	sig2Len := len(signal2)
	resultLen := sig1Len + sig2Len - 1

	// Ensure destination is large enough
	if len(dst) < resultLen {
		panic("Correlate: destination buffer too small")
	}

	// FFT size must be at least resultLen to avoid aliasing
	fftSize := fp32.NextPowerOfTwo(resultLen)
	if f.length != fftSize {
		panic("Correlate: FFT length mismatch, pad inputs or recreate FFT with proper size")
	}
	if sig1Len > f.length || sig2Len > f.length {
		panic("Correlate: input length exceeds FFT length")
	}

	// Temporary buffers (use fftSize, need 2*fftSize for packed format)
	sig1FreqBuf := fp32.Pool.Get(2 * fftSize)
	defer fp32.Pool.Put(sig1FreqBuf[:0])
	sig1Freq := vec.Vector(sig1FreqBuf)
	sig2FreqBuf := fp32.Pool.Get(2 * fftSize)
	defer fp32.Pool.Put(sig2FreqBuf[:0])
	sig2Freq := vec.Vector(sig2FreqBuf)

	// Forward FFTs (will zero-pad to fftSize)
	f.Forward(sig1Freq, signal1)
	f.Forward(sig2Freq, signal2)

	// For correlation, multiply sig1 with conjugate of sig2 in frequency domain
	// Correlation: multiply by conjugate (a+bi)*(c-di) = (ac+bd) + (bc-ad)i
	// BCE optimization: reslice and use for range
	sig1FreqData := []float32(sig1Freq)
	sig2FreqData := []float32(sig2Freq)
	size := fftSize
	if len(sig1FreqData) < 2*size {
		size = len(sig1FreqData) / 2
	}
	if len(sig2FreqData) < 2*size {
		size = len(sig2FreqData) / 2
	}
	sig1FreqData = sig1FreqData[:2*size]
	sig2FreqData = sig2FreqData[:2*size]
	for i := range size {
		// Complex multiplication with conjugate: (a+bi)*(c-di) = (ac+bd) + (bc-ad)i
		real1, imag1 := sig1FreqData[i], sig1FreqData[i+size]
		real2, imag2 := sig2FreqData[i], sig2FreqData[i+size]
		sig1FreqData[i] = real1*real2 + imag1*imag2
		sig1FreqData[i+size] = imag1*real2 - real1*imag2
	}

	// Inverse FFT to destination
	f.Backward(dst, sig1Freq)

	if resultLen > len(dst) {
		resultLen = len(dst)
	}
	dstData := []float32(dst)
	for i := resultLen; i < len(dstData); i++ {
		dstData[i] = 0
	}
}
