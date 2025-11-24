package c64

// NewFFTLUT creates a new FFT LUT buffer for the given FFT size.
// Returns a pre-allocated buffer filled with twiddle factors.
// The buffer size is determined by ComputeFFTLUT(fftSize).
func NewFFTLUT(fftSize int) []complex64 {
	lutSize := ComputeFFTLUT(fftSize)
	lut := make([]complex64, lutSize)
	FillFFTLUT(lut, fftSize)
	return lut
}
