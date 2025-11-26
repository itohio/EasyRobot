package internal

import (
	"io"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
)

var _ Processor = (*TimingRecovery)(nil)

// TimingRecovery implements Gardner timing error detection (TED) for symbol timing recovery.
// It adjusts sampling instants to optimal symbol timing by interpolating samples.
type TimingRecovery struct {
	samplesPerSymbol float32 // Nominal samples per symbol
	symbolRate       float32 // Symbol rate in Hz
	sampleRate       float32 // Sample rate in Hz

	// Loop parameters
	kp           float32 // Proportional gain
	ki           float32 // Integral gain
	dampingRatio float32

	// Timing state
	timingPhase float32 // Current timing phase (0 to 1)
	timingFreq  float32 // Timing frequency offset
	timingError float32 // Last computed timing error

	// Interpolation state
	mu float32 // Interpolation interval [0, 1)

	// Buffering for interpolation (needed for sample count changes)
	buffered      []complex64
	bufferedCount int
	batchSize     int // Batch size for reading from src
}

// NewTimingRecovery creates a new Gardner timing recovery processor.
// symbolRate: symbol rate in Hz
// sampleRate: sample rate in Hz
// loopBandwidth: loop bandwidth in Hz (typical: 0.001 to 0.01 * symbolRate)
// dampingRatio: damping ratio (typical: 0.707)
func NewTimingRecovery(symbolRate, sampleRate, loopBandwidth, dampingRatio float32) TimingRecovery {
	return NewTimingRecoveryWithBatchSize(symbolRate, sampleRate, loopBandwidth, dampingRatio, 16)
}

// NewTimingRecoveryWithBatchSize creates a new Gardner timing recovery processor with custom batch size.
func NewTimingRecoveryWithBatchSize(symbolRate, sampleRate, loopBandwidth, dampingRatio float32, batchSize int) TimingRecovery {
	if dampingRatio <= 0 {
		dampingRatio = 0.707
	}
	if loopBandwidth <= 0 {
		loopBandwidth = symbolRate * 0.001
	}
	if batchSize <= 0 {
		batchSize = 16
	}

	samplesPerSymbol := sampleRate / symbolRate

	// Calculate gains
	omegaN := 2 * math32.Pi * loopBandwidth / symbolRate
	kp := 2 * dampingRatio * omegaN
	ki := omegaN * omegaN

	return TimingRecovery{
		samplesPerSymbol: samplesPerSymbol,
		symbolRate:       symbolRate,
		sampleRate:       sampleRate,
		kp:               kp,
		ki:               ki,
		dampingRatio:     dampingRatio,
		timingPhase:      0.5, // Start at middle of symbol period
		mu:               0.5,
		buffered:         make([]complex64, int(samplesPerSymbol)*4+10), // Extra buffer for interpolation
		batchSize:        batchSize,
	}
}

// Process implements Processor.
// Adjusts sampling timing using Gardner algorithm and outputs resampled samples.
// Note: This processor changes sample count, so it reads from src (or dst if it's also a Reader) and writes to dst.
func (t *TimingRecovery) Process(dst buffer.Writer[complex64], src buffer.Reader[complex64]) int {
	if dst == nil {
		return t.estimate(src)
	}

	// Prefer reading from dst if it's also a Reader, otherwise use src
	reader := src
	if dstReader, ok := dst.(buffer.Reader[complex64]); ok {
		reader = dstReader
	}
	if reader == nil {
		return 0
	}

	total := 0
	readBuf := make([]complex64, t.batchSize)

	for {
		// Need enough samples for interpolation (at least 2 symbols worth)
		minSamples := int(t.samplesPerSymbol * 2.5)
		if t.bufferedCount < minSamples && dst.Len() < dst.Cap() {
			n, err := reader.Read(readBuf)
			if n > 0 {
				if t.bufferedCount+n > len(t.buffered) {
					newBuf := make([]complex64, (t.bufferedCount+n)*2)
					copy(newBuf, t.buffered[:t.bufferedCount])
					t.buffered = newBuf
				}
				copy(t.buffered[t.bufferedCount:], readBuf[:n])
				t.bufferedCount += n
			}
			if err != nil && err != io.EOF {
				break
			}
			if n == 0 && t.bufferedCount < minSamples {
				break
			}
		}

		if t.bufferedCount < minSamples {
			break
		}

		// Process symbols until buffer is low
		maxOutput := dst.Cap() - dst.Len()
		processed := 0
		output := make([]complex64, 0, maxOutput)

		for processed < maxOutput && t.bufferedCount >= minSamples {
			// Interpolate sample at current timing phase
			sample := t.interpolate()
			output = append(output, sample)

			// Gardner TED: timing error = Re[(x(n) - x(n-2)) * conj(x(n-1))]
			if processed >= 2 {
				prev2 := output[processed-2]
				prev1 := output[processed-1]
				current := sample

				// Gardner error calculation
				i2 := float32(real(prev2))
				q2 := float32(imag(prev2))
				i1 := float32(real(prev1))
				q1 := float32(imag(prev1))
				i0 := float32(real(current))
				q0 := float32(imag(current))

				// Error = Re[(x(n) - x(n-2)) * conj(x(n-1))]
				// = (I(n) - I(n-2)) * I(n-1) + (Q(n) - Q(n-2)) * Q(n-1)
				timingError := (i0-i2)*i1 + (q0-q2)*q1

				// Update timing loop
				t.timingFreq += t.ki * timingError
				t.timingPhase += t.kp*timingError + t.timingFreq

				// Keep phase in [0, 1)
				for t.timingPhase >= 1.0 {
					t.timingPhase -= 1.0
					// Advance buffer index
					if t.bufferedCount > 0 {
						copy(t.buffered, t.buffered[1:])
						t.bufferedCount--
					}
				}
				for t.timingPhase < 0 {
					t.timingPhase += 1.0
				}

				t.timingError = timingError
			} else {
				// Advance timing phase by symbol period
				t.timingPhase += 1.0 / t.samplesPerSymbol
				if t.timingPhase >= 1.0 {
					t.timingPhase -= 1.0
					if t.bufferedCount > 0 {
						copy(t.buffered, t.buffered[1:])
						t.bufferedCount--
					}
				}
			}

			processed++
		}

		if len(output) > 0 {
			written, _ := dst.Write(output)
			total += written
		} else {
			break
		}

		if processed == 0 || dst.Len() >= dst.Cap() {
			break
		}
	}

	return total
}

// interpolate performs linear interpolation at current timing phase.
func (t *TimingRecovery) interpolate() complex64 {
	idx := int(t.timingPhase * t.samplesPerSymbol)
	if idx+1 >= t.bufferedCount {
		idx = t.bufferedCount - 2
	}
	if idx < 0 {
		idx = 0
	}

	s0 := t.buffered[idx]
	s1 := t.buffered[idx+1]

	// Linear interpolation
	mu := t.timingPhase*t.samplesPerSymbol - float32(idx)
	i0 := float32(real(s0))
	q0 := float32(imag(s0))
	i1 := float32(real(s1))
	q1 := float32(imag(s1))

	iInterp := i0*(1-mu) + i1*mu
	qInterp := q0*(1-mu) + q1*mu

	return complex64(complex(iInterp, qInterp))
}

func (t *TimingRecovery) estimate(src buffer.Reader[complex64]) int {
	if src == nil {
		return 0
	}
	// Timing recovery outputs approximately one sample per symbol period
	samples := src.Len() + t.bufferedCount
	return int(float32(samples) / t.samplesPerSymbol)
}

// Reset clears the internal state.
func (t *TimingRecovery) Reset() {
	t.bufferedCount = 0
	t.timingPhase = 0.5
	t.timingFreq = 0
	t.timingError = 0
	t.mu = 0.5
}

// TimingPhase returns the current timing phase [0, 1).
func (t *TimingRecovery) TimingPhase() float32 {
	return t.timingPhase
}

// TimingError returns the last computed timing error.
func (t *TimingRecovery) TimingError() float32 {
	return t.timingError
}

// SetLoopBandwidth updates the loop bandwidth and recalculates gains.
func (t *TimingRecovery) SetLoopBandwidth(loopBandwidth, dampingRatio float32) {
	if dampingRatio <= 0 {
		dampingRatio = 0.707
	}
	if loopBandwidth <= 0 {
		loopBandwidth = t.symbolRate * 0.001
	}

	omegaN := 2 * math32.Pi * loopBandwidth / t.symbolRate
	t.kp = 2 * dampingRatio * omegaN
	t.ki = omegaN * omegaN
	t.dampingRatio = dampingRatio
}

