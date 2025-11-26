package iq

import (
	"io"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

var _ chain.Processor[complex64, float32] = (*Modulator)(nil)

// Modulator converts real float32 samples (from ADC) to IQ complex64.
// For baseband (CarrierFrequency=0): I = signal, Q = Hilbert transform (or 0 for simple case)
// For carrier mixing: I = signal * cos(ωt), Q = signal * -sin(ωt)
type Modulator struct {
	cfg Config

	// LUT for cos/sin tables
	cosLUT []float32
	sinLUT []float32
	useLUT bool

	// Phase accumulator for mixing
	phaseAccumulator float32
	phaseIncrement   float32 // Phase increment per sample for carrier

	// Buffering state
	buffered      []float32
	bufferedCount int
}

// NewModulator builds a streaming IQ modulator that converts real signals to IQ.
func NewModulator(cfg Config) (*Modulator, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	m := &Modulator{
		cfg:             cfg,
		useLUT:          cfg.LUTSize > 0,
		buffered:        make([]float32, 1024),
		phaseAccumulator: 0,
	}

	// Calculate phase increment for carrier mixing
	if cfg.CarrierFrequency > 0 {
		m.phaseIncrement = twoPi * cfg.CarrierFrequency / cfg.SampleRate
	}

	// Build cos/sin LUT if enabled
	if m.useLUT {
		m.cosLUT = make([]float32, cfg.LUTSize)
		m.sinLUT = make([]float32, cfg.LUTSize)
		for i := range cfg.LUTSize {
			angle := twoPi * float32(i) / float32(cfg.LUTSize)
			m.cosLUT[i] = math32.Cos(angle)
			m.sinLUT[i] = math32.Sin(angle)
		}
	}

	return m, nil
}

// Process implements chain.Processor.
func (m *Modulator) Process(dst buffer.Writer[complex64], src buffer.Reader[float32]) int {
	if dst == nil {
		return m.estimateComplex(src)
	}

	total := 0
	readBuf := make([]float32, 512)

	for {
		// Read more data if needed
		if dst.Len() >= dst.Cap() {
			break
		}

		if m.bufferedCount == 0 {
			n, err := src.Read(readBuf)
			if n > 0 {
				if m.bufferedCount+n > len(m.buffered) {
					newBuf := make([]float32, (m.bufferedCount+n)*2)
					copy(newBuf, m.buffered[:m.bufferedCount])
					m.buffered = newBuf
				}
				copy(m.buffered[m.bufferedCount:], readBuf[:n])
				m.bufferedCount += n
			}
			if err != nil && err != io.EOF {
				break
			}
			if n == 0 {
				break
			}
		}

		if m.bufferedCount == 0 {
			break
		}

		// Determine how many samples we can process
		available := m.bufferedCount
		maxOutput := dst.Cap() - dst.Len()
		if available > maxOutput {
			available = maxOutput
		}
		if available == 0 {
			break
		}

		// Prepare output buffer
		complexBuf := make([]complex64, available)

		// Convert real samples to IQ
		if m.useLUT {
			m.convertWithLUT(complexBuf, m.buffered[:available])
		} else {
			m.convertDirect(complexBuf, m.buffered[:available])
		}

		// Write complex samples
		written, _ := dst.Write(complexBuf)
		total += written

		// Update phase accumulator (for next iteration)
		m.phaseAccumulator += m.phaseIncrement * float32(written)
		// Wrap phase to [0, 2π)
		for m.phaseAccumulator >= twoPi {
			m.phaseAccumulator -= twoPi
		}
		for m.phaseAccumulator < 0 {
			m.phaseAccumulator += twoPi
		}

		// Shift remaining samples
		if written < available {
			copy(m.buffered, m.buffered[written:available])
			m.bufferedCount -= written
		} else {
			m.bufferedCount = 0
		}

		if written == 0 {
			break
		}
	}

	return total
}

// convertDirect converts real samples to IQ using direct math.
func (m *Modulator) convertDirect(dst []complex64, src []float32) {
	phase := m.phaseAccumulator
	inc := m.phaseIncrement

	if m.cfg.CarrierFrequency == 0 {
		// Baseband: I = signal, Q = 0 (or could use Hilbert transform)
		for i := range src {
			sig := src[i]
			dst[i] = complex64(complex(sig, 0))
		}
	} else {
		// Carrier mixing: I = signal * cos(ωt), Q = signal * sin(ωt)
		for i := range src {
			sig := src[i]
			cosVal := math32.Cos(phase)
			sinVal := math32.Sin(phase)
			// I = signal * cos(ωt)
			// Q = signal * sin(ωt)
			dst[i] = complex64(complex(sig*cosVal, sig*sinVal))
			phase += inc
			if phase >= twoPi {
				phase -= twoPi
			}
		}
	}
}

// convertWithLUT converts real samples to IQ using LUT with interpolation.
func (m *Modulator) convertWithLUT(dst []complex64, src []float32) {
	phase := m.phaseAccumulator
	inc := m.phaseIncrement
	scale := float32(len(m.cosLUT)) / twoPi

	if m.cfg.CarrierFrequency == 0 {
		// Baseband: I = signal, Q = 0
		for i := range src {
			sig := src[i]
			dst[i] = complex64(complex(sig, 0))
		}
	} else {
		// Carrier mixing with LUT interpolation
		for i := range src {
			sig := src[i]

			// Look up cos/sin with interpolation
			idx := phase * scale
			idx0 := int(idx) % len(m.cosLUT)
			idx1 := (idx0 + 1) % len(m.cosLUT)
			fraction := idx - float32(int(idx))

			// Linear interpolation for cos
			cosVal := m.cosLUT[idx0]*(1-fraction) + m.cosLUT[idx1]*fraction
			// Linear interpolation for sin
			sinVal := m.sinLUT[idx0]*(1-fraction) + m.sinLUT[idx1]*fraction

			// I = signal * cos(ωt)
			// Q = signal * sin(ωt)
			dst[i] = complex64(complex(sig*cosVal, sig*sinVal))

			phase += inc
			if phase >= twoPi {
				phase -= twoPi
			}
		}
	}
}

func (m *Modulator) estimateComplex(src buffer.Reader[float32]) int {
	if src == nil {
		return 0
	}
	length := src.Len()
	if length <= 0 {
		return 0
	}
	// Each real sample produces one complex sample
	return length + m.bufferedCount
}

// Reset clears the internal state.
func (m *Modulator) Reset() {
	m.bufferedCount = 0
	m.phaseAccumulator = 0
}
