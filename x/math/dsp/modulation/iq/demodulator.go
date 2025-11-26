package iq

import (
	"io"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

var _ chain.Processor[float32, complex64] = (*Demodulator)(nil)

// Demodulator converts IQ complex64 samples back to real float32 (for DAC).
// For baseband (CarrierFrequency=0): real = I (or real part of IQ)
// For carrier mixing: real = I * cos(ωt) + Q * sin(ωt) (upconversion)
type Demodulator struct {
	cfg Config

	// LUT for cos/sin tables
	cosLUT []float32
	sinLUT []float32
	useLUT bool

	// Phase accumulator for mixing
	phaseAccumulator float32
	phaseIncrement   float32 // Phase increment per sample for carrier

	// Buffering state
	buffered      []complex64
	bufferedCount int
}

// NewDemodulator builds a streaming IQ demodulator that converts IQ to real signals.
func NewDemodulator(cfg Config) (*Demodulator, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	d := &Demodulator{
		cfg:             cfg,
		useLUT:          cfg.LUTSize > 0,
		buffered:        make([]complex64, 512),
		phaseAccumulator: 0,
	}

	// Calculate phase increment for carrier mixing
	if cfg.CarrierFrequency > 0 {
		d.phaseIncrement = twoPi * cfg.CarrierFrequency / cfg.SampleRate
	}

	// Build cos/sin LUT if enabled
	if d.useLUT {
		d.cosLUT = make([]float32, cfg.LUTSize)
		d.sinLUT = make([]float32, cfg.LUTSize)
		for i := range cfg.LUTSize {
			angle := twoPi * float32(i) / float32(cfg.LUTSize)
			d.cosLUT[i] = math32.Cos(angle)
			d.sinLUT[i] = math32.Sin(angle)
		}
	}

	return d, nil
}

// Process implements chain.Processor.
func (d *Demodulator) Process(dst buffer.Writer[float32], src buffer.Reader[complex64]) int {
	if dst == nil {
		return d.estimateFloat32(src)
	}

	total := 0
	readBuf := make([]complex64, 256)

	for {
		// Read more data if needed
		if dst.Len() >= dst.Cap() {
			break
		}

		if d.bufferedCount == 0 {
			n, err := src.Read(readBuf)
			if n > 0 {
				if d.bufferedCount+n > len(d.buffered) {
					newBuf := make([]complex64, (d.bufferedCount+n)*2)
					copy(newBuf, d.buffered[:d.bufferedCount])
					d.buffered = newBuf
				}
				copy(d.buffered[d.bufferedCount:], readBuf[:n])
				d.bufferedCount += n
			}
			if err != nil && err != io.EOF {
				break
			}
			if n == 0 {
				break
			}
		}

		if d.bufferedCount == 0 {
			break
		}

		// Determine how many samples we can process
		available := d.bufferedCount
		maxOutput := dst.Cap() - dst.Len()
		if available > maxOutput {
			available = maxOutput
		}
		if available == 0 {
			break
		}

		// Prepare output buffer
		floatBuf := make([]float32, available)

		// Convert IQ to real samples
		if d.useLUT {
			d.convertWithLUT(floatBuf, d.buffered[:available])
		} else {
			d.convertDirect(floatBuf, d.buffered[:available])
		}

		// Write real samples
		written, _ := dst.Write(floatBuf)
		total += written

		// Update phase accumulator (for next iteration)
		d.phaseAccumulator += d.phaseIncrement * float32(written)
		// Wrap phase to [0, 2π)
		for d.phaseAccumulator >= twoPi {
			d.phaseAccumulator -= twoPi
		}
		for d.phaseAccumulator < 0 {
			d.phaseAccumulator += twoPi
		}

		// Shift remaining samples
		if written < available {
			copy(d.buffered, d.buffered[written:available])
			d.bufferedCount -= written
		} else {
			d.bufferedCount = 0
		}

		if written == 0 {
			break
		}
	}

	return total
}

// convertDirect converts IQ samples to real using direct math.
func (d *Demodulator) convertDirect(dst []float32, src []complex64) {
	phase := d.phaseAccumulator
	inc := d.phaseIncrement

	if d.cfg.CarrierFrequency == 0 {
		// Baseband: real = I (real part)
		for i := range src {
			dst[i] = float32(real(src[i]))
		}
	} else {
		// Upconversion: real = I * cos(ωt) + Q * sin(ωt)
		for i := range src {
			c := src[i]
			iVal := float32(real(c))
			qVal := float32(imag(c))

			cosVal := math32.Cos(phase)
			sinVal := math32.Sin(phase)

			// real = I * cos(ωt) + Q * sin(ωt)
			dst[i] = iVal*cosVal + qVal*sinVal

			phase += inc
			if phase >= twoPi {
				phase -= twoPi
			}
		}
	}
}

// convertWithLUT converts IQ samples to real using LUT with interpolation.
func (d *Demodulator) convertWithLUT(dst []float32, src []complex64) {
	phase := d.phaseAccumulator
	inc := d.phaseIncrement
	scale := float32(len(d.cosLUT)) / twoPi

	if d.cfg.CarrierFrequency == 0 {
		// Baseband: real = I (real part)
		for i := range src {
			dst[i] = float32(real(src[i]))
		}
	} else {
		// Upconversion with LUT interpolation
		for i := range src {
			c := src[i]
			iVal := float32(real(c))
			qVal := float32(imag(c))

			// Look up cos/sin with interpolation
			idx := phase * scale
			idx0 := int(idx) % len(d.cosLUT)
			idx1 := (idx0 + 1) % len(d.cosLUT)
			fraction := idx - float32(int(idx))

			// Linear interpolation for cos
			cosVal := d.cosLUT[idx0]*(1-fraction) + d.cosLUT[idx1]*fraction
			// Linear interpolation for sin
			sinVal := d.sinLUT[idx0]*(1-fraction) + d.sinLUT[idx1]*fraction

			// real = I * cos(ωt) + Q * sin(ωt)
			dst[i] = iVal*cosVal + qVal*sinVal

			phase += inc
			if phase >= twoPi {
				phase -= twoPi
			}
		}
	}
}

func (d *Demodulator) estimateFloat32(src buffer.Reader[complex64]) int {
	if src == nil {
		return 0
	}
	length := src.Len()
	if length <= 0 {
		return 0
	}
	// Each complex sample produces one real sample
	return length + d.bufferedCount
}

// Reset clears the internal state.
func (d *Demodulator) Reset() {
	d.bufferedCount = 0
	d.phaseAccumulator = 0
}
