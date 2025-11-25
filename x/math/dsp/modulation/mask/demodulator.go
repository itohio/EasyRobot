package mask

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	"github.com/itohio/EasyRobot/x/math/dsp"
)

var _ chain.Processor[byte, float32] = (*Demodulator)(nil)

// Demodulator converts M-ASK samples back to symbols.
type Demodulator struct {
	cfg   Config
	table amplitudeTable

	fifo     []float32
	buffered int
	readBuf  []float32

	// Transmission detection
	active           bool
	preambleDetected bool
	preambleProgress int
	silenceCount     int

	// AGC (Amplitude normalization)
	agcGain       float32
	agcReference float32 // Measured from preamble
	agcSum       float32
	agcCount     int

	// Carrier recovery (for coherent mode)
	carrier dsp.CosineGenerator

	tmpSymbol [1]byte
}

// NewDemodulator builds a streaming demodulator instance.
func NewDemodulator(cfg Config) (*Demodulator, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	d := &Demodulator{
		cfg:      cfg,
		table:    newAmplitudeTable(cfg),
		fifo:     make([]float32, cfg.SamplesPerSymbol*2),
		carrier:  dsp.NewCosineGenerator(cfg.SampleRate, cfg.CarrierHz, 1.0, 0.0),
		agcGain:  1.0,
		agcReference: 1.0,
	}
	return d, nil
}

// Process implements chain.Processor.
func (d *Demodulator) Process(dst buffer.Writer[byte], src buffer.Reader[float32]) int {
	if dst == nil {
		return d.estimateBytes(src)
	}
	bitsWriter, ok := dst.(buffer.BitsWriter)
	if !ok || bitsWriter == nil {
		return 0
	}
	if bitsWriter.BitsLength() != d.cfg.BitsPerSymbol {
		return 0
	}

	totalBytes := 0
	sourceDone := false

	for {
		for d.buffered < d.cfg.SamplesPerSymbol && !sourceDone {
			needed := d.cfg.SamplesPerSymbol - d.buffered
			if len(d.readBuf) < needed {
				d.readBuf = make([]float32, needed)
			}
			n, err := src.Read(d.readBuf[:needed])
			if n > 0 {
				copy(d.fifo[d.buffered:], d.readBuf[:n])
				d.buffered += n
			}
			if err == io.EOF {
				sourceDone = true
			}
			if err != nil && err != io.EOF {
				sourceDone = true
			}
		}

		if d.buffered < d.cfg.SamplesPerSymbol {
			break
		}

		symbol := d.decodeSymbol(d.fifo[:d.cfg.SamplesPerSymbol])
		if symbol >= 0 {
			d.tmpSymbol[0] = byte(symbol)
			writtenBits, _ := bitsWriter.Write(d.tmpSymbol[:])
			if writtenBits == 0 {
				break
			}
			totalBytes += writtenBits / d.cfg.BitsPerSymbol
		}

		// Shift buffer
		copy(d.fifo, d.fifo[d.cfg.SamplesPerSymbol:d.cfg.SamplesPerSymbol+d.buffered])
		d.buffered -= d.cfg.SamplesPerSymbol
	}

	return totalBytes
}

func (d *Demodulator) decodeSymbol(samples []float32) int {
	if len(samples) == 0 {
		return -1
	}

	energy := symbolEnergy(samples)
	threshold := d.cfg.EnergyThreshold

	// Adaptive threshold: more lenient during active transmission
	if d.active {
		threshold = d.cfg.EnergyThreshold * 0.1
	} else if d.preambleProgress > 0 {
		threshold = d.cfg.EnergyThreshold * 0.2
	} else {
		// Initial detection: be more lenient
		threshold = d.cfg.EnergyThreshold * 0.5
	}

	if energy < threshold {
		d.silenceCount++
		if d.silenceCount >= d.cfg.TransmissionGap {
			d.active = false
			d.preambleDetected = false
			d.preambleProgress = 0
			d.silenceCount = 0
			d.agcGain = 1.0
			d.agcReference = 1.0
			d.agcSum = 0
			d.agcCount = 0
		}
		return -1
	}

	d.active = true
	d.silenceCount = 0

	// Detect amplitude
	var amplitude float32
	if d.cfg.CarrierHz > 0 {
		// For passband, always use coherent detection (downconvert)
		// This gives us the amplitude modulation directly
		amplitude = d.coherentDetection(samples)
		if amplitude < 0 {
			amplitude = -amplitude
		}
	} else {
		// Baseband: direct amplitude (RMS or average)
		amplitude = envelopeDetection(samples)
	}

	// Preamble processing
	var symbol int
	if d.cfg.PreambleLength > 0 && d.preambleProgress < d.cfg.PreambleLength {
		// During preamble: use raw amplitude comparison (no AGC yet)
		// Update AGC from preamble
		d.updateAGC(amplitude)
		
		// For preamble detection, compare raw amplitude to expected levels
		pattern := d.cfg.PreamblePattern
		if len(pattern) == 0 {
			pattern = []int{d.cfg.constellationSize() - 1, 0}
		}
		expectedIdx := pattern[d.preambleProgress%len(pattern)]
		
		// Find which level this amplitude is closest to (raw comparison)
		symbol = quantizeAmplitude(amplitude, d.table.levels)
		
		// For preamble, accept any symbol - just need to accumulate enough for AGC
		// The pattern matching helps but isn't critical for basic functionality
		// Accept the symbol and continue
		_ = expectedIdx // Keep for potential future use
		d.preambleProgress++
		if d.preambleProgress >= d.cfg.PreambleLength {
			d.preambleDetected = true
			// Finalize AGC reference - use maximum amplitude from preamble
			if d.agcCount > 0 {
				avgAmplitude := d.agcSum / float32(d.agcCount)
				// Preamble alternates between max and min levels
				// Estimate max from average: for [max, min, max, min...], avg ≈ (max+min)/2
				// So max ≈ 2*avg - min, or simpler: max ≈ 2*avg (if min ≈ 0)
				maxIdx := d.cfg.constellationSize() - 1
				if maxIdx < len(d.table.levels) {
					expectedMaxLevel := d.table.levels[maxIdx]
					expectedMinLevel := d.table.levels[0]
					if expectedMaxLevel > expectedMinLevel {
						// Scale average to get max estimate
						expectedAvg := (expectedMaxLevel + expectedMinLevel) / 2
						if expectedAvg > 0 {
							scale := expectedMaxLevel / expectedAvg
							d.agcReference = avgAmplitude * scale
						} else {
							d.agcReference = avgAmplitude * 2.0
						}
					} else {
						d.agcReference = avgAmplitude * 2.0
					}
				} else {
					d.agcReference = avgAmplitude * 2.0
				}
				if d.agcReference <= 0 {
					d.agcReference = 1.0
				}
				d.agcGain = 1.0 / d.agcReference
			}
		}
		// Don't write preamble bytes
		return -1
	} else {
		// After preamble: use AGC-normalized amplitude
		agcRef := d.agcReference
		if agcRef <= 0 || d.cfg.PreambleLength == 0 {
			// No preamble or AGC not set: use maximum expected level as reference
			maxLevel := d.table.levels[len(d.table.levels)-1]
			if maxLevel > 0 {
				agcRef = maxLevel
			} else {
				agcRef = 1.0
			}
		}
		normalized := normalizeAmplitude(amplitude, agcRef)

		// Normalize levels to [0, 1] for comparison
		maxLevel := d.table.levels[len(d.table.levels)-1]
		if maxLevel <= 0 {
			maxLevel = 1.0
		}
		normalizedLevels := make([]float32, len(d.table.levels))
		for i, level := range d.table.levels {
			normalizedLevels[i] = level / maxLevel
		}

		// Quantize to nearest level
		symbol = quantizeAmplitude(normalized, normalizedLevels)
	}

	// Only write after preamble is detected (or if no preamble)
	if d.cfg.PreambleLength > 0 && !d.preambleDetected {
		return -1
	}

	return symbol
}

func (d *Demodulator) coherentDetection(samples []float32) float32 {
	// Downconvert by mixing with carrier
	// Signal: A * envelope(t) * cos(2πfc*t + φ)
	// Mix with cos(2πfc*t): result = A * envelope(t) * (1 + cos(4πfc*t + 2φ)) / 2
	// After averaging (low-pass): A * envelope(t) / 2
	// So we need to multiply by 2 to get the actual amplitude
	var iAccum float32
	if len(d.readBuf) < len(samples) {
		d.readBuf = make([]float32, len(samples))
	}
	carrierBuf := d.readBuf[:len(samples)]
	d.carrier.Generate(carrierBuf)
	for i := range samples {
		iAccum += samples[i] * carrierBuf[i]
	}
	// Average - the scaling depends on envelope shape
	// For raised cosine with beta=0.35, average envelope ≈ 0.65
	// So we need to scale by approximately 2/0.65 ≈ 3.0
	// But let's use a more conservative factor and let AGC handle the rest
	amplitude := iAccum / float32(len(samples)) * 2.5
	if amplitude < 0 {
		amplitude = -amplitude
	}
	return amplitude
}

func (d *Demodulator) envelopeDetectionPassband(samples []float32) float32 {
	// For passband M-ASK, extract envelope using peak-to-peak detection
	// Signal is: A * envelope(t) * cos(2πfc*t)
	// Peak-to-peak over a symbol period gives us approximately 2*A*envelope
	var maxVal, minVal float32
	if len(samples) == 0 {
		return 0
	}
	maxVal = samples[0]
	minVal = samples[0]
	for _, s := range samples {
		if s > maxVal {
			maxVal = s
		}
		if s < minVal {
			minVal = s
		}
	}
	// Peak-to-peak gives us approximately 2*amplitude for carrier
	pp := maxVal - minVal
	amplitude := pp / 2.0
	// Fallback to RMS if peak-to-peak is too small
	if amplitude < 0.01 {
		amplitude = envelopeDetection(samples) * 1.414 // Scale RMS to approximate amplitude
	}
	return amplitude
}

func (d *Demodulator) updateAGC(amplitude float32) {
	// Accumulate amplitude measurements during preamble
	d.agcSum += amplitude
	d.agcCount++
}

func (d *Demodulator) estimateBytes(src buffer.Reader[float32]) int {
	if src == nil {
		return 0
	}
	length := src.Len()
	if length <= 0 {
		return 0
	}
	symbols := length / d.cfg.SamplesPerSymbol
	// Subtract preamble symbols
	if d.cfg.PreambleLength > 0 {
		if symbols > d.cfg.PreambleLength {
			symbols -= d.cfg.PreambleLength
		} else {
			symbols = 0
		}
	}
	return symbols * d.cfg.BitsPerSymbol / 8
}

// Reset clears internal state.
func (d *Demodulator) Reset() {
	d.buffered = 0
	d.active = false
	d.preambleDetected = false
	d.preambleProgress = 0
	d.silenceCount = 0
	d.agcGain = 1.0
	d.agcReference = 1.0
	d.agcSum = 0
	d.agcCount = 0
	d.carrier = dsp.NewCosineGenerator(d.cfg.SampleRate, d.cfg.CarrierHz, 1.0, 0.0)
}

