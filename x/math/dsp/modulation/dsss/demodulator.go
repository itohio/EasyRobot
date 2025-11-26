package dsss

import (
	"errors"
	"io"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
)

var (
	_ chain.Processor[byte, float32] = (*Demodulator)(nil)
	_ modulation.Receiver            = (*Demodulator)(nil)
)

type Demodulator struct {
	config           Config
	samplesPerSymbol int

	downconverter dsp.CosineGenerator

	sampleBuffer []float32
	readBuf      []float32
	buffered     int

	active           bool
	preambleDetected bool
	preambleProgress int
	preambleErrors   int
	silenceSymbols   int

	// Statistics tracking
	totalSamplesProcessed int
	transmissionStartIdx  int
	preambleStartIdx      int
	packetStartIdx        int
	lockAcquiredIdx       int
	lockLostIdx           int
	lockCount             int
	unlockCount           int

	// SNR and signal strength tracking
	signalPowerSum float32
	noisePowerSum  float32
	snrSampleCount int
	rawSignalSum   float32
	rawSignalCount int

	// Error statistics tracking
	phaseErrorSum   float32
	phaseErrorCount int
	phaseErrorMin   float32
	phaseErrorMax   float32
	phaseErrorSumSq float32

	freqErrorSum   float32
	freqErrorCount int
	freqErrorMin   float32
	freqErrorMax   float32
	freqErrorSumSq float32

	constellationErrorSum   float32
	constellationErrorCount int
	constellationErrorMin   float32
	constellationErrorMax   float32
	constellationErrorSumSq float32
}

func NewDemodulator(config Config) (*Demodulator, error) {
	if config.SpreadingFactor <= 0 {
		return nil, errors.New("dsss: spreading factor must be positive")
	}
	if config.SamplesPerChip == 0 {
		config.SamplesPerChip = config.CalculateSamplesPerChip(config.SampleRate)
	}
	if err := config.ValidateSampleRate(config.SampleRate); err != nil {
		return nil, err
	}

	return &Demodulator{
		config:           config,
		samplesPerSymbol: config.SpreadingFactor * config.SamplesPerChip,
		downconverter:    dsp.NewCosineGenerator(config.SampleRate, config.CarrierFreq, 1.0, 0.0),
	}, nil
}

func (d *Demodulator) Process(dst buffer.Writer[byte], src buffer.Reader[float32]) int {
	if dst == nil {
		return d.estimateBytes(src)
	}

	bitsWriter, ok := dst.(buffer.BitsWriter)
	if !ok || bitsWriter == nil {
		return 0
	}
	if bitsWriter.BitsLength() != d.config.BitsPerSymbol {
		return 0
	}

	d.ensureBuffers()
	writtenSymbols := 0

	for {
		needed := d.samplesPerSymbol - d.buffered
		if needed > 0 {
			if needed > len(d.readBuf) {
				d.readBuf = make([]float32, needed)
			}
			n, err := src.Read(d.readBuf[:needed])
			if n > 0 {
				copy(d.sampleBuffer[d.buffered:], d.readBuf[:n])
				d.buffered += n
			}
			if err == io.EOF && d.buffered < d.samplesPerSymbol {
				break
			}
			if d.buffered < d.samplesPerSymbol {
				continue
			}
		}

		// Track total samples processed (before processing symbol)
		symbolsWritten := d.processSymbol(bitsWriter, d.sampleBuffer[:d.samplesPerSymbol])
		if symbolsWritten > 0 {
			d.totalSamplesProcessed += d.samplesPerSymbol
		}
		writtenSymbols += symbolsWritten
		d.buffered = 0
	}

	return writtenSymbols
}

func (d *Demodulator) ensureBuffers() {
	if len(d.sampleBuffer) < d.samplesPerSymbol {
		d.sampleBuffer = make([]float32, d.samplesPerSymbol)
	}
	if len(d.readBuf) < d.samplesPerSymbol {
		d.readBuf = make([]float32, d.samplesPerSymbol)
	}
}

func (d *Demodulator) processSymbol(dst buffer.BitsWriter, samples []float32) int {
	d.downconverter.GenerateWithOp(samples, false, func(val *float32, carrier float32) {
		*val *= carrier
	})

	energy := d.symbolEnergy(samples)
	if !d.active {
		if energy > d.config.EnergyThreshold {
			d.active = true
			d.preambleDetected = false
			d.preambleProgress = 0
			// Track transmission start
			if d.transmissionStartIdx == 0 {
				d.transmissionStartIdx = d.totalSamplesProcessed
			}
		} else {
			return 0
		}
	}

	// Track raw signal strength (before any processing)
	rawMag := d.symbolMagnitude(samples)
	d.rawSignalSum += rawMag
	d.rawSignalCount++

	symbol := d.detectSymbol(samples)

	if !d.preambleDetected && d.config.PreambleLength > 0 {
		pattern := d.config.PreamblePattern
		if len(pattern) == 0 {
			pattern = []int{1, 0}
		}
		expected := pattern[d.preambleProgress%len(pattern)]
		if symbol == expected {
			if d.preambleProgress == 0 && d.preambleStartIdx == 0 {
				d.preambleStartIdx = d.totalSamplesProcessed
			}
			d.preambleProgress++
			d.preambleErrors = 0
			if d.preambleProgress >= d.config.PreambleLength {
				d.preambleDetected = true
				d.silenceSymbols = 0
				// Track packet start (after preamble)
				if d.packetStartIdx == 0 {
					d.packetStartIdx = d.totalSamplesProcessed + d.samplesPerSymbol
				}
				// Track lock acquisition
				if d.lockAcquiredIdx == 0 {
					d.lockAcquiredIdx = d.totalSamplesProcessed
					d.lockCount++
				}
			}
		} else {
			// Allow some preamble errors before resetting (more robust with noise)
			// Allow up to 1 error per 4 preamble symbols, but reset if too many errors
			maxErrors := d.config.PreambleLength / 4
			if maxErrors < 1 {
				maxErrors = 1
			}
			d.preambleErrors++
			if d.preambleErrors > maxErrors {
				d.preambleProgress = 0
				d.preambleErrors = 0
			}
		}
		return 0
	}

	if energy < d.config.EnergyThreshold {
		d.silenceSymbols++
		if d.silenceSymbols >= d.config.TransmissionTimeout {
			d.active = false
			d.preambleDetected = false
			d.preambleProgress = 0
			d.silenceSymbols = 0
			// Track lock loss
			if d.lockAcquiredIdx > 0 {
				d.lockLostIdx = d.totalSamplesProcessed
				d.unlockCount++
				d.lockAcquiredIdx = 0 // Reset for next acquisition
			}
			return 0
		}
	} else {
		d.silenceSymbols = 0
	}

	// Track statistics for decoded symbols
	d.trackSymbolStatistics(samples, symbol)

	bits, err := dst.Write([]byte{byte(symbol)})
	if err != nil {
		return 0
	}
	if d.config.BitsPerSymbol == 0 {
		return 0
	}
	return bits / d.config.BitsPerSymbol
}

func (d *Demodulator) detectSymbol(samples []float32) int {
	if d.config.ModulationOrder == 2 {
		score := d.codeCorrelation(samples, d.config.Codes[0])
		if score >= 0 {
			return 1
		}
		return 0
	}

	bestIdx := 0
	bestScore := float32(-math32.MaxFloat32)
	for idx, code := range d.config.Codes {
		score := d.codeCorrelation(samples, code)
		if score > bestScore {
			bestScore = score
			bestIdx = idx
		}
	}
	return bestIdx
}

func (d *Demodulator) codeCorrelation(samples []float32, code []int8) float32 {
	samplesPerChip := d.config.SamplesPerChip
	var corr float32
	for chip := 0; chip < d.config.SpreadingFactor; chip++ {
		start := chip * samplesPerChip
		end := start + samplesPerChip
		if end > len(samples) {
			end = len(samples)
		}
		var sum float32
		for i := start; i < end; i++ {
			sum += samples[i]
		}
		corr += sum * float32(code[chip])
	}
	return corr
}

func (d *Demodulator) symbolEnergy(samples []float32) float32 {
	if len(samples) == 0 {
		return 0
	}
	var energy float32
	for _, sample := range samples {
		energy += sample * sample
	}
	return energy / float32(len(samples))
}

func (d *Demodulator) symbolMagnitude(samples []float32) float32 {
	if len(samples) == 0 {
		return 0
	}
	var sum float32
	for _, sample := range samples {
		sum += sample * sample
	}
	return math32.Sqrt(sum / float32(len(samples)))
}

func (d *Demodulator) trackSymbolStatistics(samples []float32, detectedSymbol int) {
	// Calculate actual correlation for detected symbol
	var idealCode []int8
	if d.config.ModulationOrder == 2 {
		idealCode = d.config.Codes[0]
	} else {
		if detectedSymbol >= 0 && detectedSymbol < len(d.config.Codes) {
			idealCode = d.config.Codes[detectedSymbol]
		} else {
			return // Invalid symbol
		}
	}

	actualCorr := d.codeCorrelation(samples, idealCode)

	// For DSSS, "constellation error" is the difference between ideal and actual correlation
	// Ideal correlation should be close to spreading factor for correct symbol
	// For binary DSSS with symbol 0, correlation is negated
	idealCorrValue := float32(d.config.SpreadingFactor)
	if d.config.ModulationOrder == 2 && detectedSymbol == 0 {
		actualCorr = -actualCorr
	}
	correlationError := math32.Abs(idealCorrValue - actualCorr)

	// Track constellation error statistics
	d.constellationErrorCount++
	d.constellationErrorSum += correlationError
	d.constellationErrorSumSq += correlationError * correlationError
	if d.constellationErrorCount == 1 {
		d.constellationErrorMin = correlationError
		d.constellationErrorMax = correlationError
	} else {
		if correlationError < d.constellationErrorMin {
			d.constellationErrorMin = correlationError
		}
		if correlationError > d.constellationErrorMax {
			d.constellationErrorMax = correlationError
		}
	}

	// For DSSS, phase error is not directly applicable (no phase modulation)
	// But we can track carrier phase offset if needed (not implemented here)
	// Frequency error tracking would require frequency offset estimation (not implemented)

	// Track SNR: signal power vs noise power
	// Signal power: ideal correlation squared
	signalPower := idealCorrValue * idealCorrValue
	// Noise power: error squared
	noisePower := correlationError * correlationError

	d.signalPowerSum += signalPower
	d.noisePowerSum += noisePower
	d.snrSampleCount++
}

func (d *Demodulator) estimateBytes(src buffer.Reader[float32]) int {
	totalSamples := src.Len()
	if d.samplesPerSymbol == 0 || totalSamples == 0 {
		return 0
	}
	symbols := totalSamples / d.samplesPerSymbol
	bits := symbols * d.config.BitsPerSymbol
	return bits / 8
}

func (d *Demodulator) Reset() {
	d.buffered = 0
	d.active = false
	d.preambleDetected = false
	d.preambleProgress = 0
	d.preambleErrors = 0
	d.silenceSymbols = 0
	d.downconverter = dsp.NewCosineGenerator(d.config.SampleRate, d.config.CarrierFreq, 1.0, 0.0)

	// Reset statistics
	d.totalSamplesProcessed = 0
	d.transmissionStartIdx = 0
	d.preambleStartIdx = 0
	d.packetStartIdx = 0
	d.lockAcquiredIdx = 0
	d.lockLostIdx = 0
	d.lockCount = 0
	d.unlockCount = 0

	d.signalPowerSum = 0
	d.noisePowerSum = 0
	d.snrSampleCount = 0
	d.rawSignalSum = 0
	d.rawSignalCount = 0

	d.phaseErrorSum = 0
	d.phaseErrorCount = 0
	d.phaseErrorMin = 0
	d.phaseErrorMax = 0
	d.phaseErrorSumSq = 0

	d.freqErrorSum = 0
	d.freqErrorCount = 0
	d.freqErrorMin = 0
	d.freqErrorMax = 0
	d.freqErrorSumSq = 0

	d.constellationErrorSum = 0
	d.constellationErrorCount = 0
	d.constellationErrorMin = 0
	d.constellationErrorMax = 0
	d.constellationErrorSumSq = 0
}

// Quality implements modulation.Receiver.
func (d *Demodulator) Quality() modulation.QualityStats {
	stats := modulation.QualityStats{}

	// Calculate SNR
	if d.snrSampleCount > 0 && d.noisePowerSum > 0 {
		signalPower := d.signalPowerSum / float32(d.snrSampleCount)
		noisePower := d.noisePowerSum / float32(d.snrSampleCount)
		if noisePower > 0 {
			snrLinear := signalPower / noisePower
			if snrLinear > 0 {
				stats.SNR = 10.0 * math32.Log10(snrLinear)
			}
		}
	}

	// Calculate signal strength (raw signal magnitude before normalization)
	if d.rawSignalCount > 0 {
		avgMagnitude := d.rawSignalSum / float32(d.rawSignalCount)
		// Convert to power and then to dB: 10*log10(power)
		power := avgMagnitude * avgMagnitude
		if power > 0 {
			stats.SignalStrength = 10.0 * math32.Log10(power)
		}
	}

	// Phase error: not applicable for DSSS (no phase modulation)
	stats.PhaseError = modulation.Stats{}

	// Frequency error: not implemented (would require frequency offset estimation)
	stats.FrequencyError = modulation.Stats{}

	// Constellation error statistics
	if d.constellationErrorCount > 0 {
		avg := d.constellationErrorSum / float32(d.constellationErrorCount)
		var stdDev float32
		if d.constellationErrorCount > 1 {
			variance := (d.constellationErrorSumSq / float32(d.constellationErrorCount)) - (avg * avg)
			if variance > 0 {
				stdDev = math32.Sqrt(variance)
			}
		}
		stats.ConstellationError = modulation.Stats{
			Average:     avg,
			Min:         d.constellationErrorMin,
			Max:         d.constellationErrorMax,
			StdDev:      stdDev,
			SampleCount: d.constellationErrorCount,
		}
	}

	return stats
}

// Transmission implements modulation.Receiver.
func (d *Demodulator) Transmission() modulation.TransmissionStats {
	isLocked := d.preambleDetected && d.active && d.lockAcquiredIdx > 0

	return modulation.TransmissionStats{
		TransmissionStart: d.transmissionStartIdx,
		PacketStart:       d.packetStartIdx,
		PreambleIndex:     d.preambleStartIdx,
		IsLocked:          isLocked,
		LockinIndex:       d.lockAcquiredIdx,
		LockoutIndex:      d.lockLostIdx,
		LockCount:         d.lockCount,
		UnlockCount:       d.unlockCount,
	}
}

// Params implements modulation.Receiver.
func (d *Demodulator) Params() modulation.TransmitterParams {
	// Calculate signal strength in dB (power: 10*log10(power/reference))
	// Use expected amplitude from config
	power := d.config.Amplitude * d.config.Amplitude
	signalStrength := float32(0)
	if power > 0 {
		signalStrength = 10.0 * math32.Log10(power)
	}

	bandwidth := d.config.CalculateRequiredBandwidth()

	return modulation.TransmitterParams{
		BaudRate:         d.config.SymbolRate,
		SampleRate:       d.config.SampleRate,
		SignalStrength:   signalStrength,
		Bandwidth:        bandwidth,
		CarrierFrequency: d.config.CarrierFreq,
	}
}
