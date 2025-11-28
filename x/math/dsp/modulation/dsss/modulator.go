package dsss

import (
	"errors"
	"fmt"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
)

var (
	_ chain.Processor[float32, byte] = (*Modulator)(nil)
	_ modulation.Transmitter         = (*Modulator)(nil)
)

// Modulator converts bit symbols into DSSS passband samples.
type Modulator struct {
	config Config

	carrier    dsp.CosineGenerator
	pulse      []float32
	chipBuffer []float32

	hasPending    bool
	pendingSymbol int
	pendingChip   int
}

func NewModulator(config Config) (*Modulator, error) {
	if config.SpreadingFactor <= 0 {
		return nil, errors.New("dsss: spreading factor must be positive")
	}
	if config.SamplesPerChip == 0 {
		config.SamplesPerChip = config.CalculateSamplesPerChip(config.SampleRate)
	}
	if config.SamplesPerChip <= 0 {
		return nil, errors.New("dsss: samples per chip must be positive")
	}
	if err := config.ValidateSampleRate(config.SampleRate); err != nil {
		return nil, err
	}
	if len(config.Codes) == 0 {
		return nil, errors.New("dsss: at least one spreading code is required")
	}

	pulse := generatePulseShape(&config)
	if len(pulse) == 0 {
		pulse = generateRectangular(config.SamplesPerChip)
	}
	if len(pulse) != config.SamplesPerChip {
		return nil, fmt.Errorf("dsss: pulse shape must have %d samples, got %d", config.SamplesPerChip, len(pulse))
	}

	return &Modulator{
		config:     config,
		carrier:    dsp.NewCosineGenerator(config.SampleRate, config.CarrierFreq, 1.0, 0.0),
		pulse:      pulse,
		chipBuffer: make([]float32, config.SamplesPerChip),
	}, nil
}

func (m *Modulator) Process(dst buffer.Writer[float32], src buffer.Reader[byte]) int {
	bitsReader, ok := src.(buffer.BitsReader)
	if !ok || bitsReader == nil {
		return 0
	}
	if bitsReader.BitsLength() != m.config.BitsPerSymbol {
		return 0
	}
	if dst == nil {
		return m.estimateSamples(bitsReader)
	}

	totalSamples := 0
	for {
		if !m.hasPending {
			symbol, ok := m.nextSymbol(bitsReader)
			if !ok {
				break
			}
			m.pendingSymbol = symbol
			m.pendingChip = 0
			m.hasPending = true
		}

		written, completed := m.emitPendingSymbol(dst)
		totalSamples += written
		if !completed {
			break
		}
	}
	return totalSamples
}

func (m *Modulator) nextSymbol(reader buffer.Reader[byte]) (int, bool) {
	var buf [1]byte
	n, err := reader.Read(buf[:])
	if err != nil && n == 0 {
		return 0, false
	}
	if n < m.config.BitsPerSymbol {
		return 0, false
	}
	return int(buf[0]), true
}

func (m *Modulator) emitPendingSymbol(dst buffer.Writer[float32]) (int, bool) {
	samplesPerChip := m.config.SamplesPerChip
	samplesPerSymbol := samplesPerChip * m.config.SpreadingFactor
	written := 0

	for m.pendingChip < m.config.SpreadingFactor {
		chipValue := m.chipValue(m.pendingSymbol, m.pendingChip)
		m.renderChip(chipValue)

		n, err := dst.Write(m.chipBuffer)
		written += n
		if err != nil || n < len(m.chipBuffer) {
			return written, false
		}
		m.pendingChip++
	}

	if written < samplesPerSymbol {
		return written, false
	}

	m.hasPending = false
	return written, true
}

func (m *Modulator) chipValue(symbol, chipIdx int) float32 {
	chipIdx %= m.config.SpreadingFactor
	if m.config.ModulationOrder == 2 {
		code := m.config.Codes[0][chipIdx]
		if symbol == 0 {
			return float32(-code)
		}
		return float32(code)
	}

	if symbol < 0 || symbol >= len(m.config.Codes) {
		symbol = symbol % len(m.config.Codes)
		if symbol < 0 {
			symbol += len(m.config.Codes)
		}
	}
	return float32(m.config.Codes[symbol][chipIdx])
}

func (m *Modulator) renderChip(chipValue float32) {
	for i := range m.chipBuffer {
		m.chipBuffer[i] = chipValue * m.pulse[i] * m.config.Amplitude
	}
	m.carrier.GenerateWithOp(m.chipBuffer, false, func(dst *float32, carrierValue float32) {
		*dst *= carrierValue
	})
}

func (m *Modulator) estimateSamples(br buffer.BitsReader) int {
	totalBits := br.Bits()
	if totalBits <= 0 {
		return 0
	}
	symbols := (totalBits + m.config.BitsPerSymbol - 1) / m.config.BitsPerSymbol
	samplesPerSymbol := m.config.SpreadingFactor * m.config.SamplesPerChip
	// DSSS doesn't use preamble - code correlation provides sync
	return symbols * samplesPerSymbol
}

func (m *Modulator) Reset() {
	m.hasPending = false
	m.pendingSymbol = 0
	m.pendingChip = 0
	m.carrier = dsp.NewCosineGenerator(m.config.SampleRate, m.config.CarrierFreq, 1.0, 0.0)
}

// Params implements modulation.Transmitter.
func (m *Modulator) Params() modulation.TransmitterParams {
	// Calculate signal strength in dB (power: 10*log10(power/reference))
	// Reference is normalized to 1.0, so power = amplitude^2
	power := m.config.Amplitude * m.config.Amplitude
	signalStrength := float32(0)
	if power > 0 {
		signalStrength = 10.0 * math32.Log10(power)
	}

	bandwidth := m.config.CalculateRequiredBandwidth()

	return modulation.TransmitterParams{
		BaudRate:         m.config.SymbolRate,
		SampleRate:       m.config.SampleRate,
		SignalStrength:   signalStrength,
		Bandwidth:        bandwidth,
		CarrierFrequency: m.config.CarrierFreq,
	}
}
