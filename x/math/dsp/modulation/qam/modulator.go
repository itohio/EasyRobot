package qam

import (
	"errors"
	"io"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
)

var (
	_ chain.Processor[float32, byte] = (*Modulator)(nil)
	_ modulation.Transmitter         = (*Modulator)(nil)
)

// Modulator converts bit symbols into M-QAM IQ samples.
type Modulator struct {
	cfg     Config
	table   constellationTable
	scratch []float32

	hasPending      bool
	pendingSymbol   int
	pendingSample   int
	preambleEmitted int

	tmp [1]byte
}

// NewModulator builds a streaming QAM modulator.
func NewModulator(cfg Config) (*Modulator, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &Modulator{
		cfg:     cfg,
		table:   newConstellationTable(cfg),
		scratch: make([]float32, cfg.SamplesPerSymbol*2),
	}, nil
}

// Process implements chain.Processor.
func (m *Modulator) Process(dst buffer.Writer[float32], src buffer.Reader[byte]) int {
	bitsReader, ok := src.(buffer.BitsReader)
	if !ok || bitsReader == nil {
		return 0
	}
	if bitsReader.BitsLength() != m.cfg.BitsPerSymbol {
		return 0
	}
	if dst == nil {
		return m.estimateSamples(bitsReader)
	}

	total := 0
	for dst.Len() < dst.Cap() {
		if !m.hasPending {
			symbol, ok := m.nextSymbol(bitsReader)
			if !ok {
				break
			}
			m.pendingSymbol = symbol
			m.pendingSample = 0
			m.hasPending = true
			m.renderSymbol(symbol)
		}

		written, _ := dst.Write(m.scratch[m.pendingSample:])
		total += written
		m.pendingSample += written
		if m.pendingSample >= len(m.scratch) {
			m.pendingSample = 0
			m.hasPending = false
		}
		if written == 0 {
			break
		}
	}

	return total
}

func (m *Modulator) estimateSamples(br buffer.BitsReader) int {
	totalBits := br.Bits()
	if totalBits <= 0 {
		return m.cfg.PreambleLength * m.cfg.SamplesPerSymbol * 2
	}
	symbols := (totalBits + m.cfg.BitsPerSymbol - 1) / m.cfg.BitsPerSymbol
	return (symbols + m.cfg.PreambleLength) * m.cfg.SamplesPerSymbol * 2
}

func (m *Modulator) nextSymbol(br buffer.BitsReader) (int, bool) {
	if m.preambleEmitted < m.cfg.PreambleLength {
		idx := 0
		if m.preambleEmitted&1 == 1 {
			idx = m.cfg.constellationSize() - 1
		}
		m.preambleEmitted++
		return idx, true
	}
	n, err := br.Read(m.tmp[:])
	if n == 0 {
		return 0, false
	}
	symbol := int(m.tmp[0]) & ((1 << m.cfg.BitsPerSymbol) - 1)
	if err != nil && !errors.Is(err, io.EOF) {
		return 0, false
	}
	return symbol, true
}

func (m *Modulator) renderSymbol(idx int) {
	point := m.table.points[idx]
	env := m.table.envelope
	for i := range env {
		base := i * 2
		scale := env[i]
		m.scratch[base] = scale * point.i
		m.scratch[base+1] = scale * point.q
	}
}

// Reset clears the streaming state.
func (m *Modulator) Reset() {
	m.hasPending = false
	m.pendingSample = 0
	m.pendingSymbol = 0
	m.preambleEmitted = 0
}

// Params implements modulation.Transmitter.
func (m *Modulator) Params() modulation.TransmitterParams {
	power := m.cfg.Amplitude * m.cfg.Amplitude
	signalStrength := float32(0)
	if power > 0 {
		signalStrength = 10.0 * math32.Log10(power)
	}
	return modulation.TransmitterParams{
		BaudRate:        m.cfg.SymbolRate(),
		SampleRate:       m.cfg.SampleRate,
		SignalStrength:   signalStrength,
		Bandwidth:        m.cfg.RequiredBandwidth(),
		CarrierFrequency: m.cfg.CarrierHz,
	}
}
