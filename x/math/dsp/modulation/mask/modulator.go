package mask

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
	_ chain.Processor[float32, byte] = (*Modulator)(nil)
	_ modulation.Transmitter         = (*Modulator)(nil)
)

// Modulator converts bit symbols into M-ASK samples.
type Modulator struct {
	cfg     Config
	table   amplitudeTable
	carrier dsp.CosineGenerator
	scratch []float32

	hasPending      bool
	pendingSymbol   int
	pendingSample   int
	preambleEmitted int

	tmp [1]byte
}

// NewModulator builds a streaming modulator from the supplied configuration.
func NewModulator(cfg Config) (*Modulator, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	m := &Modulator{
		cfg:     cfg,
		table:   newAmplitudeTable(cfg),
		carrier: dsp.NewCosineGenerator(cfg.SampleRate, cfg.CarrierHz, 1.0, 0.0),
		scratch: make([]float32, cfg.SamplesPerSymbol),
	}
	return m, nil
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
			sym, ok := m.nextSymbol(bitsReader)
			if !ok {
				break
			}
			m.pendingSymbol = sym
			m.pendingSample = 0
			m.hasPending = true
			m.renderSymbol(sym)
		}

		written, _ := dst.Write(m.scratch[m.pendingSample:])
		total += written
		m.pendingSample += written
		if m.pendingSample >= m.cfg.SamplesPerSymbol {
			m.hasPending = false
			m.pendingSample = 0
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
		return m.cfg.PreambleLength * m.cfg.SamplesPerSymbol
	}
	symbols := (totalBits + m.cfg.BitsPerSymbol - 1) / m.cfg.BitsPerSymbol
	return (symbols + m.cfg.PreambleLength) * m.cfg.SamplesPerSymbol
}

func (m *Modulator) nextSymbol(br buffer.BitsReader) (int, bool) {
	if m.preambleEmitted < m.cfg.PreambleLength {
		pattern := m.cfg.PreamblePattern
		if len(pattern) == 0 {
			// Default: highest symbol (M-1), lowest symbol (0) alternating
			M := m.cfg.constellationSize()
			pattern = []int{M - 1, 0}
		}
		idx := pattern[m.preambleEmitted%len(pattern)]
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

func (m *Modulator) renderSymbol(symbolIdx int) {
	if symbolIdx < 0 || symbolIdx >= len(m.table.levels) {
		symbolIdx = 0
	}
	amplitude := m.table.levels[symbolIdx]
	env := m.table.envelope

	// Apply amplitude and envelope
	for i := range env {
		m.scratch[i] = amplitude * env[i]
	}

	// Apply carrier for passband modulation
	if m.cfg.CarrierHz > 0 {
		m.carrier.GenerateWithOp(m.scratch, false, func(dst *float32, carrierValue float32) {
			*dst *= carrierValue
		})
	}
}

// Reset clears internal state.
func (m *Modulator) Reset() {
	m.hasPending = false
	m.pendingSample = 0
	m.pendingSymbol = 0
	m.preambleEmitted = 0
	m.carrier = dsp.NewCosineGenerator(m.cfg.SampleRate, m.cfg.CarrierHz, 1.0, 0.0)
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

