package mpsk

import (
	"errors"
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

var _ chain.Processor[float32, byte] = (*Modulator)(nil)

// Modulator converts bit symbols into MPSK samples.
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

// NewModulator builds a streaming modulator from the supplied configuration.
func NewModulator(cfg Config) (*Modulator, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	m := &Modulator{
		cfg:     cfg,
		table:   newConstellationTable(cfg),
		scratch: make([]float32, cfg.SamplesPerSymbol*2),
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
		if m.pendingSample >= m.cfg.SamplesPerSymbol*2 {
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
	return (symbols + m.cfg.PreambleLength) * m.cfg.SamplesPerSymbol * 2
}

func (m *Modulator) nextSymbol(br buffer.BitsReader) (int, bool) {
	if m.preambleEmitted < m.cfg.PreambleLength {
		idx := 0
		if m.preambleEmitted&1 == 1 {
			idx = m.cfg.constellationSize() / 2
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
	env := m.table.envelope
	c := m.table.cos[idx]
	s := m.table.sin[idx]
	for i := range env {
		base := i * 2
		val := env[i]
		m.scratch[base] = val * c
		m.scratch[base+1] = val * s
	}
}

// Reset clears internal state.
func (m *Modulator) Reset() {
	m.hasPending = false
	m.pendingSample = 0
	m.pendingSymbol = 0
	m.preambleEmitted = 0
}
