package mfsk

import (
	"errors"
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

var _ chain.Processor[float32, byte] = (*Modulator)(nil)

// Modulator converts bit symbols into MFSK samples.
type Modulator struct {
	cfg     Config
	tones   toneTable
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
		tones:   newToneTable(cfg),
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
		return 0
	}
	symbols := (totalBits + m.cfg.BitsPerSymbol - 1) / m.cfg.BitsPerSymbol
	return (m.cfg.PreambleLength + symbols) * m.cfg.SamplesPerSymbol
}

func (m *Modulator) nextSymbol(br buffer.BitsReader) (int, bool) {
	if m.preambleEmitted < m.cfg.PreambleLength {
		idx := 0
		if m.preambleEmitted&1 == 1 {
			idx = m.cfg.toneCount() - 1
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
	phase := m.tones.phases[idx]
	inc := m.tones.incr[idx]
	env := m.tones.envelope
	for i := range env {
		m.scratch[i] = env[i] * mathSin(phase)
		phase += inc
		if phase > twoPi {
			phase -= twoPi
		}
	}
	m.tones.phases[idx] = phase
}

// Reset clears internal state.
func (m *Modulator) Reset() {
	m.hasPending = false
	m.pendingSample = 0
	m.pendingSymbol = 0
	m.preambleEmitted = 0
	for i := range m.tones.phases {
		m.tones.phases[i] = 0
	}
}
