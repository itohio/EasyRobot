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
		// Optimized preamble pattern for fast PLL lock:
		// 1. First M symbols: cover all M phases (0, 1, 2, 3 for QPSK) - resolves ambiguity
		// 2. Next symbols: maximize phase transitions to help PLL track quickly
		// Pattern: 0, 1, 2, 3, 0, 2, 1, 3, 0, 1, 2, 3, ...
		// This provides: all phases covered quickly, then good transitions (0→2=180°, 1→3=180°)
		// followed by adjacent transitions (0→1=90°, 2→3=90°)
		idx := m.preamblePattern(m.preambleEmitted)
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
	m.renderSymbol(symbol)
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

// preamblePattern returns an optimized preamble symbol index for fast PLL lock.
// Pattern design:
// - First M symbols: cover all M phases (0, 1, 2, 3 for QPSK) - resolves phase ambiguity
// - Next symbols: maximize phase transitions to help PLL track quickly
// - For QPSK: 0, 1, 2, 3, 0, 2, 1, 3, 0, 1, 2, 3, ...
//   This provides: all phases covered quickly, then good transitions (0→2=180°, 1→3=180°)
//   followed by adjacent transitions (0→1=90°, 2→3=90°)
func (m *Modulator) preamblePattern(index int) int {
	M := m.cfg.constellationSize()
	
	// First M symbols: cover all phases (0, 1, 2, 3 for QPSK)
	if index < M {
		return index
	}
	
	// After covering all phases, use optimized pattern for fast PLL tracking
	// Pattern repeats every 2*M symbols: [0, 1, 2, 3, 0, 2, 1, 3] for QPSK
	// This maximizes phase transitions: 0→2 (180°), 1→3 (180°), then 0→1 (90°), 2→3 (90°)
	patternIndex := (index - M) % (2 * M)
	if patternIndex < M {
		// First half: sequential 0, 1, 2, 3
		return patternIndex
	} else {
		// Second half: optimized transitions 0, 2, 1, 3
		// Maps: 0→0, 1→2, 2→1, 3→3
		offset := patternIndex - M
		switch offset {
		case 0:
			return 0
		case 1:
			return (M / 2) % M // 2 for QPSK
		case 2:
			return 1
		case 3:
			return (M/2 + 1) % M // 3 for QPSK
		default:
			// For M > 4, use a pattern that maximizes phase differences
			if offset%2 == 0 {
				return offset / 2
			} else {
				return (offset/2 + M/2) % M
			}
		}
	}
}

// Reset clears internal state.
func (m *Modulator) Reset() {
	m.hasPending = false
	m.pendingSample = 0
	m.pendingSymbol = 0
	m.preambleEmitted = 0
}
