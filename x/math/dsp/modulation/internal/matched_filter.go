package internal

import (
	"io"

	"github.com/itohio/EasyRobot/x/buffer"
)

var _ Processor = (*MatchedFilter)(nil)

// MatchedFilter implements a matched filter integrator for symbol integration.
// It integrates samples over a symbol period using a pulse envelope (raised cosine, etc.).
type MatchedFilter struct {
	envelope       []float32 // Pulse shape envelope (one per symbol sample)
	envelopeEnergy float32   // Normalization factor
	samplesPerSymbol int

	// Buffering for symbol-period integration (needed for sample count changes)
	buffered      []complex64
	bufferedCount int
	batchSize     int // Batch size for reading from src

	// State for integration
	symbolCount int
}

// NewMatchedFilter creates a new matched filter integrator.
// envelope: pulse shape coefficients (e.g., raised cosine)
// samplesPerSymbol: number of samples per symbol period
func NewMatchedFilter(envelope []float32, samplesPerSymbol int) MatchedFilter {
	return NewMatchedFilterWithBatchSize(envelope, samplesPerSymbol, 16)
}

// NewMatchedFilterWithBatchSize creates a new matched filter integrator with custom batch size.
func NewMatchedFilterWithBatchSize(envelope []float32, samplesPerSymbol int, batchSize int) MatchedFilter {
	if samplesPerSymbol <= 0 {
		panic("samplesPerSymbol must be positive")
	}
	if batchSize <= 0 {
		batchSize = 16
	}
	if len(envelope) == 0 {
		// Default: rectangular window
		envelope = make([]float32, samplesPerSymbol)
		for i := range envelope {
			envelope[i] = 1.0
		}
	}

	// Compute envelope energy for normalization
	var energy float32
	for _, v := range envelope {
		energy += v * v
	}
	if energy == 0 {
		energy = 1
	}

	return MatchedFilter{
		envelope:         envelope,
		envelopeEnergy:   energy,
		samplesPerSymbol: samplesPerSymbol,
		buffered:         make([]complex64, samplesPerSymbol*2),
		batchSize:        batchSize,
	}
}

// Process implements Processor.
// Integrates samples over symbol periods and outputs one complex64 per symbol.
// Note: This processor changes sample count (N:1), so it reads from src (or dst if it's also a Reader) and writes to dst.
func (m *MatchedFilter) Process(dst buffer.Writer[complex64], src buffer.Reader[complex64]) int {
	if dst == nil {
		return m.estimate(src)
	}

	// Prefer reading from dst if it's also a Reader, otherwise use src
	reader := src
	if dstReader, ok := dst.(buffer.Reader[complex64]); ok {
		reader = dstReader
	}
	if reader == nil {
		return 0
	}

	total := 0
	readBuf := make([]complex64, m.batchSize)
	symbolSamples := m.samplesPerSymbol

	for {
		// Read samples until we have at least one complete symbol
		for m.bufferedCount < symbolSamples && dst.Len() < dst.Cap() {
			n, err := reader.Read(readBuf)
			if n > 0 {
				if m.bufferedCount+n > len(m.buffered) {
					newBuf := make([]complex64, (m.bufferedCount+n)*2)
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

		// Process complete symbols
		symbolsAvailable := m.bufferedCount / symbolSamples
		if symbolsAvailable == 0 {
			break
		}

		maxOutput := dst.Cap() - dst.Len()
		if symbolsAvailable > maxOutput {
			symbolsAvailable = maxOutput
		}
		if symbolsAvailable == 0 {
			break
		}

		// Integrate symbols
		integrated := make([]complex64, symbolsAvailable)
		for s := range symbolsAvailable {
			var iAcc, qAcc float32
			startIdx := s * symbolSamples

			// Integrate over symbol period with envelope
			for i := range symbolSamples {
				if startIdx+i < len(m.buffered) {
					sample := m.buffered[startIdx+i]
					envVal := m.envelope[i]
					iAcc += float32(real(sample)) * envVal
					qAcc += float32(imag(sample)) * envVal
				}
			}

			// Normalize by envelope energy
			inv := 1.0 / m.envelopeEnergy
			iAcc *= inv
			qAcc *= inv

			integrated[s] = complex64(complex(iAcc, qAcc))
		}

		// Write integrated symbols
		written, _ := dst.Write(integrated)
		total += written
		m.symbolCount += written

		// Shift remaining samples
		consumed := written * symbolSamples
		if consumed < m.bufferedCount {
			copy(m.buffered, m.buffered[consumed:m.bufferedCount])
			m.bufferedCount -= consumed
		} else {
			m.bufferedCount = 0
		}

		if written == 0 {
			break
		}

		if dst.Len() >= dst.Cap() {
			break
		}
	}

	return total
}

func (m *MatchedFilter) estimate(src buffer.Reader[complex64]) int {
	if src == nil {
		return 0
	}
	total := src.Len() + m.bufferedCount
	return total / m.samplesPerSymbol
}

// Reset clears the internal state.
func (m *MatchedFilter) Reset() {
	m.bufferedCount = 0
	m.symbolCount = 0
}

// SymbolCount returns the number of symbols processed since last reset.
func (m *MatchedFilter) SymbolCount() int {
	return m.symbolCount
}

// SamplesPerSymbol returns the number of samples per symbol.
func (m *MatchedFilter) SamplesPerSymbol() int {
	return m.samplesPerSymbol
}

