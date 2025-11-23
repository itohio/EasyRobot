package modulation

// SymbolSynchronizer handles symbol boundary detection and synchronization
// Reusable component for all digital demodulators
type SymbolSynchronizer struct {
	samplesPerSymbol int
	sampleBuffer     []float32
}

// NewSymbolSynchronizer creates a new symbol synchronizer
func NewSymbolSynchronizer(samplesPerSymbol int) SymbolSynchronizer {
	return SymbolSynchronizer{
		samplesPerSymbol: samplesPerSymbol,
		sampleBuffer:     make([]float32, 0, samplesPerSymbol*2),
	}
}

// AddSamples adds samples to the buffer
func (ss *SymbolSynchronizer) AddSamples(samples []float32) {
	ss.sampleBuffer = append(ss.sampleBuffer, samples...)
}

// ExtractSymbol extracts one complete symbol if available
// Returns the symbol samples and true if a symbol was extracted
func (ss *SymbolSynchronizer) ExtractSymbol() ([]float32, bool) {
	if len(ss.sampleBuffer) < ss.samplesPerSymbol {
		return nil, false
	}
	
	symbolSamples := ss.sampleBuffer[:ss.samplesPerSymbol]
	ss.sampleBuffer = ss.sampleBuffer[ss.samplesPerSymbol:]
	
	return symbolSamples, true
}

// HasCompleteSymbol returns true if there's at least one complete symbol
func (ss *SymbolSynchronizer) HasCompleteSymbol() bool {
	return len(ss.sampleBuffer) >= ss.samplesPerSymbol
}

// Reset resets the synchronizer state
func (ss *SymbolSynchronizer) Reset() {
	ss.sampleBuffer = ss.sampleBuffer[:0]
}

// GetRemainingSamples returns the number of remaining samples in the buffer
func (ss *SymbolSynchronizer) GetRemainingSamples() int {
	return len(ss.sampleBuffer)
}

