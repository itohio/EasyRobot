package modulation

import (
	"testing"
)

// TestSymbolSynchronizer_Basic tests basic symbol extraction
func TestSymbolSynchronizer_Basic(t *testing.T) {
	samplesPerSymbol := 48
	sync := NewSymbolSynchronizer(samplesPerSymbol)

	// Create samples for exactly one symbol
	samples := make([]float32, samplesPerSymbol)
	for i := range samples {
		samples[i] = float32(i)
	}

	sync.AddSamples(samples)

	if !sync.HasCompleteSymbol() {
		t.Error("Should have complete symbol")
	}

	extracted, ok := sync.ExtractSymbol()
	if !ok {
		t.Error("Should extract symbol")
	}

	if len(extracted) != samplesPerSymbol {
		t.Errorf("Expected %d samples, got %d", samplesPerSymbol, len(extracted))
	}

	// Verify samples match
	for i := 0; i < samplesPerSymbol; i++ {
		if extracted[i] != samples[i] {
			t.Errorf("Sample mismatch at index %d: expected %f, got %f", i, samples[i], extracted[i])
		}
	}

	if sync.HasCompleteSymbol() {
		t.Error("Should not have complete symbol after extraction")
	}
}

// TestSymbolSynchronizer_MultipleSymbols tests extracting multiple symbols
func TestSymbolSynchronizer_MultipleSymbols(t *testing.T) {
	samplesPerSymbol := 48
	sync := NewSymbolSynchronizer(samplesPerSymbol)

	// Create samples for 3 symbols
	numSymbols := 3
	samples := make([]float32, samplesPerSymbol*numSymbols)
	for i := range samples {
		samples[i] = float32(i)
	}

	sync.AddSamples(samples)

	// Extract all symbols
	for i := 0; i < numSymbols; i++ {
		if !sync.HasCompleteSymbol() {
			t.Errorf("Should have complete symbol %d", i)
		}

		extracted, ok := sync.ExtractSymbol()
		if !ok {
			t.Errorf("Should extract symbol %d", i)
		}

		if len(extracted) != samplesPerSymbol {
			t.Errorf("Symbol %d: expected %d samples, got %d", i, samplesPerSymbol, len(extracted))
		}

		// Verify samples match
		expectedStart := i * samplesPerSymbol
		for j := 0; j < samplesPerSymbol; j++ {
			expected := samples[expectedStart+j]
			if extracted[j] != expected {
				t.Errorf("Symbol %d, sample %d: expected %f, got %f", i, j, expected, extracted[j])
			}
		}
	}

	if sync.HasCompleteSymbol() {
		t.Error("Should not have complete symbol after extracting all")
	}

	remaining := sync.GetRemainingSamples()
	if remaining != 0 {
		t.Errorf("Expected 0 remaining samples, got %d", remaining)
	}
}

// TestSymbolSynchronizer_PartialSymbol tests handling of partial symbols
func TestSymbolSynchronizer_PartialSymbol(t *testing.T) {
	samplesPerSymbol := 48
	sync := NewSymbolSynchronizer(samplesPerSymbol)

	// Add less than one symbol
	partialSamples := samplesPerSymbol - 10
	samples := make([]float32, partialSamples)
	for i := range samples {
		samples[i] = float32(i)
	}

	sync.AddSamples(samples)

	if sync.HasCompleteSymbol() {
		t.Error("Should not have complete symbol with partial samples")
	}

	extracted, ok := sync.ExtractSymbol()
	if ok {
		t.Error("Should not extract symbol from partial samples")
	}

	if extracted != nil {
		t.Error("Extracted should be nil for partial symbol")
	}

	remaining := sync.GetRemainingSamples()
	if remaining != partialSamples {
		t.Errorf("Expected %d remaining samples, got %d", partialSamples, remaining)
	}

	// Add more samples to complete the symbol
	additionalSamples := make([]float32, 10)
	for i := range additionalSamples {
		additionalSamples[i] = float32(partialSamples + i)
	}

	sync.AddSamples(additionalSamples)

	if !sync.HasCompleteSymbol() {
		t.Error("Should have complete symbol after adding remaining samples")
	}

	extracted, ok = sync.ExtractSymbol()
	if !ok {
		t.Error("Should extract symbol after completing it")
	}

	if len(extracted) != samplesPerSymbol {
		t.Errorf("Expected %d samples, got %d", samplesPerSymbol, len(extracted))
	}
}

// TestSymbolSynchronizer_ChunkedInput tests adding samples in chunks
func TestSymbolSynchronizer_ChunkedInput(t *testing.T) {
	samplesPerSymbol := 48
	sync := NewSymbolSynchronizer(samplesPerSymbol)

	// Create samples for 2.5 symbols
	totalSamples := samplesPerSymbol*2 + samplesPerSymbol/2
	allSamples := make([]float32, totalSamples)
	for i := range allSamples {
		allSamples[i] = float32(i)
	}

	// Add samples in chunks
	chunk1 := allSamples[:samplesPerSymbol]
	chunk2 := allSamples[samplesPerSymbol : samplesPerSymbol*2]
	chunk3 := allSamples[samplesPerSymbol*2:]

	sync.AddSamples(chunk1)
	if !sync.HasCompleteSymbol() {
		t.Error("Should have complete symbol after first chunk")
	}

	extracted1, ok := sync.ExtractSymbol()
	if !ok {
		t.Error("Should extract first symbol")
	}
	if len(extracted1) != samplesPerSymbol {
		t.Errorf("Expected %d samples in first symbol, got %d", samplesPerSymbol, len(extracted1))
	}

	sync.AddSamples(chunk2)
	if !sync.HasCompleteSymbol() {
		t.Error("Should have complete symbol after second chunk")
	}

	extracted2, ok := sync.ExtractSymbol()
	if !ok {
		t.Error("Should extract second symbol")
	}
	if len(extracted2) != samplesPerSymbol {
		t.Errorf("Expected %d samples in second symbol, got %d", samplesPerSymbol, len(extracted2))
	}

	sync.AddSamples(chunk3)
	if sync.HasCompleteSymbol() {
		t.Error("Should not have complete symbol with only partial third chunk")
	}

	remaining := sync.GetRemainingSamples()
	expectedRemaining := samplesPerSymbol / 2
	if remaining != expectedRemaining {
		t.Errorf("Expected %d remaining samples, got %d", expectedRemaining, remaining)
	}
}

// TestSymbolSynchronizer_Reset tests reset functionality
func TestSymbolSynchronizer_Reset(t *testing.T) {
	samplesPerSymbol := 48
	sync := NewSymbolSynchronizer(samplesPerSymbol)

	// Add some samples
	samples := make([]float32, samplesPerSymbol+10)
	for i := range samples {
		samples[i] = float32(i)
	}

	sync.AddSamples(samples)

	if !sync.HasCompleteSymbol() {
		t.Error("Should have complete symbol")
	}

	remaining := sync.GetRemainingSamples()
	if remaining != samplesPerSymbol+10 {
		t.Errorf("Expected %d remaining samples, got %d", samplesPerSymbol+10, remaining)
	}

	sync.Reset()

	if sync.HasCompleteSymbol() {
		t.Error("Should not have complete symbol after reset")
	}

	remaining = sync.GetRemainingSamples()
	if remaining != 0 {
		t.Errorf("Expected 0 remaining samples after reset, got %d", remaining)
	}
}
