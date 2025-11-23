package mfsk

import (
	"math"

	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
	"github.com/itohio/EasyRobot/x/math/filter"
	agcFilter "github.com/itohio/EasyRobot/x/math/filter/agc"
	"github.com/itohio/EasyRobot/x/math/filter/fir"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// Demodulator handles Frequency Shift Keying demodulation
// Uses reusable components: FrequencyDetector, SymbolSynchronizer, BitExtractor
type Demodulator struct {
	sampleRate  float32
	frequencies []float32 // Frequencies for each symbol
	symbolRate  float32

	// Reusable components (SOLID principle - single responsibility)
	frequencyDetector  modulation.FrequencyDetector
	symbolSynchronizer *modulation.SymbolSynchronizer
	bitExtractor       *modulation.BitExtractor

	// Low-pass filter to remove noise before frequency detection
	lowPassFilter filter.Processor[float32]

	// Signal detection and AGC
	signalDetector *modulation.SignalDetector
	agc            *agcFilter.Filter

	// State
	samplesPerSymbol int
	modulationOrder  int
	bitsPerSymbol    int

	// Lock state
	locked bool
}

// NewDemodulator creates a new FSK demodulator
func NewDemodulator(sampleRate, freqLow, freqHigh, symbolRate float32) *Demodulator {
	return NewMDemodulator(sampleRate, symbolRate, freqLow, freqHigh)
}

// NewMDemodulator creates a new M-FSK demodulator with multiple frequencies
func NewMDemodulator(sampleRate, symbolRate float32, frequencies ...float32) *Demodulator {
	if len(frequencies) < 2 {
		panic("FSK decoder requires at least 2 frequencies")
	}

	samplesPerSymbol := int(sampleRate / symbolRate)
	modulationOrder := len(frequencies)
	bitsPerSymbol := int(math.Log2(float64(modulationOrder)))

	// Create reusable components (SOLID principle)
	// Use Goertzel with proper low-pass filtering (better than naive I/Q correlation)
	frequencyDetector := modulation.NewGoertzelFrequencyDetector(sampleRate, frequencies...)
	symbolSynchronizer := modulation.NewSymbolSynchronizer(samplesPerSymbol)
	bitExtractor := modulation.NewBitExtractor()

	// Create low-pass filter to remove noise before frequency detection
	// Use a very high cutoff (close to Nyquist) to minimize distortion of FSK frequencies
	// The filter will still help with high-frequency noise but won't affect the signal much
	maxFreq := float32(0)
	for _, freq := range frequencies {
		if freq > maxFreq {
			maxFreq = freq
		}
	}
	// Use a very high cutoff (4x the highest frequency or 80% of Nyquist, whichever is lower)
	// This minimizes distortion of FSK frequencies while still filtering high-frequency noise
	cutoffFreq := maxFreq * 4.0
	nyquist := sampleRate / 2.0
	if cutoffFreq > nyquist*0.8 {
		cutoffFreq = nyquist * 0.8 // 80% of Nyquist - very high, minimal distortion
	}
	// Use a lower order filter to reduce phase distortion
	lowPassFilter := fir.NewLowPass(16, cutoffFreq, sampleRate)

	// Initialize signal detector
	// Use a larger window and more selective threshold to avoid locking onto noise
	// The threshold is set higher to require stronger signal (FSK-like patterns)
	detectorWindow := samplesPerSymbol * 4 // Larger window for better noise rejection
	signalDetector := modulation.NewSignalDetector(detectorWindow, modulation.MinSignalEnergy*10)
	// Require minimum 4dB SNR to lock (signal must be ~2.5x stronger than noise)
	// This prevents locking onto noise while still allowing signal detection
	signalDetector.SetMinSNR(4.0)

	// Initialize AGC
	agc := agcFilter.New(0.5, 0.01) // 0.01 is AGC time constant
	agc.SetSampleRate(sampleRate)

	return &Demodulator{
		sampleRate:         sampleRate,
		frequencies:        frequencies,
		symbolRate:         symbolRate,
		frequencyDetector:  frequencyDetector,
		symbolSynchronizer: symbolSynchronizer,
		bitExtractor:       bitExtractor,
		lowPassFilter:      lowPassFilter,
		samplesPerSymbol:   samplesPerSymbol,
		modulationOrder:    modulationOrder,
		bitsPerSymbol:      bitsPerSymbol,
		signalDetector:     signalDetector,
		agc:                agc,
	}
}

// SetFilter sets a filter for frequency detection (if detector supports it)
// Note: Filters should be applied to signal before frequency detection, not to magnitudes
// SetFilter sets the low-pass filter for demodulation
//
// Deprecated: This method is not part of the Demodulator interface.
// Filters should be applied to the input signal before calling Decode.
func (fsk *Demodulator) SetFilter(f filter.Processor[float32]) {
	// Filtering is not directly supported in the new component-based design
	// Filters should be applied to the input signal before calling Decode
	_ = f
}

// Reset resets the demodulator state
func (fsk *Demodulator) Reset() {
	fsk.locked = false
	fsk.frequencyDetector.Reset()
	fsk.symbolSynchronizer.Reset()
	fsk.bitExtractor.Reset()
	if fsk.lowPassFilter != nil {
		fsk.lowPassFilter.Reset()
	}
	if fsk.signalDetector != nil {
		fsk.signalDetector.Reset()
	}
	if fsk.agc != nil {
		fsk.agc.Reset()
	}
}

// IsLocked returns true if demodulator is locked onto signal
//
// Deprecated: This method is not part of the Demodulator interface.
func (fsk *Demodulator) IsLocked() bool {
	return fsk.locked && fsk.signalDetector != nil && fsk.signalDetector.IsLocked()
}

// FrequencyDetector returns the frequency detector (for debugging)
func (fsk *Demodulator) FrequencyDetector() modulation.FrequencyDetector {
	return fsk.frequencyDetector
}

// SymbolSynchronizer returns the symbol synchronizer (for debugging)
func (fsk *Demodulator) SymbolSynchronizer() *modulation.SymbolSynchronizer {
	return fsk.symbolSynchronizer
}

// BitExtractor returns the bit extractor (for debugging)
func (fsk *Demodulator) BitExtractor() *modulation.BitExtractor {
	return fsk.bitExtractor
}

// SignalDetector returns the signal detector (for debugging)
func (fsk *Demodulator) SignalDetector() *modulation.SignalDetector {
	return fsk.signalDetector
}

// Demodulate demodulates FSK signal into symbols
//
// Deprecated: This method is not part of the Demodulator interface.
func (fsk *Demodulator) Demodulate(signal vecTypes.Vector) vec.Vector {
	signalData := []float32(signal.(vec.Vector))

	// Apply AGC if enabled
	if fsk.agc != nil {
		for i := range signalData {
			signalData[i] = fsk.agc.Process(signalData[i])
		}
	}

	numSymbols := len(signalData) / fsk.samplesPerSymbol
	symbols := vec.New(numSymbols)
	symbolData := []float32(symbols)

	// Reset frequency detector for fresh start
	fsk.frequencyDetector.Reset()

	// Process each symbol
	for i := 0; i < numSymbols; i++ {
		startSample := i * fsk.samplesPerSymbol
		endSample := (i + 1) * fsk.samplesPerSymbol
		if endSample > len(signalData) {
			endSample = len(signalData)
		}

		symbolSamples := signalData[startSample:endSample]

		// Use frequency detector (phase-independent)
		freqIdx, _ := fsk.frequencyDetector.DetectFrequency(symbolSamples)

		// Map frequency index to symbol value
		symbolData[i] = float32(freqIdx)
	}

	return symbols
}

// ProcessChunk processes a chunk of samples for streaming decoding
//
// Deprecated: This method is not part of the Demodulator interface.
// Use Decode instead.
func (fsk *Demodulator) ProcessChunk(samples []float32, output []byte) (bytesDecoded int, hasMore bool) {
	// Apply AGC and signal detection
	for i := range samples {
		if fsk.agc != nil {
			samples[i] = fsk.agc.Process(samples[i])
		}
		if fsk.signalDetector != nil {
			fsk.signalDetector.Update(samples[i])
			fsk.locked = fsk.signalDetector.IsLocked()
		}
	}

	// If not locked, don't decode
	if !fsk.locked {
		return 0, false
	}

	// Add samples to symbol synchronizer
	fsk.symbolSynchronizer.AddSamples(samples)

	// Process complete symbols
	bytesDecoded = 0

	for fsk.symbolSynchronizer.HasCompleteSymbol() {
		// Extract one symbol using synchronizer
		symbolSamples, ok := fsk.symbolSynchronizer.ExtractSymbol()
		if !ok {
			break
		}

		// Detect frequency using frequency detector
		freqIdx, _ := fsk.frequencyDetector.DetectFrequency(symbolSamples)

		// Add symbol to bit extractor
		fsk.bitExtractor.AddSymbol(freqIdx, fsk.bitsPerSymbol)

		// Extract bytes from bit extractor
		remaining := output[bytesDecoded:]
		extracted := fsk.bitExtractor.ExtractBytes(remaining)
		bytesDecoded += extracted
	}

	return bytesDecoded, fsk.symbolSynchronizer.HasCompleteSymbol()
}

// Decode decodes amplitude vector into bytes
// Implements modulation.Demodulator interface
// Returns the number of bytes written
// Returns expected data length if data is nil
func (fsk *Demodulator) Decode(data []byte, amplitudes vecTypes.Vector) int {
	if data == nil {
		// Calculate expected length
		bitsPerSymbol := int(math.Log2(float64(fsk.modulationOrder)))
		// Estimate based on available samples in synchronizer
		symbolsAvailable := 0
		if fsk.symbolSynchronizer != nil {
			// Rough estimate - actual depends on internal buffer state
			symbolsAvailable = 100 // Conservative estimate
		}
		bitsAvailable := symbolsAvailable * bitsPerSymbol
		return bitsAvailable / 8
	}
	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)

	// Apply AGC and signal detection
	wasLocked := fsk.locked
	everLocked := fsk.locked // Track if we were ever locked during this decode call
	if fsk.agc != nil || fsk.signalDetector != nil {
		for i := range ampData {
			if fsk.agc != nil {
				ampData[i] = fsk.agc.Process(ampData[i])
			}
			if fsk.signalDetector != nil {
				fsk.signalDetector.Update(ampData[i])
				fsk.locked = fsk.signalDetector.IsLocked()
				if fsk.locked {
					everLocked = true // Remember we were locked
				}
			}
		}
	}

	// If we just unlocked, clear buffers to avoid partial symbols
	if wasLocked && !fsk.locked {
		// Signal just ended - clear buffers to avoid partial decoding
		fsk.symbolSynchronizer.Reset()
		fsk.bitExtractor.Reset()
		fsk.frequencyDetector.Reset()
		if fsk.lowPassFilter != nil {
			fsk.lowPassFilter.Reset()
		}
	}

	// Apply low-pass filter to remove noise before frequency detection
	// This is critical for robust FSK demodulation with Goertzel
	if fsk.lowPassFilter != nil {
		for i := range ampData {
			ampData[i] = fsk.lowPassFilter.Process(ampData[i])
		}
	}

	// Always add samples to symbol synchronizer (even if not locked yet)
	// This allows signal detector to accumulate samples and lock
	// We'll only decode symbols when locked
	fsk.symbolSynchronizer.AddSamples(ampData)

	// Only decode symbols if locked (signal detector must lock first)
	// Use everLocked to handle cases where detector locks then unlocks due to trailing zeros
	// This happens with signals that have padding - detector locks during signal, unlocks on zeros
	if !fsk.locked && !everLocked {
		return 0
	}
	// If we were ever locked, process symbols even if currently unlocked (trailing zeros case)

	// Process complete symbols
	bytesWritten := 0
	symbolsAddedToExtractor := 0
	symbolsExtractedFromSync := 0

	for fsk.symbolSynchronizer.HasCompleteSymbol() {
		symbolsExtractedFromSync++
		// Extract one symbol using synchronizer
		symbolSamples, ok := fsk.symbolSynchronizer.ExtractSymbol()
		if !ok {
			break
		}

		// Detect frequency using Goertzel (phase-independent, efficient)
		// Low-pass filter was already applied to symbolSamples via ampData
		freqIdx, magnitude := fsk.frequencyDetector.DetectFrequency(symbolSamples)

		// Check if detection is valid (freqIdx == -1 indicates noise/ambiguous)
		// With proper low-pass filtering, the detector should be more reliable
		if freqIdx < 0 {
			// Ambiguous detection or noise - use lowest frequency as default
			freqIdx = 0
		}

		// Additional magnitude check (with proper filtering, valid signals should be stronger)
		// Use adaptive threshold based on SNR - lower threshold for noisy signals
		// For clean signals (>20dB SNR), use 0.05 threshold
		// For noisy signals (<20dB SNR), use lower threshold (0.02)
		magnitudeThreshold := float32(0.05)
		if magnitude < 0.1 {
			// Low magnitude - might be noisy signal, use lower threshold
			magnitudeThreshold = float32(0.02)
		}

		if magnitude < magnitudeThreshold {
			// Very low magnitude even after filtering - likely noise
			// Skip this symbol but continue processing (signal detector is locked)
			continue
		}

		// Decode symbol - decoder doesn't care about preamble, higher level code handles that
		// Add symbol to bit extractor
		// DO NOT extract bytes here - wait until all symbols are processed
		// Extracting bytes incrementally causes bit misalignment
		fsk.bitExtractor.AddSymbol(freqIdx, fsk.bitsPerSymbol)
		symbolsAddedToExtractor++
	}

	// After processing all available symbols, extract bytes from bit extractor
	// Extract bytes only after all symbols are processed to avoid bit misalignment
	// Keep extracting until buffer is full or no more bytes available
	for fsk.bitExtractor.HasCompleteByte() && bytesWritten < len(data) {
		remaining := data[bytesWritten:]
		extracted := fsk.bitExtractor.ExtractBytes(remaining)
		if extracted == 0 {
			break // No more bytes to extract
		}
		bytesWritten += extracted
	}

	// Return the number of bytes written
	return bytesWritten
}

// OnIsEndOfPacket sets a callback to check if packet is complete
func (fsk *Demodulator) OnIsEndOfPacket(callback func([]byte) bool) {
	_ = callback
}
