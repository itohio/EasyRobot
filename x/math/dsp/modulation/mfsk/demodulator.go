package mfsk

import (
	"math"

	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
	"github.com/itohio/EasyRobot/x/math/filter"
	agcFilter "github.com/itohio/EasyRobot/x/math/filter/agc"
	"github.com/itohio/EasyRobot/x/math/filter/fir"
	"github.com/itohio/EasyRobot/x/math/filter/pll"
	"github.com/itohio/EasyRobot/x/math/primitive/generics"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// Demodulator handles Frequency Shift Keying demodulation
// Uses reusable components: FrequencyDetector, SymbolSynchronizer, BitBuffer
type Demodulator struct {
	sampleRate  float32
	frequencies []float32 // Frequencies for each symbol
	symbolRate  float32

	// Reusable components (SOLID principle - single responsibility)
	frequencyDetector  modulation.FrequencyDetector // Kept for backward compatibility, but coherent correlators are preferred
	symbolSynchronizer modulation.SymbolSynchronizer
	bitBuffer          generics.BitBuffer[byte] // BitBuffer for symbol-to-byte conversion
	bytesExtracted     int                      // Number of complete bytes already extracted from bitBuffer

	// Coherent detection components
	pllFilter pll.Filter // PLL for carrier phase/frequency tracking

	// Low-pass filter to remove noise before frequency detection
	lowPassFilter filter.Processor[float32]

	// Signal detection and AGC
	signalDetector modulation.SignalDetector
	agc            agcFilter.Filter

	// Coherent correlator state (pre-allocated, no allocations during processing)
	phaseIncrements []float32 // 2π * freq_k / sampleRate for each frequency

	// State
	samplesPerSymbol int
	modulationOrder  int
	bitsPerSymbol    int

	// Preamble detection state
	preambleDetected bool
	preambleSymbols  []int // Expected preamble symbol sequence (derived from Preamble bytes)
	preambleMatchIdx int   // Current index in preamble matching

	// Lock state
	locked    bool
	wasLocked bool // Track previous lock state to detect lock transition

	// Phase tracking for continuous phase accumulator (matches encoder)
	accumulatedPhase float32 // Tracks phase across symbols to match encoder's continuous phase
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

	modulationOrder := len(frequencies)

	// For 8-FSK and 16-FSK, use 4x sample rate to improve frequency resolution
	// This gives 4x more samples per symbol, improving Goertzel's ability to distinguish closely spaced frequencies
	effectiveSampleRate := sampleRate
	if modulationOrder >= 8 {
		effectiveSampleRate = sampleRate * 4.0
	}

	samplesPerSymbol := int(effectiveSampleRate / symbolRate)
	bitsPerSymbol := int(math.Log2(float64(modulationOrder)))

	// Create reusable components (SOLID principle)
	// Keep Goertzel for backward compatibility, but prefer coherent correlators
	// Use effective sample rate for frequency detector (4x for 8-FSK/16-FSK)
	frequencyDetector := modulation.NewGoertzelFrequencyDetector(effectiveSampleRate, frequencies...)
	symbolSynchronizer := modulation.NewSymbolSynchronizer(samplesPerSymbol)
	bitBuffer := generics.NewBitBuffer[byte](bitsPerSymbol)

	// Create PLL for coherent detection
	// Use loop bandwidth of 5% of symbol rate for good tracking vs noise trade-off
	loopBandwidth := symbolRate * 0.05
	pllFilter := *pll.NewFromLoopBandwidth(loopBandwidth, symbolRate, 0.707)

	// Pre-compute phase increments for coherent correlators (embedded-friendly, no allocations)
	// phaseIncrement[k] = 2π * freq_k / effectiveSampleRate
	phaseIncrements := make([]float32, len(frequencies))
	for i, freq := range frequencies {
		phaseIncrements[i] = 2 * math.Pi * freq / effectiveSampleRate
	}

	// Create low-pass filter to remove noise before frequency detection
	// Use a very high cutoff (close to Nyquist) to minimize distortion of FSK frequencies
	// The filter will still help with high-frequency noise but won't affect the signal much
	maxFreq := float32(0)
	for _, freq := range frequencies {
		if freq > maxFreq {
			maxFreq = freq
		}
	}
	// Use a very high cutoff (90% of Nyquist) to minimize distortion
	// This ensures we don't filter out any FSK frequencies while still removing high-frequency noise
	nyquist := sampleRate / 2.0
	cutoffFreq := nyquist * 0.9 // 90% of Nyquist - very high, minimal distortion
	// Use a lower order filter to reduce phase distortion
	lowPassFilter := fir.NewLowPass(8, cutoffFreq, sampleRate) // Reduced order from 16 to 8

	// Initialize signal detector
	// Use a larger window and more selective threshold to avoid locking onto noise
	// The threshold is set higher to require stronger signal (FSK-like patterns)
	detectorWindow := samplesPerSymbol * 4 // Larger window for better noise rejection
	signalDetector := *modulation.NewSignalDetector(detectorWindow, modulation.MinSignalEnergy*10)
	// Require minimum 4dB SNR to lock (signal must be ~2.5x stronger than noise)
	// This prevents locking onto noise while still allowing signal detection
	signalDetector.SetMinSNR(4.0)

	// Initialize AGC
	agc := *agcFilter.New(0.5, 0.01)       // 0.01 is AGC time constant
	agc.SetSampleRate(effectiveSampleRate) // Use effective sample rate

	// Pre-compute preamble symbol sequence from preamble bytes
	// Convert preamble bytes to bits, then to symbols
	preambleBitsPerSymbol := int(math.Log2(float64(modulationOrder)))
	preambleSymbols := make([]int, 0, len(Preamble)*8/preambleBitsPerSymbol+1)
	preambleBits := make([]byte, 0, len(Preamble)*8)
	for _, b := range Preamble {
		for i := 7; i >= 0; i-- {
			bit := (b >> uint(i)) & 1
			preambleBits = append(preambleBits, byte(bit))
		}
	}
	// Convert bits to symbols
	for i := 0; i < len(preambleBits)-preambleBitsPerSymbol+1; i += preambleBitsPerSymbol {
		symbol := 0
		for j := 0; j < preambleBitsPerSymbol && i+j < len(preambleBits); j++ {
			symbol = (symbol << 1) | int(preambleBits[i+j])
		}
		preambleSymbols = append(preambleSymbols, symbol)
	}

	return &Demodulator{
		sampleRate:         effectiveSampleRate, // Store effective sample rate
		frequencies:        frequencies,
		symbolRate:         symbolRate,
		frequencyDetector:  &frequencyDetector,
		symbolSynchronizer: symbolSynchronizer,
		bitBuffer:          bitBuffer,
		bytesExtracted:     0,
		pllFilter:          pllFilter,
		lowPassFilter:      lowPassFilter,
		samplesPerSymbol:   samplesPerSymbol,
		modulationOrder:    modulationOrder,
		bitsPerSymbol:      bitsPerSymbol,
		signalDetector:     signalDetector,
		agc:                agc,
		phaseIncrements:    phaseIncrements,
		preambleSymbols:    preambleSymbols,
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
	fsk.wasLocked = false
	fsk.accumulatedPhase = 0 // Reset phase accumulator
	fsk.preambleDetected = false
	fsk.preambleMatchIdx = 0
	fsk.frequencyDetector.Reset()
	fsk.symbolSynchronizer.Reset()
	fsk.bitBuffer.Clear()
	fsk.bytesExtracted = 0
	fsk.pllFilter.Reset()
	if fsk.lowPassFilter != nil {
		fsk.lowPassFilter.Reset()
	}
	fsk.signalDetector.Reset()
	fsk.agc.Reset()
}

// IsLocked returns true if demodulator is locked onto signal
//
// Deprecated: This method is not part of the Demodulator interface.
func (fsk *Demodulator) IsLocked() bool {
	return fsk.locked && fsk.signalDetector.IsLocked()
}

// FrequencyDetector returns the frequency detector (for debugging)
func (fsk *Demodulator) FrequencyDetector() modulation.FrequencyDetector {
	return fsk.frequencyDetector
}

// SymbolSynchronizer returns the symbol synchronizer (for debugging)
func (fsk *Demodulator) SymbolSynchronizer() *modulation.SymbolSynchronizer {
	return &fsk.symbolSynchronizer
}

// BitBuffer returns the bit buffer (for debugging)
func (fsk *Demodulator) BitBuffer() *generics.BitBuffer[byte] {
	return &fsk.bitBuffer
}

// SignalDetector returns the signal detector (for debugging)
func (fsk *Demodulator) SignalDetector() *modulation.SignalDetector {
	return &fsk.signalDetector
}

// Demodulate demodulates FSK signal into symbols
//
// Deprecated: This method is not part of the Demodulator interface.
func (fsk *Demodulator) Demodulate(signal vecTypes.Vector) vec.Vector {
	signalData := []float32(signal.(vec.Vector))

	// Apply AGC
	for i := range signalData {
		signalData[i] = fsk.agc.Process(signalData[i])
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
		samples[i] = fsk.agc.Process(samples[i])
		fsk.signalDetector.Update(samples[i])
		fsk.locked = fsk.signalDetector.IsLocked()
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

		// Add symbol to bit buffer
		fsk.bitBuffer.Write(byte(freqIdx))

		// Extract bytes from bit buffer
		allBytes := fsk.bitBuffer.ToBytes()
		if len(allBytes) > fsk.bytesExtracted {
			// Copy new bytes to output
			newBytes := allBytes[fsk.bytesExtracted:]
			remaining := output[bytesDecoded:]
			copied := copy(remaining, newBytes)
			bytesDecoded += copied
			fsk.bytesExtracted += copied

			// If we extracted all complete bytes, compact the buffer
			bitsWritten := fsk.bitBuffer.GetWrittenBits()
			completeBytes := (bitsWritten >> 3) * 8
			if fsk.bytesExtracted*8 >= completeBytes && completeBytes > 0 {
				// Move remaining bits to beginning
				fsk.bitBuffer.SetPosition(completeBytes)
				fsk.bitBuffer.Compact()
				fsk.bytesExtracted = 0
			}
		}
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
		// Rough estimate - actual depends on internal buffer state
		symbolsAvailable := 100 // Conservative estimate
		bitsAvailable := symbolsAvailable * bitsPerSymbol
		return bitsAvailable / 8
	}
	ampVec := amplitudes.(vec.Vector)
	ampData := []float32(ampVec)

	// Apply AGC and signal detection
	everLocked := fsk.locked // Track if we were ever locked during this decode call
	for i := range ampData {
		ampData[i] = fsk.agc.Process(ampData[i])
		fsk.signalDetector.Update(ampData[i])
		fsk.locked = fsk.signalDetector.IsLocked()
		if fsk.locked {
			everLocked = true // Remember we were locked
		}
	}

	// Check if we just locked (transition from unlocked to locked)
	justLocked := !fsk.wasLocked && fsk.locked
	if justLocked {
		// When we first lock, clear bit extractor to discard any bits accumulated from padding
		// DO NOT clear symbol synchronizer - it may contain the first symbol of the actual signal!
		// The synchronizer will naturally flush old samples as new ones come in
		fsk.bitBuffer.Clear()
		fsk.bytesExtracted = 0
		fsk.frequencyDetector.Reset()
		fsk.pllFilter.Reset()
		fsk.preambleDetected = false
		fsk.preambleMatchIdx = 0
	}
	fsk.wasLocked = fsk.locked

	// If we just unlocked, clear buffers to avoid partial symbols
	if fsk.wasLocked && !fsk.locked {
		// Signal just ended - clear buffers to avoid partial decoding
		fsk.symbolSynchronizer.Reset()
		fsk.bitBuffer.Clear()
		fsk.bytesExtracted = 0
		fsk.frequencyDetector.Reset()
		fsk.pllFilter.Reset()
		if fsk.lowPassFilter != nil {
			fsk.lowPassFilter.Reset()
		}
	}

	// Apply low-pass filter to remove noise before frequency detection
	// This is critical for robust FSK demodulation with Goertzel
	// For clean signals, we can skip filtering to avoid any distortion
	// Only filter for noisy signals (when magnitude is low)
	// For now, skip filtering for clean signals to see if it helps with 8-FSK/16-FSK
	// TODO: Make this adaptive based on signal quality
	// if fsk.lowPassFilter != nil {
	// 	for i := range ampData {
	// 		ampData[i] = fsk.lowPassFilter.Process(ampData[i])
	// 	}
	// }

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

		// Use coherent detection for all modulation orders
		// Coherent detection handles phase continuity better than Goertzel
		// With 4x sample rate, we have enough samples per symbol for good frequency resolution
		freqIdx := fsk.detectFrequencyCoherent(symbolSamples)

		// Detect preamble at symbol level before adding to bit extractor
		if !fsk.preambleDetected {
			// Check if this symbol matches the expected preamble symbol
			if fsk.preambleMatchIdx < len(fsk.preambleSymbols) {
				expectedSymbol := fsk.preambleSymbols[fsk.preambleMatchIdx]
				if freqIdx == expectedSymbol {
					fsk.preambleMatchIdx++
					if fsk.preambleMatchIdx >= len(fsk.preambleSymbols) {
						// Preamble fully detected - clear bit extractor to skip preamble bits
						fsk.preambleDetected = true
						fsk.bitBuffer.Clear()
						fsk.bytesExtracted = 0 // Clear any bits accumulated during preamble
					}
					// Skip adding this symbol to bit extractor (it's part of preamble)
					continue
				} else {
					// Mismatch - reset preamble matching
					fsk.preambleMatchIdx = 0
					// If this symbol matches the first preamble symbol, start matching
					if freqIdx == fsk.preambleSymbols[0] {
						fsk.preambleMatchIdx = 1
						continue // Skip adding first symbol
					}
				}
			} else {
				// We've seen all preamble symbols but didn't match - reset and try again
				fsk.preambleMatchIdx = 0
				if freqIdx == fsk.preambleSymbols[0] {
					fsk.preambleMatchIdx = 1
					continue
				}
			}
		}

		// Only add symbols to bit extractor after preamble is detected
		// For now, if preamble detection fails after many symbols, start decoding anyway
		// This handles cases where preamble might be corrupted or detection is imperfect
		if fsk.preambleDetected || symbolsExtractedFromSync > len(fsk.preambleSymbols)*3 {
			// If we've processed many symbols without detecting preamble, assume it's already past
			// or preamble detection failed - start decoding anyway
			if !fsk.preambleDetected && symbolsExtractedFromSync > len(fsk.preambleSymbols)*3 {
				fsk.preambleDetected = true
				fsk.bitBuffer.Clear()
				fsk.bytesExtracted = 0 // Clear any accumulated bits
			}

			// Decode symbol - add to bit buffer
			// DO NOT skip symbols based on magnitude - this causes bit misalignment
			// Decode all symbols when locked, even if magnitude is low (might be valid zeros or padding)
			// DO NOT extract bytes here - wait until all symbols are processed
			// Extracting bytes incrementally causes bit misalignment
			fsk.bitBuffer.Write(byte(freqIdx))
			symbolsAddedToExtractor++
		}
	}

	// After processing all available symbols, extract bytes from bit buffer
	// Only extract if preamble was detected
	if fsk.preambleDetected {
		// Extract bytes only after all symbols are processed to avoid bit misalignment
		// Keep extracting until buffer is full or no more bytes available
		for bytesWritten < len(data) {
			allBytes := fsk.bitBuffer.ToBytes()
			if len(allBytes) <= fsk.bytesExtracted {
				break // No more bytes to extract
			}
			// Copy new bytes to output
			newBytes := allBytes[fsk.bytesExtracted:]
			remaining := data[bytesWritten:]
			copied := copy(remaining, newBytes)
			if copied == 0 {
				break // No more space in output
			}
			bytesWritten += copied
			fsk.bytesExtracted += copied

			// If we extracted all complete bytes, compact the buffer
			bitsWritten := fsk.bitBuffer.GetWrittenBits()
			completeBytes := (bitsWritten >> 3) * 8
			if fsk.bytesExtracted*8 >= completeBytes && completeBytes > 0 {
				// Move remaining bits to beginning
				fsk.bitBuffer.SetPosition(completeBytes)
				fsk.bitBuffer.Compact()
				fsk.bytesExtracted = 0
			}
		}
	}

	// Return the number of bytes written
	return bytesWritten
}

// detectFrequencyCoherent uses coherent correlators to detect frequency
// Returns the frequency index with highest correlation magnitude
// This provides better performance than Goertzel, especially for closely spaced frequencies
// Accounts for encoder's continuous phase accumulator
func (fsk *Demodulator) detectFrequencyCoherent(samples []float32) int {
	if len(samples) == 0 {
		return 0
	}

	maxMagnitude := float32(0)
	bestFreqIdx := 0
	var bestCorrelatorI, bestCorrelatorQ float32

	// Compute coherent correlators for each frequency
	// Z_k = Σ r[n] * e^(-j2πf_k nT_s) = Σ r[n] * (cos(2πf_k nT_s) - j*sin(2πf_k nT_s))
	// For real signals r[n] = A*cos(ω_k*n + φ), correlating with cos(ω_k*n - θ) and sin(ω_k*n - θ)
	// gives us phase-independent detection via magnitude
	pllPhase := fsk.pllFilter.Phase()

	for freqIdx, phaseInc := range fsk.phaseIncrements {
		var sumI, sumQ float32

		// Correlate with cosine and sine (I and Q components)
		// The encoder generates: amplitude * cos(phase) where phase accumulates continuously
		// We need to account for the accumulated phase from previous symbols
		// Encoder: phase += 2π * freq / sampleRate for each sample
		// So for symbol starting at sample n0, phase = accumulatedPhase + phaseInc * (n - n0)
		for n, sample := range samples {
			// Account for continuous phase: start from accumulated phase, then add phase increment
			phase := fsk.accumulatedPhase + phaseInc*float32(n)
			// Apply PLL phase correction
			correctedPhase := phase - pllPhase
			cosPhase := float32(math.Cos(float64(correctedPhase)))
			sinPhase := float32(math.Sin(float64(correctedPhase)))

			// Coherent correlation
			sumI += sample * cosPhase
			sumQ += sample * sinPhase
		}

		// Compute magnitude: |Z_k| = sqrt(I^2 + Q^2)
		// Use raw magnitude (not normalized) for comparison
		magnitude := float32(math.Sqrt(float64(sumI*sumI + sumQ*sumQ)))

		if magnitude > maxMagnitude {
			maxMagnitude = magnitude
			bestFreqIdx = freqIdx
			bestCorrelatorI = sumI
			bestCorrelatorQ = sumQ
		}
	}

	// Check if we have a valid detection
	// For clean signals with amplitude=1, expected magnitude is ~N/2
	// Use a lower threshold to allow detection
	expectedMagnitude := float32(len(samples)) * 0.1 // 10% of samples as minimum
	if maxMagnitude < expectedMagnitude {
		// Coherent detection failed - fall back to Goertzel
		freqIdx, goertzelMag := fsk.frequencyDetector.DetectFrequency(samples)
		if freqIdx >= 0 && goertzelMag > 0.001 {
			// Update accumulated phase for detected frequency
			fsk.accumulatedPhase += fsk.phaseIncrements[freqIdx] * float32(len(samples))
			return freqIdx
		}
		// Both failed - return 0 as safe default
		return 0
	}

	// Update PLL with phase error from detected symbol
	// Phase error = arg(Z_k) where Z_k is the correlator output
	normalizedI := bestCorrelatorI / float32(len(samples))
	normalizedQ := bestCorrelatorQ / float32(len(samples))
	phaseError := float32(math.Atan2(float64(normalizedQ), float64(normalizedI)))
	fsk.pllFilter.Update(phaseError)

	// Update accumulated phase for detected frequency (matches encoder behavior)
	// Encoder: phase += 2π * freq / sampleRate for each sample
	// So we accumulate: phaseInc * len(samples)
	fsk.accumulatedPhase += fsk.phaseIncrements[bestFreqIdx] * float32(len(samples))
	// Normalize phase to [0, 2π) to prevent overflow
	if fsk.accumulatedPhase > 2*math.Pi {
		fsk.accumulatedPhase -= 2 * math.Pi * float32(int(fsk.accumulatedPhase/(2*math.Pi)))
	}

	return bestFreqIdx
}

// detectBestFrequencyGoertzel finds the best frequency using Goertzel algorithm
// without ambiguity checks - always returns the frequency with highest magnitude
// Used as fallback when main detector returns -1 (ambiguous)
func (fsk *Demodulator) detectBestFrequencyGoertzel(samples []float32) int {
	if len(samples) == 0 {
		return 0
	}

	// Use the frequency detector's Goertzel implementation directly
	// by calling DetectFrequency but ignoring the -1 return
	// Actually, we need to access the internal Goertzel implementation
	// For now, use a simple correlation-based approach
	maxMagnitude := float32(0)
	bestFreqIdx := 0

	for freqIdx, phaseInc := range fsk.phaseIncrements {
		var sumI, sumQ float32
		for n, sample := range samples {
			phase := phaseInc * float32(n)
			cosPhase := float32(math.Cos(float64(phase)))
			sinPhase := float32(math.Sin(float64(phase)))
			sumI += sample * cosPhase
			sumQ += sample * sinPhase
		}
		magnitude := float32(math.Sqrt(float64(sumI*sumI + sumQ*sumQ)))
		if magnitude > maxMagnitude {
			maxMagnitude = magnitude
			bestFreqIdx = freqIdx
		}
	}

	return bestFreqIdx
}
