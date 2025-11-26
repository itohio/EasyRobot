package mask

import (
	"io"
	"log"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
	"github.com/itohio/EasyRobot/x/math/dsp"
)

var _ chain.Processor[byte, float32] = (*Demodulator)(nil)

// Demodulator converts M-ASK samples back to symbols.
type Demodulator struct {
	cfg   Config
	table amplitudeTable

	fifo     []float32
	buffered int
	readBuf  []float32

	// Transmission detection
	active           bool
	preambleDetected bool
	preambleProgress int
	silenceCount     int
	zeroSymbolCount  int // Consecutive zero symbols for lock loss detection
	lockLost         bool
	targetSNR        float32 // Target SNR for threshold calculation

	// AGC (Amplitude normalization)
	agcGain      float32
	agcReference float32   // Measured from preamble (max level)
	agcMinLevel  float32   // Minimum level from preamble
	agcMaxLevel  float32   // Maximum level from preamble
	agcSamples   []float32 // Store preamble samples for min/max calculation

	// Carrier recovery (for coherent mode)
	carrier dsp.CosineGenerator

	tmpSymbol [1]byte

	// Debug logging
	debugLogging bool
	symbolIndex  int // Track symbol index for debug output
}

// NewDemodulator builds a streaming demodulator instance.
func NewDemodulator(cfg Config) (*Demodulator, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	d := &Demodulator{
		cfg:          cfg,
		table:        newAmplitudeTable(cfg),
		fifo:         make([]float32, cfg.SamplesPerSymbol*2),
		carrier:      dsp.NewCosineGenerator(cfg.SampleRate, cfg.CarrierHz, 1.0, 0.0),
		agcGain:      1.0,
		agcReference: 1.0,
		agcMinLevel:  0.0,
		agcMaxLevel:  1.0,
		targetSNR:    35.0, // Default target SNR for threshold calculation (higher for M-ASK)
	}
	// Preamble is always required for normalization, so always allocate AGC samples buffer
	d.agcSamples = make([]float32, 0, cfg.PreambleLength)
	return d, nil
}

// Process implements chain.Processor.
func (d *Demodulator) Process(dst buffer.Writer[byte], src buffer.Reader[float32]) int {
	if dst == nil {
		return d.estimateBytes(src)
	}
	bitsWriter, ok := dst.(buffer.BitsWriter)
	if !ok || bitsWriter == nil {
		return 0
	}
	if bitsWriter.BitsLength() != d.cfg.BitsPerSymbol {
		return 0
	}

	totalBytes := 0
	sourceDone := false

	for {
		for d.buffered < d.cfg.SamplesPerSymbol && !sourceDone {
			needed := d.cfg.SamplesPerSymbol - d.buffered
			if len(d.readBuf) < needed {
				d.readBuf = make([]float32, needed)
			}
			n, err := src.Read(d.readBuf[:needed])
			if n > 0 {
				copy(d.fifo[d.buffered:], d.readBuf[:n])
				d.buffered += n
			}
			if err == io.EOF {
				sourceDone = true
			}
			if err != nil && err != io.EOF {
				sourceDone = true
			}
		}

		if d.buffered < d.cfg.SamplesPerSymbol {
			break
		}

		symbol := d.decodeSymbol(d.fifo[:d.cfg.SamplesPerSymbol])
		if symbol >= 0 {
			d.tmpSymbol[0] = byte(symbol)
			writtenBits, _ := bitsWriter.Write(d.tmpSymbol[:])
			if writtenBits == 0 {
				break
			}
			totalBytes += writtenBits / d.cfg.BitsPerSymbol
		}

		// Advance buffer by one symbol period
		// Copy remaining samples (from SamplesPerSymbol to end) to beginning
		remaining := d.buffered - d.cfg.SamplesPerSymbol
		if remaining > 0 {
			copy(d.fifo, d.fifo[d.cfg.SamplesPerSymbol:d.buffered])
		}
		d.buffered = remaining
	}

	return totalBytes
}

func (d *Demodulator) decodeSymbol(samples []float32) int {
	if len(samples) == 0 {
		return -1
	}

	energy := symbolEnergy(samples)

	// Debug: log symbol index and energy
	if d.debugLogging {
		if d.preambleDetected {
			// Only log after preamble (payload symbols)
		}
	}

	// Full energy check with SNR-based adaptive threshold
	baseThreshold := d.cfg.EnergyThresholdForSNR(d.targetSNR)
	// Fallback to fixed threshold if SNR-based calculation fails or for infinite SNR
	if baseThreshold <= 0 {
		baseThreshold = d.cfg.EnergyThreshold
	}

	// Coarse detection: quick energy check to avoid wasting CPU on noise
	// When no lock on a signal - coarse detection avoids calculations
	// When coarse detection triggers and lock is achieved - disable coarse detection until lock is lost
	// Coarse detection should NOT be active during active lock
	// Only enable coarse detection if we have a valid SNR-based threshold (not infinite SNR)
	useCoarseDetection := !d.active && baseThreshold > 0 && baseThreshold != d.cfg.EnergyThreshold
	if useCoarseDetection {
		coarseThreshold := baseThreshold * 0.1 // Very lenient for coarse check
		if energy < coarseThreshold {
			// Not active and energy too low - likely noise, skip expensive processing
			return -1
		}
	}

	// Use adaptive threshold based on state
	var threshold float32
	if baseThreshold <= 0 || baseThreshold == d.cfg.EnergyThreshold {
		// Infinite SNR or using fixed threshold - use very lenient thresholds
		// For infinite SNR, any non-zero signal should be detected (even symbol 0)
		if d.active {
			// During active transmission: be very lenient (symbol 0 might have low energy)
			threshold = d.cfg.EnergyThreshold * 0.01 // Very low to catch symbol 0
		} else if d.preambleProgress > 0 {
			threshold = d.cfg.EnergyThreshold * 0.1
		} else {
			// Initial detection: use lenient threshold to catch first symbol
			threshold = d.cfg.EnergyThreshold * 0.1 // Low enough to catch symbol 0
		}
	} else {
		// SNR-based threshold
		if d.active {
			// During active transmission: be lenient (noise might reduce energy)
			threshold = baseThreshold * 0.3
		} else if d.preambleProgress > 0 {
			// During preamble: moderate threshold
			threshold = baseThreshold * 0.5
		} else {
			// Initial detection: use full SNR-based threshold
			threshold = baseThreshold
		}
	}

	// After preamble is detected, always try to decode (ignore energy threshold)
	// Lock is lost only when we get consecutive zero symbols
	if d.preambleDetected && !d.lockLost {
		// After preamble: always decode, don't check energy threshold
		// We'll detect lock loss based on consecutive zero symbols
		// Skip energy threshold check - decode all symbols
	} else {
		// Before preamble or after lock lost: use energy threshold
		if energy < threshold {
			// Only reset if we're NOT in active transmission
			// During active transmission, low energy might be due to noise
			if !d.active {
				d.silenceCount++
				if d.silenceCount >= d.cfg.TransmissionGap {
					// Enough silence detected - reset for new transmission
					d.resetLock()
				}
				return -1
			}
			// During active transmission, low energy might be due to noise
			// Still try to decode - might still work
		}
	}

	// Signal detected or continuing active transmission
	if !d.active {
		d.active = true
	}
	d.silenceCount = 0

	// Detect amplitude
	var amplitude float32
	if d.cfg.CarrierHz > 0 {
		// For passband, always use coherent detection (downconvert)
		// This gives us the amplitude modulation directly
		amplitude = d.coherentDetection(samples)
		if amplitude < 0 {
			amplitude = -amplitude
		}
	} else {
		// Baseband: direct amplitude (RMS or average)
		amplitude = envelopeDetection(samples)
	}

	// Preamble processing (always required for normalization)
	var symbol int

	// Check if we should process this symbol as preamble
	// CRITICAL: We collect exactly (PreambleLength-1) samples for AGC, then detect completion
	// The last preamble symbol might overlap with first payload, so we're conservative
	// After collecting (PreambleLength-1) samples, we mark preamble as detected and decode the next symbol as payload
	if d.preambleProgress < d.cfg.PreambleLength {
		// During preamble: collect amplitude samples for normalization
		// Store amplitude for later min/max calculation
		if d.agcSamples != nil {
			d.agcSamples = append(d.agcSamples, amplitude)
		}

		// Expected preamble pattern: highest (M-1), lowest (0) alternating
		pattern := d.cfg.PreamblePattern
		if len(pattern) == 0 {
			M := d.cfg.constellationSize()
			pattern = []int{M - 1, 0}
		}
		expectedIdx := pattern[d.preambleProgress%len(pattern)]

		// Debug: log preamble detection
		if d.debugLogging {
			log.Printf("[PREAMBLE %d] amplitude=%.6f energy=%.6f expected_symbol=%d",
				d.preambleProgress, amplitude, energy, expectedIdx)
		}

		// Track preamble progress - symbol quantization not needed during preamble
		// We just collect amplitude samples for AGC calibration
		d.preambleProgress++

		// Check if we've collected enough for AGC calibration
		// After collecting PreambleLength samples, mark as detected (next symbol will be payload)
		if d.preambleProgress >= d.cfg.PreambleLength {
			d.preambleDetected = true
			// Finalize AGC normalization based on preamble min/max
			if len(d.agcSamples) > 0 {
				// Find min and max from preamble samples
				d.agcMinLevel = d.agcSamples[0]
				d.agcMaxLevel = d.agcSamples[0]
				for _, amp := range d.agcSamples {
					if amp < d.agcMinLevel {
						d.agcMinLevel = amp
					}
					if amp > d.agcMaxLevel {
						d.agcMaxLevel = amp
					}
				}

				// Calculate normalization reference
				// Expected: preamble alternates between lowest and highest levels
				expectedMinLevel := d.table.levels[0]
				expectedMaxLevel := d.table.levels[len(d.table.levels)-1]
				expectedRange := expectedMaxLevel - expectedMinLevel
				measuredRange := d.agcMaxLevel - d.agcMinLevel

				if expectedRange > 0 && measuredRange > 0 {
					// Scale factor to normalize measured range to expected range
					d.agcGain = expectedRange / measuredRange
					// Store expected minimum for offset in normalization formula
					d.agcReference = expectedMinLevel
					// d.agcMinLevel already stores measured minimum
				} else {
					// Fallback: use maximum level as reference
					if d.agcMaxLevel > 0 {
						d.agcReference = expectedMaxLevel / d.agcMaxLevel
						d.agcGain = 1.0 / d.agcReference
					} else {
						d.agcReference = expectedMaxLevel
						d.agcGain = 1.0
					}
				}

				// Debug: log AGC calibration
				if d.debugLogging {
					log.Printf("[AGC] preamble_detected min=%.6f max=%.6f gain=%.6f ref=%.6f",
						d.agcMinLevel, d.agcMaxLevel, d.agcGain, d.agcReference)
				}
			}
		}
		// Don't write preamble bytes
		return -1
	} else {
		// After preamble: use AGC-normalized amplitude
		if d.preambleDetected {
			// Normalize using measured min/max from preamble
			// Remove offset: (amplitude - measured_min)
			// Scale: (amplitude - measured_min) * gain
			// Add expected offset: normalized = (amplitude - measured_min) * gain + expected_min
			normalized := (amplitude-d.agcMinLevel)*d.agcGain + d.agcReference

			// Debug logging: log amplitude levels, normalization, and symbol distances
			if d.debugLogging && d.symbolIndex < 50 {
				// Log every symbol for first 50 payload symbols
				d.logAmplitudeDebug(d.symbolIndex, amplitude, normalized, energy)
			}

			// Quantize using expected levels (already in correct range)
			symbol = quantizeAmplitude(normalized, d.table.levels)

			if d.debugLogging && d.symbolIndex < 50 {
				d.logSymbolQuantization(d.symbolIndex, normalized, symbol, d.table.levels)
			}

			// Increment symbol index for debug logging (only count payload symbols after preamble)
			d.symbolIndex++
		} else {
			// Preamble not detected yet - this shouldn't happen in normal operation
			// but fallback to raw amplitude
			symbol = quantizeAmplitude(amplitude, d.table.levels)
		}
	}

	// Only write after preamble is detected (preamble is always required)
	if !d.preambleDetected {
		return -1
	}

	// Check for lock loss: consecutive zero symbols
	if d.preambleDetected && !d.lockLost {
		if symbol == 0 {
			d.zeroSymbolCount++
			if d.zeroSymbolCount >= d.cfg.LockLossThreshold {
				// Lock lost: too many consecutive zero symbols
				d.lockLost = true
				d.resetLock()
				return -1
			}
		} else {
			// Non-zero symbol: reset counter
			d.zeroSymbolCount = 0
		}
	} else if !d.preambleDetected {
		// Not in lock yet: reset zero counter
		d.zeroSymbolCount = 0
	}

	// If lock is lost, don't return symbols
	if d.lockLost {
		return -1
	}

	return symbol
}

func (d *Demodulator) coherentDetection(samples []float32) float32 {
	// Downconvert by mixing with carrier
	// Signal: A * envelope(t) * cos(2πfc*t + φ)
	// Mix with cos(2πfc*t): result = A * envelope(t) * (1 + cos(4πfc*t + 2φ)) / 2
	// After averaging (low-pass): A * envelope(t) / 2
	// So we need to multiply by 2 to get the actual amplitude
	var iAccum float32
	if len(d.readBuf) < len(samples) {
		d.readBuf = make([]float32, len(samples))
	}
	carrierBuf := d.readBuf[:len(samples)]
	d.carrier.Generate(carrierBuf)
	for i := range samples {
		iAccum += samples[i] * carrierBuf[i]
	}
	// Average - the scaling depends on envelope shape
	// For raised cosine with beta=0.35, average envelope ≈ 0.65
	// So we need to scale by approximately 2/0.65 ≈ 3.0
	// But let's use a more conservative factor and let AGC handle the rest
	amplitude := iAccum / float32(len(samples)) * 2.5
	if amplitude < 0 {
		amplitude = -amplitude
	}
	return amplitude
}

func (d *Demodulator) envelopeDetectionPassband(samples []float32) float32 {
	// For passband M-ASK, extract envelope using peak-to-peak detection
	// Signal is: A * envelope(t) * cos(2πfc*t)
	// Peak-to-peak over a symbol period gives us approximately 2*A*envelope
	var maxVal, minVal float32
	if len(samples) == 0 {
		return 0
	}
	maxVal = samples[0]
	minVal = samples[0]
	for _, s := range samples {
		if s > maxVal {
			maxVal = s
		}
		if s < minVal {
			minVal = s
		}
	}
	// Peak-to-peak gives us approximately 2*amplitude for carrier
	pp := maxVal - minVal
	amplitude := pp / 2.0
	// Fallback to RMS if peak-to-peak is too small
	if amplitude < 0.01 {
		amplitude = envelopeDetection(samples) * 1.414 // Scale RMS to approximate amplitude
	}
	return amplitude
}

func (d *Demodulator) estimateBytes(src buffer.Reader[float32]) int {
	if src == nil {
		return 0
	}
	length := src.Len()
	if length <= 0 {
		return 0
	}
	symbols := length / d.cfg.SamplesPerSymbol
	// Subtract preamble symbols
	if d.cfg.PreambleLength > 0 {
		if symbols > d.cfg.PreambleLength {
			symbols -= d.cfg.PreambleLength
		} else {
			symbols = 0
		}
	}
	return symbols * d.cfg.BitsPerSymbol / 8
}

// resetLock resets lock-related state (called when lock is lost)
func (d *Demodulator) resetLock() {
	d.active = false
	d.preambleDetected = false
	d.preambleProgress = 0
	d.silenceCount = 0
	d.zeroSymbolCount = 0
	d.lockLost = false
	d.symbolIndex = 0
	d.agcGain = 1.0
	d.agcReference = 1.0
	d.agcMinLevel = 0.0
	d.agcMaxLevel = 1.0
	if d.agcSamples != nil {
		d.agcSamples = d.agcSamples[:0] // Reset but keep capacity
	}
	d.carrier = dsp.NewCosineGenerator(d.cfg.SampleRate, d.cfg.CarrierHz, 1.0, 0.0)
}

// Reset clears internal state.
func (d *Demodulator) Reset() {
	d.buffered = 0
	d.resetLock()
	d.symbolIndex = 0
	d.debugLogging = false
}

// SetDebugLogging enables or disables debug logging for amplitude detection
func (d *Demodulator) SetDebugLogging(enabled bool) {
	d.debugLogging = enabled
}

// logAmplitudeDebug logs amplitude information including level differences and noise impact
func (d *Demodulator) logAmplitudeDebug(symbolIdx int, rawAmplitude, normalizedAmplitude, energy float32) {
	// Calculate symbol level differences
	levels := d.table.levels
	if len(levels) < 2 {
		return
	}

	// Find nearest level to normalized amplitude
	nearestIdx := quantizeAmplitude(normalizedAmplitude, levels)
	var levelDistance float32 = 1000.0
	if nearestIdx >= 0 && nearestIdx < len(levels) {
		levelDistance = math32.Abs(normalizedAmplitude - levels[nearestIdx])
	}

	// Calculate minimum symbol distance (spacing between adjacent levels)
	minSymbolDistance := float32(1000.0)
	for i := 1; i < len(levels); i++ {
		dist := levels[i] - levels[i-1]
		if dist < minSymbolDistance && dist > 0 {
			minSymbolDistance = dist
		}
	}

	log.Printf("[SYMBOL %d] raw_amp=%.6f norm_amp=%.6f energy=%.6f nearest_level_idx=%d level_distance=%.6f min_symbol_dist=%.6f",
		symbolIdx, rawAmplitude, normalizedAmplitude, energy, nearestIdx, levelDistance, minSymbolDistance)

	// Log if noise is comparable to symbol distance
	if minSymbolDistance > 0 && minSymbolDistance < 1000.0 {
		noiseRatio := levelDistance / minSymbolDistance
		if noiseRatio > 0.5 {
			log.Printf("[SYMBOL %d] WARNING: level_distance=%.6f is %.1f%% of min_symbol_dist=%.6f (noise may cause errors)",
				symbolIdx, levelDistance, noiseRatio*100, minSymbolDistance)
		}
	}
}

// logSymbolQuantization logs quantization details including level differences
func (d *Demodulator) logSymbolQuantization(symbolIdx int, normalizedAmplitude float32, quantizedSymbol int, levels []float32) {
	if quantizedSymbol < 0 || quantizedSymbol >= len(levels) {
		return
	}

	expectedLevel := levels[quantizedSymbol]
	quantizationError := math32.Abs(normalizedAmplitude - expectedLevel)

	// Calculate distances to adjacent levels
	var distToLower, distToUpper float32 = 1000.0, 1000.0
	if quantizedSymbol > 0 {
		distToLower = math32.Abs(normalizedAmplitude - levels[quantizedSymbol-1])
	}
	if quantizedSymbol < len(levels)-1 {
		distToUpper = math32.Abs(normalizedAmplitude - levels[quantizedSymbol+1])
	}

	log.Printf("[SYMBOL %d] quantized: symbol=%d expected_level=%.6f quantization_error=%.6f dist_to_lower=%.6f dist_to_upper=%.6f",
		symbolIdx, quantizedSymbol, expectedLevel, quantizationError, distToLower, distToUpper)

	// Log level differences for all levels once at the start
	if symbolIdx == 0 {
		log.Printf("[LEVELS] Constellation levels and distances:")
		for i := 0; i < len(levels); i++ {
			var nextDist float32 = 0
			if i < len(levels)-1 {
				nextDist = levels[i+1] - levels[i]
			}
			log.Printf("[LEVELS] level[%d]=%.6f (next_distance=%.6f)", i, levels[i], nextDist)
		}
	}
}
