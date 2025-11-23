package dsss

import (
	"math"
)

// Acquisition handles signal acquisition/locking for DSSS
// Extracted for testability (SOLID principle)
type Acquisition struct {
	spreadingFactor  int
	samplesPerChip   int
	samplesPerSymbol int
	modulationOrder  int
	bufferSize       int
	code             Code
	codeTable        CodeTable
	carrierFreq      float32
	sampleRate       float32
}

// NewAcquisition creates a new acquisition component
func NewAcquisition(spreadingFactor, samplesPerChip, samplesPerSymbol, modulationOrder, bufferSize int,
	code Code, codeTable CodeTable, carrierFreq, sampleRate float32) Acquisition {
	return Acquisition{
		spreadingFactor:  spreadingFactor,
		samplesPerChip:   samplesPerChip,
		samplesPerSymbol: samplesPerSymbol,
		modulationOrder:  modulationOrder,
		bufferSize:       bufferSize,
		code:             code,
		codeTable:        codeTable,
		carrierFreq:      carrierFreq,
		sampleRate:       sampleRate,
	}
}

// AttemptAcquisition attempts to acquire the spreading code
// Returns: acquired (bool), codePhase (int), maxCorrelation (float32)
func (acq *Acquisition) AttemptAcquisition(sampleBuffer []float32, bufferIndex, totalSamples int) (bool, int, float32) {
	// Need at least one symbol period of samples
	if totalSamples < acq.samplesPerSymbol {
		return false, 0, 0
	}

	maxCorrelation := float32(0)
	secondMaxCorrelation := float32(0)
	bestPhase := 0

	// Search over possible code phases
	searchRange := min(acq.samplesPerSymbol*2, acq.bufferSize-acq.samplesPerSymbol)
	if searchRange <= 0 {
		searchRange = acq.samplesPerSymbol
	}

	// Search over code phases using simple correlation
	// Try multiple carrier phase hypotheses
	carrierPhaseOffsets := []float32{0, math.Pi / 4, math.Pi / 2, 3 * math.Pi / 4, math.Pi, 5 * math.Pi / 4, 3 * math.Pi / 2, 7 * math.Pi / 4}

	for _, carrierPhaseOffset := range carrierPhaseOffsets {
		for phase := 0; phase < searchRange; phase++ {
			// Try both I/Q correlation and simple correlation
			iCorr, qCorr := acq.computeIQCorrelationWithPhaseOffset(sampleBuffer, bufferIndex, totalSamples, phase, carrierPhaseOffset)
			magnitude := float32(math.Sqrt(float64(iCorr*iCorr + qCorr*qCorr)))

			// Also try simple correlation (no phase offset)
			simpleCorr := acq.computeSimpleCorrelation(sampleBuffer, bufferIndex, totalSamples, phase)
			if simpleCorr > magnitude {
				magnitude = simpleCorr
			}

			if magnitude > maxCorrelation {
				secondMaxCorrelation = maxCorrelation
				maxCorrelation = magnitude
				bestPhase = phase
			} else if magnitude > secondMaxCorrelation {
				secondMaxCorrelation = magnitude
			}
		}
	}

	// Use adaptive threshold: require that max correlation is significantly larger than second max
	// Lower the peak ratio requirement - if we have a strong absolute correlation, accept it
	peakRatio := float32(1.2) // More lenient
	noiseThreshold := float32(acq.spreadingFactor*acq.samplesPerChip) * 0.05
	if secondMaxCorrelation > noiseThreshold && maxCorrelation/secondMaxCorrelation < peakRatio {
		// Only fail if correlation is also low - if it's high, accept it anyway
		strongCorrelationThreshold := float32(acq.spreadingFactor*acq.samplesPerChip) * 0.3
		if maxCorrelation < strongCorrelationThreshold {
			return false, 0, maxCorrelation // Not a clear peak
		}
		// High correlation even without clear peak - accept it
	}

	// Absolute threshold - lower it to be more lenient
	minAbsoluteThreshold := float32(acq.spreadingFactor*acq.samplesPerChip) * 0.05

	if maxCorrelation > minAbsoluteThreshold {
		return true, bestPhase, maxCorrelation
	}

	return false, 0, maxCorrelation
}

// computeSimpleCorrelation computes simple correlation with carrier phase = 0
func (acq *Acquisition) computeSimpleCorrelation(sampleBuffer []float32, bufferIndex, totalSamples, phase int) float32 {
	var correlation float32

	// Get code to correlate with
	var code Code
	if acq.modulationOrder == 2 {
		code = acq.code
	} else {
		code = acq.codeTable[0]
	}

	codeLength := code.length
	lookback := acq.samplesPerSymbol + phase
	if lookback > acq.bufferSize {
		lookback = acq.bufferSize
	}
	bufferStart := (bufferIndex - lookback + acq.bufferSize) % acq.bufferSize

	correlationStartSample := totalSamples - lookback
	if correlationStartSample < 0 {
		correlationStartSample = 0
	}

	// Correlate: multiply samples by code and accumulate
	for chipIdx := 0; chipIdx < codeLength; chipIdx++ {
		chipStartOffset := chipIdx * acq.samplesPerChip
		chipValue := code.GetBipolar(chipIdx)

		for s := 0; s < acq.samplesPerChip; s++ {
			sampleOffset := chipStartOffset + s
			bufferIdx := (bufferStart + sampleOffset) % acq.bufferSize
			sample := sampleBuffer[bufferIdx]

			// Compute carrier phase and mix down
			absoluteSampleIdx := correlationStartSample + sampleOffset
			carrierPhase := 2 * math.Pi * acq.carrierFreq * float32(absoluteSampleIdx) / acq.sampleRate
			carrierMix := float32(math.Cos(float64(carrierPhase)))

			// Correlate: sample * code * carrier
			correlation += sample * chipValue * carrierMix
		}
	}

	// Return absolute value for magnitude
	if correlation < 0 {
		return -correlation
	}
	return correlation
}

// computeIQCorrelationWithPhaseOffset computes I/Q correlation with a carrier phase offset
func (acq *Acquisition) computeIQCorrelationWithPhaseOffset(sampleBuffer []float32, bufferIndex, totalSamples, phase int, carrierPhaseOffset float32) (float32, float32) {
	var iCorr, qCorr float32

	// Get code to correlate with
	var code Code
	if acq.modulationOrder == 2 {
		code = acq.code
	} else {
		code = acq.codeTable[0]
	}

	codeLength := code.length
	lookback := acq.samplesPerSymbol + phase
	if lookback > acq.bufferSize {
		lookback = acq.bufferSize
	}
	bufferStart := (bufferIndex - lookback + acq.bufferSize) % acq.bufferSize

	correlationStartSample := totalSamples - lookback
	if correlationStartSample < 0 {
		correlationStartSample = 0
	}

	// Integrate over all samples
	for chipIdx := 0; chipIdx < codeLength; chipIdx++ {
		chipStartOffset := chipIdx * acq.samplesPerChip
		chipValue := code.GetBipolar(chipIdx)

		for s := 0; s < acq.samplesPerChip; s++ {
			sampleOffset := chipStartOffset + s
			bufferIdx := (bufferStart + sampleOffset) % acq.bufferSize
			sample := sampleBuffer[bufferIdx]

			// Compute carrier phase with offset
			absoluteSampleIdx := correlationStartSample + sampleOffset
			carrierPhase := 2 * math.Pi * acq.carrierFreq * float32(absoluteSampleIdx) / acq.sampleRate
			correctedPhase := carrierPhase - carrierPhaseOffset

			iCorrLocal := float32(math.Cos(float64(correctedPhase)))
			qCorrLocal := float32(math.Sin(float64(correctedPhase)))

			iCorr += sample * chipValue * iCorrLocal
			qCorr += sample * chipValue * qCorrLocal
		}
	}

	return iCorr, qCorr
}
