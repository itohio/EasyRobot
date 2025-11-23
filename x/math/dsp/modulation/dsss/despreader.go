package dsss

import (
	"math"
)

// Despreader handles symbol despreading for DSSS
// Extracted for testability (SOLID principle)
type Despreader struct {
	spreadingFactor  int
	samplesPerChip    int
	samplesPerSymbol  int
	modulationOrder   int
	bufferSize        int
	code              Code
	codeTable         CodeTable
	carrierFreq       float32
	sampleRate        float32
}

// NewDespreader creates a new despreader component
func NewDespreader(spreadingFactor, samplesPerChip, samplesPerSymbol, modulationOrder, bufferSize int,
	code Code, codeTable CodeTable, carrierFreq, sampleRate float32) Despreader {
	return Despreader{
		spreadingFactor: spreadingFactor,
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

// DespreadSymbol despreads one symbol from the received signal
// Returns symbol index, or -1 if not enough samples or error
func (dsp *Despreader) DespreadSymbol(sampleBuffer []float32, bufferIndex, totalSamples, codePhase int, pllPhase float32) int {
	// Compute correlation at current code phase
	correlation := dsp.computeCorrelation(sampleBuffer, bufferIndex, totalSamples, codePhase, pllPhase)

	if dsp.modulationOrder == 2 {
		// Binary DSSS: correlate with single code
		// Decision: positive -> 1, negative -> 0
		if correlation > 0 {
			return 1
		}
		return 0
	} else {
		// M-ary DSSS: correlate with all codes, choose maximum
		maxCorrelation := float32(0)
		bestSymbol := 0

		for symbolIdx, code := range dsp.codeTable {
			// Compute correlation with this code
			corr := dsp.computeCorrelationWithCode(sampleBuffer, bufferIndex, totalSamples, codePhase, code, pllPhase)

			if corr > maxCorrelation {
				maxCorrelation = corr
				bestSymbol = symbolIdx
			}
		}

		return bestSymbol
	}
}

// computeCorrelation computes I/Q correlation at given code phase and updates PLL
// Returns in-phase (I) correlation value
func (dsp *Despreader) computeCorrelation(sampleBuffer []float32, bufferIndex, totalSamples, phase int, pllPhase float32) float32 {
	iCorr, _ := dsp.computeIQCorrelation(sampleBuffer, bufferIndex, totalSamples, phase, pllPhase)

	// Return in-phase correlation (this is what we use for binary DSSS decision)
	return iCorr
}

// computeIQCorrelation computes I/Q correlation at given code phase
func (dsp *Despreader) computeIQCorrelation(sampleBuffer []float32, bufferIndex, totalSamples, phase int, pllPhase float32) (float32, float32) {
	var iCorr, qCorr float32

	// Get code to correlate with
	var code Code
	if dsp.modulationOrder == 2 {
		code = dsp.code
	} else {
		code = dsp.codeTable[0]
	}

	codeLength := code.length
	lookback := dsp.samplesPerSymbol + phase
	if lookback > dsp.bufferSize {
		lookback = dsp.bufferSize
	}
	bufferStart := (bufferIndex - lookback + dsp.bufferSize) % dsp.bufferSize

	correlationStartSample := totalSamples - lookback
	if correlationStartSample < 0 {
		correlationStartSample = 0
	}

	// Integrate over all samples in the symbol period
	for chipIdx := 0; chipIdx < codeLength; chipIdx++ {
		chipStartOffset := chipIdx * dsp.samplesPerChip
		chipValue := code.GetBipolar(chipIdx)

		for s := 0; s < dsp.samplesPerChip; s++ {
			sampleOffset := chipStartOffset + s
			bufferIdx := (bufferStart + sampleOffset) % dsp.bufferSize
			sample := sampleBuffer[bufferIdx]

			// Compute carrier phase using absolute sample index
			absoluteSampleIdx := correlationStartSample + sampleOffset
			carrierPhase := 2 * math.Pi * dsp.carrierFreq * float32(absoluteSampleIdx) / dsp.sampleRate

			// Apply PLL phase correction
			correctedPhase := carrierPhase - pllPhase

			// I/Q correlation
			iCorrLocal := float32(math.Cos(float64(correctedPhase)))
			qCorrLocal := float32(math.Sin(float64(correctedPhase)))

			iCorr += sample * chipValue * iCorrLocal
			qCorr += sample * chipValue * qCorrLocal
		}
	}

	return iCorr, qCorr
}

// computeCorrelationWithCode computes I/Q correlation with a specific code
// Returns magnitude of complex correlation
func (dsp *Despreader) computeCorrelationWithCode(sampleBuffer []float32, bufferIndex, totalSamples, phase int, code Code, pllPhase float32) float32 {
	var iCorr, qCorr float32

	codeLength := code.length
	lookback := dsp.samplesPerSymbol + phase
	if lookback > dsp.bufferSize {
		lookback = dsp.bufferSize
	}
	bufferStart := (bufferIndex - lookback + dsp.bufferSize) % dsp.bufferSize

	correlationStartSample := totalSamples - lookback
	if correlationStartSample < 0 {
		correlationStartSample = 0
	}

	// Integrate over all samples in the symbol period
	for chipIdx := 0; chipIdx < codeLength; chipIdx++ {
		chipStartOffset := chipIdx * dsp.samplesPerChip
		chipValue := code.GetBipolar(chipIdx)

		for s := 0; s < dsp.samplesPerChip; s++ {
			sampleOffset := chipStartOffset + s
			bufferIdx := (bufferStart + sampleOffset) % dsp.bufferSize
			sample := sampleBuffer[bufferIdx]

			// Compute carrier phase using absolute sample index
			absoluteSampleIdx := correlationStartSample + sampleOffset
			carrierPhase := 2 * math.Pi * dsp.carrierFreq * float32(absoluteSampleIdx) / dsp.sampleRate

			// Apply PLL phase correction
			correctedPhase := carrierPhase - pllPhase

			// I/Q correlation
			iCorrLocal := float32(math.Cos(float64(correctedPhase)))
			qCorrLocal := float32(math.Sin(float64(correctedPhase)))

			iCorr += sample * chipValue * iCorrLocal
			qCorr += sample * chipValue * qCorrLocal
		}
	}

	// Return magnitude for M-ary DSSS
	return float32(math.Sqrt(float64(iCorr*iCorr + qCorr*qCorr)))
}

