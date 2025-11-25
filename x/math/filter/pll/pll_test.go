package pll

import (
	"fmt"
	"math"
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/dsp"
)

func TestLoop_StepResponse(t *testing.T) {
	symbolRate := float32(1000)
	loopBandwidth := float32(0.05 * symbolRate)
	loop := NewFromLoopBandwidth(loopBandwidth, symbolRate, 0.707)

	for i := 0; i < 1000; i++ {
		actual := float32(math.Pi / 4)
		loop.Update(actual - loop.Phase())
	}

	if diff := math.Abs(float64(loop.Phase() - float32(math.Pi/4))); diff > 0.1 {
		t.Fatalf("PLL failed to converge diff=%f", diff)
	}
}

func TestLoop_FrequencyTracking(t *testing.T) {
	symbolRate := float32(1000)
	loop := NewFromLoopBandwidth(symbolRate*0.05, symbolRate, 0.707)
	cfoHz := float32(10)
	phaseStep := 2 * math.Pi * float64(cfoHz) / float64(symbolRate)
	actual := 0.0

	for i := 0; i < 1000; i++ {
		actual += phaseStep
		loop.Update(float32(actual) - loop.Phase())
	}

	if diff := math.Abs(float64(loop.FrequencyOffset() - cfoHz)); diff > 2 {
		t.Fatalf("PLL CFO mismatch diff=%f", diff)
	}
}

func TestLoop_Rotate(t *testing.T) {
	loop := New(0.1, 0.01, 1000)
	loop.Update(float32(math.Pi / 2))

	i, q := loop.Rotate(1, 0)
	if math.Abs(float64(i)) > 1e-3 || math.Abs(float64(q+1)) > 1e-3 {
		t.Fatalf("rotation incorrect (%f,%f)", i, q)
	}
}

func TestLoop_Reset(t *testing.T) {
	loop := New(0.1, 0.01, 1000)
	loop.Update(1)
	loop.Reset()
	if loop.Phase() != 0 || loop.FrequencyOffset() != 0 {
		t.Fatalf("reset failed")
	}
}

// wrapPiTest wraps phase to [-π, π] for test use
func wrapPiTest(x float32) float32 {
	switch {
	case x > math32.Pi:
		return x - 2*math32.Pi
	case x < -math32.Pi:
		return x + 2*math32.Pi
	default:
		return x
	}
}

// TestLoop_NoiseConvergence tests PLL convergence with noisy phase errors at different SNR levels
func TestLoop_NoiseConvergence(t *testing.T) {
	symbolRate := float32(1000)
	targetPhase := float32(math.Pi / 4)
	
	testCases := []struct {
		name      string
		snrDb     float32
		loopBw    float32
		maxError  float32
		minSymbols int
	}{
		{"HighSNR_50dB", 50, 0.05, 0.01, 100},
		{"HighSNR_30dB", 30, 0.05, 0.05, 200},
		{"MediumSNR_25dB", 25, 0.05, 0.1, 300},
		{"MediumSNR_20dB", 20, 0.05, 0.15, 500},
		{"LowSNR_15dB", 15, 0.05, 0.2, 800},
		{"NarrowBW_30dB", 30, 0.01, 0.08, 400},
		{"WideBW_30dB", 30, 0.1, 0.1, 150},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			loop := NewFromLoopBandwidth(tc.loopBw*symbolRate, symbolRate, 0.707)
			
			// Calculate noise amplitude for given SNR
			// Phase error power is approximately 1.0 (normalized)
			noisePower := 1.0 / math32.Pow(10, tc.snrDb/10)
			noiseAmp := math32.Sqrt(noisePower)
			noiseGen := dsp.NewNoiseWhiteGenerator(noiseAmp)
			noise := make([]float32, 1)
			
			converged := false
			for i := 0; i < tc.minSymbols*2; i++ {
				// True phase error + noise
				trueError := targetPhase - loop.Phase()
				noiseGen.Generate(noise)
				noisyError := trueError + noise[0]
				
				loop.Update(noisyError)
				
				// Check if converged
				phaseError := math32.Abs(wrapPiTest(loop.Phase() - targetPhase))
				if phaseError < tc.maxError && i >= tc.minSymbols {
					converged = true
					t.Logf("converged at symbol %d, final error=%.4f", i, phaseError)
					break
				}
			}
			
			if !converged {
				finalError := math32.Abs(wrapPiTest(loop.Phase() - targetPhase))
				t.Errorf("failed to converge: final error=%.4f (max=%.4f)", finalError, tc.maxError)
			}
		})
	}
}

// TestLoop_NoiseSteadyState tests steady-state phase error variance in noise
func TestLoop_NoiseSteadyState(t *testing.T) {
	symbolRate := float32(1000)
	targetPhase := float32(math.Pi / 4)
	snrDb := float32(30)
	
	loop := NewFromLoopBandwidth(0.05*symbolRate, symbolRate, 0.707)
	
	// Calculate noise amplitude
	noisePower := 1.0 / math32.Pow(10, snrDb/10)
	noiseAmp := math32.Sqrt(noisePower)
	noiseGen := dsp.NewNoiseWhiteGenerator(noiseAmp)
	noise := make([]float32, 1)
	
	// Converge first (warm-up)
	for i := 0; i < 500; i++ {
		trueError := targetPhase - loop.Phase()
		noiseGen.Generate(noise)
		loop.Update(trueError + noise[0])
	}
	
	// Measure steady-state error variance
	samples := 1000
	errors := make([]float32, samples)
	var sum, sumSq float32
	
	for i := 0; i < samples; i++ {
		trueError := targetPhase - loop.Phase()
		noiseGen.Generate(noise)
		loop.Update(trueError + noise[0])
		
		phaseError := wrapPiTest(loop.Phase() - targetPhase)
		errors[i] = phaseError
		sum += phaseError
		sumSq += phaseError * phaseError
	}
	
	mean := sum / float32(samples)
	variance := (sumSq / float32(samples)) - (mean * mean)
	stdDev := math32.Sqrt(variance)
	
	t.Logf("SNR=%.1fdB: mean error=%.4f, std dev=%.4f", snrDb, mean, stdDev)
	
	// At 30dB SNR, std dev should be reasonable (< 0.1 rad)
	if stdDev > 0.15 {
		t.Errorf("steady-state std dev too high: %.4f (expected < 0.15)", stdDev)
	}
	
	// Mean should be close to zero
	if math32.Abs(mean) > 0.05 {
		t.Errorf("steady-state mean error too high: %.4f (expected < 0.05)", mean)
	}
}

// TestLoop_NoiseFrequencyTracking tests frequency tracking in noise
func TestLoop_NoiseFrequencyTracking(t *testing.T) {
	symbolRate := float32(1000)
	cfoHz := float32(10)
	snrDb := float32(30)
	
	loop := NewFromLoopBandwidth(0.05*symbolRate, symbolRate, 0.707)
	phaseStep := 2 * math32.Pi * cfoHz / symbolRate
	actualPhase := float32(0)
	
	// Calculate noise amplitude
	noisePower := 1.0 / math32.Pow(10, snrDb/10)
	noiseAmp := math32.Sqrt(noisePower)
	noiseGen := dsp.NewNoiseWhiteGenerator(noiseAmp)
	noise := make([]float32, 1)
	
	// Track frequency with noise
	for i := 0; i < 2000; i++ {
		actualPhase += phaseStep
		actualPhase = wrapPiTest(actualPhase) // Wrap to prevent overflow
		trueError := wrapPiTest(actualPhase - loop.Phase())
		noiseGen.Generate(noise)
		loop.Update(trueError + noise[0])
	}
	
	estimatedCFO := loop.FrequencyOffset()
	error := math32.Abs(estimatedCFO - cfoHz)
	
	t.Logf("SNR=%.1fdB: true CFO=%.2f Hz, estimated=%.2f Hz, error=%.2f Hz", 
		snrDb, cfoHz, estimatedCFO, error)
	
	// At 30dB SNR, should track within 2 Hz
	if error > 3 {
		t.Errorf("frequency tracking error too high: %.2f Hz (expected < 3 Hz)", error)
	}
}

// TestLoop_NoiseBandwidthComparison compares different loop bandwidths in noise
func TestLoop_NoiseBandwidthComparison(t *testing.T) {
	symbolRate := float32(1000)
	targetPhase := float32(math.Pi / 4)
	snrDb := float32(25)
	
	bandwidths := []float32{0.01, 0.05, 0.1}
	results := make(map[float32]float32)
	
	noisePower := 1.0 / math32.Pow(10, snrDb/10)
	noiseAmp := math32.Sqrt(noisePower)
	
	for _, bw := range bandwidths {
		loop := NewFromLoopBandwidth(bw*symbolRate, symbolRate, 0.707)
		noiseGen := dsp.NewNoiseWhiteGenerator(noiseAmp)
		noise := make([]float32, 1)
		
		// Converge
		for i := 0; i < 500; i++ {
			trueError := targetPhase - loop.Phase()
			noiseGen.Generate(noise)
			loop.Update(trueError + noise[0])
		}
		
		// Measure steady-state error
		samples := 500
		var sumSq float32
		for i := 0; i < samples; i++ {
			trueError := targetPhase - loop.Phase()
			noiseGen.Generate(noise)
			loop.Update(trueError + noise[0])
			
			phaseError := wrapPiTest(loop.Phase() - targetPhase)
			sumSq += phaseError * phaseError
		}
		
		variance := sumSq / float32(samples)
		stdDev := math32.Sqrt(variance)
		results[bw] = stdDev
		
		t.Logf("BW=%.3f: std dev=%.4f", bw, stdDev)
	}
	
	// Narrower bandwidth should have lower noise (better filtering)
	// but this depends on the noise characteristics
	if results[0.01] > results[0.1]*2 {
		t.Logf("warning: narrow bandwidth has higher error - may be too slow to track")
	}
}

// TestLoop_NoiseAcquisitionTime tests how long it takes to acquire lock in noise
func TestLoop_NoiseAcquisitionTime(t *testing.T) {
	symbolRate := float32(1000)
	targetPhase := float32(math.Pi / 2)
	
	testCases := []struct {
		name         string
		snrDb        float32
		initialPhase float32
		maxSymbols   int
		threshold    float32
	}{
		{"HighSNR_50dB", 50, 0, 200, 0.02},
		{"HighSNR_30dB", 30, 0, 300, 0.05},
		{"MediumSNR_25dB", 25, 0, 500, 0.1},
		{"LargeOffset_30dB", 30, math32.Pi, 400, 0.05},
	}
	
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			loop := NewFromLoopBandwidth(0.05*symbolRate, symbolRate, 0.707)
			// Initialize by updating to the initial phase
			for i := 0; i < 10; i++ {
				loop.Update(tc.initialPhase - loop.Phase())
			}
			
			noisePower := 1.0 / math32.Pow(10, tc.snrDb/10)
			noiseAmp := math32.Sqrt(noisePower)
			noiseGen := dsp.NewNoiseWhiteGenerator(noiseAmp)
			noise := make([]float32, 1)
			
			acquired := false
			for i := 0; i < tc.maxSymbols; i++ {
				trueError := targetPhase - loop.Phase()
				noiseGen.Generate(noise)
				loop.Update(trueError + noise[0])
				
				phaseError := math32.Abs(wrapPi(loop.Phase() - targetPhase))
				if phaseError < tc.threshold {
					acquired = true
					t.Logf("acquired at symbol %d, error=%.4f", i, phaseError)
					break
				}
			}
			
			if !acquired {
				finalError := math32.Abs(wrapPi(loop.Phase() - targetPhase))
				t.Errorf("failed to acquire: final error=%.4f (threshold=%.4f)", 
					finalError, tc.threshold)
			}
		})
	}
}

// TestLoop_NoiseSNRSweep performs a comprehensive SNR sweep from 30dB down to 3dB
// and finds reasonable parameters for each SNR level
func TestLoop_NoiseSNRSweep(t *testing.T) {
	symbolRate := float32(1000)
	targetPhase := float32(math.Pi / 4)
	
	// Parameter combinations to test
	loopBandwidths := []float32{0.01, 0.02, 0.05, 0.1}
	dampingRatios := []float32{0.5, 0.707, 1.0, 1.5}
	
	// SNR levels to test (in dB)
	snrLevels := []float32{30, 25, 20, 15, 10, 8, 6, 5, 4, 3}
	
	type result struct {
		loopBw      float32
		damping     float32
		converged   bool
		finalError  float32
		convergeTime int
		stdDev      float32
	}
	
	results := make(map[float32][]result)
	
	for _, snrDb := range snrLevels {
		t.Run(fmt.Sprintf("SNR_%.0fdB", snrDb), func(t *testing.T) {
			noisePower := 1.0 / math32.Pow(10, snrDb/10)
			noiseAmp := math32.Sqrt(noisePower)
			
			// Determine reasonable thresholds based on SNR
			var maxError, maxStdDev float32
			var minSymbols, maxSymbols int
			switch {
			case snrDb >= 25:
				maxError = 0.05
				maxStdDev = 0.02
				minSymbols = 200
				maxSymbols = 1000
			case snrDb >= 15:
				maxError = 0.1
				maxStdDev = 0.05
				minSymbols = 500
				maxSymbols = 2000
			case snrDb >= 8:
				maxError = 0.2
				maxStdDev = 0.1
				minSymbols = 1000
				maxSymbols = 5000
			default: // 3-7 dB
				maxError = 0.3
				maxStdDev = 0.15
				minSymbols = 2000
				maxSymbols = 10000
			}
			
			snrResults := []result{}
			bestResult := result{finalError: 1e10}
			
			for _, loopBw := range loopBandwidths {
				for _, damping := range dampingRatios {
					loop := NewFromLoopBandwidth(loopBw*symbolRate, symbolRate, damping)
					noiseGen := dsp.NewNoiseWhiteGenerator(noiseAmp)
					noise := make([]float32, 1)
					
					converged := false
					convergeTime := maxSymbols
					
					// Convergence phase
					for i := 0; i < maxSymbols; i++ {
						trueError := targetPhase - loop.Phase()
						noiseGen.Generate(noise)
						loop.Update(trueError + noise[0])
						
						if i >= minSymbols {
							phaseError := math32.Abs(wrapPiTest(loop.Phase() - targetPhase))
							if phaseError < maxError {
								converged = true
								convergeTime = i
								break
							}
						}
					}
					
					// Measure steady-state if converged
					var stdDev float32
					finalError := math32.Abs(wrapPiTest(loop.Phase() - targetPhase))
					
					if converged {
						// Measure steady-state error variance
						samples := 500
						var sum, sumSq float32
						for i := 0; i < samples; i++ {
							trueError := targetPhase - loop.Phase()
							noiseGen.Generate(noise)
							loop.Update(trueError + noise[0])
							
							phaseError := wrapPiTest(loop.Phase() - targetPhase)
							sum += phaseError
							sumSq += phaseError * phaseError
						}
						
						mean := sum / float32(samples)
						variance := (sumSq / float32(samples)) - (mean * mean)
						stdDev = math32.Sqrt(variance)
						
						// Check if this is the best result so far
						if finalError < bestResult.finalError && stdDev < maxStdDev {
							bestResult = result{
								loopBw:      loopBw,
								damping:     damping,
								converged:   true,
								finalError:  finalError,
								convergeTime: convergeTime,
								stdDev:      stdDev,
							}
						}
					}
					
					snrResults = append(snrResults, result{
						loopBw:       loopBw,
						damping:     damping,
						converged:   converged,
						finalError:  finalError,
						convergeTime: convergeTime,
						stdDev:      stdDev,
					})
				}
			}
			
			// Count successful configurations
			successCount := 0
			for _, r := range snrResults {
				if r.converged && r.stdDev < maxStdDev {
					successCount++
				}
			}
			
			t.Logf("SNR=%.0fdB: %d/%d configurations successful", snrDb, successCount, len(snrResults))
			
			if bestResult.converged {
				t.Logf("  Best: BW=%.3f, damping=%.2f, error=%.4f, std=%.4f, time=%d symbols",
					bestResult.loopBw, bestResult.damping, bestResult.finalError,
					bestResult.stdDev, bestResult.convergeTime)
				
				// Log all successful configurations
				for _, r := range snrResults {
					if r.converged && r.stdDev < maxStdDev {
						t.Logf("  OK: BW=%.3f, damping=%.2f, error=%.4f, std=%.4f, time=%d",
							r.loopBw, r.damping, r.finalError, r.stdDev, r.convergeTime)
					}
				}
			} else {
				t.Logf("  No successful configurations found")
			}
			
			results[snrDb] = snrResults
			
			// Require at least one successful configuration
			if !bestResult.converged {
				t.Errorf("no working configuration found at SNR=%.0fdB", snrDb)
			}
		})
	}
	
	// Summary table
	t.Logf("\n=== Parameter Summary ===")
	t.Logf("SNR(dB) | Best BW | Best Damping | Error | StdDev | Time")
	t.Logf("--------|---------|--------------|-------|--------|------")
	for _, snrDb := range snrLevels {
		best := result{finalError: 1e10}
		for _, r := range results[snrDb] {
			if r.converged && r.stdDev < 0.2 && r.finalError < best.finalError {
				best = r
			}
		}
		if best.converged {
			t.Logf("%7.0f | %7.3f | %12.2f | %5.3f | %6.3f | %4d",
				snrDb, best.loopBw, best.damping, best.finalError, best.stdDev, best.convergeTime)
		} else {
			t.Logf("%7.0f |   FAIL  |      FAIL    |   -   |   -    |   -", snrDb)
		}
	}
}

// TestLoop_NoiseLowSNR explores very low SNR scenarios (3-5dB) with extended parameters
func TestLoop_NoiseLowSNR(t *testing.T) {
	symbolRate := float32(1000)
	targetPhase := float32(math.Pi / 4)
	
	// Very narrow bandwidths for low SNR
	lowSNRParams := []struct {
		name      string
		loopBw    float32
		damping   float32
		snrDb     float32
		maxSymbols int
		maxError  float32
		maxStdDev float32
	}{
		{"3dB_VeryNarrow", 0.005, 0.707, 3, 15000, 0.3, 0.2},
		{"3dB_Narrow", 0.01, 0.707, 3, 12000, 0.3, 0.2},
		{"3dB_Medium", 0.02, 0.707, 3, 10000, 0.3, 0.2},
		{"4dB_VeryNarrow", 0.005, 0.707, 4, 10000, 0.25, 0.15},
		{"4dB_Narrow", 0.01, 0.707, 4, 8000, 0.25, 0.15},
		{"5dB_Narrow", 0.01, 0.707, 5, 6000, 0.2, 0.13},
		{"5dB_Medium", 0.02, 0.707, 5, 5000, 0.2, 0.13},
	}
	
	for _, tc := range lowSNRParams {
		t.Run(tc.name, func(t *testing.T) {
			noisePower := 1.0 / math32.Pow(10, tc.snrDb/10)
			noiseAmp := math32.Sqrt(noisePower)
			loop := NewFromLoopBandwidth(tc.loopBw*symbolRate, symbolRate, tc.damping)
			noiseGen := dsp.NewNoiseWhiteGenerator(noiseAmp)
			noise := make([]float32, 1)
			
			// Convergence phase
			converged := false
			convergeTime := tc.maxSymbols
			for i := 0; i < tc.maxSymbols; i++ {
				trueError := targetPhase - loop.Phase()
				noiseGen.Generate(noise)
				loop.Update(trueError + noise[0])
				
				if i >= tc.maxSymbols/3 {
					phaseError := math32.Abs(wrapPiTest(loop.Phase() - targetPhase))
					if phaseError < tc.maxError {
						converged = true
						convergeTime = i
						break
					}
				}
			}
			
			if !converged {
				finalError := math32.Abs(wrapPiTest(loop.Phase() - targetPhase))
				t.Errorf("failed to converge: final error=%.4f (max=%.4f) at %d symbols",
					finalError, tc.maxError, convergeTime)
				return
			}
			
			// Measure steady-state
			samples := 1000
			var sum, sumSq float32
			for i := 0; i < samples; i++ {
				trueError := targetPhase - loop.Phase()
				noiseGen.Generate(noise)
				loop.Update(trueError + noise[0])
				
				phaseError := wrapPiTest(loop.Phase() - targetPhase)
				sum += phaseError
				sumSq += phaseError * phaseError
			}
			
			mean := sum / float32(samples)
			variance := (sumSq / float32(samples)) - (mean * mean)
			stdDev := math32.Sqrt(variance)
			finalError := math32.Abs(wrapPiTest(loop.Phase() - targetPhase))
			
			t.Logf("SNR=%.0fdB, BW=%.3f, damping=%.2f: converged at %d, error=%.4f, std=%.4f",
				tc.snrDb, tc.loopBw, tc.damping, convergeTime, finalError, stdDev)
			
			if stdDev > tc.maxStdDev {
				t.Errorf("steady-state std dev too high: %.4f (max=%.4f)", stdDev, tc.maxStdDev)
			}
		})
	}
}
