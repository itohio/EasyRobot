package pll

import (
	"math"
	"testing"
)

func TestFilter_StepResponse(t *testing.T) {
	// Test PLL step response with known phase offset
	symbolRate := float32(1000)
	loopBandwidth := float32(0.05 * symbolRate) // 5% of symbol rate

	filter := NewFromLoopBandwidth(loopBandwidth, symbolRate, 0.707)

	// Update PLL multiple times
	// As PLL tracks, the phase error should decrease
	for i := 0; i < 1000; i++ {
		// Simulate: actual phase is π/4, estimated phase is filter.Phase()
		// Phase error = actual - estimated
		actualPhase := float32(math.Pi / 4)
		estimatedPhase := filter.Phase()
		currentError := actualPhase - estimatedPhase
		
		// Normalize error to [-π, π]
		if currentError > math.Pi {
			currentError -= 2 * math.Pi
		} else if currentError < -math.Pi {
			currentError += 2 * math.Pi
		}
		
		filter.Update(currentError)
	}

	// PLL should track the phase, so estimated phase should approach actual phase
	// Phase error should be small
	expectedPhase := float32(math.Pi / 4)
	estimatedPhase := filter.Phase()
	finalPhaseError := expectedPhase - estimatedPhase
	
	// Normalize
	if finalPhaseError > math.Pi {
		finalPhaseError -= 2 * math.Pi
	} else if finalPhaseError < -math.Pi {
		finalPhaseError += 2 * math.Pi
	}
	
	if math.Abs(float64(finalPhaseError)) > 0.1 {
		t.Errorf("PLL did not track phase: expected=%f, estimated=%f, error=%f", expectedPhase, estimatedPhase, finalPhaseError)
	}
}

func TestFilter_FrequencyTracking(t *testing.T) {
	// Test PLL frequency tracking with constant CFO
	symbolRate := float32(1000)
	loopBandwidth := float32(0.05 * symbolRate)

	filter := NewFromLoopBandwidth(loopBandwidth, symbolRate, 0.707)

	// Simulate constant frequency offset
	// Each symbol accumulates phase due to CFO
	cfoHz := float32(10) // 10 Hz offset
	phaseAccumulationPerSymbol := 2 * math.Pi * cfoHz / symbolRate

	// Simulate actual phase accumulating due to CFO
	actualPhase := float32(0)

	// Update PLL for many symbols
	for i := 0; i < 1000; i++ {
		// Actual phase accumulates due to CFO
		actualPhase += phaseAccumulationPerSymbol
		if actualPhase > math.Pi {
			actualPhase -= 2 * math.Pi
		} else if actualPhase < -math.Pi {
			actualPhase += 2 * math.Pi
		}

		// Phase error = actual - estimated
		estimatedPhase := filter.Phase()
		phaseError := actualPhase - estimatedPhase

		// Normalize error to [-π, π]
		if phaseError > math.Pi {
			phaseError -= 2 * math.Pi
		} else if phaseError < -math.Pi {
			phaseError += 2 * math.Pi
		}

		filter.Update(phaseError)
	}

	// PLL should track the frequency offset
	estimatedCFO := filter.FrequencyOffset()
	if math.Abs(float64(estimatedCFO-cfoHz)) > 2.0 {
		t.Errorf("PLL did not track frequency offset: expected ~%f Hz, got %f Hz", cfoHz, estimatedCFO)
	}
}

func TestFilter_RotatePhase(t *testing.T) {
	filter := New(0.1, 0.01, 1000)

	// Set phase to π/2
	filter.phase = float32(math.Pi / 2)

	// Rotate a sample at 0° (I=1, Q=0)
	i, q := filter.RotatePhase(1.0, 0.0)

	// Should rotate to -90° (I=0, Q=-1)
	expectedI := float32(0.0)
	expectedQ := float32(-1.0)

	tolerance := float32(0.001)
	if math.Abs(float64(i-expectedI)) > float64(tolerance) || math.Abs(float64(q-expectedQ)) > float64(tolerance) {
		t.Errorf("Phase rotation incorrect: expected (%f, %f), got (%f, %f)", expectedI, expectedQ, i, q)
	}
}

func TestFilter_Reset(t *testing.T) {
	filter := New(0.1, 0.01, 1000)

	// Update PLL
	filter.Update(0.1)
	filter.Update(0.1)

	// Reset
	filter.Reset()

	// State should be zero
	if filter.Phase() != 0 {
		t.Errorf("Phase not reset: got %f", filter.Phase())
	}
	if filter.FrequencyOffset() != 0 {
		t.Errorf("Frequency offset not reset: got %f", filter.FrequencyOffset())
	}
}

