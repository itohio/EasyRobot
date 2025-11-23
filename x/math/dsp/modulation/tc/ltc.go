package tc

import (
	"errors"
)

// EncodeBiPhase encodes bits to bi-phase mark audio samples
// Each bit is encoded as:
//   - Bit 0: transition at start, no transition at middle
//   - Bit 1: transition at start AND middle
// Returns samples for one frame period
func EncodeBiPhase(bits []byte, sampleRate float32, frameRate FrameRate) []float32 {
	if len(bits) < 10 {
		return nil
	}

	// Calculate samples per bit
	// One frame period = 1 / frameRate seconds
	// 80 bits per frame
	// Samples per bit = sampleRate / (frameRate * 80)
	framePeriod := 1.0 / float32(frameRate)
	samplesPerBit := int(sampleRate * framePeriod / 80.0)
	if samplesPerBit < 2 {
		samplesPerBit = 2 // Minimum 2 samples per bit for transitions
	}

	totalSamples := 80 * samplesPerBit
	samples := make([]float32, totalSamples)

	// Current signal level (start high)
	level := float32(1.0)

	// Encode each bit
	for bitIdx := 0; bitIdx < 80; bitIdx++ {
		byteIdx := bitIdx / 8
		bitPos := bitIdx % 8
		bit := (bits[byteIdx] >> bitPos) & 1

		bitStart := bitIdx * samplesPerBit
		bitMiddle := bitStart + samplesPerBit/2
		bitEnd := bitStart + samplesPerBit

		// Always transition at start of bit
		level = -level

		// Set first half of bit
		for i := bitStart; i < bitMiddle && i < totalSamples; i++ {
			samples[i] = level
		}

		// If bit is 1, transition at middle
		if bit == 1 {
			level = -level
		}

		// Set second half of bit
		for i := bitMiddle; i < bitEnd && i < totalSamples; i++ {
			samples[i] = level
		}
	}

	return samples
}

// DecodeBiPhase decodes bi-phase mark audio samples to bits
// Performs clock recovery and bit decision
func DecodeBiPhase(samples []float32, sampleRate float32, frameRate FrameRate) ([]byte, error) {
	if len(samples) == 0 {
		return nil, errors.New("no samples provided")
	}

	// Calculate expected samples per bit
	framePeriod := 1.0 / float32(frameRate)
	expectedSamplesPerBit := sampleRate * framePeriod / 80.0

	// Detect transitions (zero crossings)
	transitions := detectTransitions(samples)

	if len(transitions) < 80 {
		return nil, errors.New("insufficient transitions detected")
	}

	// Recover bits from transitions
	bits := make([]byte, 10) // 80 bits = 10 bytes

	// Bi-phase mark decoding:
	// - Transition at start of bit period = always present
	// - Transition at middle = bit is 1, no transition = bit is 0
	for bitIdx := 0; bitIdx < 80; bitIdx++ {
		expectedStart := float32(bitIdx) * expectedSamplesPerBit
		expectedMiddle := expectedStart + expectedSamplesPerBit/2

		// Find nearest transition to start
		startTransIdx := findNearestTransition(transitions, expectedStart)
		if startTransIdx < 0 || startTransIdx >= len(transitions) {
			continue
		}

		// Find nearest transition to middle
		middleTransIdx := findNearestTransition(transitions, expectedMiddle)

		// If transition found near middle, bit is 1
		bit := byte(0)
		if middleTransIdx >= 0 && middleTransIdx < len(transitions) {
			middleTrans := transitions[middleTransIdx]
			// Check if transition is close to expected middle
			diff := middleTrans - expectedMiddle
			if diff < 0 {
				diff = -diff
			}
			if diff < expectedSamplesPerBit/4 {
				bit = 1
			}
		}

		// Set bit in output
		byteIdx := bitIdx / 8
		bitPos := bitIdx % 8
		bits[byteIdx] |= bit << bitPos
	}

	return bits, nil
}

// detectTransitions finds zero crossings in samples
func detectTransitions(samples []float32) []float32 {
	if len(samples) < 2 {
		return nil
	}

	transitions := make([]float32, 0, len(samples)/10) // Estimate

	for i := 1; i < len(samples); i++ {
		// Check for sign change (zero crossing)
		if (samples[i-1] >= 0 && samples[i] < 0) || (samples[i-1] < 0 && samples[i] >= 0) {
			// Linear interpolation for more accurate transition point
			if samples[i-1] != samples[i] {
				t := -samples[i-1] / (samples[i] - samples[i-1])
				transitions = append(transitions, float32(i-1)+t)
			} else {
				transitions = append(transitions, float32(i))
			}
		}
	}

	return transitions
}

// findNearestTransition finds the transition index closest to the given sample position
func findNearestTransition(transitions []float32, pos float32) int {
	if len(transitions) == 0 {
		return -1
	}

	// Binary search for nearest transition
	left, right := 0, len(transitions)-1
	bestIdx := 0
	bestDist := abs(transitions[0] - pos)

	for left <= right {
		mid := (left + right) / 2
		dist := abs(transitions[mid] - pos)

		if dist < bestDist {
			bestDist = dist
			bestIdx = mid
		}

		if transitions[mid] < pos {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}

	return bestIdx
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

