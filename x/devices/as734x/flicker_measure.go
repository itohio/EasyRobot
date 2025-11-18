package as734x

import (
	"errors"
	"time"
)

// MeasureFlickerFrequency measures the actual flicker frequency by sampling
// the sensor over time and analyzing the signal. This provides more accurate
// frequency measurement than the built-in flicker detection which only detects
// common AC line frequencies (100Hz/120Hz).
//
// Parameters:
//   - duration: How long to sample (typically 100-500ms for 50-120Hz flicker)
//   - sampleRate: Target samples per second (higher = more accurate but slower)
//
// Returns the measured frequency in Hz, or 0 if no flicker detected.
func (d *Device) MeasureFlickerFrequency(duration time.Duration, sampleRate int) (float32, error) {
	if d.variant == VariantUnknown {
		return 0, errUnknownDevice
	}

	// Calculate sample interval
	interval := time.Second / time.Duration(sampleRate)
	numSamples := int(duration / interval)
	if numSamples < 10 {
		return 0, errors.New("as734x: duration too short for frequency measurement")
	}

	// Sample the clear channel
	samples := make([]uint16, numSamples)
	startTime := time.Now()

	for i := 0; i < numSamples; i++ {
		// Read a quick measurement
		measurement, err := d.Read()
		if err != nil {
			return 0, err
		}

		// Get clear channel value
		// AS7341: clear is at index 8 (10 channels total)
		// AS7343: clear is typically at index 0 (18 channels total)
		var clearValue uint16
		if d.variant == VariantAS7341 {
			if len(measurement.Channels) > 8 {
				clearValue = measurement.Channels[8] // Clear channel
			} else {
				// Fallback: use first available channel
				if len(measurement.Channels) > 0 {
					clearValue = measurement.Channels[0]
				}
			}
		} else if d.variant == VariantAS7343 {
			if len(measurement.Channels) > 0 {
				// AS7343: channel 0 is typically the clear/broadband channel
				clearValue = measurement.Channels[0]
			}
		}

		samples[i] = clearValue

		// Wait for next sample, accounting for measurement time
		elapsed := time.Since(startTime)
		nextSampleTime := time.Duration(i+1) * interval
		if elapsed < nextSampleTime {
			time.Sleep(nextSampleTime - elapsed)
		}
	}

	// Calculate frequency using zero-crossing detection
	frequency := calculateFrequency(samples, interval)
	return frequency, nil
}

// calculateFrequency uses zero-crossing detection to calculate frequency from samples.
// This method finds points where the signal crosses its mean value and calculates
// the average period between crossings to determine frequency.
func calculateFrequency(samples []uint16, interval time.Duration) float32 {
	if len(samples) < 4 {
		return 0
	}

	// Calculate mean to find zero crossings
	var sum uint32
	for _, s := range samples {
		sum += uint32(s)
	}
	mean := float32(sum) / float32(len(samples))

	// Find zero crossings (points where signal crosses the mean)
	// Use a small threshold to avoid noise-induced false crossings
	threshold := mean * 0.02 // 2% of mean as threshold
	var crossings []int
	prevValue := float32(samples[0])
	prevAbove := prevValue > mean+threshold
	prevBelow := prevValue < mean-threshold

	for i := 1; i < len(samples); i++ {
		value := float32(samples[i])
		above := value > mean+threshold
		below := value < mean-threshold

		// Detect crossing: going from above threshold to below threshold or vice versa
		if (prevAbove && below) || (prevBelow && above) {
			crossings = append(crossings, i)
		}

		prevAbove = above
		prevBelow = below
		prevValue = value
	}

	// Need at least 2 crossings to calculate frequency
	if len(crossings) < 2 {
		return 0
	}

	// Calculate average period between crossings
	// Each period is between two crossings (rising or falling)
	var totalPeriod time.Duration
	validPeriods := 0
	for i := 1; i < len(crossings); i++ {
		period := time.Duration(crossings[i]-crossings[i-1]) * interval
		// Filter out periods that are too short or too long (likely noise)
		periodMs := float32(period) / float32(time.Millisecond)
		if periodMs > 1 && periodMs < 1000 { // 1ms to 1000ms (1Hz to 1000Hz)
			totalPeriod += period
			validPeriods++
		}
	}

	if validPeriods == 0 {
		return 0
	}

	avgPeriod := totalPeriod / time.Duration(validPeriods)

	// Frequency = 1 / period
	if avgPeriod == 0 {
		return 0
	}
	frequencyHz := float32(time.Second) / float32(avgPeriod)

	// Filter out unrealistic frequencies (too high or too low)
	if frequencyHz < 10 || frequencyHz > 2000 {
		return 0
	}

	return frequencyHz
}

// MeasureFlickerFrequencyQuick is a convenience method that uses reasonable defaults
// for measuring flicker frequency (200ms duration, 1000 samples/sec).
func (d *Device) MeasureFlickerFrequencyQuick() (float32, error) {
	return d.MeasureFlickerFrequency(200*time.Millisecond, 1000)
}
