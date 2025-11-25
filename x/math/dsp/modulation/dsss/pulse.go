package dsss

import (
	"github.com/itohio/EasyRobot/x/math/dsp"
)

// generateRectangular generates rectangular pulse shape samples
func generateRectangular(samplesPerChip int) []float32 {
	pulse := make([]float32, samplesPerChip)
	for i := range samplesPerChip {
		pulse[i] = 1.0
	}
	return pulse
}

// generatePulseShape generates pulse shape samples based on configuration
func generatePulseShape(config *Config) []float32 {
	switch config.PulseShape {
	case "raised_cosine", "root_raised_cosine":
		pulse := make([]float32, config.SamplesPerChip)
		center := float32(config.SamplesPerChip) / 2.0
		scale := 1 / float32(config.SamplesPerChip)
		gen := dsp.NewRaisedCosineGenerator(center, scale, config.Rolloff, 1.0)
		gen.Generate(pulse)
		return pulse
	case "rectangular":
		return generateRectangular(config.SamplesPerChip)
	default:
		return generateRectangular(config.SamplesPerChip)
	}
}
