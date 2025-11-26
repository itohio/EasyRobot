package iq

import (
	"errors"
	"fmt"

	"github.com/chewxy/math32"
)

// Config holds all tunable parameters for the IQ modulator/demodulator.
type Config struct {
	SampleRate       float32
	CarrierFrequency float32 // Carrier frequency for mixing (0 = baseband, no mixing)
	LUTSize          int     // Size of lookup table for cos/sin (0 = use direct math)
	UseInterpolation bool    // Enable interpolation for LUT lookups
}

// DefaultConfig returns a safe baseline configuration.
func DefaultConfig() Config {
	return Config{
		SampleRate:       96_000,
		CarrierFrequency: 0,      // Baseband by default
		LUTSize:          4096,   // Default LUT size for cos/sin
		UseInterpolation: true,
	}
}

// Validate checks the configuration and applies defaults.
func (c *Config) Validate() error {
	if c.SampleRate <= 0 {
		return errors.New("iq: sample rate must be positive")
	}
	if c.CarrierFrequency < 0 {
		return errors.New("iq: carrier frequency must be non-negative")
	}
	if c.CarrierFrequency > c.SampleRate/2 {
		return fmt.Errorf("iq: carrier frequency %.2fHz exceeds Nyquist %.2fHz", c.CarrierFrequency, c.SampleRate/2)
	}
	if c.LUTSize < 0 {
		return errors.New("iq: LUT size must be non-negative")
	}
	if c.LUTSize == 0 && c.UseInterpolation {
		// If interpolation is enabled, we need a LUT
		c.LUTSize = 4096
	}
	if c.LUTSize > 0 && c.LUTSize < 4 {
		// Minimum LUT size for meaningful interpolation
		return fmt.Errorf("iq: LUT size must be at least 4 for interpolation, got %d", c.LUTSize)
	}
	return nil
}

const twoPi = 2 * math32.Pi
