package qam

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
	"github.com/stretchr/testify/assert"
)

// TestEncoder_ImplementsInterface verifies that Encoder implements modulation.Modulator interface
func TestEncoder_ImplementsInterface(t *testing.T) {
	var _ modulation.Modulator = (*Encoder)(nil)
	assert.True(t, true, "Encoder implements modulation.Modulator interface")
}

// TestDecoder_ImplementsInterface verifies that Decoder implements modulation.Demodulator interface
func TestDecoder_ImplementsInterface(t *testing.T) {
	var _ modulation.Demodulator = (*Decoder)(nil)
	assert.True(t, true, "Decoder implements modulation.Demodulator interface")
}

