package mask

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/dsp/modulation"
	"github.com/stretchr/testify/assert"
)

// TestModulator_ImplementsInterface verifies that Modulator implements modulation.Modulator interface
func TestModulator_ImplementsInterface(t *testing.T) {
	var _ modulation.Modulator = (*Modulator)(nil)
	assert.True(t, true, "Modulator implements modulation.Modulator interface")
}

// TestDemodulator_ImplementsInterface verifies that Demodulator implements modulation.Demodulator interface
func TestDemodulator_ImplementsInterface(t *testing.T) {
	var _ modulation.Demodulator = (*Demodulator)(nil)
	assert.True(t, true, "Demodulator implements modulation.Demodulator interface")
}

