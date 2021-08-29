// +build sam,xiao

package servos

import (
	"machine"
)

func timerMapping(pin machine.Pin) (*machine.TCC, bool) {
	switch pin {
	case machine.D9:
		return machine.TCC0, true
	case machine.D8:
		return machine.TCC1, true
	case machine.D10:
		return machine.TCC1, true
	default:
		return nil, false
	}
}
