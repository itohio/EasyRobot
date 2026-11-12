//go:build tinygo

package devices

import "machine"

// Use machine Pin directly

type PinChange = machine.PinChange

// Pin change interrupt constants for SetInterrupt.
const (
	// Edge falling
	PinFalling = machine.PinFalling
	// Edge rising
	PinRising = machine.PinRising

	PinToggle = machine.PinToggle
)

// TinyGoPin adapts a machine.Pin to Pin: machine.Pin has Get/Set/High/Low,
// but its SetInterrupt takes a func(machine.Pin), not a func(Pin).
type TinyGoPin struct{ machine.Pin }

// SetInterrupt calls callback with this TinyGoPin on change; a nil callback
// disables the interrupt (machine.Pin's rule). The Pin value is made here,
// so the interrupt handler converts nothing.
func (p TinyGoPin) SetInterrupt(change PinChange, callback func(Pin)) error {
	if callback == nil {
		return p.Pin.SetInterrupt(change, nil)
	}
	var self Pin = p
	return p.Pin.SetInterrupt(change, func(machine.Pin) { callback(self) })
}

// NewTinyGoPin wraps a machine.Pin as a Pin.
func NewTinyGoPin(pin machine.Pin) Pin {
	return TinyGoPin{pin}
}
