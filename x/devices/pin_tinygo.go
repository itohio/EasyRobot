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

// NewTinyGoPin wraps a machine.Pin to implement devices.Pin interface.
// This allows using machine.Pin directly with device drivers.
func NewTinyGoPin(pin machine.Pin) Pin {
	return pin
}
