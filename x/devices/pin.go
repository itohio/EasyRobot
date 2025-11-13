package devices

// Pin represents a GPIO pin. It is implemented by machine.Pin in TinyGo,
// and can be implemented by Linux GPIO drivers for Raspberry Pi.
// Configuration is done by concrete implementations, not through this interface.
type Pin interface {
	PinInterrupt

	// Get returns the current pin state (high = true, low = false).
	Get() bool

	// Set sets the pin state (high = true, low = false).
	Set(value bool)

	// High sets the pin to high (true).
	High()

	// Low sets the pin to low (false).
	Low()
}

// PinInterrupt allows configuring an interrupt callback on a pin.
type PinInterrupt interface {
	// SetInterrupt sets up an interrupt on the pin for the selected change type.
	// The callback is called with the pin as its argument.
	SetInterrupt(change PinChange, callback func(Pin)) error
}
