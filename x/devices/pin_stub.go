//go:build !tinygo && !linux

package devices

// PinChange represents one or more trigger events that can happen on a given GPIO pin
// on the RP2040. ORed PinChanges are valid input to most IRQ functions.
type PinChange uint8

// Pin change interrupt constants for SetInterrupt.
const (
	// Edge falling
	PinFalling PinChange = 4 << iota
	// Edge rising
	PinRising

	PinToggle = PinFalling | PinRising
)

// stubPin is a stub implementation that conforms to the Pin interface.
type stubPin struct {
	num uint8
}

// NewPin creates a stub Pin that conforms to the Pin interface.
func NewPin(num uint8) *stubPin {
	return &stubPin{num: num}
}

func (p *stubPin) High() error {
	return nil
}

func (p *stubPin) Low() error {
	return nil
}

func (p *stubPin) Read() (bool, error) {
	return false, nil
}

func (p *stubPin) SetInterrupt(change PinChange, callback func()) error {
	return nil
}
