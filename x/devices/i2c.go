// Package devices provides platform-agnostic interfaces for I2C, SPI, Serial (UART), and Pin.
//
// These interfaces are compatible with TinyGo's machine.I2C, machine.SPI, machine.UART, and machine.Pin,
// but also support standard Go implementations for Linux (Raspberry Pi).
package devices

// I2C represents an I2C bus in controller/master mode.
// It is notably implemented by machine.I2C in TinyGo,
// and can be implemented by Linux I2C drivers for Raspberry Pi.
type I2C interface {
	// ReadRegister reads a register value from the device.
	ReadRegister(addr uint8, r uint8, buf []byte) error

	// WriteRegister writes a register value to the device.
	WriteRegister(addr uint8, r uint8, buf []byte) error

	// Tx performs a generic I2C transaction.
	// addr is the 7-bit I2C address (without R/W bit).
	// w is the write buffer (can be nil for read-only).
	// r is the read buffer (can be nil for write-only).
	Tx(addr uint16, w, r []byte) error
}

// I2CTarget represents an I2C bus in target/slave mode.
// This interface is separate from I2C as not all implementations support target mode.
type I2CTarget interface {
	// Listen starts listening for I2C transactions at the given address.
	Listen(addr uint16) error

	// WaitForEvent waits for an I2C target event and reads data into buf.
	// Returns the event type, number of bytes read, and any error.
	WaitForEvent(buf []byte) (evt I2CTargetEvent, count int, err error)

	// Reply sends a reply to the I2C controller.
	Reply(buf []byte) error
}
