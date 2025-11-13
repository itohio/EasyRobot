package devices

import "io"

// Serial represents a serial/UART connection. It is implemented by machine.UART
// in TinyGo, and can be implemented by Linux serial drivers for Raspberry Pi.
type Serial interface {
	io.Reader
	io.Writer

	// Buffered returns the number of bytes currently in the receive buffer.
	Buffered() int
}
