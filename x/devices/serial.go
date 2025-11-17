package devices

import "io"

// SerialConfig holds configuration for serial ports.
type SerialConfig struct {
	// BaudRate is the serial port baud rate (e.g., 115200, 230400, 9600).
	// Default: 115200
	BaudRate int

	// EnableBuffering enables the Buffered() method to query available bytes.
	// WARNING: Enabling buffering can cause issues with devices like Arduino
	// that expect immediate responses. Only enable if you need to check buffer
	// status and understand the implications.
	EnableBuffering bool
}

// DefaultSerialConfig returns a default serial configuration with buffering disabled.
func DefaultSerialConfig() SerialConfig {
	return SerialConfig{
		BaudRate:        115200, // Default baud rate
		EnableBuffering: false,  // Disabled by default for reliable device communication
	}
}

// Serial represents a serial/UART connection. It is implemented by machine.UART
// in TinyGo, and can be implemented by Linux/Windows serial drivers.
//
// Note: The Buffered() method is opt-in via SerialConfig.EnableBuffering.
// Buffering is disabled by default to avoid issues with devices like Arduino
// that expect immediate, unbuffered communication.
type Serial interface {
	io.Reader
	io.Writer

	// Buffered returns the number of bytes currently in the receive buffer.
	// Returns 0 if buffering is disabled (default) to avoid communication issues.
	Buffered() int
}
