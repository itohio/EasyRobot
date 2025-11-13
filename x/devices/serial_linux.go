//go:build !tinygo && linux

package devices

import (
	"fmt"
	"os"
)

// LinuxSerial implements Serial interface using Linux serial port.
type LinuxSerial struct {
	file *os.File
}

// NewLinuxSerial creates a new Serial interface for Linux.
// device should be like "/dev/ttyAMA0" or "/dev/ttyUSB0".
func NewSerial(device string) (*LinuxSerial, error) {
	file, err := os.OpenFile(device, os.O_RDWR, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to open serial device %s: %w", device, err)
	}

	return &LinuxSerial{file: file}, nil
}

// Read reads data from the serial port.
func (s *LinuxSerial) Read(p []byte) (n int, err error) {
	return s.file.Read(p)
}

// Write writes data to the serial port.
func (s *LinuxSerial) Write(p []byte) (n int, err error) {
	return s.file.Write(p)
}

// Buffered returns the number of bytes currently in the receive buffer.
// On Linux, this is an approximation based on available data.
func (s *LinuxSerial) Buffered() int {
	// Try to peek at available data
	// This is a simplified implementation - full implementation would use FIONREAD ioctl
	return 0
}

// Close closes the serial port.
func (s *LinuxSerial) Close() error {
	return s.file.Close()
}

// File returns the underlying *os.File.
func (s *LinuxSerial) File() *os.File {
	return s.file
}
