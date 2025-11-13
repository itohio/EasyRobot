//go:build !tinygo && !linux

package devices

import "errors"

// StubSPI is a stub implementation that returns errors.
// This allows code to compile on platforms without SPI support.
type StubSPI struct{}

// NewStubSPI creates a stub SPI bus that always returns errors.
func NewSPI() *StubSPI {
	return &StubSPI{}
}

// Tx always returns an error.
func (b *StubSPI) Tx(w, r []byte) error {
	return errors.New("SPI not supported on this platform")
}

// Transfer always returns an error.
func (b *StubSPI) Transfer(byteVal byte) (byte, error) {
	return 0, errors.New("SPI not supported on this platform")
}
