//go:build !tinygo && linux

package devices

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

// LinuxSPI implements SPI bus using Linux spidev interface.
type LinuxSPI struct {
	fd *os.File
}

// NewLinuxSPI creates a new SPI bus for Linux (Raspberry Pi).
// device should be like "/dev/spidev0.0" for SPI bus 0, chip select 0.
func NewSPI(device string) (*LinuxSPI, error) {
	fd, err := os.OpenFile(device, os.O_RDWR, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to open SPI device %s: %w", device, err)
	}

	return &LinuxSPI{fd: fd}, nil
}

// Tx transmits and receives data.
func (b *LinuxSPI) Tx(w, r []byte) error {
	if len(w) != len(r) && w != nil && r != nil {
		return fmt.Errorf("SPI Tx: write and read buffers must be same length")
	}

	length := len(w)
	if length == 0 {
		length = len(r)
	}

	// Prepare spi_ioc_transfer structure
	type spiIocTransfer struct {
		txBuf       uint64
		rxBuf       uint64
		length      uint32
		speedHz     uint32
		delayUsecs  uint16
		bitsPerWord uint8
		csChange    uint8
		txNbits     uint8
		rxNbits     uint8
		pad         uint16
	}

	transfer := spiIocTransfer{
		length: uint32(length),
	}

	if w != nil {
		transfer.txBuf = uint64(uintptr(unsafe.Pointer(&w[0])))
	}
	if r != nil {
		transfer.rxBuf = uint64(uintptr(unsafe.Pointer(&r[0])))
	}

	// SPI_IOC_MESSAGE(1) ioctl
	const SPI_IOC_MESSAGE = 0x40006B00
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, b.fd.Fd(), SPI_IOC_MESSAGE, uintptr(unsafe.Pointer(&transfer)))
	if errno != 0 {
		return fmt.Errorf("SPI transfer failed: %v", errno)
	}

	return nil
}

// Transfer writes a single byte and receives a byte.
func (b *LinuxSPI) Transfer(byteVal byte) (byte, error) {
	w := []byte{byteVal}
	r := make([]byte, 1)
	if err := b.Tx(w, r); err != nil {
		return 0, err
	}
	return r[0], nil
}

// Close closes the SPI bus.
func (b *LinuxSPI) Close() error {
	return b.fd.Close()
}
