//go:build !tinygo && linux

package devices

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

// SPIMode represents SPI mode (clock polarity and phase).
type SPIMode uint8

const (
	// SPIMode0: CPOL=0, CPHA=0 (clock idle low, data sampled on rising edge)
	SPIMode0 SPIMode = 0
	// SPIMode1: CPOL=0, CPHA=1 (clock idle low, data sampled on falling edge)
	SPIMode1 SPIMode = 1
	// SPIMode2: CPOL=1, CPHA=0 (clock idle high, data sampled on falling edge)
	SPIMode2 SPIMode = 2
	// SPIMode3: CPOL=1, CPHA=1 (clock idle high, data sampled on rising edge)
	SPIMode3 SPIMode = 3
)

// SPIConfig holds SPI configuration for Raspberry Pi.
type SPIConfig struct {
	// SpeedHz is the SPI clock speed in Hz (default: 500000 = 500kHz)
	SpeedHz uint32
	// Mode is the SPI mode (default: SPIMode0)
	Mode SPIMode
	// BitsPerWord is the number of bits per word (default: 8)
	BitsPerWord uint8
}

// DefaultSPIConfig returns a default SPI configuration suitable for most devices.
func DefaultSPIConfig() SPIConfig {
	return SPIConfig{
		SpeedHz:     500000, // 500kHz
		Mode:        SPIMode0,
		BitsPerWord: 8,
	}
}

// LinuxSPI implements SPI bus using Linux spidev interface (Raspberry Pi).
type LinuxSPI struct {
	fd     *os.File
	device string
	config SPIConfig
}

// NewSPI creates a new SPI bus for Linux (Raspberry Pi).
// device should be like "/dev/spidev0.0" for SPI bus 0, chip select 0.
// Common Raspberry Pi SPI devices:
//   - /dev/spidev0.0 (SPI0, CS0) - Primary SPI
//   - /dev/spidev0.1 (SPI0, CS1) - Primary SPI, alternate CS
//   - /dev/spidev1.0 (SPI1, CS0) - Auxiliary SPI (if available)
func NewSPI(device string) (*LinuxSPI, error) {
	return NewSPIWithConfig(device, DefaultSPIConfig())
}

// NewSPIWithConfig creates a new SPI bus with custom configuration.
func NewSPIWithConfig(device string, config SPIConfig) (*LinuxSPI, error) {
	fd, err := os.OpenFile(device, os.O_RDWR, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to open SPI device %s: %w (ensure SPI is enabled in raspi-config)", device, err)
	}

	spi := &LinuxSPI{
		fd:     fd,
		device: device,
		config: config,
	}

	// Apply configuration
	if err := spi.configure(); err != nil {
		fd.Close()
		return nil, fmt.Errorf("failed to configure SPI: %w", err)
	}

	return spi, nil
}

// configure applies SPI configuration using ioctls.
func (b *LinuxSPI) configure() error {
	// Set SPI mode
	const SPI_IOC_WR_MODE = 0x40016B01
	mode := uint8(b.config.Mode)
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, b.fd.Fd(), SPI_IOC_WR_MODE, uintptr(unsafe.Pointer(&mode)))
	if errno != 0 {
		return fmt.Errorf("failed to set SPI mode: %v", errno)
	}

	// Set bits per word
	const SPI_IOC_WR_BITS_PER_WORD = 0x40016B03
	bits := uint8(b.config.BitsPerWord)
	_, _, errno = syscall.Syscall(syscall.SYS_IOCTL, b.fd.Fd(), SPI_IOC_WR_BITS_PER_WORD, uintptr(unsafe.Pointer(&bits)))
	if errno != 0 {
		return fmt.Errorf("failed to set SPI bits per word: %v", errno)
	}

	// Set max speed
	const SPI_IOC_WR_MAX_SPEED_HZ = 0x40046B04
	speed := b.config.SpeedHz
	_, _, errno = syscall.Syscall(syscall.SYS_IOCTL, b.fd.Fd(), SPI_IOC_WR_MAX_SPEED_HZ, uintptr(unsafe.Pointer(&speed)))
	if errno != 0 {
		return fmt.Errorf("failed to set SPI speed: %v", errno)
	}

	return nil
}

// SetConfig updates the SPI configuration.
func (b *LinuxSPI) SetConfig(config SPIConfig) error {
	b.config = config
	return b.configure()
}

// GetConfig returns the current SPI configuration.
func (b *LinuxSPI) GetConfig() SPIConfig {
	return b.config
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
		length:      uint32(length),
		speedHz:     b.config.SpeedHz,
		bitsPerWord: b.config.BitsPerWord,
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
