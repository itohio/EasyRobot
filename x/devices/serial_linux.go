//go:build !tinygo && linux

package devices

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
)

// LinuxSerial implements Serial interface using Linux serial port.
type LinuxSerial struct {
	file   *os.File
	config SerialConfig
}

// NewSerial creates a new Serial interface for Linux.
// Common Linux serial devices:
//   - /dev/ttyUSB0, /dev/ttyUSB1, etc. - USB-to-serial adapters
//   - /dev/ttyAMA0 - Primary UART (on Raspberry Pi, GPIO 14/15)
//   - /dev/ttyS0 - Serial port (on Raspberry Pi, Mini UART)
//   - /dev/serial0 - Symlink to primary UART (on Raspberry Pi)
func NewSerial(device string) (*LinuxSerial, error) {
	return NewSerialWithConfig(device, DefaultSerialConfig())
}

// NewSerialWithConfig creates a new Serial interface with custom configuration.
func NewSerialWithConfig(device string, config SerialConfig) (*LinuxSerial, error) {
	file, err := os.OpenFile(device, os.O_RDWR|syscall.O_NOCTTY|syscall.O_NONBLOCK, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to open serial device %s: %w (ensure user is in dialout group)", device, err)
	}

	// Set to blocking mode (clear O_NONBLOCK flag)
	flags, _, errno := syscall.Syscall(syscall.SYS_FCNTL, file.Fd(), syscall.F_GETFL, 0)
	if errno != 0 {
		file.Close()
		return nil, fmt.Errorf("failed to get file flags: %v", errno)
	}
	flags &^= syscall.O_NONBLOCK
	_, _, errno = syscall.Syscall(syscall.SYS_FCNTL, file.Fd(), syscall.F_SETFL, flags)
	if errno != 0 {
		file.Close()
		return nil, fmt.Errorf("failed to set blocking mode: %v", errno)
	}

	// Configure serial port parameters using termios
	termios, err := unix.IoctlGetTermios(int(file.Fd()), unix.TCGETS)
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to get termios: %w", err)
	}

	// Set baud rate
	baudRate := config.BaudRate
	if baudRate == 0 {
		baudRate = 115200 // Default if not set
	}
	baudConst := baudRateToConstant(baudRate)
	if baudConst == 0 {
		// Try to set arbitrary baud rate using BOTHER
		// Clear old baud rate flags
		termios.Cflag &^= unix.CBAUD
		termios.Cflag |= unix.BOTHER
		termios.Ispeed = uint32(baudRate)
		termios.Ospeed = uint32(baudRate)
	} else {
		termios.Ispeed = baudConst
		termios.Ospeed = baudConst
	}

	// Set 8N1 (8 data bits, no parity, 1 stop bit)
	termios.Cflag &^= unix.CSIZE | unix.PARENB | unix.CSTOPB
	termios.Cflag |= unix.CS8
	termios.Cflag |= unix.CREAD | unix.CLOCAL // Enable receiver, ignore modem control lines

	// Disable canonical mode and echo
	termios.Lflag &^= unix.ICANON | unix.ECHO | unix.ECHOE | unix.ISIG

	// Set minimum characters and timeout
	termios.Cc[unix.VMIN] = 0
	termios.Cc[unix.VTIME] = 0

	if err := unix.IoctlSetTermios(int(file.Fd()), unix.TCSETS, termios); err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to set termios: %w", err)
	}

	return &LinuxSerial{
		file:   file,
		config: config,
	}, nil
}

// baudRateToConstant converts a baud rate to the corresponding termios constant.
func baudRateToConstant(baud int) uint32 {
	switch baud {
	case 50:
		return unix.B50
	case 75:
		return unix.B75
	case 110:
		return unix.B110
	case 134:
		return unix.B134
	case 150:
		return unix.B150
	case 200:
		return unix.B200
	case 300:
		return unix.B300
	case 600:
		return unix.B600
	case 1200:
		return unix.B1200
	case 1800:
		return unix.B1800
	case 2400:
		return unix.B2400
	case 4800:
		return unix.B4800
	case 9600:
		return unix.B9600
	case 19200:
		return unix.B19200
	case 38400:
		return unix.B38400
	case 57600:
		return unix.B57600
	case 115200:
		return unix.B115200
	case 230400:
		return unix.B230400
	case 460800:
		return unix.B460800
	case 500000:
		return unix.B500000
	case 576000:
		return unix.B576000
	case 921600:
		return unix.B921600
	case 1000000:
		return unix.B1000000
	case 1152000:
		return unix.B1152000
	case 1500000:
		return unix.B1500000
	case 2000000:
		return unix.B2000000
	case 2500000:
		return unix.B2500000
	case 3000000:
		return unix.B3000000
	case 3500000:
		return unix.B3500000
	case 4000000:
		return unix.B4000000
	default:
		return 0
	}
}

// Read reads data from the serial port.
// On Linux, os.File.Read() can incorrectly return EOF for serial ports.
// We use syscall.Read directly to avoid this issue.
func (s *LinuxSerial) Read(p []byte) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}
	n, err = syscall.Read(int(s.file.Fd()), p)
	if err != nil {
		// syscall.Read returns syscall.Errno, convert to error
		if err == syscall.EAGAIN || err == syscall.EWOULDBLOCK {
			// No data available, return 0, nil (not an error)
			return 0, nil
		}
		return n, err
	}
	if n == 0 {
		// No data read, but not an error
		return 0, nil
	}
	return n, nil
}

// Write writes data to the serial port.
func (s *LinuxSerial) Write(p []byte) (n int, err error) {
	return s.file.Write(p)
}

// Buffered returns the number of bytes currently in the receive buffer.
// Only works if EnableBuffering was set to true in the configuration.
// Returns 0 if buffering is disabled (default) to avoid issues with devices
// like Arduino that expect immediate, unbuffered communication.
func (s *LinuxSerial) Buffered() int {
	if !s.config.EnableBuffering {
		return 0
	}

	const FIONREAD = 0x541B
	var n int32
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, s.file.Fd(), FIONREAD, uintptr(unsafe.Pointer(&n)))
	if errno != 0 {
		return 0
	}
	return int(n)
}

// Close closes the serial port.
func (s *LinuxSerial) Close() error {
	return s.file.Close()
}

// File returns the underlying *os.File.
func (s *LinuxSerial) File() *os.File {
	return s.file
}
