//go:build !tinygo && windows

package devices

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

// WindowsSerial implements Serial interface using Windows COM ports.
type WindowsSerial struct {
	file   *os.File
	handle windows.Handle
	config SerialConfig
}

// NewSerial creates a new Serial interface for Windows.
// Common Windows serial devices:
//   - COM1, COM2, COM3, etc. - Serial ports
//   - COM ports are typically named "COM1", "COM2", etc.
func NewSerial(device string) (*WindowsSerial, error) {
	return NewSerialWithConfig(device, DefaultSerialConfig())
}

// NewSerialWithConfig creates a new Serial interface with custom configuration.
func NewSerialWithConfig(device string, config SerialConfig) (*WindowsSerial, error) {
	// Windows COM ports need the "\\\\.\\" prefix for ports above COM9
	// and for reliable access
	devicePath := device
	if len(device) > 4 && device[:4] == "COM" {
		devicePath = "\\\\.\\" + device
	}

	// Open the COM port using CreateFile
	deviceUTF16, err := windows.UTF16PtrFromString(devicePath)
	if err != nil {
		return nil, fmt.Errorf("invalid device path: %w", err)
	}

	handle, err := windows.CreateFile(
		deviceUTF16,
		windows.GENERIC_READ|windows.GENERIC_WRITE,
		0,
		nil,
		windows.OPEN_EXISTING,
		0,
		0,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to open serial device %s: %w", device, err)
	}

	file := os.NewFile(uintptr(handle), device)

	// Configure serial port parameters
	dcb := &windows.DCB{}
	if err := windows.GetCommState(handle, dcb); err != nil {
		windows.CloseHandle(handle)
		return nil, fmt.Errorf("failed to get serial state: %w", err)
	}

	// Set common defaults (115200 baud, 8N1)
	dcb.BaudRate = 115200
	dcb.ByteSize = 8
	dcb.Parity = windows.NOPARITY
	dcb.StopBits = windows.ONESTOPBIT

	if err := windows.SetCommState(handle, dcb); err != nil {
		windows.CloseHandle(handle)
		return nil, fmt.Errorf("failed to set serial state: %w", err)
	}

	// Set timeouts (no timeout for reads, immediate return)
	timeouts := &windows.CommTimeouts{
		ReadIntervalTimeout:        0xFFFFFFFF, // No interval timeout
		ReadTotalTimeoutMultiplier: 0,
		ReadTotalTimeoutConstant:   0,
		WriteTotalTimeoutMultiplier: 0,
		WriteTotalTimeoutConstant:  0,
	}
	if err := windows.SetCommTimeouts(handle, timeouts); err != nil {
		windows.CloseHandle(handle)
		return nil, fmt.Errorf("failed to set serial timeouts: %w", err)
	}

	return &WindowsSerial{
		file:   file,
		handle: handle,
		config: config,
	}, nil
}

// Read reads data from the serial port.
func (s *WindowsSerial) Read(p []byte) (n int, err error) {
	return s.file.Read(p)
}

// Write writes data to the serial port.
func (s *WindowsSerial) Write(p []byte) (n int, err error) {
	return s.file.Write(p)
}

// Buffered returns the number of bytes currently in the receive buffer.
// Only works if EnableBuffering was set to true in the configuration.
// Returns 0 if buffering is disabled (default) to avoid issues with devices
// like Arduino that expect immediate, unbuffered communication.
func (s *WindowsSerial) Buffered() int {
	if !s.config.EnableBuffering {
		return 0
	}

	var stat windows.ComStat
	if err := windows.ClearCommError(s.handle, nil, &stat); err != nil {
		return 0
	}
	return int(stat.InQue)
}

// Close closes the serial port.
func (s *WindowsSerial) Close() error {
	if s.handle != 0 {
		windows.CloseHandle(s.handle)
		s.handle = 0
	}
	return s.file.Close()
}

// File returns the underlying *os.File.
func (s *WindowsSerial) File() *os.File {
	return s.file
}

