//go:build darwin || linux

package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"golang.org/x/sys/unix"
)

// port is the board's USB CDC serial port, raw, polled reads (0, nil after
// ~100 ms without data).
type port struct{ fd int }

// openPort opens path raw and exclusively (TIOCEXCL) without resetting the
// board: O_NONBLOCK open, DTR/RTS deasserted at once, HUPCL cleared (an
// ESP32-S3 USB-Serial/JTAG resets on the DTR/RTS sequence a plain open
// produces).
func openPort(path string) (*port, error) {
	fd, err := unix.Open(path, unix.O_RDWR|unix.O_NOCTTY|unix.O_NONBLOCK|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	if err := setup(fd); err != nil {
		unix.Close(fd)
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &port{fd: fd}, nil
}

func setup(fd int) error {
	// Exclusive: a second open (grab, another client) fails with EBUSY
	// instead of sending commands into the stream and stealing its bytes.
	if err := unix.IoctlSetInt(fd, unix.TIOCEXCL, 0); err != nil {
		return fmt.Errorf("exclusive access: %w", err)
	}
	if err := unix.IoctlSetPointerInt(fd, unix.TIOCMBIC, unix.TIOCM_DTR|unix.TIOCM_RTS); err != nil {
		return fmt.Errorf("deassert DTR/RTS: %w", err)
	}
	t, err := unix.IoctlGetTermios(fd, ioctlGetTermios)
	if err != nil {
		return fmt.Errorf("tcgetattr: %w", err)
	}
	t.Iflag &^= unix.IGNBRK | unix.BRKINT | unix.PARMRK | unix.ISTRIP | unix.INLCR | unix.IGNCR | unix.ICRNL | unix.IXON | unix.IXOFF | unix.IXANY | unix.INPCK
	t.Oflag &^= unix.OPOST
	t.Lflag &^= unix.ECHO | unix.ECHONL | unix.ICANON | unix.ISIG | unix.IEXTEN
	t.Cflag &^= unix.CSIZE | unix.PARENB | unix.CSTOPB | unix.CRTSCTS | unix.HUPCL
	t.Cflag |= unix.CS8 | unix.CLOCAL | unix.CREAD
	t.Cc[unix.VMIN], t.Cc[unix.VTIME] = 0, 1
	if err := unix.IoctlSetTermios(fd, ioctlSetTermios, t); err != nil {
		return fmt.Errorf("tcsetattr: %w", err)
	}
	if err := unix.IoctlSetPointerInt(fd, unix.TIOCMBIC, unix.TIOCM_DTR|unix.TIOCM_RTS); err != nil {
		return fmt.Errorf("deassert DTR/RTS: %w", err)
	}
	if err := unix.SetNonblock(fd, false); err != nil {
		return fmt.Errorf("clear O_NONBLOCK: %w", err)
	}
	return nil
}

func (p *port) Read(b []byte) (int, error) {
	for {
		n, err := unix.Read(p.fd, b)
		if err == unix.EINTR {
			continue
		}
		return max(n, 0), err
	}
}

func (p *port) Write(b []byte) (int, error) {
	done := 0
	for done < len(b) {
		n, err := unix.Write(p.fd, b[done:])
		if err == unix.EINTR {
			continue
		}
		if err != nil {
			return done, err
		}
		done += n
	}
	return done, nil
}

func (p *port) Close() error { return unix.Close(p.fd) }

// findPort returns the serial device of the USB device with serial number
// sn: on macOS its IOCalloutDevice from ioreg, on Linux the ttyACM/ttyUSB
// whose USB device has that serial.
func findPort(sn string) (string, error) {
	if runtime.GOOS == "darwin" {
		out, err := exec.Command("ioreg", "-r", "-c", "IOUSBHostDevice", "-l").Output()
		if err != nil {
			return "", fmt.Errorf("ioreg: %w", err)
		}
		if p := calloutFor(out, sn); p != "" {
			return p, nil
		}
	} else {
		ttys, _ := filepath.Glob("/sys/class/tty/tty*/device")
		for _, d := range ttys {
			for _, rel := range []string{"../serial", "../../serial"} {
				b, err := os.ReadFile(filepath.Join(d, rel))
				if err == nil && strings.TrimSpace(string(b)) == sn {
					return "/dev/" + filepath.Base(filepath.Dir(d)), nil
				}
			}
		}
	}
	return "", fmt.Errorf("no USB device with serial %s (board not plugged in?)", sn)
}
