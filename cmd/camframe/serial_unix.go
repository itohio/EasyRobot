//go:build darwin || linux

package main

import (
	"fmt"

	"golang.org/x/sys/unix"
)

// port is a serial port opened raw 8N1 without flow control. Read polls:
// it returns 0, nil after ~100 ms without data (VMIN=0, VTIME=1).
type port struct{ fd int }

// openPort opens path at baud. It never writes to the device; DTR and RTS
// are deasserted right after open and again after configuration.
func openPort(path string, baud int) (*port, error) {
	if baud <= 0 {
		return nil, fmt.Errorf("serial %s: invalid baud %d", path, baud)
	}
	fd, err := unix.Open(path, unix.O_RDWR|unix.O_NOCTTY|unix.O_NONBLOCK|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("serial %s: open: %w", path, err)
	}
	p := &port{fd: fd}
	if err := p.setup(baud); err != nil {
		unix.Close(fd)
		return nil, fmt.Errorf("serial %s: %w", path, err)
	}
	return p, nil
}

func (p *port) setup(baud int) error {
	if err := p.clearModem(); err != nil {
		return err
	}
	if err := configure(p.fd, baud); err != nil {
		return err
	}
	if err := p.clearModem(); err != nil {
		return err
	}
	if err := unix.SetNonblock(p.fd, false); err != nil {
		return fmt.Errorf("clear O_NONBLOCK: %w", err)
	}
	return nil
}

// clearModem deasserts DTR and RTS.
func (p *port) clearModem() error {
	if err := unix.IoctlSetPointerInt(p.fd, unix.TIOCMBIC, unix.TIOCM_DTR|unix.TIOCM_RTS); err != nil {
		return fmt.Errorf("deassert DTR/RTS: %w", err)
	}
	return nil
}

func (p *port) Read(b []byte) (int, error) {
	for {
		n, err := unix.Read(p.fd, b)
		if err == unix.EINTR {
			continue
		}
		if n < 0 {
			n = 0
		}
		return n, err
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
