package main

import (
	"fmt"

	"golang.org/x/sys/unix"
)

// iossiospeed is IOSSIOSPEED = _IOW('T', 2, speed_t) from <IOKit/serial/ioss.h>:
// it sets an arbitrary baud rate after tcsetattr.
const iossiospeed = 0x80085402

// configure sets fd raw 8N1, no flow control, CLOCAL|CREAD, HUPCL cleared,
// VMIN=0/VTIME=1 polling reads, then the baud rate via IOSSIOSPEED.
func configure(fd, baud int) error {
	t, err := unix.IoctlGetTermios(fd, unix.TIOCGETA)
	if err != nil {
		return fmt.Errorf("tcgetattr: %w", err)
	}
	t.Iflag &^= unix.IGNBRK | unix.BRKINT | unix.PARMRK | unix.ISTRIP | unix.INLCR | unix.IGNCR | unix.ICRNL |
		unix.IXON | unix.IXOFF | unix.IXANY | unix.INPCK
	t.Oflag &^= unix.OPOST
	t.Lflag &^= unix.ECHO | unix.ECHONL | unix.ICANON | unix.ISIG | unix.IEXTEN
	t.Cflag &^= unix.CSIZE | unix.PARENB | unix.CSTOPB | unix.CRTSCTS | unix.HUPCL
	t.Cflag |= unix.CS8 | unix.CLOCAL | unix.CREAD
	t.Cc[unix.VMIN] = 0
	t.Cc[unix.VTIME] = 1
	// tcsetattr only takes the classic rates; use a placeholder and set the
	// real one with IOSSIOSPEED below.
	t.Ispeed, t.Ospeed = unix.B9600, unix.B9600
	if err := unix.IoctlSetTermios(fd, unix.TIOCSETA, t); err != nil {
		return fmt.Errorf("tcsetattr: %w", err)
	}
	// IOSSIOSPEED reads an 8-byte speed_t through its pointer argument.
	// x/sys exposes no 8-byte pointer ioctl, but Winsize is four uint16s
	// (8 bytes) and goes through the same libc ioctl path; lay the speed
	// out little-endian in it (darwin runs only on little-endian CPUs).
	s := uint64(baud)
	ws := unix.Winsize{Row: uint16(s), Col: uint16(s >> 16), Xpixel: uint16(s >> 32), Ypixel: uint16(s >> 48)}
	if err := unix.IoctlSetWinsize(fd, iossiospeed, &ws); err != nil {
		return fmt.Errorf("IOSSIOSPEED %d: %w", baud, err)
	}
	return nil
}
