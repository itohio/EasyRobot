package main

import (
	"fmt"

	"golang.org/x/sys/unix"
)

// configure sets fd raw 8N1, no flow control, CLOCAL|CREAD, HUPCL cleared,
// VMIN=0/VTIME=1 polling reads, and an arbitrary baud rate through termios2
// with BOTHER.
func configure(fd, baud int) error {
	t, err := unix.IoctlGetTermios(fd, unix.TCGETS2)
	if err != nil {
		return fmt.Errorf("TCGETS2: %w", err)
	}
	t.Iflag &^= unix.IGNBRK | unix.BRKINT | unix.PARMRK | unix.ISTRIP | unix.INLCR | unix.IGNCR | unix.ICRNL |
		unix.IXON | unix.IXOFF | unix.IXANY | unix.INPCK
	t.Oflag &^= unix.OPOST
	t.Lflag &^= unix.ECHO | unix.ECHONL | unix.ICANON | unix.ISIG | unix.IEXTEN
	t.Cflag &^= unix.CSIZE | unix.PARENB | unix.CSTOPB | unix.CRTSCTS | unix.HUPCL | unix.CBAUD | unix.CIBAUD
	t.Cflag |= unix.CS8 | unix.CLOCAL | unix.CREAD | unix.BOTHER | unix.BOTHER<<unix.IBSHIFT
	t.Cc[unix.VMIN] = 0
	t.Cc[unix.VTIME] = 1
	t.Ispeed = uint32(baud)
	t.Ospeed = uint32(baud)
	if err := unix.IoctlSetTermios(fd, unix.TCSETS2, t); err != nil {
		return fmt.Errorf("TCSETS2 baud %d: %w", baud, err)
	}
	return nil
}
