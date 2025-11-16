//go:build !tinygo && linux

package devices

import (
	"fmt"
	"os"
	"sync"
	"syscall"
)

// PinChange represents one or more trigger events that can happen on a given GPIO pin
// on the RP2040. ORed PinChanges are valid input to most IRQ functions.
type PinChange uint8

// Pin change interrupt constants for SetInterrupt.
const (
	// Edge falling
	PinFalling PinChange = 4 << iota
	// Edge rising
	PinRising

	PinToggle = PinFalling | PinRising
)

// LinuxPin implements Pin interface using Linux sysfs GPIO.
type LinuxPin struct {
	pinNum int
	value  *os.File

	// Interrupt handling
	mu       sync.Mutex
	callback func(Pin)
	stopCh   chan struct{}
	doneCh   chan struct{}
}

// NewLinuxPin creates a new Pin interface for Linux GPIO.
// pinNum is the GPIO pin number (e.g., 18 for GPIO18).
// The pin must be exported first (e.g., echo 18 > /sys/class/gpio/export).
func NewPin(pinNum int) (*LinuxPin, error) {
	valuePath := fmt.Sprintf("/sys/class/gpio/gpio%d/value", pinNum)
	value, err := os.OpenFile(valuePath, os.O_RDWR, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to open GPIO pin %d: %w (ensure pin is exported)", pinNum, err)
	}

	return &LinuxPin{
		pinNum: pinNum,
		value:  value,
	}, nil
}

// Get returns the current pin state.
func (p *LinuxPin) Get() bool {
	buf := make([]byte, 1)
	_, err := p.value.ReadAt(buf, 0)
	if err != nil {
		return false
	}
	return buf[0] == '1'
}

// Set sets the pin state.
func (p *LinuxPin) Set(value bool) {
	var b byte = '0'
	if value {
		b = '1'
	}
	p.value.WriteAt([]byte{b}, 0)
}

// High sets the pin to high.
func (p *LinuxPin) High() {
	p.value.WriteAt([]byte{'1'}, 0)
}

// Low sets the pin to low.
func (p *LinuxPin) Low() {
	p.value.WriteAt([]byte{'0'}, 0)
}

// SetInterrupt sets up an interrupt on the pin for the selected change type.
// The callback is called with the pin as its argument when the edge is detected.
func (p *LinuxPin) SetInterrupt(change PinChange, callback func(Pin)) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Stop existing interrupt handler if any
	if p.stopCh != nil {
		close(p.stopCh)
		<-p.doneCh
		p.stopCh = nil
		p.doneCh = nil
	}

	if callback == nil {
		// Disable interrupt by setting edge to "none"
		edgePath := fmt.Sprintf("/sys/class/gpio/gpio%d/edge", p.pinNum)
		return os.WriteFile(edgePath, []byte("none"), 0)
	}

	// Map PinChange to sysfs edge value
	var edge string
	switch {
	case change&PinToggle == PinToggle:
		edge = "both"
	case change&PinRising != 0:
		edge = "rising"
	case change&PinFalling != 0:
		edge = "falling"
	default:
		return fmt.Errorf("invalid PinChange value: %d", change)
	}

	// Set edge trigger mode
	edgePath := fmt.Sprintf("/sys/class/gpio/gpio%d/edge", p.pinNum)
	if err := os.WriteFile(edgePath, []byte(edge), 0); err != nil {
		return fmt.Errorf("failed to set edge trigger for GPIO %d: %w", p.pinNum, err)
	}

	// Store callback and start interrupt handler
	p.callback = callback
	p.stopCh = make(chan struct{})
	p.doneCh = make(chan struct{})

	go p.interruptLoop()

	return nil
}

// interruptLoop monitors the GPIO value file for edge changes using epoll.
func (p *LinuxPin) interruptLoop() {
	defer close(p.doneCh)

	// Create epoll instance
	epfd, err := syscall.EpollCreate1(0)
	if err != nil {
		return
	}
	defer syscall.Close(epfd)

	// Get file descriptor for the value file
	fd := int(p.value.Fd())

	// Add file descriptor to epoll
	// Combine epoll events: EPOLLIN (readable), EPOLLET (edge-triggered), EPOLLPRI (priority data)
	// Use int64 intermediate to handle negative constants, then cast to uint32
	eventsInt := int64(syscall.EPOLLIN) | int64(syscall.EPOLLET) | int64(syscall.EPOLLPRI)
	event := syscall.EpollEvent{
		Events: uint32(eventsInt),
		Fd:     int32(fd),
	}
	if err := syscall.EpollCtl(epfd, syscall.EPOLL_CTL_ADD, fd, &event); err != nil {
		return
	}

	// Buffer for reading (to clear the event)
	buf := make([]byte, 1)
	epollEvents := make([]syscall.EpollEvent, 1)

	for {
		select {
		case <-p.stopCh:
			return
		default:
			// Wait for event with 100ms timeout
			n, err := syscall.EpollWait(epfd, epollEvents, 100)
			if err != nil {
				if err == syscall.EINTR {
					continue
				}
				return
			}

			if n > 0 {
				// Read value to clear the event
				p.value.ReadAt(buf, 0)

				// Call callback
				p.mu.Lock()
				cb := p.callback
				p.mu.Unlock()

				if cb != nil {
					cb(p)
				}
			}
		}
	}
}

// Close closes the GPIO pin file and stops interrupt handling.
func (p *LinuxPin) Close() error {
	p.mu.Lock()
	if p.stopCh != nil {
		close(p.stopCh)
		done := p.doneCh
		p.mu.Unlock()
		<-done
		p.mu.Lock()
	}
	p.mu.Unlock()

	return p.value.Close()
}

// PinNum returns the GPIO pin number.
func (p *LinuxPin) PinNum() int {
	return p.pinNum
}
