// Package encoder provides a quadrature encoder driver that uses pin interrupts
// to efficiently track motor position and calculate rotation speed.
//
// The encoder uses 4x decoding (counts all edges) for maximum resolution.
package encoder

import (
	"sync/atomic"
	"time"

	"github.com/itohio/EasyRobot/x/devices"
)

// Encoder represents a quadrature encoder with position and speed tracking.
type Encoder struct {
	pinA, pinB devices.Pin
	position   int64 // Atomic access
	lastTime   int64 // Atomic access - microseconds since epoch

	// For RPM calculation
	lastPosition int64 // Last position used for RPM calculation
	lastRPMTime  int64 // Last time RPM was calculated
	rpm          int64 // Atomic access - RPM * 1000 (fixed point)

	// Configuration
	countsPerRevolution int64         // Number of encoder counts per full revolution
	updateInterval      time.Duration // How often to update RPM

	// Internal state (protected by atomic operations)
	// Store state as uint32: bits 0-1 = last state (A=bit0, B=bit1)
	lastState uint32 // Atomic access
}

// Config holds configuration for an encoder.
type Config struct {
	CountsPerRevolution int64         // Number of encoder counts per full revolution (default: 2048 for typical 512 PPR encoder with 4x decoding)
	UpdateInterval      time.Duration // How often to update RPM calculation (default: 100ms)
}

// DefaultConfig returns a default configuration.
func DefaultConfig() Config {
	return Config{
		CountsPerRevolution: 2048, // 512 PPR * 4 (4x decoding)
		UpdateInterval:      100 * time.Millisecond,
	}
}

// New creates a new quadrature encoder.
// The pins must support interrupts (typically any GPIO pin on most MCUs).
func New(pinA, pinB devices.Pin, config Config) *Encoder {
	if config.CountsPerRevolution == 0 {
		config.CountsPerRevolution = 2048
	}
	if config.UpdateInterval == 0 {
		config.UpdateInterval = 100 * time.Millisecond
	}
	enc := &Encoder{
		pinA:                pinA,
		pinB:                pinB,
		countsPerRevolution: config.CountsPerRevolution,
		updateInterval:      config.UpdateInterval,
	}
	return enc
}

// Configure sets up the encoder pins and enables interrupts.
func (e *Encoder) Configure() error {
	configurePins(e.pinA, e.pinB)

	// Read initial state and store atomically
	// Pack A and B state into uint32: bit 0 = A, bit 1 = B
	initialState := uint32(0)
	if e.pinA.Get() {
		initialState |= 0x01
	}
	if e.pinB.Get() {
		initialState |= 0x02
	}
	atomic.StoreUint32(&e.lastState, initialState)

	// Set up interrupt handlers for both pins, both rising and falling edges
	if err := e.pinA.SetInterrupt(devices.PinRising|devices.PinFalling, e.handleA); err != nil {
		return err
	}
	if err := e.pinB.SetInterrupt(devices.PinRising|devices.PinFalling, e.handleB); err != nil {
		return err
	}

	// Initialize timing
	now := time.Now().UnixMicro()
	atomic.StoreInt64(&e.lastTime, now)
	atomic.StoreInt64(&e.lastRPMTime, now)

	return nil
}

// Position returns the current encoder position in counts.
func (e *Encoder) Position() int64 {
	return atomic.LoadInt64(&e.position)
}

// Reset sets the encoder position to zero.
func (e *Encoder) Reset() {
	atomic.StoreInt64(&e.position, 0)
	atomic.StoreInt64(&e.lastPosition, 0)
	now := time.Now().UnixMicro()
	atomic.StoreInt64(&e.lastRPMTime, now)
	atomic.StoreInt64(&e.rpm, 0)
}

// RPM returns the current rotation speed in RPM (revolutions per minute).
// The value is calculated based on position changes over time.
func (e *Encoder) RPM() float64 {
	e.updateRPM()
	rpmFixed := atomic.LoadInt64(&e.rpm)
	return float64(rpmFixed) / 1000.0
}

// updateRPM calculates RPM based on position change over time.
// This should be called periodically or after position changes.
func (e *Encoder) updateRPM() {
	now := time.Now().UnixMicro()
	lastRPMTime := atomic.LoadInt64(&e.lastRPMTime)

	if time.Duration(now-lastRPMTime) < e.updateInterval {
		return
	}

	currentPos := atomic.LoadInt64(&e.position)
	lastPos := atomic.LoadInt64(&e.lastPosition)
	deltaPos := currentPos - lastPos
	deltaTime := now - lastRPMTime

	if deltaTime > 0 {
		rpmFixed := (deltaPos * 60_000_000_000) / (e.countsPerRevolution * deltaTime)
		atomic.StoreInt64(&e.rpm, rpmFixed)
	}

	atomic.StoreInt64(&e.lastPosition, currentPos)
	atomic.StoreInt64(&e.lastRPMTime, now)
}

// handleA is the interrupt handler for pin A.
func (e *Encoder) handleA(pin devices.Pin) {
	aState := pin.Get()
	bState := e.pinB.Get()

	newState := uint32(0)
	if aState {
		newState |= 0x01
	}
	if bState {
		newState |= 0x02
	}

	lastState := atomic.LoadUint32(&e.lastState)
	lastA := (lastState & 0x01) != 0
	if aState != lastA {
		var delta int64
		if aState {
			if bState {
				delta = -1
			} else {
				delta = 1
			}
		} else {
			if bState {
				delta = 1
			} else {
				delta = -1
			}
		}
		atomic.AddInt64(&e.position, delta)
		now := time.Now().UnixMicro()
		atomic.StoreInt64(&e.lastTime, now)
		atomic.StoreUint32(&e.lastState, newState)
	}
}

// handleB is the interrupt handler for pin B.
func (e *Encoder) handleB(pin devices.Pin) {
	aState := e.pinA.Get()
	bState := pin.Get()

	newState := uint32(0)
	if aState {
		newState |= 0x01
	}
	if bState {
		newState |= 0x02
	}

	lastState := atomic.LoadUint32(&e.lastState)
	lastB := (lastState & 0x02) != 0
	if bState != lastB {
		var delta int64
		if bState {
			if aState {
				delta = 1
			} else {
				delta = -1
			}
		} else {
			if aState {
				delta = -1
			} else {
				delta = 1
			}
		}
		atomic.AddInt64(&e.position, delta)
		now := time.Now().UnixMicro()
		atomic.StoreInt64(&e.lastTime, now)
		atomic.StoreUint32(&e.lastState, newState)
	}
}

// SetCountsPerRevolution updates the counts per revolution setting.
func (e *Encoder) SetCountsPerRevolution(counts int64) {
	e.countsPerRevolution = counts
}

// CountsPerRevolution returns the configured counts per revolution.
func (e *Encoder) CountsPerRevolution() int64 {
	return e.countsPerRevolution
}
