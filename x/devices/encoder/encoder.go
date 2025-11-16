// Package encoder provides a quadrature encoder driver that uses pin interrupts
// to efficiently track motor position and calculate rotation speed.
//
// Implementation based on TinyGo drivers:
// https://raw.githubusercontent.com/tinygo-org/drivers/refs/heads/release/encoders/quadrature_interrupt.go
//
// The encoder uses 4x decoding (counts all edges) for maximum resolution.
package encoder

import (
	"sync/atomic"
	"time"

	"github.com/itohio/EasyRobot/x/devices"
)

// Quadrature state transition lookup table.
// Maps 4-bit state transition (oldAB << 2 | newAB) to delta: -1, 0, or 1.
// Based on TinyGo implementation.
var states = [16]int8{0, -1, 1, 0, 1, 0, 0, -1, -1, 0, 0, 1, 0, 1, -1, 0}

// Device represents a quadrature encoder with position and speed tracking.
type Device struct {
	pinA, pinB devices.Pin

	// Position tracking (atomic access - read from multiple goroutines)
	position int64

	// State tracking (only accessed from interrupt handler, no atomics needed)
	// oldAB stores the last 4 bits of state transitions (2 bits per transition)
	// Bits: [oldA, oldB, newA, newB] where A=bit1, B=bit0
	oldAB uint32

	// For RPM calculation (atomic access - read from multiple goroutines)
	lastPosition int64 // Last position used for RPM calculation
	lastRPMTime  int64 // Last time RPM was calculated (microseconds)
	rpm          int64 // RPM * 1000 (fixed point)

	// Configuration
	countsPerRevolution int64         // Number of encoder counts per full revolution
	updateInterval      time.Duration // How often to update RPM
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
func New(pinA, pinB devices.Pin, config Config) *Device {
	if config.CountsPerRevolution == 0 {
		config.CountsPerRevolution = 2048
	}
	if config.UpdateInterval == 0 {
		config.UpdateInterval = 100 * time.Millisecond
	}
	return &Device{
		pinA:                pinA,
		pinB:                pinB,
		countsPerRevolution: config.CountsPerRevolution,
		updateInterval:      config.UpdateInterval,
	}
}

// Configure sets up the encoder pins and enables interrupts.
func (d *Device) Configure() error {
	if err := configurePins(d.pinA, d.pinB); err != nil {
		return err
	}

	// Read initial state and pack into oldAB
	// Initial state: both pins read, stored as bits [A, B] = [bit1, bit0]
	initialAB := uint32(0)
	if d.pinA.Get() {
		initialAB |= 0x02 // A = bit 1
	}
	if d.pinB.Get() {
		initialAB |= 0x01 // B = bit 0
	}
	// Set initial state as if we've already seen this state twice
	// (so first transition will work correctly)
	d.oldAB = initialAB | (initialAB << 2)

	// Set up interrupt handlers for both pins using PinToggle (both edges)
	// Both pins use the same interrupt handler
	if err := d.pinA.SetInterrupt(devices.PinToggle, d.interrupt); err != nil {
		return err
	}
	if err := d.pinB.SetInterrupt(devices.PinToggle, d.interrupt); err != nil {
		return err
	}

	// Initialize timing
	now := time.Now().UnixMicro()
	atomic.StoreInt64(&d.lastRPMTime, now)

	return nil
}

// interrupt is the single interrupt handler for both pins.
// Based on TinyGo implementation using lookup table.
// Only accessed from interrupt context (single goroutine), so oldAB doesn't need atomics.
func (d *Device) interrupt(pin devices.Pin) {
	// Read both pins
	aHigh := d.pinA.Get()
	bHigh := d.pinB.Get()

	// Shift old state left by 2 bits and add new state
	d.oldAB <<= 2
	if aHigh {
		d.oldAB |= 0x02 // A = bit 1
	}
	if bHigh {
		d.oldAB |= 0x01 // B = bit 0
	}

	// Look up delta from state transition table
	delta := int64(states[d.oldAB&0x0f])

	// Update position atomically (may be read from other goroutines)
	if delta != 0 {
		atomic.AddInt64(&d.position, delta)
	}
}

// Position returns the current encoder position in counts.
func (d *Device) Position() int64 {
	return atomic.LoadInt64(&d.position)
}

// Reset sets the encoder position to zero.
func (d *Device) Reset() {
	atomic.StoreInt64(&d.position, 0)
	atomic.StoreInt64(&d.lastPosition, 0)
	now := time.Now().UnixMicro()
	atomic.StoreInt64(&d.lastRPMTime, now)
	atomic.StoreInt64(&d.rpm, 0)
}

// RPM returns the current rotation speed in RPM (revolutions per minute).
// The value is calculated based on position changes over time.
func (d *Device) RPM() float64 {
	d.updateRPM()
	rpmFixed := atomic.LoadInt64(&d.rpm)
	return float64(rpmFixed) / 1000.0
}

// updateRPM calculates RPM based on position change over time.
// This should be called periodically or after position changes.
func (d *Device) updateRPM() {
	now := time.Now().UnixMicro()
	lastRPMTime := atomic.LoadInt64(&d.lastRPMTime)

	if time.Duration(now-lastRPMTime) < d.updateInterval {
		return
	}

	currentPos := atomic.LoadInt64(&d.position)
	lastPos := atomic.LoadInt64(&d.lastPosition)
	deltaPos := currentPos - lastPos
	deltaTime := now - lastRPMTime

	if deltaTime > 0 {
		// RPM = (deltaPos / countsPerRevolution) / (deltaTime / 60_000_000 microseconds)
		//     = (deltaPos * 60_000_000) / (countsPerRevolution * deltaTime)
		// Store as fixed point (RPM * 1000)
		rpmFixed := (deltaPos * 60_000_000_000) / (d.countsPerRevolution * deltaTime)
		atomic.StoreInt64(&d.rpm, rpmFixed)
	}

	atomic.StoreInt64(&d.lastPosition, currentPos)
	atomic.StoreInt64(&d.lastRPMTime, now)
}

// SetCountsPerRevolution updates the counts per revolution setting.
func (d *Device) SetCountsPerRevolution(counts int64) {
	d.countsPerRevolution = counts
}

// CountsPerRevolution returns the configured counts per revolution.
func (d *Device) CountsPerRevolution() int64 {
	return d.countsPerRevolution
}
