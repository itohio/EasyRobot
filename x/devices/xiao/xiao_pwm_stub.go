//go:build !sam || !xiao

package xiao

import (
	"github.com/itohio/EasyRobot/x/devices"
)

// Stub implementation for non-XIAO platforms

// NewPWMDevice returns a stub PWM device that always returns errors.
func NewPWMDevice() devices.PWMDevice {
	return &stubPWMDevice{}
}

type stubPWMDevice struct{}

func (s *stubPWMDevice) Channel(pin devices.Pin) (devices.PWM, error) {
	return nil, devices.ErrNotSupported
}

func (s *stubPWMDevice) Configure(frequency uint32) error {
	return devices.ErrNotSupported
}

func (s *stubPWMDevice) SetFrequency(frequency uint32) error {
	return devices.ErrNotSupported
}

