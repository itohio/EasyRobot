package tca9548a

import (
	"github.com/itohio/EasyRobot/x/devices"
)

// Router manages a TCA9548A I2C multiplexer and provides channel-specific I2C interfaces.
type Router struct {
	mux            *Device
	baseBus        devices.I2C
	channels       [8]*ChannelI2C
	currentChannel uint8
}

// NewRouter creates a new TCA9548A router.
// The base I2C bus must already be configured.
// The bus can be either a TinyGo machine.I2C (wrapped with devices.NewTinyGoI2C)
// or a Linux I2C bus (created with devices.NewLinuxI2C).
func NewRouter(baseBus devices.I2C, muxAddress uint8) *Router {
	mux := New(baseBus, muxAddress)
	router := &Router{
		mux:            mux,
		baseBus:        baseBus,
		currentChannel: 0xFF, // Invalid channel to force first switch
	}

	// Pre-create all channel wrappers
	for i := uint8(0); i < 8; i++ {
		router.channels[i] = &ChannelI2C{
			router:  router,
			channel: i,
		}
	}

	return router
}

// Configure initializes the router. If init is true, resets all channels.
func (r *Router) Configure(init bool) error {
	return r.mux.Configure(init)
}

// Channel returns an I2C interface wrapper for the specified channel (0-7).
// This wrapper implements devices.I2C and automatically routes traffic to the correct channel.
func (r *Router) Channel(channel uint8) (devices.I2C, error) {
	if channel > 7 {
		return nil, ErrInvalidChannel
	}
	return r.channels[channel], nil
}

// Reset disables all channels.
func (r *Router) Reset() error {
	r.currentChannel = 0xFF // Reset to invalid channel
	return r.mux.Reset()
}

// setChannel switches to the specified channel if not already active.
func (r *Router) setChannel(channel uint8) error {
	if r.currentChannel == channel {
		return nil // Already on the correct channel
	}

	if err := r.mux.SetChannel(channel); err != nil {
		return err
	}

	r.currentChannel = channel
	return nil
}

// ChannelI2C is an I2C wrapper that automatically routes traffic through a specific
// TCA9548A channel. It implements the devices.I2C interface.
type ChannelI2C struct {
	router  *Router
	channel uint8
}

// ReadRegister reads a register value from the device.
func (c *ChannelI2C) ReadRegister(addr uint8, r uint8, buf []byte) error {
	// Switch to this channel before the transaction
	if err := c.router.setChannel(c.channel); err != nil {
		return err
	}

	// Perform the actual I2C transaction on the base bus
	return c.router.baseBus.ReadRegister(addr, r, buf)
}

// WriteRegister writes a register value to the device.
func (c *ChannelI2C) WriteRegister(addr uint8, r uint8, buf []byte) error {
	// Switch to this channel before the transaction
	if err := c.router.setChannel(c.channel); err != nil {
		return err
	}

	// Perform the actual I2C transaction on the base bus
	return c.router.baseBus.WriteRegister(addr, r, buf)
}

// Tx performs an I2C transaction, automatically switching to the correct channel first.
func (c *ChannelI2C) Tx(addr uint16, w, r []byte) error {
	// Switch to this channel before the transaction
	if err := c.router.setChannel(c.channel); err != nil {
		return err
	}

	// Perform the actual I2C transaction on the base bus
	return c.router.baseBus.Tx(addr, w, r)
}
