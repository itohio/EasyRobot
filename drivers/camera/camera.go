// Package camera captures images from an SCCB-configured image sensor. The
// API is the same on every host; each host has its own backend file chosen
// by build tag (camera_k210.go: the K210 DVP). A host without a backend has
// no Identify/Open, so a firmware build for it fails at compile time.
//
// Per-host API (identical names; Pins is the host's own type):
//
//	func Identify(cfg Config, pins Pins) (sensor.ID, error)
//	func Open(cfg Config, pins Pins) (*Camera, error)
//	func (c *Camera) Capture(buf []byte) (Frame, error)
//	func (c *Camera) Close() error
package camera

import (
	"errors"

	"github.com/itohio/EasyRobot/drivers/camera/sensor"
)

// Config selects the sensor output and the sensor clock.
type Config struct {
	sensor.Mode
	XCLK uint32 // sensor clock in Hz; 0 = backend default
}

// Frame is one captured image; Data aliases the caller's buffer.
type Frame struct {
	Width, Height uint16
	Format        sensor.Format
	Seq           uint32
	Data          []byte
}

var (
	ErrTimeout    = errors.New("camera: timeout")
	ErrBufSize    = errors.New("camera: buffer too small for the frame")
	ErrNoDriver   = errors.New("camera: no driver for this sensor")
	ErrNoControls = errors.New("camera: sensor driver has no exposure/gain/white-balance controls")
)

// FrameSize is the buffer size Capture needs for cfg, or 0 for JPEG (whose
// size the sensor decides).
func FrameSize(cfg Config) int {
	px := int(cfg.W) * int(cfg.H)
	switch cfg.Format {
	case sensor.Y8:
		return px
	case sensor.RGB565, sensor.YUV422:
		return 2 * px
	}
	return 0
}
