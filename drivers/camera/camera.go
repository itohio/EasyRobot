// Package camera captures images from an SCCB-configured image sensor. The
// API is the same on every host; each host has its own backend file chosen
// by build tag (camera_k210.go: the K210 DVP; camera_esp32s3.go: the
// ESP32-S3 LCD_CAM). A host without a backend has no Identify/Open, so a
// firmware build for it fails at compile time.
//
// # Backend contract
//
// Every backend file defines, with these exact names:
//
//	type Pins struct{ … }                                  // host-specific pin set
//	func Identify(cfg Config, pins Pins) (sensor.ID, error) // power up, read the sensor id, configure nothing
//	func Open(cfg Config, pins Pins) (*Camera, error)       // identify, configure the sensor and the capture path
//	type Camera struct{ … }                                // implements Backend
//	func (c *Camera) Driver() sensor.Sensor                 // the sensor driver, for sensor-specific settings
//
// and asserts the method set at compile time:
//
//	var _ Backend = (*Camera)(nil)
//
// Shared rules:
//   - Errors: unsupported format or size ErrFormat (also when the sensor
//     driver rejects the mode); buffer too small
//     ErrBufSize; buffer misaligned or in memory the capture engine cannot
//     write ErrBufAlign; no frame in time ErrTimeout; unknown sensor
//     ErrNoDriver; driver without controls ErrNoControls; the capture engine
//     lost data (FIFO overflow/underflow, where the host reports it)
//     ErrOverflow.
//   - A failed Open leaves the sensor powered down.
//   - Raw formats (Y8, RGB565, YUV422): Capture needs FrameSize(cfg) bytes;
//     RGB565 comes out little-endian, pixels in raster order.
//   - JPEG: FrameSize returns 0 because the sensor decides the size. The
//     caller passes a buffer sized for the configured resolution and
//     quality, up to the backend's maximum (documented per backend;
//     ESP32-S3: JPEGBufSize(cfg), at most MaxJPEG, PSRAM for large ones);
//     Frame.Data is truncated to the JPEG's length, from FF D8 to FF D9
//     inclusive. A JPEG larger than the buffer gives ErrBufSize.
//   - Frame.Data aliases the caller's buffer (or a host alias of it) and is
//     valid until the next Capture.
//   - Capture does not allocate. The controls and Identify go over SCCB and
//     may allocate where the host's I2C driver does (ESP32-S3: TinyGo 0.43's
//     machine.I2C allocates per transfer).
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

// Backend is the method set every host's Camera implements.
type Backend interface {
	Sensor() sensor.ID
	Capture(buf []byte) (Frame, error)
	SetControls(sensor.Controls) error
	ReadControls() (sensor.Controls, error)
	Close() error
}

var (
	ErrTimeout    = errors.New("camera: timeout")
	ErrBufSize    = errors.New("camera: buffer too small for the frame")
	ErrBufAlign   = errors.New("camera: buffer misaligned, or in memory this host's capture engine cannot write")
	ErrFormat     = errors.New("camera: format or size not supported by this backend")
	ErrNoDriver   = errors.New("camera: no driver for this sensor")
	ErrNoControls = errors.New("camera: sensor driver has no exposure/gain/white-balance controls")
	ErrOverflow   = errors.New("camera: capture FIFO overflowed or underflowed; frame data lost")
)

// FrameSize is the buffer size Capture needs for cfg. It is 0 for JPEG
// (whose size the sensor decides), for unknown formats, and for sizes that
// do not fit an int (32-bit hosts).
func FrameSize(cfg Config) int {
	px := uint64(cfg.W) * uint64(cfg.H)
	var n uint64
	switch cfg.Format {
	case sensor.Y8:
		n = px
	case sensor.RGB565, sensor.YUV422:
		n = 2 * px
	default:
		return 0
	}
	if n > uint64(maxInt) {
		return 0
	}
	return int(n)
}

const maxInt = int(^uint(0) >> 1)
