// Package sensor is the host-independent part of the camera driver: the SCCB
// bus interface every capture backend implements, image modes, and sensor
// identification. It has no build tag, so it builds and tests on any host.
package sensor

import "errors"

// Format is the pixel format a sensor is asked to produce.
type Format uint8

const (
	Y8     Format = iota + 1 // 8-bit luma
	RGB565                   // 16-bit RGB
	YUV422                   // YUYV
	JPEG                     // on-sensor JPEG (OV2640, OV3660, OV5640)
)

// Resolution is an image size in pixels.
type Resolution struct{ W, H uint16 }

var (
	QQVGA = Resolution{160, 120}
	QVGA  = Resolution{320, 240}
	VGA   = Resolution{640, 480}
)

// Mode is what a sensor is configured to output.
type Mode struct {
	Resolution
	Format Format
	// Crop selects a centred W×H window at 1:1 sensor pixels instead of the
	// whole field of view scaled down to W×H (the default).
	Crop        bool
	TestPattern bool // sensor-generated colour bars instead of the scene (bring-up checks)
}

// Bus is an SCCB master. dev is the 7-bit device address. Sensors with 8-bit
// register addresses (OV7740, OV2640) use the Reg8 calls; OV3660 and OV5640
// use 16-bit register addresses.
type Bus interface {
	ReadReg8(dev, reg uint8) (uint8, error)
	WriteReg8(dev, reg, val uint8) error
	ReadReg16(dev uint8, reg uint16) (uint8, error)
	WriteReg16(dev uint8, reg uint16, val uint8) error
}

// Sensor is a configured sensor driver (for example ov2640.Sensor), bound
// to a Bus by its constructor.
type Sensor interface {
	Name() string
	Init(Mode) error // soft reset, then configure for the mode
}

// Controls are a sensor's exposure, gain and white-balance settings. On
// write, the manual values apply where the matching Auto flag is false. On
// read-back, every value is what the sensor currently uses, including the
// values its auto loops chose.
type Controls struct {
	AutoExposure bool
	// ExposureUS is the exposure time in µs. Sensors expose in whole line
	// periods; the driver converts with its line period and reports the
	// rounded value back.
	ExposureUS uint32
	AutoGain   bool
	// Gain is the analogue gain (1 = unity). ISO equivalent ≈ 100 × Gain is a
	// nominal convention, not a calibration.
	Gain   float32
	AutoWB bool
	// WB are the red, green and blue white-balance gains (1 = unity).
	WB [3]float32
}

// Controller is implemented by sensor drivers that expose Controls.
type Controller interface {
	SetControls(Controls) error
	ReadControls() (Controls, error)
}

// Model is a sensor model. The numeric values equal the frame format's
// sensor ids (drivers/camera/frame).
type Model uint8

const (
	Unknown Model = iota
	OV7740
	OV2640
	OV3660
	OV5640
)

func (m Model) String() string {
	switch m {
	case OV7740:
		return "OV7740"
	case OV2640:
		return "OV2640"
	case OV3660:
		return "OV3660"
	case OV5640:
		return "OV5640"
	}
	return "unknown"
}

// ID is what Identify read from a sensor.
type ID struct {
	Model Model
	Addr  uint8 // 7-bit SCCB address that answered
	PID   uint8 // product id high byte (register 0x0A, or 0x300A)
	VER   uint8 // product id low byte (register 0x0B, or 0x300B)
}

// ErrNotFound means no known sensor answered at any probed address.
var ErrNotFound = errors.New("sensor: no camera sensor answered on SCCB")

// SCCB addresses (7-bit) and id registers of the supported sensors.
const (
	addrOV7740 = 0x21 // OV7740 (and other OV76xx/77xx parts)
	addrOV2640 = 0x30
	addrOV5640 = 0x3C // OV3660 and OV5640

	regPID     = 0x0A
	regVER     = 0x0B
	regPID16   = 0x300A
	regVER16   = 0x300B
	regBankSel = 0xFF // OV2640 register bank select; 1 = sensor bank
)

// absent reports an id byte that an unanswered SCCB read returns.
func absent(b uint8) bool { return b == 0x00 || b == 0xFF }

// Identify probes the known sensor addresses and returns the first sensor
// that answers with a plausible id. An answering device with an unknown id
// is returned with Model Unknown and a nil error, so the caller can print it.
func Identify(bus Bus) (ID, error) {
	// OV7740: 8-bit registers 0x0A/0x0B, PID 0x77.
	if pid, ver, ok := read8(bus, addrOV7740); ok {
		id := ID{Addr: addrOV7740, PID: pid, VER: ver}
		if pid == 0x77 {
			id.Model = OV7740
		}
		return id, nil
	}
	// OV2640: select the sensor bank first, then PID 0x26.
	if err := bus.WriteReg8(addrOV2640, regBankSel, 0x01); err == nil {
		if pid, ver, ok := read8(bus, addrOV2640); ok {
			id := ID{Addr: addrOV2640, PID: pid, VER: ver}
			if pid == 0x26 {
				id.Model = OV2640
			}
			return id, nil
		}
	}
	// OV3660 (0x3660) / OV5640 (0x5640): 16-bit registers.
	pid, err1 := bus.ReadReg16(addrOV5640, regPID16)
	ver, err2 := bus.ReadReg16(addrOV5640, regVER16)
	if err1 == nil && err2 == nil && !(absent(pid) && absent(ver)) {
		id := ID{Addr: addrOV5640, PID: pid, VER: ver}
		switch {
		case pid == 0x36 && ver == 0x60:
			id.Model = OV3660
		case pid == 0x56 && ver == 0x40:
			id.Model = OV5640
		}
		return id, nil
	}
	return ID{}, ErrNotFound
}

func read8(bus Bus, dev uint8) (pid, ver uint8, ok bool) {
	pid, err := bus.ReadReg8(dev, regPID)
	if err != nil {
		return 0, 0, false
	}
	ver, err = bus.ReadReg8(dev, regVER)
	if err != nil {
		return 0, 0, false
	}
	if absent(pid) && absent(ver) {
		return 0, 0, false
	}
	return pid, ver, true
}
