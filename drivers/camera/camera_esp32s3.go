//go:build esp32s3

package camera

import (
	"errors"
	"machine"
	"time"
	"unsafe"

	"github.com/itohio/EasyRobot/drivers/camera/sensor"
	"github.com/itohio/EasyRobot/drivers/camera/sensor/ov3660"
)

// Pins are the ESP32-S3 GPIOs of an 8-bit DVP camera. Any GPIO works: the
// data and sync lines go through the GPIO matrix into LCD_CAM.
type Pins struct {
	D                 [8]machine.Pin // D0…D7 (sensor Y2…Y9)
	PCLK, VSYNC, HREF machine.Pin    // from the sensor
	XCLK              machine.Pin    // sensor clock, driven by an LEDC PWM
	SDA, SCL          machine.Pin    // SCCB
	// I2C is the SCCB master (nil = machine.I2C0).
	I2C *machine.I2C
	// XCLKTimer is the LEDC timer for XCLK (nil = machine.PWM0).
	XCLKTimer *machine.LEDCPWM
	// DMA is the GDMA RX channel, 0…4.
	DMA uint8
}

// ErrPins is returned by Open for a pin set it cannot use (a GDMA channel
// above 4). ESP32-S3 only.
var ErrPins = errors.New("camera: pin set invalid (GDMA channel must be 0–4)")

const (
	defaultXCLK = 20_000_000
	sccbHz      = 100_000
	// MaxJPEG is the largest buffer Capture uses for JPEG: maxNodes
	// descriptors of nodeBytes. In internal SRAM the practical limit is the
	// free RAM (M3/M45 evidence: 290–322 KB alongside FatFs).
	MaxJPEG = maxNodes * nodeBytes
)

func (p Pins) i2c() *machine.I2C {
	if p.I2C == nil {
		return machine.I2C0
	}
	return p.I2C
}

func xclkHz(cfg Config) uint32 {
	if cfg.XCLK == 0 {
		return defaultXCLK
	}
	return cfg.XCLK
}

// powerUp starts XCLK and SCCB; the sensor has no power or reset pin on the
// supported boards, so it is already running.
func powerUp(cfg Config, pins Pins, bus *sccbBus) error {
	if _, err := startXCLK(pins, xclkHz(cfg)); err != nil {
		return err
	}
	bus.i2c = pins.i2c()
	if err := bus.i2c.Configure(machine.I2CConfig{SCL: pins.SCL, SDA: pins.SDA, Frequency: sccbHz}); err != nil {
		return err
	}
	time.Sleep(20 * time.Millisecond)
	return nil
}

// powerDown puts an OV3660/OV5640 into software power-down
// (SYSTEM_CTROL0 0x42) and stops XCLK.
func powerDown(pins Pins, bus *sccbBus, id sensor.ID) {
	if bus.i2c != nil && (id.Model == sensor.OV3660 || id.Model == sensor.OV5640) {
		bus.WriteReg16(id.Addr, 0x3008, 0x42)
	}
	pins.XCLK.Configure(machine.PinConfig{Mode: machine.PinOutput})
	pins.XCLK.Low()
}

// identBus is Identify's SCCB master; a package variable so repeated
// Identify calls (camid polls once a second) don't allocate a bus each time.
// TinyGo 0.43's machine.I2C itself allocates per transfer on the ESP32-S3.
var identBus sccbBus

// Identify starts XCLK and SCCB and reads the sensor id, without
// configuring the sensor.
func Identify(cfg Config, pins Pins) (sensor.ID, error) {
	if err := powerUp(cfg, pins, &identBus); err != nil {
		return sensor.ID{}, err
	}
	return sensor.Identify(&identBus)
}

// The ESP32-S3 backend captures Y8, YUV422, RGB565 and JPEG through LCD_CAM
// and one GDMA channel, in sizes the sensor driver accepts (OV3660: see
// ov3660.Sensor.Init). Buffers must be 4-byte aligned and in DMA-capable
// internal SRAM (ErrBufAlign); PSRAM buffers come with E7. Capture waits for
// a VSYNC, so it takes about two frame periods.
var _ Backend = (*Camera)(nil)

// Camera is an opened ESP32-S3 camera.
type Camera struct {
	cfg         Config
	pins        Pins
	id          sensor.ID
	drv         sensor.Sensor
	bus         sccbBus
	ch          *gdmaIn
	vsyncActive bool
	seq         uint32
	descs       [maxNodes]dmaDesc // in the Camera allocation: internal SRAM, word aligned
}

// Open starts the sensor clock, identifies the sensor, configures it for
// cfg and sets LCD_CAM and the GDMA channel up. It allocates the Camera
// (with its descriptor table) once.
func Open(cfg Config, pins Pins) (*Camera, error) {
	switch cfg.Format {
	case sensor.Y8, sensor.YUV422, sensor.RGB565, sensor.JPEG:
	default:
		return nil, ErrFormat
	}
	if cfg.Format != sensor.JPEG && FrameSize(cfg) > maxNodes*nodeBytes {
		return nil, ErrFormat
	}
	if pins.DMA > 4 {
		return nil, ErrPins
	}
	c := &Camera{cfg: cfg, pins: pins, ch: gdmaChannel(pins.DMA)}
	fail := func(err error) (*Camera, error) {
		powerDown(c.pins, &c.bus, c.id)
		return nil, err
	}
	if err := powerUp(cfg, pins, &c.bus); err != nil {
		return fail(err)
	}
	id, err := sensor.Identify(&c.bus)
	if err != nil {
		return fail(err)
	}
	c.id = id
	switch id.Model {
	case sensor.OV3660:
		if !ov3660.Supported(cfg.Mode) { // the shared ErrFormat, not the driver's ErrMode
			return fail(ErrFormat)
		}
		c.drv = &ov3660.Sensor{Bus: &c.bus, Opts: ov3660.Options{XCLKHz: xclkHz(cfg)}}
	default:
		return fail(ErrNoDriver)
	}
	if err := c.drv.Init(cfg.Mode); err != nil {
		return fail(err)
	}
	camSetup(pins, c.ch)
	active, pulses := vsyncLevel(pins.VSYNC, 400*time.Millisecond)
	if pulses == 0 {
		return fail(ErrTimeout)
	}
	c.vsyncActive = active
	return c, nil
}

// Sensor returns the identified sensor.
func (c *Camera) Sensor() sensor.ID { return c.id }

// Driver returns the sensor driver (for sensor-specific settings, e.g.
// (*ov3660.Sensor).SetQuality).
func (c *Camera) Driver() sensor.Sensor { return c.drv }

// SetControls sets exposure, gain and white balance, if the sensor driver
// supports them.
func (c *Camera) SetControls(ctl sensor.Controls) error {
	d, ok := c.drv.(sensor.Controller)
	if !ok {
		return ErrNoControls
	}
	return d.SetControls(ctl)
}

// ReadControls reads the sensor's current exposure, gain and white balance.
func (c *Camera) ReadControls() (sensor.Controls, error) {
	d, ok := c.drv.(sensor.Controller)
	if !ok {
		return sensor.Controls{}, ErrNoControls
	}
	return d.ReadControls()
}

func inDMARAM(b []byte) bool {
	p := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	return p >= dmaLow && p+uintptr(len(b)) <= dmaHigh
}

// Capture grabs the next whole frame into buf.
//   - Raw formats: buf needs FrameSize(cfg) bytes; RGB565 is swapped to
//     little-endian in place (the OV3660 sends it big-endian).
//   - JPEG: up to MaxJPEG bytes of buf are used; Frame.Data runs from FF D8
//     to FF D9 (a complete JPEG counts even when the sensor's padding after
//     the EOI overran the buffer). A JPEG larger than the buffer gives
//     ErrBufSize.
//
// buf must be 4-byte aligned and in DMA-capable internal SRAM (ErrBufAlign).
// A frame that does not complete (no VSYNC or EOF within 2 s, wrong length,
// JPEG without SOI/EOI) gives ErrTimeout. Frame.Data aliases buf.
func (c *Camera) Capture(buf []byte) (Frame, error) {
	jpeg := c.cfg.Format == sensor.JPEG
	n := FrameSize(c.cfg)
	if jpeg {
		n = min(len(buf), MaxJPEG) &^ 3
		if n < nodeBytes {
			return Frame{}, ErrBufSize
		}
	} else if len(buf) < n {
		return Frame{}, ErrBufSize
	}
	buf = buf[:n]
	if uintptr(unsafe.Pointer(unsafe.SliceData(buf)))&3 != 0 || !inDMARAM(buf) {
		return Frame{}, ErrBufAlign
	}
	used := buildChain(&c.descs, buf, n)
	got, ended := captureOnce(c.pins, c.vsyncActive, c.ch, &c.descs, used, time.Now().Add(2*time.Second))
	data, err := frameResult(jpeg, buf, got, n, ended)
	if err != nil {
		return Frame{}, err
	}
	if c.cfg.Format == sensor.RGB565 {
		swap16(data)
	}
	c.seq++
	return Frame{Width: c.cfg.W, Height: c.cfg.H, Format: c.cfg.Format, Seq: c.seq, Data: data}, nil
}

// Close stops capture, powers the sensor down and stops XCLK.
func (c *Camera) Close() error {
	powerDown(c.pins, &c.bus, c.id)
	return nil
}
