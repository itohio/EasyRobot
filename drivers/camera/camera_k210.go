//go:build k210

package camera

import (
	"machine"
	"time"
	"unsafe"

	"device/kendryte"

	"github.com/itohio/EasyRobot/drivers/camera/sensor"
	"github.com/itohio/EasyRobot/drivers/camera/sensor/ov2640"
)

// Pins are the K210 FPIOA pins of the camera's control signals. The eight
// data lines are the K210's dedicated DVP pins and need no mapping.
type Pins struct {
	SDA, SCL          machine.Pin // SCCB
	RST, PWDN, XCLK   machine.Pin // driven by the DVP
	VSYNC, HREF, PCLK machine.Pin // from the sensor
}

const (
	defaultXCLK = 24_000_000
	sccbHz      = 100_000
)

var clocks Clocks

// ClockInfo returns the clock rates of the last Identify/Open (K210 only).
func ClockInfo() Clocks { return clocks }

func (p Pins) route() {
	p.SDA.SetFPIOAFunction(machine.FUNC_SCCB_SDA)
	p.SCL.SetFPIOAFunction(machine.FUNC_SCCB_SCLK)
	p.RST.SetFPIOAFunction(machine.FUNC_CMOS_RST)
	p.PWDN.SetFPIOAFunction(machine.FUNC_CMOS_PWDN)
	p.XCLK.SetFPIOAFunction(machine.FUNC_CMOS_XCLK)
	p.VSYNC.SetFPIOAFunction(machine.FUNC_CMOS_VSYNC)
	p.HREF.SetFPIOAFunction(machine.FUNC_CMOS_HREF)
	p.PCLK.SetFPIOAFunction(machine.FUNC_CMOS_PCLK)
}

// powerUp routes the pins, starts the DVP, XCLK and SCCB, and power-cycles
// the sensor.
func powerUp(cfg Config, pins Pins) {
	pins.route()
	xclk := cfg.XCLK
	if xclk == 0 {
		xclk = defaultXCLK
	}
	clocks.SCCB = dvpInit(sccbHz)
	clocks.XCLK = setXCLK(xclk)
	clocks.PLL0, clocks.ACLK, clocks.APB1 = pll0Hz(), aclkHz(), apb1Hz()
	sensorPowerCycle()
}

// Identify powers the sensor up and reads its id over SCCB, without
// configuring it.
func Identify(cfg Config, pins Pins) (sensor.ID, error) {
	powerUp(cfg, pins)
	return sensor.Identify(sccbBus{})
}

// The K210 DVP path captures Y8 and RGB565 at widths that are multiples of
// 32; other formats and sizes give ErrFormat. Buffers must be 8-byte aligned
// (ErrBufAlign).
var _ Backend = (*Camera)(nil)

// Camera is an opened K210 DVP camera.
type Camera struct {
	cfg    Config
	id     sensor.ID
	drv    sensor.Sensor
	seq    uint32
	planes []byte // G and B planes for Y8 (the DVP's AI output writes three)
}

// Cached general SRAM (6 MiB at 0x8000_0000). It is also mapped uncached at
// 0x4000_0000; the DVP writes memory behind the D-cache, so the CPU reads
// frames through that alias.
const (
	sramStart      = 0x8000_0000
	sramEnd        = 0x8060_0000
	uncachedOffset = 0x4000_0000
)

// inSRAM reports whether b lies in cached general SRAM. A previous
// Frame.Data (already the uncached alias) does not, and must not be handed
// to the DVP.
func inSRAM(b []byte) bool {
	p := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	return p >= sramStart && p+uintptr(len(b)) <= sramEnd
}

// uncached returns b's uncached alias; b must satisfy inSRAM.
func uncached(b []byte) []byte {
	p := uintptr(unsafe.Pointer(unsafe.SliceData(b))) - uncachedOffset
	return unsafe.Slice((*byte)(unsafe.Pointer(p)), len(b))
}

func physAddr(b []byte) uint32 { return uint32(uintptr(unsafe.Pointer(unsafe.SliceData(b)))) }

// powerDown stops the sensor clock and powers the sensor down.
func powerDown() {
	kendryte.DVP.CMOS_CFG.SetBits(kendryte.DVP_CMOS_CFG_POWER_DOWN)
	kendryte.DVP.CMOS_CFG.ClearBits(kendryte.DVP_CMOS_CFG_CLK_ENABLE)
}

// Open powers the sensor up, identifies it, configures it for cfg and sets
// the DVP up for that frame size. Formats: Y8 (sensor YUV422, the DVP keeps
// Y) and RGB565.
func Open(cfg Config, pins Pins) (*Camera, error) {
	if cfg.Format != sensor.Y8 && cfg.Format != sensor.RGB565 {
		return nil, ErrFormat
	}
	if cfg.W == 0 || cfg.W%32 != 0 || cfg.H == 0 {
		return nil, ErrFormat
	}
	id, err := Identify(cfg, pins)
	if err != nil {
		powerDown()
		return nil, err
	}
	var s sensor.Sensor
	switch id.Model {
	case sensor.OV2640:
		s = &ov2640.Sensor{Bus: sccbBus{}, Opts: ov2640.Options{
			ClockDouble: true, ClockDiv: 3, YUVLowByteFirst: true,
			LineNS: ov2640.LineNS(clocks.XCLK),
		}}
	default:
		powerDown()
		return nil, ErrNoDriver
	}
	if err := s.Init(cfg.Mode); err != nil {
		powerDown()
		return nil, modeErr(err, ov2640.ErrMode)
	}
	c := &Camera{cfg: cfg, id: id, drv: s}
	if cfg.Format == sensor.Y8 {
		c.planes = make([]byte, 2*int(cfg.W)*int(cfg.H))
		dvpConfigure(kendryte.DVP_DVP_CFG_FORMAT_Y, cfg.W, cfg.H, true, false)
	} else {
		dvpConfigure(kendryte.DVP_DVP_CFG_FORMAT_RGB, cfg.W, cfg.H, false, true)
	}
	return c, nil
}

// Sensor returns the identified sensor.
func (c *Camera) Sensor() sensor.ID { return c.id }

// Driver returns the sensor driver (for sensor-specific settings).
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

// Capture grabs the next whole frame into buf (FrameSize(cfg) bytes,
// 8-byte aligned, in cached SRAM: not a previous Frame.Data, which is the
// uncached alias). Frame.Data is buf's uncached alias. RGB565 comes out as
// little-endian 16-bit pixels in raster order.
func (c *Camera) Capture(buf []byte) (Frame, error) {
	n := FrameSize(c.cfg)
	if len(buf) < n {
		return Frame{}, ErrBufSize
	}
	buf = buf[:n]
	if physAddr(buf)&7 != 0 || !inSRAM(buf) {
		return Frame{}, ErrBufAlign
	}
	if c.cfg.Format == sensor.Y8 {
		kendryte.DVP.R_ADDR.Set(physAddr(buf))
		kendryte.DVP.G_ADDR.Set(physAddr(c.planes[:n]))
		kendryte.DVP.B_ADDR.Set(physAddr(c.planes[n:]))
	} else {
		kendryte.DVP.RGB_ADDR.Set(physAddr(buf))
	}
	if err := dvpCaptureOne(time.Second); err != nil {
		return Frame{}, err
	}
	data := uncached(buf)
	if c.cfg.Format == sensor.RGB565 {
		// The DVP display output stores each pixel pair with the two 16-bit
		// pixels swapped within the 32-bit word (observed, ER-K210-M4).
		words := unsafe.Slice((*uint32)(unsafe.Pointer(unsafe.SliceData(data))), n/4)
		for i, w := range words {
			words[i] = w>>16 | w<<16
		}
	}
	c.seq++
	return Frame{Width: c.cfg.W, Height: c.cfg.H, Format: c.cfg.Format, Seq: c.seq, Data: data}, nil
}

// Close stops the sensor clock and powers the sensor down.
func (c *Camera) Close() error {
	powerDown()
	return nil
}
