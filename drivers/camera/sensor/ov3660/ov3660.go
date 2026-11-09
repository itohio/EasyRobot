// Package ov3660 configures an OmniVision OV3660 (3 MP, on-sensor JPEG) over
// SCCB with 16-bit register addresses. Register values are transcribed from
// espressif/esp32-camera v2.1.9 (Apache-2.0): sensors/ov3660.c (reset,
// set_ae_level, set_pixformat, set_framesize, set_image_options, set_pll,
// calc_sysclk, set_quality, set_colorbar, controls) and
// sensors/private_include/ov3660_settings.h (default registers, format
// tables, ratio table). Brought up on a XIAO ESP32-S3 Sense at XCLK 20 MHz
// (ER-ESP32CAM E2–E5, C1, M3–M5 evidence, outside the repo).
package ov3660

import (
	"errors"
	"time"

	"github.com/itohio/EasyRobot/drivers/camera/sensor"
)

// Addr is the OV3660's 7-bit SCCB address.
const Addr = 0x3C

// Registers (ov3660_regs.h names).
const (
	regSystemCtrl0 = 0x3008 // SYSTEM_CTROL0: 0x82 soft reset, 0x42 power down, 0x02 run
	regPLLCtrl0    = 0x303A // SC_PLLS_CTRL0..3
	regPLLCtrl1    = 0x303B
	regPLLCtrl2    = 0x303C
	regPLLCtrl3    = 0x303D
	regXAddrSt     = 0x3800 // X_ADDR_ST_H; Y at +2; X/Y_ADDR_END at 0x3804/6
	regXAddrEnd    = 0x3804
	regXOutput     = 0x3808 // X_OUTPUT_SIZE_H; Y at +2
	regXTotal      = 0x380C // X_TOTAL_SIZE_H (HTS); Y (VTS) at +2
	regXOffset     = 0x3810 // ISP X offset; Y at +2
	regXInc        = 0x3814 // X_INCREMENT; Y at 0x3815
	regYInc        = 0x3815
	regTiming20    = 0x3820 // TIMING_TC_REG20: bit 6 no-binning timing, bit 0 vertical binning
	regTiming21    = 0x3821 // TIMING_TC_REG21: bit 5 compression, bit 0 horizontal binning
	regPCLKRatio   = 0x3824
	regCompress07  = 0x4407 // COMPRESSION_CTRL07: JPEG quantisation scale 0..63
	regFormat00    = 0x4300 // FORMAT_CTRL00
	regVFIFO0C     = 0x460C
	regTestSetting = 0x503D // PRE_ISP_TEST_SETTING_1
	regISPCtrl01   = 0x5001 // ISP_CONTROL_01: bit 5 scale, bit 0 AWB
	regFormatCtrl  = 0x501F // FORMAT_CTRL (ISP mux)
	regPID         = 0x300A
	regVER         = 0x300B
	testColorBar   = 0xC0
	isp01Default   = 0x83
	isp01Scale     = 0x20
	maxW, maxH     = 2048, 1536
	arrayW, arrayH = 2080, 1548 // pixel array window of the 4:3 ratio entry (X/Y_ADDR_END + 1)
	ispOffX        = 16         // ISP offsets without binning
	ispOffY        = 6
)

type reg struct {
	addr uint16
	val  uint8
}

// regDelay in a table means "wait val milliseconds".
const regDelay = 0xFFFF

// Options are host-specific choices.
type Options struct {
	// XCLKHz is the sensor clock the backend supplies (0 = 20 MHz). It sets
	// the line period used for exposure in µs.
	XCLKHz uint32
	// Quality is the JPEG quantisation scale, 0 (best) … 63; 0 means 12.
	// Smaller values give larger JPEGs (M3 evidence: 2048×1536 needs ≥ 40 to
	// fit 290 KB of internal RAM).
	Quality uint8
}

// Sensor drives one OV3660.
type Sensor struct {
	Bus  sensor.Bus
	Opts Options

	lineNS uint32 // line period of the configured mode
}

// ErrMode is returned for a size or format this driver does not set up.
var ErrMode = errors.New("ov3660: unsupported mode (sizes of an esp32-camera aspect ratio up to its maximum, or a crop up to 2048×1536; multiples of 4; Y8, RGB565, YUV422, JPEG)")

// ErrID means the device at Addr is not an OV3660.
var ErrID = errors.New("ov3660: product id is not 0x3660")

func (s *Sensor) Name() string { return "OV3660" }

// LineNS is the line period of the mode set by the last Init, in ns
// (0 before Init). C1 evidence: 115,000 ns measured for QVGA YUV at 20 MHz.
func (s *Sensor) LineNS() uint32 { return s.lineNS }

func (s *Sensor) xclk() uint32 {
	if s.Opts.XCLKHz == 0 {
		return 20_000_000
	}
	return s.Opts.XCLKHz
}

func (s *Sensor) write(r uint16, v uint8) error { return s.Bus.WriteReg16(Addr, r, v) }

func (s *Sensor) read(r uint16) (uint8, error) { return s.Bus.ReadReg16(Addr, r) }

func (s *Sensor) writeTable(t []reg) error {
	for _, r := range t {
		if r.addr == regDelay {
			sleep(time.Duration(r.val) * time.Millisecond)
			continue
		}
		if err := s.write(r.addr, r.val); err != nil {
			return err
		}
	}
	return nil
}

// write16 writes a 16-bit value to the register pair hi, hi+1.
func (s *Sensor) write16(hi uint16, v uint16) error {
	if err := s.write(hi, uint8(v>>8)); err != nil {
		return err
	}
	return s.write(hi+1, uint8(v))
}

func (s *Sensor) read16(hi uint16) (uint16, error) {
	h, err := s.read(hi)
	if err != nil {
		return 0, err
	}
	l, err := s.read(hi + 1)
	return uint16(h)<<8 | uint16(l), err
}

// Init checks the product id, soft-resets the sensor and configures it for
// m. Full field (m.Crop false): m.W×m.H must have the aspect ratio of one of
// esp32-camera's ratio entries (4:3, 3:2, 16:10, 5:3, 16:9, 21:9, 5:4, 1:1,
// 9:16) and be no larger than its maximum; sizes up to half the maximum use
// 2×2 binning. Crop: a centred m.W×m.H window at 1:1 sensor pixels, up to
// 2048×1536. Y8 is the sensor's own 1-byte luma output. RGB565 comes out
// big-endian on the bus (esp32-camera's FORMAT_CTRL00 0x61); the backend
// swaps it.
func (s *Sensor) Init(m sensor.Mode) error {
	p, err := plan(m, s.xclk())
	if err != nil {
		return err
	}
	pid, err := s.read(regPID)
	if err != nil {
		return err
	}
	ver, err := s.read(regVER)
	if err != nil {
		return err
	}
	if pid != 0x36 || ver != 0x60 {
		return ErrID
	}
	if err := s.write(regSystemCtrl0, 0x82); err != nil { // reset(): soft reset, then defaults
		return err
	}
	sleep(100 * time.Millisecond)
	if err := s.writeTable(defaults[:]); err != nil {
		return err
	}
	if err := s.writeTable(aeLevel0[:]); err != nil {
		return err
	}
	sleep(100 * time.Millisecond)
	if err := s.writeTable(formatTable(m.Format)); err != nil {
		return err
	}
	if err := s.apply(p); err != nil {
		return err
	}
	if m.Format == sensor.JPEG {
		q := s.Opts.Quality
		if q == 0 {
			q = 12
		}
		if err := s.write(regCompress07, q&0x3F); err != nil {
			return err
		}
	}
	if m.TestPattern {
		if err := s.write(regTestSetting, testColorBar); err != nil {
			return err
		}
	}
	s.lineNS = p.lineNS
	return nil
}

// SetQuality changes the JPEG quantisation scale (0 best … 63) of a running
// sensor; it applies from the next frames (esp32-camera set_quality).
func (s *Sensor) SetQuality(q uint8) error {
	if q > 63 {
		return ErrMode
	}
	s.Opts.Quality = q
	return s.write(regCompress07, q)
}

func formatTable(f sensor.Format) []reg {
	switch f {
	case sensor.Y8:
		return fmtY8[:]
	case sensor.RGB565:
		return fmtRGB565[:]
	case sensor.JPEG:
		return fmtJPEG[:]
	}
	return fmtYUV422[:]
}

// ratio is one esp32-camera ratio_table entry.
type ratio struct {
	mw, mh, sx, sy, ex, ey, tx, ty uint16
}

var ratios = [...]ratio{
	{2048, 1536, 0, 0, 2079, 1547, 2300, 1564},    // 4:3
	{1920, 1280, 64, 128, 2015, 1419, 2172, 1436}, // 3:2
	{2048, 1280, 0, 128, 2079, 1419, 2300, 1436},  // 16:10
	{1920, 1152, 64, 192, 2015, 1355, 2172, 1372}, // 5:3
	{1920, 1080, 64, 242, 2015, 1333, 2172, 1322}, // 16:9
	{2048, 880, 0, 328, 2079, 1219, 2300, 1236},   // 21:9
	{1920, 1536, 64, 0, 2015, 1547, 2172, 1564},   // 5:4
	{1536, 1536, 256, 0, 1823, 1547, 2044, 1564},  // 1:1
	{864, 1536, 592, 0, 1487, 1547, 2044, 1564},   // 9:16
}

// pll is a set_pll argument set (bypass off, PCLK manual).
type pll struct {
	mult, sysDiv, preDiv uint8
	root2x               bool
	seld5, pclkDiv       uint8
}

// sysclk is calc_sysclk's SYSCLK in Hz for xclk.
func (p pll) sysclk(xclk uint32) uint32 {
	preDiv2x := [4]uint64{2, 3, 4, 6}[p.preDiv&3]
	seld52x := [4]uint64{2, 2, 4, 5}[p.seld5&3]
	root := uint64(1)
	if p.root2x {
		root = 2
	}
	sysDiv := uint64(p.sysDiv)
	if sysDiv == 0 {
		sysDiv = 1
	}
	vcoKHz := uint64(xclk/1000) * uint64(p.mult) * root * 2 / preDiv2x
	pllclk := vcoKHz * 1000 * 2 / sysDiv / seld52x
	return uint32(pllclk / 4)
}

// framePlan is everything set_framesize writes, computed without I/O.
type framePlan struct {
	sx, sy, ex, ey uint16 // sensor array window
	w, h           uint16 // output size
	tx, ty         uint16 // HTS, VTS
	ox, oy         uint16 // ISP offsets
	binning, scale bool
	jpeg           bool
	pll            pll
	lineNS         uint32
}

func plan(m sensor.Mode, xclk uint32) (framePlan, error) {
	var p framePlan
	switch m.Format {
	case sensor.Y8, sensor.YUV422, sensor.RGB565, sensor.JPEG:
	default:
		return p, ErrMode
	}
	w, h := m.W, m.H
	if w == 0 || h == 0 || w%4 != 0 || h%4 != 0 || w > maxW || h > maxH {
		return p, ErrMode
	}
	p.w, p.h, p.jpeg = w, h, m.Format == sensor.JPEG
	full := false
	if m.Crop {
		// The window is the output plus the ISP margins, centred in the array.
		x0 := (arrayW - (w + 2*ispOffX)) / 2 &^ 1
		y0 := (arrayH - (h + 2*ispOffY)) / 2 &^ 1
		p.sx, p.sy = x0, y0
		p.ex, p.ey = x0+w+2*ispOffX-1, y0+h+2*ispOffY-1
		p.tx, p.ty = ratios[0].tx, ratios[0].ty
		p.ox, p.oy = ispOffX, ispOffY
		full = w == maxW && h == maxH
	} else {
		r, ok := ratioFor(w, h)
		if !ok {
			return p, ErrMode
		}
		p.sx, p.sy, p.ex, p.ey = r.sx, r.sy, r.ex, r.ey
		p.binning = w <= r.mw/2 && h <= r.mh/2
		p.scale = !(w == r.mw && h == r.mh || w == r.mw/2 && h == r.mh/2)
		p.tx, p.ty = r.tx, r.ty
		p.ox, p.oy = ispOffX, ispOffY
		if p.binning {
			p.ty = r.ty/2 + 1
			p.ox, p.oy = 8, 2
		}
		full = w == maxW && h == maxH
	}
	switch {
	case p.jpeg && (full || xclk == 16_000_000):
		p.pll = pll{24, 1, 3, false, 0, 8} // 40 MHz SYSCLK at 20 MHz XCLK
	case p.jpeg:
		p.pll = pll{30, 1, 3, false, 0, 10} // 50 MHz SYSCLK
	case uint32(w)*uint32(h) > 480*320: // larger than HVGA
		p.pll = pll{4, 1, 0, false, 2, 2} // 10 MHz SYSCLK
	case uint32(w)*uint32(h) >= 320*240: // QVGA … HVGA
		p.pll = pll{8, 1, 0, false, 2, 4} // 20 MHz SYSCLK
	default:
		p.pll = pll{8, 1, 0, false, 0, 8} // 40 MHz SYSCLK
	}
	if sc := p.pll.sysclk(xclk); sc != 0 {
		p.lineNS = uint32(uint64(p.tx) * 1_000_000_000 / uint64(sc))
	}
	return p, nil
}

// ratioFor returns the ratio entry with exactly w:h whose maximum holds w×h.
func ratioFor(w, h uint16) (ratio, bool) {
	for _, r := range ratios {
		if uint32(w)*uint32(r.mh) == uint32(h)*uint32(r.mw) && w <= r.mw && h <= r.mh {
			return r, true
		}
	}
	return ratio{}, false
}

// apply writes p: set_framesize, set_image_options (no flip or mirror) and set_pll.
func (s *Sensor) apply(p framePlan) error {
	pairs := [...][2]uint16{
		{regXAddrSt, p.sx}, {regXAddrSt + 2, p.sy},
		{regXAddrEnd, p.ex}, {regXAddrEnd + 2, p.ey},
		{regXOutput, p.w}, {regXOutput + 2, p.h},
		{regXTotal, p.tx}, {regXTotal + 2, p.ty},
		{regXOffset, p.ox}, {regXOffset + 2, p.oy},
	}
	for _, pr := range pairs {
		if err := s.write16(pr[0], pr[1]); err != nil {
			return err
		}
	}
	isp := uint8(isp01Default)
	if p.scale {
		isp |= isp01Scale
	}
	r20, r21, r4514, r4520, inc := uint8(0x40), uint8(0), uint8(0x88), uint8(0xb0), uint8(0x11)
	if p.binning {
		r20, r21, r4514, r4520, inc = 0x01, 0x01, 0xaa, 0x0b, 0x31
	}
	if p.jpeg {
		r21 |= 0x20
	}
	c := p.pll
	ctrl3 := c.preDiv&3<<4 | c.seld5&3
	if c.root2x {
		ctrl3 |= 0x40
	}
	t := [...]reg{
		{regISPCtrl01, isp},
		{regTiming20, r20}, {regTiming21, r21}, {0x4514, r4514},
		{0x4520, r4520}, {regXInc, inc}, {regYInc, inc},
		{regPLLCtrl0, 0x00}, {regPLLCtrl1, c.mult & 0x1F}, {regPLLCtrl2, 0x10 | c.sysDiv&0x0F},
		{regPLLCtrl3, ctrl3}, {regPCLKRatio, c.pclkDiv & 0x1F}, {regVFIFO0C, 0x22},
	}
	return s.writeTable(t[:])
}

// sleep waits at least d, checking a deadline (a bare time.Sleep can return
// early on some TinyGo targets; see drivers/camera/sensor/ov2640).
func sleep(d time.Duration) {
	end := time.Now().Add(d)
	for left := time.Until(end); left > 0; left = time.Until(end) {
		time.Sleep(left)
	}
}
