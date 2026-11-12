// Package ov2640 configures an OmniVision OV2640 over SCCB. It is
// host-independent; capture backends call Init after sensor.Identify.
//
// Register tables are from espressif/esp32-camera v2.1.9 (the SVGA and UXGA
// mode tables from its master branch, read 2026-11-10)
// (sensors/private_include/ov2640_settings.h, Apache-2.0, Copyright
// 2015-2016 Espressif Systems) and register names from OpenMV's
// ov2640_regs.h (MIT). The window/clock sequence follows esp32-camera's
// ov2640.c set_window/set_framesize/set_pixformat.
package ov2640

import (
	"errors"
	"time"

	"github.com/itohio/EasyRobot/drivers/camera/sensor"
)

// Addr is the OV2640's 7-bit SCCB address.
const Addr = 0x30

// DSP bank (BANK_SEL = 0) registers.
const (
	rBypass   = 0x05
	ctrlI     = 0x50
	hSize     = 0x51
	vSize     = 0x52
	xOffL     = 0x53
	yOffL     = 0x54
	vhyx      = 0x55
	test      = 0x57
	zmow      = 0x5A
	zmoh      = 0x5B
	zmhh      = 0x5C
	bpAddr    = 0x7C
	bpData    = 0x7D
	qs        = 0x44
	ctrl2     = 0x86
	ctrl3     = 0x87
	sizeL     = 0x8C
	hSize8    = 0xC0
	vSize8    = 0xC1
	ctrl1     = 0xC3
	rDVPSP    = 0xD3
	imageMode = 0xDA
	reset     = 0xE0
	mcBIST    = 0xF9
	bankSel   = 0xFF

	bankDSP    = 0x00
	bankSensor = 0x01
)

// Sensor bank (BANK_SEL = 1) registers.
const (
	com1      = 0x03
	reg04     = 0x04
	com2      = 0x09
	com4      = 0x0D
	clkrc     = 0x11
	com7      = 0x12
	com8      = 0x13
	com9      = 0x14
	hStart    = 0x17
	hStop     = 0x18
	vStart    = 0x19
	vStop     = 0x1A
	aew       = 0x24
	aeb       = 0x25
	vv        = 0x26
	reg32     = 0x32
	arcom2    = 0x34
	bd50      = 0x4F
	bd60      = 0x50
	histoLow  = 0x61
	histoHigh = 0x62

	com7SRST     = 0x80
	com7ResUXGA  = 0x00
	com7ResSVGA  = 0x40
	com7ResCIF   = 0x20
	com7ColorBar = 0x02

	ctrl2DCW  = 0x20 // CTRL2 DCW_EN
	ctrlILPDP = 0x80 // CTRLI LP_DP
)

// Image modes (DSP IMAGE_MODE).
const (
	imageModeYUV422     = 0x00
	imageModeRGB565     = 0x08
	imageModeJPEG       = 0x10
	imageModeHV         = 0x02 // HREF/VSYNC timing for JPEG
	imageModeLByteFirst = 0x01
	resetDVP            = 0x04
	resetJPEG           = 0x10
)

type reg struct{ addr, val uint8 }

// settingsCIF is esp32-camera's ov2640_settings_cif: full init, CIF
// (400×296) sensor mode, 30 fps at 24 MHz XCLK.
var settingsCIF = [...]reg{
	{bankSel, bankDSP}, {0x2c, 0xff}, {0x2e, 0xdf},
	{bankSel, bankSensor}, {0x3c, 0x32}, {clkrc, 0x01}, {com2, 0x02},
	{reg04, 0x28}, {com8, 0xE5}, {com9, 0x48},
	{0x2c, 0x0c}, {0x33, 0x78}, {0x3a, 0x33}, {0x3b, 0xfB}, {0x3e, 0x00},
	{0x43, 0x11}, {0x16, 0x10}, {0x39, 0x92}, {0x35, 0xda}, {0x22, 0x1a},
	{0x37, 0xc3}, {0x23, 0x00}, {arcom2, 0xc0}, {0x06, 0x88}, {0x07, 0xc0},
	{com4, 0x87}, {0x0e, 0x41}, {0x4c, 0x00}, {0x4a, 0x81}, {0x21, 0x99},
	{aew, 0x40}, {aeb, 0x38}, {vv, 0x82}, {0x5c, 0x00}, {0x63, 0x00},
	{histoLow, 0x70}, {histoHigh, 0x80}, {0x7c, 0x05}, {0x20, 0x80},
	{0x28, 0x30}, {0x6c, 0x00}, {0x6d, 0x80}, {0x6e, 0x00}, {0x70, 0x02},
	{0x71, 0x94}, {0x73, 0xc1}, {0x3d, 0x34}, {0x5a, 0x57}, {bd50, 0xbb},
	{bd60, 0x9c}, {com7, com7ResCIF}, {hStart, 0x11}, {hStop, 0x43},
	{vStart, 0x00}, {vStop, 0x25}, {reg32, 0x89}, {0x37, 0xc0},
	{bd50, 0xca}, {bd60, 0xa8}, {0x6d, 0x00}, {0x3d, 0x38},
	{bankSel, bankDSP}, {0xe5, 0x7f}, {mcBIST, 0xC0}, {0x41, 0x24},
	{reset, resetJPEG | resetDVP}, {0x76, 0xff}, {0x33, 0xa0}, {0x42, 0x20},
	{0x43, 0x18}, {0x4c, 0x00}, {ctrl3, 0x50}, {0x88, 0x3f}, {0xd7, 0x03},
	{0xd9, 0x10}, {rDVPSP, 0x82}, {0xc8, 0x08}, {0xc9, 0x80},
	{bpAddr, 0x00}, {bpData, 0x00}, {bpAddr, 0x03}, {bpData, 0x48},
	{bpData, 0x48}, {bpAddr, 0x08}, {bpData, 0x20}, {bpData, 0x10},
	{bpData, 0x0e},
	{0x90, 0x00}, {0x91, 0x0e}, {0x91, 0x1a}, {0x91, 0x31}, {0x91, 0x5a},
	{0x91, 0x69}, {0x91, 0x75}, {0x91, 0x7e}, {0x91, 0x88}, {0x91, 0x8f},
	{0x91, 0x96}, {0x91, 0xa3}, {0x91, 0xaf}, {0x91, 0xc4}, {0x91, 0xd7},
	{0x91, 0xe8}, {0x91, 0x20},
	{0x92, 0x00}, {0x93, 0x06}, {0x93, 0xe3}, {0x93, 0x05}, {0x93, 0x05},
	{0x93, 0x00}, {0x93, 0x04}, {0x93, 0x00}, {0x93, 0x00}, {0x93, 0x00},
	{0x93, 0x00}, {0x93, 0x00}, {0x93, 0x00}, {0x93, 0x00},
	{0x96, 0x00}, {0x97, 0x08}, {0x97, 0x19}, {0x97, 0x02}, {0x97, 0x0c},
	{0x97, 0x24}, {0x97, 0x30}, {0x97, 0x28}, {0x97, 0x26}, {0x97, 0x02},
	{0x97, 0x98}, {0x97, 0x80}, {0x97, 0x00}, {0x97, 0x00},
	{0xa4, 0x00}, {0xa8, 0x00}, {0xc5, 0x11}, {0xc6, 0x51}, {0xbf, 0x80},
	{0xc7, 0x10}, {0xb6, 0x66}, {0xb8, 0xA5}, {0xb7, 0x64}, {0xb9, 0x7C},
	{0xb3, 0xaf}, {0xb4, 0x97}, {0xb5, 0xFF}, {0xb0, 0xC5}, {0xb1, 0x94},
	{0xb2, 0x0f}, {0xc4, 0x5c}, {ctrl1, 0xfd}, {0x7f, 0x00}, {0xe5, 0x1f},
	{0xe1, 0x67}, {0xdd, 0x7f}, {imageMode, 0x00}, {reset, 0x00},
	{rBypass, 0x00},
}

// settingsToCIF is esp32-camera's ov2640_settings_to_cif.
var settingsToCIF = [...]reg{
	{bankSel, bankSensor}, {com7, com7ResCIF}, {com1, 0x0A}, {reg32, 0x89},
	{hStart, 0x11}, {hStop, 0x43}, {vStart, 0x00}, {vStop, 0x25},
	{bd50, 0xca}, {bd60, 0xa8}, {0x5a, 0x23}, {0x6d, 0x00}, {0x3d, 0x38},
	{0x39, 0x92}, {0x35, 0xda}, {0x22, 0x1a}, {0x37, 0xc3}, {0x23, 0x00},
	{arcom2, 0xc0}, {0x06, 0x88}, {0x07, 0xc0}, {com4, 0x87}, {0x0e, 0x41},
	{0x4c, 0x00},
	{bankSel, bankDSP}, {reset, resetDVP},
	{hSize8, 0x32}, {vSize8, 0x25}, {sizeL, 0x00},
	{hSize, 0x64}, {vSize, 0x4a}, {xOffL, 0x00}, {yOffL, 0x00}, {vhyx, 0x00},
	{test, 0x00},
	{ctrl2, 0x3D}, {ctrlI, 0x80},
}

// settingsToSVGA is esp32-camera's ov2640_settings_to_svga.
var settingsToSVGA = [...]reg{
	{bankSel, bankSensor}, {com7, com7ResSVGA}, {com1, 0x0A}, {reg32, 0x09},
	{hStart, 0x11}, {hStop, 0x43}, {vStart, 0x00}, {vStop, 0x4b},
	{0x37, 0xc0}, {bd50, 0xca}, {bd60, 0xa8}, {0x5a, 0x23}, {0x6d, 0x00},
	{0x3d, 0x38}, {0x39, 0x92}, {0x35, 0xda}, {0x22, 0x1a}, {0x37, 0xc3},
	{0x23, 0x00}, {arcom2, 0xc0}, {0x06, 0x88}, {0x07, 0xc0}, {com4, 0x87},
	{0x0e, 0x41}, {0x42, 0x03}, {0x4c, 0x00},
	{bankSel, bankDSP}, {reset, resetDVP},
	{hSize8, 0x64}, {vSize8, 0x4B}, {sizeL, 0x00},
	{hSize, 0xC8}, {vSize, 0x96}, {xOffL, 0x00}, {yOffL, 0x00}, {vhyx, 0x00},
	{test, 0x00},
	{ctrl2, ctrl2DCW | 0x1D}, {ctrlI, ctrlILPDP},
}

// settingsToUXGA is esp32-camera's ov2640_settings_to_uxga.
var settingsToUXGA = [...]reg{
	{bankSel, bankSensor}, {com7, com7ResUXGA}, {com1, 0x0F}, {reg32, 0x36},
	{hStart, 0x11}, {hStop, 0x75}, {vStart, 0x01}, {vStop, 0x97},
	{0x3d, 0x34}, {bd50, 0xbb}, {bd60, 0x9c}, {0x5a, 0x57}, {0x6d, 0x80},
	{0x39, 0x82}, {0x23, 0x00}, {0x07, 0xc0}, {0x4c, 0x00}, {0x35, 0x88},
	{0x22, 0x0a}, {0x37, 0x40}, {arcom2, 0xa0}, {0x06, 0x02}, {com4, 0xb7},
	{0x0e, 0x01}, {0x42, 0x83},
	{bankSel, bankDSP}, {reset, resetDVP},
	{hSize8, 0xc8}, {vSize8, 0x96}, {sizeL, 0x00},
	{hSize, 0x90}, {vSize, 0x2c}, {xOffL, 0x00}, {yOffL, 0x00}, {vhyx, 0x88},
	{test, 0x00},
	{ctrl2, ctrl2DCW | 0x1d}, {ctrlI, 0x00},
}

// Sensor modes: the sensor's readout (full field of view in each) before
// the DSP scales or crops to the output size.
type sensorMode struct {
	w, h  uint16 // field size in pixels at this mode
	com7  uint8
	table []reg
}

var (
	modeCIF  = sensorMode{400, 296, com7ResCIF, settingsToCIF[:]}
	modeSVGA = sensorMode{800, 600, com7ResSVGA, settingsToSVGA[:]}
	modeUXGA = sensorMode{1600, 1200, com7ResUXGA, settingsToUXGA[:]}
)

// modeFor picks the smallest sensor mode that holds w×h, as esp32-camera's
// set_framesize (CIF up to 400×296, SVGA up to 800×600, else UXGA).
func modeFor(w, h uint16) sensorMode {
	switch {
	case w <= modeCIF.w && h <= modeCIF.h:
		return modeCIF
	case w <= modeSVGA.w && h <= modeSVGA.h:
		return modeSVGA
	}
	return modeUXGA
}

var (
	settingsYUV422 = [...]reg{{bankSel, bankDSP}, {reset, resetDVP}, {imageMode, imageModeYUV422}, {0xD7, 0x01}, {0xE1, 0x67}, {reset, 0x00}}
	settingsRGB565 = [...]reg{{bankSel, bankDSP}, {reset, resetDVP}, {imageMode, imageModeRGB565}, {0xD7, 0x03}, {0xE1, 0x77}, {reset, 0x00}}
	settingsJPEG   = [...]reg{
		{bankSel, bankDSP}, {reset, resetJPEG | resetDVP}, {imageMode, imageModeJPEG | imageModeHV},
		{0xD7, 0x03}, {0xE1, 0x77}, {0xE5, 0x1F}, {0xD9, 0x10}, {0xDF, 0x80}, {0x33, 0x80},
		{0x3C, 0x10}, {0xEB, 0x30}, {0xDD, 0x7F}, {reset, 0x00},
	}
)

// ErrMode is returned for a size or format this driver does not set up.
var ErrMode = errors.New("ov2640: unsupported mode (sizes up to 1600×1200, multiples of 4; Y8, RGB565, RGB888Planar, YUV422, JPEG)")

// Options are host-specific choices.
type Options struct {
	// ClockDouble sets CLKRC's 2× PLL bit for raw (non-JPEG) output
	// (esp32-camera: on for every target except the original ESP32).
	ClockDouble bool
	// ClockDiv is CLKRC's divider for raw output in CIF mode (esp32-camera: 3).
	ClockDiv uint8
	// YUVLowByteFirst sets IMAGE_MODE's LBYTE_FIRST bit for YUV422/Y8, which
	// swaps the byte order on the bus. The K210 DVP's Y mode needs it
	// (observed, ER-K210-M4 evidence, outside the repo).
	YUVLowByteFirst bool
	// JPEGSensorHREF keeps HREF as the sensor's line timing in JPEG mode
	// (IMAGE_MODE bit 1 clear) instead of esp32-camera's HREF = VSYNC.
	// The K210 DVP needs it: each line then carries one 256-byte chunk and
	// every frame arrives whole; with HREF = VSYNC about half the frames
	// lose their tail (ER-K210-REC evidence, R3).
	JPEGSensorHREF bool
	// LineNS is the sensor's line period in ns for this clock setup; it
	// converts exposure between µs and AEC line periods. The backend measures
	// or computes it (0 = unknown: manual exposure is refused).
	LineNS uint32
}

// Sensor drives one OV2640.
type Sensor struct {
	Bus  sensor.Bus
	Opts Options
}

func (s *Sensor) Name() string { return "OV2640" }

func (s *Sensor) write(regs []reg) error {
	for _, r := range regs {
		if err := s.Bus.WriteReg8(Addr, r.addr, r.val); err != nil {
			return err
		}
	}
	return nil
}

// Init soft-resets the sensor and configures it for m. Y8 uses the YUV422
// sensor output; the capture backend keeps only Y. RGB888Planar uses the
// RGB565 output; the capture backend splits it into planes. m.TestPattern turns on
// the sensor's colour bars (COM7 bit 1). The sensor mode is the smallest of
// CIF (400×296), SVGA (800×600) and UXGA (1600×1200) that holds m.
func (s *Sensor) Init(m sensor.Mode) error {
	switch m.Format {
	case sensor.RGB565, sensor.RGB888Planar, sensor.YUV422, sensor.Y8, sensor.JPEG:
	default:
		return ErrMode
	}
	if m.W == 0 || m.H == 0 || m.W > modeUXGA.w || m.H > modeUXGA.h || m.W%4 != 0 || m.H%4 != 0 {
		return ErrMode
	}
	sm := modeFor(m.W, m.H)
	if err := s.write([]reg{{bankSel, bankSensor}, {com7, com7SRST}}); err != nil {
		return err
	}
	sleep(10 * time.Millisecond)
	if err := s.write(settingsCIF[:]); err != nil {
		return err
	}
	if err := s.setWindow(m, sm); err != nil {
		return err
	}
	if err := s.setFormat(m.Format); err != nil {
		return err
	}
	if m.TestPattern {
		return s.write([]reg{{bankSel, bankSensor}, {com7, sm.com7 | com7ColorBar}})
	}
	return nil
}

// SetQuality sets the JPEG quantisation scale (DSP register QS, as
// esp32-camera's set_quality): lower is better quality and larger frames.
func (s *Sensor) SetQuality(q uint8) error {
	return s.write([]reg{{bankSel, bankDSP}, {qs, q}})
}

func (s *Sensor) setFormat(f sensor.Format) error {
	var regs []reg
	switch f {
	case sensor.RGB565, sensor.RGB888Planar:
		regs = settingsRGB565[:]
	case sensor.YUV422, sensor.Y8:
		regs = settingsYUV422[:]
	case sensor.JPEG:
		regs = settingsJPEG[:]
	default:
		return ErrMode
	}
	if err := s.write(regs); err != nil {
		return err
	}
	if s.Opts.JPEGSensorHREF && f == sensor.JPEG {
		if err := s.write([]reg{{bankSel, bankDSP}, {reset, resetJPEG | resetDVP}, {imageMode, imageModeJPEG}, {reset, 0x00}}); err != nil {
			return err
		}
	}
	if s.Opts.YUVLowByteFirst && (f == sensor.YUV422 || f == sensor.Y8) {
		if err := s.write([]reg{{bankSel, bankDSP}, {reset, resetDVP}, {imageMode, imageModeYUV422 | imageModeLByteFirst}, {reset, 0x00}}); err != nil {
			return err
		}
	}
	sleep(10 * time.Millisecond)
	return nil
}

// setWindow programs sensor mode sm (its whole field of view, as
// esp32-camera's 4:3 ratio-table entry scaled to the mode; CIF's max_y is
// capped at 296) and the DSP window and output size. Full field (m.Crop
// false): the window is the whole field and the DSP scaler shrinks it to
// m.W×m.H. Crop: the window is the centred m.W×m.H region at 1:1 sensor
// pixels of the mode.
func (s *Sensor) setWindow(m sensor.Mode, sm sensorMode) error {
	maxX, maxY := sm.w/4, sm.h/4 // window size in 4-pixel units
	var offX, offY uint16        // window offset in pixels
	if m.Crop {
		maxX, maxY = m.W/4, m.H/4
		offX, offY = (sm.w-m.W)/2, (sm.h-m.H)/2
	}
	w, h := m.W/4, m.H/4
	win := []reg{
		{bankSel, bankDSP},
		{hSize, uint8(maxX)},
		{vSize, uint8(maxY)},
		{xOffL, uint8(offX)},
		{yOffL, uint8(offY)},
		{vhyx, uint8((maxY>>1)&0x80 | (offY>>4)&0x70 | (maxX>>5)&0x08 | (offX>>8)&0x07)},
		{test, uint8((maxX >> 2) & 0x80)},
		{zmow, uint8(w)},
		{zmoh, uint8(h)},
		{zmhh, uint8((h>>6)&0x04 | (w>>8)&0x03)},
	}
	// Clock: CLKRC (sensor bank) and R_DVP_SP (DSP bank) as esp32-camera:
	// raw output divides by ClockDiv in CIF and by 7 above it, with auto
	// PCLK ÷8 (÷12 in UXGA); JPEG runs undivided with manual PCLK ÷8 (÷12
	// in UXGA).
	div := uint8(7)
	if sm.com7 == com7ResCIF {
		div = s.Opts.ClockDiv
	}
	clk := div & 0x3F
	pclkDiv := uint8(8)
	if sm.com7 == com7ResUXGA {
		pclkDiv = 12
	}
	pclk := 0x80 | pclkDiv // auto PCLK
	if s.Opts.ClockDouble {
		clk |= 0x80
	}
	if m.Format == sensor.JPEG {
		clk, pclk = 0, pclkDiv
	}
	if err := s.write([]reg{{bankSel, bankDSP}, {rBypass, 0x01}}); err != nil {
		return err
	}
	if err := s.write(sm.table); err != nil {
		return err
	}
	if err := s.write(win); err != nil {
		return err
	}
	if err := s.write([]reg{{bankSel, bankSensor}, {clkrc, clk}, {bankSel, bankDSP}, {rDVPSP, pclk}, {rBypass, 0x00}}); err != nil {
		return err
	}
	sleep(10 * time.Millisecond)
	return nil
}

// sleep waits at least d, checking a deadline (a bare time.Sleep can return
// early on TinyGo 0.43/K210, ER-K210-M1 evidence §4, outside the repo).
func sleep(d time.Duration) {
	end := time.Now().Add(d)
	for left := time.Until(end); left > 0; left = time.Until(end) {
		time.Sleep(left)
	}
}
