package ov3660

import (
	"errors"
	"math"

	"github.com/itohio/EasyRobot/drivers/camera/sensor"
)

// Control registers (esp32-camera ov3660.c set_exposure_ctrl, set_gain_ctrl,
// set_aec_value, set_agc_gain, set_wb_mode; C1 evidence for the live-value
// and freeze behaviour).
const (
	regAECManual = 0x3503 // bit 0 AEC manual, bit 1 AGC manual
	regExposure  = 0x3500 // [3:0] 0x3501 0x3502[7:4]: exposure lines in register bits 19:4
	regGain      = 0x350A // [1:0] 0x350B: gain × 16
	regAWBManual = 0x3406 // bit 0: manual white-balance gains
	regWBGain    = 0x3400 // R, G, B at 0x3400/02/04: 12 bit, 0x400 = 1.0
	regAWBLive   = 0x519F // R, G, B at 0x519F/A1/A3: the AWB loop's current gains
	aecManual    = 0x01
	agcManual    = 0x02
	wbUnity      = 0x400
	vtsMargin    = 4                         // frame lines beyond the exposure: VTS ≥ exposure lines + 4
	maxExposure  = (0xFFFF - vtsMargin) << 4 // longest exposure the 16-bit VTS can hold, whole lines × 16
	maxGain16    = 0x3FF
)

// ErrNotInit means Init has not run, so the line period is unknown.
var ErrNotInit = errors.New("ov3660: controls need Init first (line period unknown)")

var _ sensor.Controller = (*Sensor)(nil)

// exposureReg converts µs to the exposure register value: whole lines
// (rounded to the nearest) in bits 19:4, clamped to the register. 0x3502
// stores only bits 7:4, so the sensor exposes in whole line periods
// (esp32-camera set_aec_value writes lines << 4; C1 read-back 5552 = 347 lines).
func exposureReg(us, lineNS uint32) uint32 {
	lines := (uint64(us)*1000 + uint64(lineNS)/2) / uint64(lineNS)
	if lines<<4 > maxExposure {
		return maxExposure
	}
	return uint32(lines << 4)
}

func exposureUS(v, lineNS uint32) uint32 { return uint32(uint64(v>>4) * uint64(lineNS) / 1000) }

// gainReg converts an analogue gain to the register (× 16), clamped to
// 1 … 63.9; NaN gives 1.
func gainReg(g float32) uint16 {
	if !(g >= 1) {
		return 16
	}
	v := math.Round(float64(g) * 16)
	if v > maxGain16 {
		return maxGain16
	}
	return uint16(v)
}

// wbReg converts a white-balance gain to the 12-bit register, clamped;
// negative and NaN give 0.
func wbReg(g float32) uint16 {
	if !(g > 0) {
		return 0
	}
	v := math.Round(float64(g) * wbUnity)
	if v > 0xFFF {
		return 0xFFF
	}
	return uint16(v)
}

// SetControls applies c. A manual exposure is rounded to whole lines and
// silently clamped to 65,531 lines (≈ 3.77 s at 2048×1536); one longer than
// the mode's frame stretches the frame to exposure + 4 lines, so the frame
// rate drops to match; auto exposure or a shorter
// one restores the mode's frame length. Turning AutoWB off with c.WB from ReadControls
// freezes the current gains (the AWB block stays on; clearing ISP_CONTROL_01
// bit 0, as esp32-camera's set_whitebal(0) does, removes white balance
// altogether: C1 finding). It does not allocate.
func (s *Sensor) SetControls(c sensor.Controls) error {
	if s.lineNS == 0 {
		return ErrNotInit
	}
	mode := uint8(0)
	if !c.AutoExposure {
		mode |= aecManual
	}
	if !c.AutoGain {
		mode |= agcManual
	}
	if err := s.write(regAECManual, mode); err != nil {
		return err
	}
	// The sensor exposes at most one frame: a longer manual exposure
	// stretches the frame (VTS) to fit; otherwise the mode's VTS applies.
	vts := s.vts
	if !c.AutoExposure {
		v := exposureReg(c.ExposureUS, s.lineNS)
		vts = max(vts, uint16(v>>4)+vtsMargin)
		if err := s.write(regExposure, uint8(v>>16&0x0F)); err != nil {
			return err
		}
		if err := s.write(regExposure+1, uint8(v>>8)); err != nil {
			return err
		}
		if err := s.write(regExposure+2, uint8(v)&0xF0); err != nil {
			return err
		}
	}
	if err := s.write16(regXTotal+2, vts); err != nil {
		return err
	}
	if !c.AutoGain {
		if err := s.write16(regGain, gainReg(c.Gain)); err != nil {
			return err
		}
	}
	if !c.AutoWB {
		// Step a register variable rather than add 2·i to the constant:
		// TinyGo 0.43's Xtensa backend fails at -opt=z on
		// regWBGain+uint16(2*i) ("Cannot select: i32 = Constant<13312>").
		r := uint16(regWBGain)
		for _, g := range c.WB {
			if err := s.write16(r, wbReg(g)); err != nil {
				return err
			}
			r += 2
		}
	}
	isp, err := s.read(regISPCtrl01)
	if err != nil {
		return err
	}
	if err := s.write(regISPCtrl01, isp|0x01); err != nil { // AWB block on
		return err
	}
	awb := uint8(0)
	if !c.AutoWB {
		awb = 1
	}
	return s.write(regAWBManual, awb)
}

// ReadControls reads the current settings, including the values the auto
// loops chose (the exposure and gain registers hold the loop's values while
// AE/AGC are on; the AWB loop's gains are read from 0x519F/A1/A3).
func (s *Sensor) ReadControls() (sensor.Controls, error) {
	var c sensor.Controls
	if s.lineNS == 0 {
		return c, ErrNotInit
	}
	mode, err := s.read(regAECManual)
	if err != nil {
		return c, err
	}
	c.AutoExposure = mode&aecManual == 0
	c.AutoGain = mode&agcManual == 0
	var e [3]uint8
	for i := range e {
		if e[i], err = s.read(regExposure + uint16(i)); err != nil {
			return c, err
		}
	}
	c.ExposureUS = exposureUS(uint32(e[0]&0x0F)<<16|uint32(e[1])<<8|uint32(e[2]&0xF0), s.lineNS)
	g, err := s.read16(regGain)
	if err != nil {
		return c, err
	}
	c.Gain = float32(g&0x3FF) / 16
	awb, err := s.read(regAWBManual)
	if err != nil {
		return c, err
	}
	c.AutoWB = awb&0x01 == 0
	r := uint16(regWBGain)
	if c.AutoWB {
		r = regAWBLive
	}
	for i := range c.WB { // register stepped, not computed from i (see SetControls)
		v, err := s.read16(r)
		if err != nil {
			return c, err
		}
		c.WB[i] = float32(v&0xFFF) / wbUnity
		r += 2
	}
	return c, nil
}
