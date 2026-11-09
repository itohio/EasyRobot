package ov2640

import (
	"errors"

	"github.com/itohio/EasyRobot/drivers/camera/sensor"
)

// Control registers (OV2640 datasheet register names; WB gain registers and
// their 0x40 = unity scale as used by OpenMV's ov2640.c).
const (
	gain  = 0x00 // sensor bank: AGC gain
	aecLo = 0x04 // sensor bank REG04 bits 1:0 = AEC[1:0]
	aec   = 0x10 // sensor bank AEC[9:2]
	reg45 = 0x45 // sensor bank bits 5:0 = AEC[15:10]

	com8AEC = 0x01
	com8AGC = 0x04
	ctrl1WB = 0x08 // DSP bank CTRL1: AWB enable

	wbR, wbG, wbB = 0xCC, 0xCD, 0xCE // DSP bank manual WB gains
	wbUnity       = 0x40
)

// ErrNoLineTime means Options.LineNS is 0, so exposure cannot be converted
// between µs and line periods.
var ErrNoLineTime = errors.New("ov2640: line period unknown (Options.LineNS)")

// LineNS estimates the line period for the CIF setup this driver programs
// (CLKRC 2×, ÷4, DVP PCLK auto) at XCLK xclkHz. The init tables' banding
// value BD50 = 0xCA says 202 line periods per 10 ms (half a 50 Hz cycle) at
// the 24 MHz XCLK they were tuned for; line time scales with 1/XCLK.
// Inferred from the tables, not measured; the C1 run checks it.
func LineNS(xclkHz uint32) uint32 {
	if xclkHz == 0 {
		return 0
	}
	return uint32(uint64(10_000_000) * 24_000_000 / (202 * uint64(xclkHz)))
}

func (s *Sensor) read(bank, r uint8) (uint8, error) {
	if err := s.Bus.WriteReg8(Addr, bankSel, bank); err != nil {
		return 0, err
	}
	return s.Bus.ReadReg8(Addr, r)
}

// encodeGain returns the GAIN register value for g: bits 7:4 are four ×2
// stages (set from bit 4 up), bits 3:0 add (1 + n/16).
func encodeGain(g float32) uint8 {
	if g < 1 {
		g = 1
	}
	var stages uint8
	for g >= 2 && stages < 4 {
		g /= 2
		stages++
	}
	n := int((g-1)*16 + 0.5)
	if n > 15 {
		n = 15
	}
	return uint8(1<<stages-1)<<4 | uint8(n)
}

func decodeGain(v uint8) float32 {
	g := 1 + float32(v&0x0F)/16
	for bit := uint8(0x10); bit != 0; bit <<= 1 {
		if v&bit != 0 {
			g *= 2
		}
	}
	return g
}

// SetControls applies c. Manual exposure needs Options.LineNS.
func (s *Sensor) SetControls(c sensor.Controls) error {
	com8v, err := s.read(bankSensor, com8)
	if err != nil {
		return err
	}
	com8v &^= com8AEC | com8AGC
	if c.AutoExposure {
		com8v |= com8AEC
	}
	if c.AutoGain {
		com8v |= com8AGC
	}
	regs := []reg{{bankSel, bankSensor}, {com8, com8v}}
	if !c.AutoExposure {
		if s.Opts.LineNS == 0 {
			return ErrNoLineTime
		}
		lines := (uint64(c.ExposureUS)*1000 + uint64(s.Opts.LineNS)/2) / uint64(s.Opts.LineNS)
		if lines < 1 {
			lines = 1
		}
		if lines > 0xFFFF {
			lines = 0xFFFF
		}
		lo, err := s.read(bankSensor, aecLo)
		if err != nil {
			return err
		}
		hi, err := s.read(bankSensor, reg45)
		if err != nil {
			return err
		}
		regs = append(regs,
			reg{bankSel, bankSensor},
			reg{aecLo, lo&^0x03 | uint8(lines&0x03)},
			reg{aec, uint8(lines >> 2)},
			reg{reg45, hi&^0x3F | uint8(lines>>10)&0x3F})
	}
	if !c.AutoGain {
		regs = append(regs, reg{bankSel, bankSensor}, reg{gain, encodeGain(c.Gain)})
	}
	ctrl1v, err := s.read(bankDSP, ctrl1)
	if err != nil {
		return err
	}
	ctrl1v &^= ctrl1WB
	if c.AutoWB {
		ctrl1v |= ctrl1WB
	}
	regs = append(regs, reg{bankSel, bankDSP}, reg{ctrl1, ctrl1v})
	if !c.AutoWB {
		for i, r := range [3]uint8{wbR, wbG, wbB} {
			v := c.WB[i]*wbUnity + 0.5
			if v > 255 {
				v = 255
			}
			regs = append(regs, reg{r, uint8(v)})
		}
	}
	return s.write(regs)
}

// ReadControls reads the current settings, including the values the auto
// loops chose. ExposureUS is 0 when Options.LineNS is unknown.
func (s *Sensor) ReadControls() (sensor.Controls, error) {
	var c sensor.Controls
	var v [8]uint8
	for i, r := range []struct{ bank, reg uint8 }{
		{bankSensor, com8}, {bankSensor, aecLo}, {bankSensor, aec}, {bankSensor, reg45},
		{bankSensor, gain}, {bankDSP, ctrl1}, {bankDSP, wbR}, {bankDSP, wbG},
	} {
		x, err := s.read(r.bank, r.reg)
		if err != nil {
			return c, err
		}
		v[i] = x
	}
	b, err := s.read(bankDSP, wbB)
	if err != nil {
		return c, err
	}
	c.AutoExposure = v[0]&com8AEC != 0
	c.AutoGain = v[0]&com8AGC != 0
	lines := uint32(v[1]&0x03) | uint32(v[2])<<2 | uint32(v[3]&0x3F)<<10
	c.ExposureUS = uint32(uint64(lines) * uint64(s.Opts.LineNS) / 1000)
	c.Gain = decodeGain(v[4])
	c.AutoWB = v[5]&ctrl1WB != 0
	c.WB = [3]float32{float32(v[6]) / wbUnity, float32(v[7]) / wbUnity, float32(b) / wbUnity}
	return c, nil
}
