package as734x

import "time"

func (d *Device) configureAS7343() error {
	if d.cfg.Gain > Gain2048x {
		return errUnsupportedGain
	}
	if err := d.writeReg(regATimeAS7343, d.cfg.ATime); err != nil {
		return err
	}
	if err := d.writeAstStepAS7343(); err != nil {
		return err
	}
	if err := d.writeReg(regCfg1AS7343, byte(d.cfg.Gain)); err != nil {
		return err
	}
	if err := d.setAutoSmuxAS7343(autoSmux18); err != nil {
		return err
	}
	return d.configureFlickerAS7343()
}

func (d *Device) writeAstStepAS7343() error {
	low := byte(d.cfg.AStep & 0xFF)
	high := byte(d.cfg.AStep >> 8)
	if err := d.writeReg(regAStepAS7343, low); err != nil {
		return err
	}
	return d.writeReg(regAStepAS7343+1, high)
}

func (d *Device) setAutoSmuxAS7343(mode byte) error {
	reg, err := d.readReg(regCfg20AS7343)
	if err != nil {
		return err
	}
	reg &^= 0x60
	reg |= (mode & 0x03) << 5
	return d.writeReg(regCfg20AS7343, reg)
}

func (d *Device) configureFlickerAS7343() error {
	reg, err := d.readReg(regEnable)
	if err != nil {
		return err
	}
	if d.cfg.FlickerEnable {
		reg |= 0x20
	} else {
		reg &^= 0x20
	}
	return d.writeReg(regEnable, reg)
}

func (d *Device) readAS7343() (RawMeasurement, error) {
	if err := d.enableSpectral(true); err != nil {
		return RawMeasurement{}, err
	}
	if err := d.waitForDataReady(regStatus2AS7343, 0x40, time.Millisecond*200); err != nil {
		return RawMeasurement{}, err
	}
	buf := make([]byte, 36)
	if err := d.readRegs(regData0AS7343, buf); err != nil {
		return RawMeasurement{}, err
	}
	_ = d.enableSpectral(false)
	channels := decodeAS7343Channels(buf)
	sat := d.as7343Saturation()
	flicker := d.readFlickerAS7343()
	return RawMeasurement{
		Timestamp:     time.Now(),
		Variant:       VariantAS7343,
		Channels:      channels,
		IntegrationUs: d.integrationTimeUs(),
		Gain:          d.cfg.Gain,
		Flicker:       flicker,
		Saturated:     sat,
	}, nil
}

func decodeAS7343Channels(buf []byte) []uint16 {
	out := make([]uint16, 18)
	for i := 0; i < 18; i++ {
		lo := buf[i*2]
		hi := buf[i*2+1]
		out[i] = uint16(hi)<<8 | uint16(lo)
	}
	return out
}

func (d *Device) as7343Saturation() bool {
	status, err := d.readReg(regStatus2AS7343)
	if err != nil {
		return false
	}
	return status&0x10 != 0 || status&0x08 != 0
}

func (d *Device) readFlickerAS7343() Flicker {
	if !d.cfg.FlickerEnable {
		return Flicker{}
	}
	status, err := d.readReg(regFdStatusAS7343)
	if err != nil {
		return Flicker{}
	}
	return Flicker{
		Frequency: decodeAS7343Flicker(status),
		Valid:     status&0x20 != 0,
		Saturated: status&0x10 != 0,
		Raw:       status,
	}
}

func decodeAS7343Flicker(status byte) uint16 {
	switch {
	case status&0x01 != 0 && status&0x04 != 0:
		return 100
	case status&0x02 != 0 && status&0x08 != 0:
		return 120
	default:
		return 0
	}
}

