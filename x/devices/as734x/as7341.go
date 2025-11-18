package as734x

import (
	"errors"
	"time"
)

func (d *Device) configureAS7341() error {
	if d.cfg.Gain > Gain512x {
		return errUnsupportedGain
	}
	if err := d.writeReg(regATime, d.cfg.ATime); err != nil {
		return err
	}
	if err := d.writeAstStepAS7341(); err != nil {
		return err
	}
	if err := d.writeReg(regCfg1, byte(d.cfg.Gain)); err != nil {
		return err
	}
	// ensure spectral engine disabled until read
	return d.enableSpectral(false)
}

func (d *Device) writeAstStepAS7341() error {
	low := byte(d.cfg.AStep & 0xFF)
	high := byte(d.cfg.AStep >> 8)
	if err := d.writeReg(regAStepL, low); err != nil {
		return err
	}
	return d.writeReg(regAStepH, high)
}

func (d *Device) setSmuxCommand(cmd byte) error {
	reg, err := d.readReg(regCfg6)
	if err != nil {
		return err
	}
	reg &^= regSmuxWriteMask
	reg |= (cmd << 3) & regSmuxWriteMask
	return d.writeReg(regCfg6, reg)
}

func (d *Device) enableSmux() error {
	reg, err := d.readReg(regEnable)
	if err != nil {
		return err
	}
	reg |= regSmuxEnable
	if err := d.writeReg(regEnable, reg); err != nil {
		return err
	}
	for i := 0; i < 50; i++ {
		val, err := d.readReg(regEnable)
		if err != nil {
			return err
		}
		if val&regSmuxEnable == 0 {
			return nil
		}
		time.Sleep(time.Millisecond * 2)
	}
	return errors.New("as7341: smux enable timeout")
}

func (d *Device) loadSmux(config [20]byte) error {
	if err := d.enableSpectral(false); err != nil {
		return err
	}
	if err := d.setSmuxCommand(smuxCmdWrite); err != nil {
		return err
	}
	if err := d.writeRegs(0x00, config[:]); err != nil {
		return err
	}
	return d.enableSmux()
}

func (d *Device) readAS7341() (RawMeasurement, error) {
	low, err := d.captureBlock(smuxLowConfig)
	if err != nil {
		return RawMeasurement{}, err
	}
	high, err := d.captureBlock(smuxHighConfig)
	if err != nil {
		return RawMeasurement{}, err
	}
	channels := mergeAS7341Channels(low, high)
	flicker := d.detectFlickerAS7341()
	sat := (low.status|high.status)&regStatus2Saturation != 0
	return RawMeasurement{
		Timestamp:     time.Now(),
		Variant:       VariantAS7341,
		Channels:      channels,
		IntegrationUs: d.integrationTimeUs(),
		Gain:          d.cfg.Gain,
		Flicker:       flicker,
		Saturated:     sat,
	}, nil
}

type channelBlock struct {
	values [6]uint16
	status byte
}

func (d *Device) captureBlock(cfg [20]byte) (channelBlock, error) {
	if err := d.prepareCapture(cfg); err != nil {
		return channelBlock{}, err
	}
	return d.readChannelBlock()
}

func (d *Device) prepareCapture(cfg [20]byte) error {
	if err := d.loadSmux(cfg); err != nil {
		return err
	}
	if err := d.enableSpectral(true); err != nil {
		return err
	}
	return d.waitForDataReady(regStatus2AS7341, regStatus2AValid, time.Millisecond*200)
}

func (d *Device) readChannelBlock() (channelBlock, error) {
	buf := make([]byte, 12)
	if err := d.readRegs(regCh0DataL, buf); err != nil {
		return channelBlock{}, err
	}
	var cb channelBlock
	for i := 0; i < 6; i++ {
		lo := buf[i*2]
		hi := buf[i*2+1]
		cb.values[i] = uint16(hi)<<8 | uint16(lo)
	}
	status, err := d.readReg(regStatus2AS7341)
	if err != nil {
		return channelBlock{}, err
	}
	cb.status = status
	return cb, nil
}

func mergeAS7341Channels(low, high channelBlock) []uint16 {
	out := make([]uint16, 10)
	copy(out[0:4], low.values[0:4])
	copy(out[4:8], high.values[0:4])
	clear := averageUint16(low.values[4], high.values[4])
	nir := averageUint16(low.values[5], high.values[5])
	out[8] = clear
	out[9] = nir
	return out
}

func averageUint16(a, b uint16) uint16 {
	return uint16((uint32(a) + uint32(b)) / 2)
}

func (d *Device) detectFlickerAS7341() Flicker {
	if !d.cfg.FlickerEnable {
		return Flicker{}
	}
	reg, err := d.startFlickerAS7341()
	if err != nil {
		return Flicker{}
	}
	status, err := d.finishFlickerAS7341(reg)
	if err != nil {
		return Flicker{}
	}
	frequency := decodeFlickerFrequency(status)
	return Flicker{
		Frequency: frequency,
		Valid:     status&0x20 != 0,
		Saturated: status&0x10 != 0,
		Raw:       status,
	}
}

func (d *Device) startFlickerAS7341() (byte, error) {
	_ = d.enableSpectral(false)
	_ = d.writeReg(regEnable, 0)
	if err := d.writeReg(regEnable, 0x01); err != nil {
		return 0, err
	}
	if err := d.loadSmux(smuxFlickerConfig); err != nil {
		return 0, err
	}
	reg, err := d.readReg(regEnable)
	if err != nil {
		return 0, err
	}
	reg |= regSpectralEnable | regFlickerEnable
	if err := d.writeReg(regEnable, reg); err != nil {
		return 0, err
	}
	return reg, nil
}

func (d *Device) finishFlickerAS7341(reg byte) (byte, error) {
	time.Sleep(time.Millisecond * 200)
	status, err := d.readReg(regFdStatus)
	if err != nil {
		return 0, err
	}
	return status, d.writeReg(regEnable, reg&^regFlickerEnable)
}

func decodeFlickerFrequency(status byte) uint16 {
	switch {
	case status&0x02 != 0:
		return 120
	case status&0x01 != 0:
		return 100
	case status&0x04 != 0:
		return 100
	case status&0x08 != 0:
		return 120
	case status&0x0C != 0:
		return 0
	case status&0x2C != 0:
		return 0
	case status&0x2C == 0 && status != 0:
		return 1
	default:
		return 0
	}
}

