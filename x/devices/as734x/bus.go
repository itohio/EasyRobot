package as734x

func (d *Device) readReg(reg byte) (byte, error) {
	if d.variant == VariantAS7343 {
		if err := d.setRegisterBank(reg); err != nil {
			return 0, err
		}
	}
	return d.readRegRaw(reg)
}

func (d *Device) writeReg(reg byte, value byte) error {
	if d.variant == VariantAS7343 {
		if err := d.setRegisterBank(reg); err != nil {
			return err
		}
	}
	return d.writeRegRaw(reg, value)
}

func (d *Device) readRegs(reg byte, buf []byte) error {
	if d.variant == VariantAS7343 {
		if err := d.setRegisterBank(reg); err != nil {
			return err
		}
	}
	return d.readRegsRaw(reg, buf)
}

func (d *Device) writeRegs(reg byte, buf []byte) error {
	if d.variant == VariantAS7343 {
		if err := d.setRegisterBank(reg); err != nil {
			return err
		}
	}
	return d.writeRegsRaw(reg, buf)
}

func (d *Device) readRegRaw(reg byte) (byte, error) {
	w := [1]byte{reg}
	var r [1]byte
	if err := d.bus.Tx(uint16(d.address), w[:], r[:]); err != nil {
		return 0, err
	}
	return r[0], nil
}

func (d *Device) writeRegRaw(reg byte, value byte) error {
	buf := [2]byte{reg, value}
	return d.bus.Tx(uint16(d.address), buf[:], nil)
}

func (d *Device) readRegsRaw(reg byte, buf []byte) error {
	w := [1]byte{reg}
	return d.bus.Tx(uint16(d.address), w[:], buf)
}

func (d *Device) writeRegsRaw(reg byte, buf []byte) error {
	payload := make([]byte, len(buf)+1)
	payload[0] = reg
	copy(payload[1:], buf)
	return d.bus.Tx(uint16(d.address), payload, nil)
}

