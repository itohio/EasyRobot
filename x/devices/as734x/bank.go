package as734x

func (d *Device) setRegisterBank(reg byte) error {
	if reg >= bank0Limit {
		return d.setBankAS7343(0)
	}
	return d.setBankAS7343(1)
}

func (d *Device) setBankAS7343(bank uint8) error {
	if d.variant != VariantAS7343 {
		return nil
	}
	if d.bank == bank {
		return nil
	}

	cfg0, err := d.readRegRaw(regCfg0AS7343)
	if err != nil {
		return err
	}
	if bank == 1 {
		cfg0 |= regBankMask
	} else {
		cfg0 &^= regBankMask
	}
	if err := d.writeRegRaw(regCfg0AS7343, cfg0); err != nil {
		return err
	}
	d.bank = bank
	return nil
}

func (d *Device) forceBankAS7343(bank uint8) error {
	cfg0, err := d.readRegRaw(regCfg0AS7343)
	if err != nil {
		return err
	}
	if bank == 1 {
		cfg0 |= regBankMask
	} else {
		cfg0 &^= regBankMask
	}
	return d.writeRegRaw(regCfg0AS7343, cfg0)
}

