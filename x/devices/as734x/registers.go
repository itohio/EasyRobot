package as734x

// AS7341 register addresses
const (
	as7341ChipID         = 0x09
	regWhoAmIAS7341      = 0x92
	regEnable            = 0x80
	regATime             = 0x81
	regAStepL            = 0xCA
	regAStepH            = 0xCB
	regCfg1              = 0xAA
	regStatus2AS7341     = 0xA3
	regCh0DataL          = 0x95
	regCfg6              = 0xAF
	regSmuxWriteMask     = 0x18
	regSmuxEnableBit     = 0x10
	regSpectralEnable    = 0x02
	regSmuxEnable        = 0x10
	regFlickerEnable     = 0x40
	regStatus2AValid     = 0x40
	regStatus2Saturation = 0x10
	regFdStatus          = 0xDB
)

// AS7343 register addresses
const (
	as7343ChipID      = 0x81
	regIDAS7343       = 0x5A
	regATimeAS7343    = 0x81
	regAStepAS7343    = 0xD4
	regCfg1AS7343     = 0xC6
	regCfg20AS7343    = 0xD6
	regStatus2AS7343  = 0x90
	regData0AS7343    = 0x95
	regFdStatusAS7343 = 0xE3
	regCfg0AS7343     = 0xBF
	regBankMask       = 0x10
)

// SMUX commands
const (
	smuxCmdRomReset byte = 0
	smuxCmdRead     byte = 1
	smuxCmdWrite    byte = 2
)

// AS7343 auto SMUX mode
const (
	autoSmux18 = 0x03
)

// Bank limit for AS7343
const (
	bank0Limit byte = 0x80
)

