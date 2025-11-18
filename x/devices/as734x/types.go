package as734x

import "time"

// DefaultAddress is the default I2C address for AS734x devices
const DefaultAddress = 0x39

// Variant represents the AS734x device variant.
type Variant uint8

const (
	VariantUnknown Variant = iota
	VariantAS7341
	VariantAS7343
)

func (v Variant) String() string {
	switch v {
	case VariantAS7341:
		return "AS7341"
	case VariantAS7343:
		return "AS7343"
	default:
		return "UNKNOWN"
	}
}

// Gain represents the gain setting for the device.
type Gain uint8

const (
	Gain0p5x Gain = iota
	Gain1x
	Gain2x
	Gain4x
	Gain8x
	Gain16x
	Gain32x
	Gain64x
	Gain128x
	Gain256x
	Gain512x
	Gain1024x
	Gain2048x
)

// Multiplier returns the gain multiplier as a float32.
func (g Gain) Multiplier() float32 {
	switch g {
	case Gain0p5x:
		return 0.5
	case Gain1x:
		return 1
	case Gain2x:
		return 2
	case Gain4x:
		return 4
	case Gain8x:
		return 8
	case Gain16x:
		return 16
	case Gain32x:
		return 32
	case Gain64x:
		return 64
	case Gain128x:
		return 128
	case Gain256x:
		return 256
	case Gain512x:
		return 512
	case Gain1024x:
		return 1024
	case Gain2048x:
		return 2048
	default:
		return 1
	}
}

// Config holds the configuration for the AS734x device.
type Config struct {
	Address       uint8
	Gain          Gain
	ATime         uint8
	AStep         uint16
	FlickerEnable bool
}

// DefaultConfig returns a default configuration.
func DefaultConfig() Config {
	return Config{
		Address:       DefaultAddress,
		Gain:          Gain16x,
		ATime:         29,
		AStep:         599,
		FlickerEnable: true,
	}
}

// Flicker contains flicker detection data.
type Flicker struct {
	Frequency uint16
	Valid     bool
	Saturated bool
	Raw       uint8
}

// RawMeasurement contains a raw spectral measurement.
type RawMeasurement struct {
	Timestamp     time.Time
	Variant       Variant
	Channels      []uint16
	IntegrationUs uint32
	Gain          Gain
	Flicker       Flicker
	Saturated     bool
}

