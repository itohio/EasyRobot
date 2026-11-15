package camera

import "github.com/itohio/EasyRobot/drivers/camera/sensor"

// dvpMaxLines is the most lines the K210 DVP captures per frame: its
// LINE_NUM field is 10 bits (1024–1200-line raw
// frames were truncated silently).
const dvpMaxLines = 1023

// checkDVPFormat reports whether the K210 DVP path captures cfg: Y8, RGB565
// and RGB888Planar at widths that are multiples of 32 and at most
// dvpMaxLines lines; JPEG at any size the sensor makes (its lines are
// chunks, see JPEGBufSize).
func checkDVPFormat(cfg Config) error {
	switch cfg.Format {
	case sensor.Y8, sensor.RGB565, sensor.RGB888Planar:
		if cfg.W == 0 || cfg.H == 0 || cfg.W%32 != 0 || cfg.H > dvpMaxLines {
			return ErrFormat
		}
	case sensor.JPEG:
		if cfg.W == 0 || cfg.H == 0 {
			return ErrFormat
		}
	default:
		return ErrFormat
	}
	return nil
}
