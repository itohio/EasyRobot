package camera

import (
	"testing"

	"github.com/itohio/EasyRobot/drivers/camera/sensor"
)

func TestFrameSize(t *testing.T) {
	cfg := func(w, h uint16, f sensor.Format) Config {
		return Config{Mode: sensor.Mode{Resolution: sensor.Resolution{W: w, H: h}, Format: f}}
	}
	// 2·65535² = 8.6e9 bytes: an int only on 64-bit hosts, else FrameSize
	// must report 0 instead of a wrapped size.
	bigU := uint64(2) * 65535 * 65535
	big := 0
	if bigU <= uint64(maxInt) {
		big = int(bigU)
	}
	for _, tt := range []struct {
		cfg  Config
		want int
	}{
		{cfg(160, 120, sensor.Y8), 19200},
		{cfg(320, 240, sensor.RGB565), 153600},
		{cfg(320, 240, sensor.YUV422), 153600},
		{cfg(256, 256, sensor.RGB888Planar), 196608},
		{cfg(320, 240, sensor.JPEG), 0},
		{cfg(320, 240, 0), 0},
		{cfg(0, 240, sensor.Y8), 0},
		{cfg(65535, 65535, sensor.RGB565), big},
	} {
		if got := FrameSize(tt.cfg); got != tt.want {
			t.Errorf("FrameSize(%v) = %d, want %d", tt.cfg.Mode, got, tt.want)
		}
	}
}
