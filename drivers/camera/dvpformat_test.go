package camera

import (
	"errors"
	"testing"

	"github.com/itohio/EasyRobot/drivers/camera/sensor"
)

// TestCheckFormatDVPLines: raw frames are limited to the DVP's 1,023
// lines (10-bit LINE_NUM); JPEG frames are not (their DVP lines are 256-byte
// chunks, bounded by JPEGBufSize).
func TestCheckFormatDVPLines(t *testing.T) {
	cfg := func(w, h uint16, f sensor.Format) Config {
		return Config{Mode: sensor.Mode{Resolution: sensor.Resolution{W: w, H: h}, Format: f}}
	}
	for _, c := range []struct {
		cfg Config
		ok  bool
	}{
		{cfg(1600, 1023, sensor.RGB565), true},
		{cfg(1600, 1024, sensor.RGB565), false},
		{cfg(1600, 1200, sensor.Y8), false},
		{cfg(1600, 1200, sensor.RGB888Planar), false},
		{cfg(1600, 1200, sensor.JPEG), true},
		{cfg(320, 240, sensor.RGB565), true},
		{cfg(330, 240, sensor.RGB565), false}, // width not a multiple of 32
	} {
		err := checkFormat(c.cfg)
		if c.ok != (err == nil) || (err != nil && !errors.Is(err, ErrFormat)) {
			t.Errorf("%v: err %v, want ok %v", c.cfg.Mode, err, c.ok)
		}
	}
}
