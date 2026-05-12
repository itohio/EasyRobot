// ELRS receiver driver: reads RC channels from a CRSF/ELRS receiver,
// outputs channels to PWM (embedded) or console (desktop),
// and sends battery telemetry back.
//
// Build:
//   tinygo flash -target=pico -tags rp2040 .
//   tinygo flash -target=esp32-coreboard-v2 -tags esp32 .
//   tinygo flash -target=xiao -tags "sam xiao" .
//   go build -tags linux .
//   go build -tags windows .

package main

import (
	"context"
	"time"

	"github.com/itohio/EasyRobot/x/devices/crsf"
	"github.com/itohio/EasyRobot/x/devices/crsf/rx"
)

const telemetryInterval = 500 * time.Millisecond

func main() {
	initHardware()

	ctx := context.Background()
	r, w := getStream()

	dev := rx.New(ctx, r, w)
	dev.OnChannels(onChannels)
	dev.OnLinkStats(onLinkStats)
	dev.Start()

	ticker := time.NewTicker(telemetryInterval)
	for range ticker.C {
		heartbeat()
		v := readBatteryVoltage()
		dev.SendBattery(&crsf.Battery{
			Voltage:   v,
			Remaining: estimatePercent(v),
		})
	}
}

func estimatePercent(v float32) uint8 {
	switch {
	case v >= 12.6:
		return 100
	case v <= 9.0:
		return 0
	default:
		return uint8((v - 9.0) / 3.6 * 100)
	}
}
