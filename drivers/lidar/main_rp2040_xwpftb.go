//go:build rp2040 && xwpftb

package main

import (
	"context"

	devio "github.com/itohio/EasyRobot/x/devices"
	"github.com/itohio/EasyRobot/x/devices/lidar/xwpftb"
)

func createLIDARImpl(ctx context.Context, ser devio.Serial, motor devio.PWM) lidarDevice {
	return xwpftb.New(ctx, ser, motor, targetPoints, maxPoints)
}

func getBaudRateImpl() uint32 {
	return 115200 // XWPFTB typical baud rate
}

