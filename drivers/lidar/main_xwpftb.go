//go:build xwpftb && rp2040

package main

import (
	"context"

	devio "github.com/itohio/EasyRobot/x/devices"
	"github.com/itohio/EasyRobot/x/devices/lidar/xwpftb"
)

// LiDAR configuration for XWPFTB
const (
	targetPoints = 0 // 0 = auto-calibrate
	maxPoints    = 2048
)

func createLIDARImpl(ctx context.Context, ser devio.Serial, motor devio.PWM) lidarDevice {
	return xwpftb.New(ctx, ser, motor, targetPoints, maxPoints)
}

func getBaudRateImpl() uint32 {
	return 115200 // XWPFTB typical baud rate
}
