//go:build ld06 && rp2040

package main

import (
	"context"

	devio "github.com/itohio/EasyRobot/x/devices"
	"github.com/itohio/EasyRobot/x/devices/lidar/ld06"
)

// LiDAR configuration for LD06
const (
	targetPoints = 0 // 0 = auto-calibrate
)

func createLIDARImpl(ctx context.Context, ser devio.Serial, motor devio.PWM) lidarDevice {
	return ld06.New(ctx, ser, motor, targetPoints, 3600)
}

func getBaudRateImpl() uint32 {
	return 230400 // LD06 baud rate
}
