//go:build linux || windows

package main

import (
	"context"
	"fmt"

	devio "github.com/itohio/EasyRobot/x/devices"
	"github.com/itohio/EasyRobot/x/devices/lidar/ld06"
	"github.com/itohio/EasyRobot/x/devices/lidar/xwpftb"
)

// DeviceFactory creates LiDAR devices based on type.
type DeviceFactory struct{}

// NewDeviceFactory creates a new device factory.
func NewDeviceFactory() *DeviceFactory {
	return &DeviceFactory{}
}

// CreateDevice creates a LiDAR device based on the configuration.
func (f *DeviceFactory) CreateDevice(ctx context.Context, cfg *Config, serial devio.Serial) (lidarDevice, error) {
	switch cfg.LidarType {
	case "xwpftb":
		return f.createXWPFTB(ctx, serial, cfg.TargetPts)
	case "ld06":
		return f.createLD06(ctx, serial, cfg.TargetPts)
	default:
		return nil, fmt.Errorf("unknown LiDAR type: %s", cfg.LidarType)
	}
}

func (f *DeviceFactory) createXWPFTB(ctx context.Context, serial devio.Serial, targetPts int) (lidarDevice, error) {
	dev := xwpftb.New(ctx, serial, nil, targetPts, 2048)
	if err := dev.Configure(true); err != nil {
		return nil, fmt.Errorf("failed to configure XWPFTB: %w", err)
	}
	return dev, nil
}

func (f *DeviceFactory) createLD06(ctx context.Context, serial devio.Serial, targetPts int) (lidarDevice, error) {
	dev := ld06.New(ctx, serial, nil, targetPts, 3600)
	if err := dev.Configure(true); err != nil {
		return nil, fmt.Errorf("failed to configure LD06: %w", err)
	}
	return dev, nil
}

