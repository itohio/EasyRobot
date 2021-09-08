// +build planar

package main

import (
	"github.com/foxis/EasyRobot/pkg/robot/actuator/kinematics"
	"github.com/foxis/EasyRobot/pkg/robot/transport"
)

var (
	kine kinematics.Kinematics
)

func setState(packet transport.PacketData) {
}

func configKinematics(packet transport.PacketData) {
	defer configMotionKinematics(packet)

	var cfg kinematics.Config
	err := cfg.Unmarshal(packet.Data)
	if err != nil {
		return
	}
	if cfg.Planar == nil || len(cfg.Planar) != len(manipulatorConfig) {
		return
	}

	cfg := [3]Config{
		{Min: -90, Max: -90, Length: 0},
		{Min: -90, Max: -90, Length: 0},
		{Min: -90, Max: -90, Length: 0},
	}
	kine = planar.New3DOF()
}
