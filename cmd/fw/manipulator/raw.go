// +build !kinematics

package main

import (
	"github.com/foxis/EasyRobot/pkg/robot/actuator/servos"
	"github.com/foxis/EasyRobot/pkg/robot/transport"
)

func setState(packet transport.PacketData) {
	var state servos.State
	err := state.Unmarshal(packet.Data)
	if err != nil {
		return
	}
	if len(state.Params) != len(manipulatorConfig) {
		return
	}

	manipulator.Set(state.Params)
}

func configKinematics(packet transport.PacketData) {
}
