//go:build !planar || !dh
// +build !planar !dh

package main

import (
	servos "github.com/itohio/EasyRobot/pkg/control/actuator/servos/fw"
	"github.com/itohio/EasyRobot/pkg/control/transport"
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

	motionLock.Lock()
	defer motionLock.Unlock()
	for i := range state.Params {
		motion[i].Target = state.Params[i]
	}
}

func configKinematics(packet transport.PacketData) {
	configMotionKinematics(packet)
}
