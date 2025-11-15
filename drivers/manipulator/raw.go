//go:build sam && xiao && !planar && !dh

package main

import (
	types "github.com/itohio/EasyRobot/types/control"
)

// Raw mode - no kinematics, direct joint angle control

func handleKinematicsConfig(config *types.ManipulatorConfig) {
	// Raw mode: ignore joints config, only use motors
	// No kinematics initialization needed
	kinematics = nil
}

func getIntentPathImpl() string {
	return "easyrobot.manipulator.raw"
}
