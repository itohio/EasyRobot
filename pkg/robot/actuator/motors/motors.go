package motors

import (
	"github.com/foxis/EasyRobot/pkg/robot/actuator/motors/pb"
	"github.com/foxis/EasyRobot/pkg/robot/kinematics"
)

//go:generate protoc -I=./ -I=${GOPATH}/pkg/mod/ -I=${GOPATH}/src --gogofaster_out=./ --proto_path=../../ pb/types.proto
//go:generate go run ../../../../cmd/codegen -i pb/types.pb.go -c ../../proto/proto.json -m re

const (
	// Device ID for robot.transport
	ID = 0x00000200
)

type Motor = pb.Motor
type Config = pb.Config
type State = kinematics.State
