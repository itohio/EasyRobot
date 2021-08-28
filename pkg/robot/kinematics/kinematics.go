package kinematics

//go:generate protoc -I=./ -I=${GOPATH}/pkg/mod/ -I=${GOPATH}/src --gogofaster_out=./ types.proto
//go:generate go run ../../../cmd/codegen -i types.pb.go -c ../proto/proto.json -m re

import "github.com/foxis/EasyRobot/pkg/core/math/vec"

type Kinematics interface {
	DOF() int
	Params() vec.Vector
	Effector() vec.Vector

	Forward() bool
	Inverse() bool
}
