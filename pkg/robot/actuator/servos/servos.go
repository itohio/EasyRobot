package servos

//go:generate protoc -I=./ -I=${GOPATH}/pkg/mod/ -I=${GOPATH}/src --gogofaster_out=./ types.proto
//go:generate go run ../../../../cmd/codegen -i types.pb.go -c ../../proto/proto.json -m re

type Actuator interface {
	Configure([]Motor) error
	Get() ([]float32, error)
	Set(params []float32) error
}

func NewDefaultConfig(pin uint32) Motor {
	return Motor{
		Pin:     pin,
		Min:     -90,
		Max:     90,
		Default: 0,
		Scale:   500 / 90.0,
		Offset:  1500,
	}
}
