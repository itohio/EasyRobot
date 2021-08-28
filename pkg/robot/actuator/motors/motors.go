package motors

//go:generate protoc -I=./ -I=${GOPATH}/pkg/mod/ -I=${GOPATH}/src --gogofaster_out=./ types.proto
//go:generate go run ../../../../cmd/codegen -i types.pb.go -c ../../proto/proto.json -m re

type Actuator interface {
	Configure([]Motor) error
	GetSpeed() ([]float32, error)
	SetSpeed(params []float32) error
	SetTorque(params []float32) error
}
