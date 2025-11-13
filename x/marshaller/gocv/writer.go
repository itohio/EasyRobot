package gocv

import (
	"errors"

	"github.com/itohio/EasyRobot/pkg/core/marshaller/types"
)

var errStopLoop = errors.New("gocv: stop loop")

type frameWriter interface {
	Write(frame types.Frame) error
	Close() error
}
