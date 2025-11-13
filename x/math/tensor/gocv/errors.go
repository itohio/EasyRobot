package gocv

import (
	"errors"
	"fmt"
)

var (
	// ErrNilMat indicates that a tensor was created without an underlying Mat.
	ErrNilMat = errors.New("gocv tensor: nil mat")
	// ErrReleased reports an operation attempted after Release.
	ErrReleased = errors.New("gocv tensor: mat already released")
	// ErrUnsupported reports unimplemented tensor operations.
	ErrUnsupported = errors.New("gocv tensor: unsupported operation")
	// ErrUnsupportedDepth indicates an OpenCV Mat depth that we do not handle.
	ErrUnsupportedDepth = errors.New("gocv tensor: unsupported mat depth")
)

func panicUnsupported(op string) {
	panic(fmt.Errorf("%w: %s", ErrUnsupported, op))
}
