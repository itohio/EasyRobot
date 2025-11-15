package source

import (
	"context"
	"errors"
	"fmt"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// Source is the interface for frame sources.
type Source interface {
	// RegisterFlags registers command-line flags for this source.
	RegisterFlags()

	// Start initializes the source and begins producing frames.
	Start(ctx context.Context) error

	// ReadFrame reads the next frame from the source.
	// Returns an error when source is exhausted or context is cancelled.
	ReadFrame() (types.Frame, error)

	// Close closes the source and cleans up resources.
	Close() error
}

var (
	// ErrSourceExhausted is returned when the source has no more frames.
	ErrSourceExhausted = errors.New("source exhausted")

	// ErrSourceNotStarted is returned when ReadFrame is called before Start.
	ErrSourceNotStarted = errors.New("source not started")
)

// baseSource provides common functionality for sources.
type baseSource struct {
	ctx     context.Context
	stream  types.FrameStream
	started bool
	frameCh <-chan types.Frame
	lastErr error
}

func (s *baseSource) Start(ctx context.Context) error {
	if s.started {
		return fmt.Errorf("source already started")
	}
	s.ctx = ctx
	s.started = true
	s.frameCh = s.stream.C
	return nil
}

func (s *baseSource) ReadFrame() (types.Frame, error) {
	if !s.started {
		return types.Frame{}, ErrSourceNotStarted
	}

	select {
	case <-s.ctx.Done():
		return types.Frame{}, s.ctx.Err()
	case frame, ok := <-s.frameCh:
		if !ok {
			return types.Frame{}, ErrSourceExhausted
		}
		// Check if frame has error in metadata
		if err, hasErr := frame.Metadata["error"]; hasErr {
			if errVal, ok := err.(error); ok {
				s.lastErr = errVal
				return types.Frame{}, errVal
			}
		}
		return frame, nil
	}
}

func (s *baseSource) Close() error {
	s.stream.Close()
	s.started = false
	return nil
}
