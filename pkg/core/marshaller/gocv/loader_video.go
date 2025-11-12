package gocv

import (
	"context"
	"fmt"
	"time"

	cv "gocv.io/x/gocv"

	"github.com/itohio/EasyRobot/pkg/core/marshaller/types"
)

type videoFileLoader struct {
	path    string
	cfg     config
	capture *cv.VideoCapture
	index   int
}

func newVideoFileLoader(path string, cfg config) (sourceStream, error) {
	cap, err := cv.VideoCaptureFile(path)
	if err != nil {
		return nil, fmt.Errorf("gocv: open video file %s: %w", path, err)
	}
	return &videoFileLoader{
		path:    path,
		cfg:     cfg,
		capture: cap,
	}, nil
}

func (l *videoFileLoader) Next(ctx context.Context) (frameItem, bool, error) {
	if l.capture == nil {
		return frameItem{}, false, fmt.Errorf("gocv: video capture not initialised")
	}

	select {
	case <-ctx.Done():
		return frameItem{}, false, ctx.Err()
	default:
	}

	frame := cv.NewMat()
	if ok := l.capture.Read(&frame); !ok || frame.Empty() {
		frame.Close()
		return frameItem{}, false, nil
	}

	tensor, err := matToTensor(frame, l.cfg, types.DT_UNKNOWN)
	if err != nil {
		frame.Close()
		return frameItem{}, false, err
	}

	meta := map[string]any{
		"path":        l.path,
		"timestamp":   time.Now().UnixNano(),
		"source":      "video",
		"frame_index": l.index,
	}
	l.index++

	return frameItem{
		tensors:  []types.Tensor{tensor},
		metadata: meta,
	}, true, nil
}

func (l *videoFileLoader) Close() error {
	if l.capture != nil {
		l.capture.Close()
		l.capture = nil
	}
	return nil
}
