package gocv

import (
	"context"
	"fmt"
	"time"

	cv "gocv.io/x/gocv"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

type videoDeviceLoader struct {
	spec    deviceSpec
	cfg     config
	capture *cv.VideoCapture
	index   int
}

func newVideoDeviceLoader(spec deviceSpec, cfg config) (sourceStream, error) {
	cap, err := cv.OpenVideoCapture(spec.ID)
	if err != nil {
		return nil, fmt.Errorf("gocv: open video device %d: %w", spec.ID, err)
	}
	if spec.Width > 0 {
		cap.Set(cv.VideoCaptureFrameWidth, float64(spec.Width))
	}
	if spec.Height > 0 {
		cap.Set(cv.VideoCaptureFrameHeight, float64(spec.Height))
	}
	return &videoDeviceLoader{
		spec:    spec,
		cfg:     cfg,
		capture: cap,
	}, nil
}

func (l *videoDeviceLoader) Next(ctx context.Context) (frameItem, bool, error) {
	if l.capture == nil {
		return frameItem{}, false, fmt.Errorf("gocv: device capture closed")
	}

	for {
		select {
		case <-ctx.Done():
			return frameItem{}, false, ctx.Err()
		default:
		}

		frame := cv.NewMat()
		if ok := l.capture.Read(&frame); !ok {
			frame.Close()
			if l.cfg.stream.allowBestEffort {
				time.Sleep(10 * time.Millisecond)
				continue
			}
			return frameItem{}, false, nil
		}
		if frame.Empty() {
			frame.Close()
			if l.cfg.stream.allowBestEffort {
				time.Sleep(5 * time.Millisecond)
				continue
			}
			return frameItem{}, false, nil
		}

		tensor, err := matToTensor(frame, l.cfg, types.DT_UNKNOWN)
		if err != nil {
			frame.Close()
			return frameItem{}, false, err
		}

		filename := fmt.Sprintf("device_%d_%06d.png", l.spec.ID, l.index)
		meta := map[string]any{
			"device":      l.spec.ID,
			"timestamp":   time.Now().UnixNano(),
			"source":      "video_device",
			"frame_index": l.index,
			"index":       l.index,
			"filename":    filename,
			"name":        []string{filename},
		}
		l.index++

		return frameItem{
			tensors:  []types.Tensor{tensor},
			metadata: meta,
		}, true, nil
	}
}

func (l *videoDeviceLoader) Close() error {
	if l.capture != nil {
		l.capture.Close()
		l.capture = nil
	}
	return nil
}
