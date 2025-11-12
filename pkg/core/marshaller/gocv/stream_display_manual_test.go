//go:build gocv_display

package gocv

import (
	"context"
	"io"
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/marshaller/types"
)

// TestManualDisplayWebcam exercises the webcam->display pipeline. It requires a
// connected camera and X/GUI environment. Run with:
//
//	go test ./pkg/core/marshaller/gocv -tags gocv_display -run TestManualDisplayWebcam
//
// Press ESC (or close the window) to exit the loop.
func TestManualDisplayWebcam(t *testing.T) {
	unmarshaller := NewUnmarshaller(
		WithVideoDevice(0, 0, 0),
	)

	var stream types.FrameStream
	if err := unmarshaller.Unmarshal(nil, &stream); err != nil {
		t.Skipf("skipping manual display test: %v", err)
	}
	defer stream.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	marshaller := NewMarshaller(
		WithDisplay(),
		WithTitle("EasyRobot GoCV Display (Press ESC to exit)"),
		WithWindowSize(640, 480),
		WithOnKey(func(key int) bool {
			if key == 27 { // ESC
				cancel()
				return false
			}
			return true
		}),
		WithEventLoop(func(ctx context.Context, step func() bool) {
			for {
				select {
				case <-ctx.Done():
					return
				default:
				}
				if !step() {
					return
				}
			}
		}),
	)

	if err := marshaller.Marshal(io.Discard, stream); err != nil {
		t.Fatalf("marshal stream: %v", err)
	}
}
