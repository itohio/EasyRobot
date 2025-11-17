package gocv

import (
	"sync"

	cv "gocv.io/x/gocv"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

type displayWriter struct {
	cfg     config
	window  *cv.Window
	once    sync.Once
	stopped bool
}

func newDisplayWriter(cfg config) (frameWriter, error) {
	return &displayWriter{
		cfg: cfg,
	}, nil
}

func (dw *displayWriter) ensureWindow() {
	dw.once.Do(func() {
		title := dw.cfg.display.title
		if title == "" {
			title = "GoCV Display"
		}
		dw.window = cv.NewWindow(title)
		if dw.cfg.display.width > 0 && dw.cfg.display.height > 0 {
			dw.window.ResizeWindow(dw.cfg.display.width, dw.cfg.display.height)
		}
		if dw.cfg.display.onMouse != nil {
			dw.window.SetMouseHandler(func(event int, x int, y int, flags int, _ interface{}) {
				if !dw.cfg.display.onMouse(event, x, y, flags) {
					dw.stopped = true
				}
			}, nil)
		}
	})
}

// WriteFrame implements StreamSink interface.
func (dw *displayWriter) WriteFrame(frame types.Frame) error {
	return dw.Write(frame)
}

// Write displays a frame (legacy method name, kept for backward compatibility).
func (dw *displayWriter) Write(frame types.Frame) error {
	if dw.stopped {
		return errStopLoop
	}
	if len(frame.Tensors) == 0 {
		return nil
	}

	dw.ensureWindow()
	if dw.window == nil {
		return errStopLoop
	}
	if !dw.window.IsOpen() {
		dw.stopped = true
		return errStopLoop
	}

	mat, err := tensorToMat(frame.Tensors[0])
	if err != nil {
		return err
	}
	defer mat.Close()

	if err := dw.window.IMShow(mat); err != nil {
		return err
	}

	key := dw.window.WaitKey(1)
	if key >= 0 && dw.cfg.display.onKey != nil {
		if !dw.cfg.display.onKey(key) {
			dw.stopped = true
			return errStopLoop
		}
	} else if key == 27 && dw.cfg.display.onKey == nil {
		dw.stopped = true
		return errStopLoop
	}

	if !dw.window.IsOpen() {
		dw.stopped = true
		return errStopLoop
	}

	if dw.stopped {
		return errStopLoop
	}

	return nil
}

func (dw *displayWriter) Close() error {
	if dw.window != nil {
		dw.window.Close()
		dw.window = nil
	}
	return nil
}
