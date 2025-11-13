package gocv

import (
	"sync"

	cv "gocv.io/x/gocv"

	"github.com/itohio/EasyRobot/pkg/core/marshaller/types"
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
		title := dw.cfg.displayTitle
		if title == "" {
			title = "GoCV Display"
		}
		dw.window = cv.NewWindow(title)
		if dw.cfg.displayWidth > 0 && dw.cfg.displayHeight > 0 {
			dw.window.ResizeWindow(dw.cfg.displayWidth, dw.cfg.displayHeight)
		}
		if dw.cfg.onMouse != nil {
			dw.window.SetMouseHandler(func(event int, x int, y int, flags int, _ interface{}) {
				if !dw.cfg.onMouse(event, x, y, flags) {
					dw.stopped = true
				}
			}, nil)
		}
	})
}

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
	if key >= 0 && dw.cfg.onKey != nil {
		if !dw.cfg.onKey(key) {
			dw.stopped = true
			return errStopLoop
		}
	} else if key == 27 && dw.cfg.onKey == nil {
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
