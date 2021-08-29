package display

import (
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"time"

	. "github.com/foxis/EasyRobot/pkg/core/logger"
	"github.com/foxis/EasyRobot/pkg/core/options"
	"github.com/foxis/EasyRobot/pkg/core/pipeline"
	"github.com/foxis/EasyRobot/pkg/core/plugin"
	"github.com/foxis/EasyRobot/pkg/core/store"

	"gocv.io/x/gocv"
)

type display struct {
	Options
	ch  <-chan pipeline.Data
	out chan pipeline.Data
}

func init() {
	pipeline.Register(NAME+"GoCV", NewGoCV)
}

func NewGoCV(opts ...options.Option) (pipeline.Step, error) {
	step := &display{
		Options: Options{
			base: plugin.DefaultOptions(),
			keys: []store.FQDNType{},
		},
	}
	options.ApplyOptions(&step.Options, opts...)
	options.ApplyOptions(&step.Options.base, plugin.WithName(NAME+"GoCV"))
	options.ApplyOptions(&step.Options.base, opts...)
	step.Reset()
	return step, nil
}

func (s *display) In(ch <-chan pipeline.Data) {
	s.ch = ch
}

func (s *display) Out() <-chan pipeline.Data {
	return s.out
}

func (s *display) Run(ctx context.Context) {
	Log.Debug().Str("name", s.base.Name).Msg("Run")
	defer Log.Debug().Str("name", s.base.Name).Msg("Exit")

	defer close(s.out)
	windows := make([]*gocv.Window, len(s.keys))
	for i, key := range s.keys {
		windows[i] = gocv.NewWindow(fmt.Sprint(s.base.Name, ": ", key.String()))
		defer func(w *gocv.Window) {
			w.Close()
		}(windows[i])
	}

	data := store.NewWithName(s.base.Name)
	index := 0

	for {
		key := gocv.WaitKey(10)
		data.Set(store.USER_KEY_CODE, key)
		data.SetIndex(int64(index))
		data.SetTimestamp(int64(time.Now().UnixNano()))
		index++

		err := pipeline.StepSend(ctx, s.base, s.out, data)
		if err != nil && !errors.Is(err, pipeline.ErrDrop) {
			Log.Error().Err(err).Msg("Send")
			return
		}

		in, err := pipeline.StepReceive(ctx, s.base, s.ch)
		if err != nil {
			return
		}

		for idx, key := range s.keys {
			s.draw(windows, in, idx, key)

			// if s.base.Close {
			// 	in.Close(key)
			// }
		}
	}
}

func (s *display) Reset() {
	s.out = pipeline.StepMakeChan(s.base)
}

func (s *display) draw(windows []*gocv.Window, data store.Store, idx int, key store.FQDNType) {
	switch key {
	case store.IMAGE:
		fallthrough
	case store.IMAGE_GRAYSCALE:
		fallthrough
	case store.DEPTH_IMAGE:
		fallthrough
	case store.MAP_IMAGE:
		s.drawImage(windows, data, idx, key)
	case store.KEY_POINTS:
		s.drawImage(windows, data, idx, key)
	default:
	}
}

func (s *display) drawImage(windows []*gocv.Window, data store.Store, idx int, key store.FQDNType) {
	val, ok := data.Get(key)
	if !ok {
		Log.Debug().Str("key", key.String()).Msg("No data")
		return
	}

	origImg, ok := val.(*gocv.Mat)
	if !ok {
		Log.Debug().Str("key", key.String()).Msg("No Image")
		return
	}

	img := origImg.Clone()
	defer img.Close()

	fps, err := data.FPS()
	fpsStr := ""
	if err == nil {
		fpsStr = fmt.Sprint("fps: ", fps)
	} else {
		Log.Debug().Msg("No FPS")
	}
	drop, err := data.DropCount()
	if err == nil {
		fpsStr = fmt.Sprint(fpsStr, "   drop: ", drop)
	} else {
		Log.Debug().Msg("No dropped frames")
	}

	gocv.PutText(&img, fpsStr, image.Pt(10, 20), gocv.FontHersheyPlain, 1.2, color.RGBA{0, 255, 0, 0}, 2)
	size := img.Size()
	gocv.PutText(&img, fmt.Sprint(size[0], "x", size[1]), image.Pt(10, 40), gocv.FontHersheyPlain, 1.2, color.RGBA{0, 255, 0, 0}, 2)
	windows[idx].IMShow(img)
}
