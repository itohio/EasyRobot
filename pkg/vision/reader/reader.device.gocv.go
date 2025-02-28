package reader

import (
	"time"

	"github.com/itohio/EasyRobot/pkg/backend"
	. "github.com/itohio/EasyRobot/pkg/core/logger"
	"github.com/itohio/EasyRobot/pkg/core/options"
	"github.com/itohio/EasyRobot/pkg/core/pipeline"
	"github.com/itohio/EasyRobot/pkg/core/pipeline/steps"

	"gocv.io/x/gocv"
)

const DEVICE_NAME = "rdev"

func init() {
	pipeline.Register(DEVICE_NAME, NewDeviceGoCV)
}

type readerDeviceGoCV struct {
	readerOpts
	dev *gocv.VideoCapture
}

func NewDeviceGoCV(opts ...options.Option) (pipeline.Step, error) {
	o := readerOpts{}
	options.ApplyOptions(&o, opts...)
	newOpts := opts
	newOpts = append(newOpts, WithDeviceReaderGoCVResolution(o.id, o.width, o.height))
	return steps.NewReader(newOpts...)
}

func WithDeviceReaderGoCVDefault(id int) options.Option {
	return steps.WithSourceReader(&readerDeviceGoCV{readerOpts: readerOpts{id: id}})
}

func WithDeviceReaderGoCVResolution(id, width, height int) options.Option {
	return steps.WithSourceReader(&readerDeviceGoCV{readerOpts: readerOpts{id: id, width: width, height: height}})
}

func (s *readerDeviceGoCV) Read(o steps.SourceOptions) (img interface{}, path string, index int, timestamp int64, err error) {
	mat := backend.NewGoCVMat()
	s.dev.Read(mat)
	if !mat.Empty() {
		img = mat
	}

	timestamp = time.Now().UnixNano()
	index = s.index
	s.index++
	return
}

func (s *readerDeviceGoCV) Reset() {
	s.index = 0
}

func (s *readerDeviceGoCV) Open() error {
	Log.Debug().Int("device", s.id).Msg("Open")
	dev, err := gocv.OpenVideoCapture(s.id)
	if err != nil {
		Log.Error().Err(err).Int("device", s.id)
		return err
	}
	if s.width != 0 && s.height != 0 {
		dev.Set(gocv.VideoCaptureFrameWidth, float64(s.width))
		dev.Set(gocv.VideoCaptureFrameHeight, float64(s.height))

	}
	s.dev = dev
	s.index = 0
	return nil
}

func (s *readerDeviceGoCV) Close() {
	if s.dev != nil {
		s.dev.Close()
	}
}

func (s *readerDeviceGoCV) Name() string {
	return DEVICE_NAME
}
