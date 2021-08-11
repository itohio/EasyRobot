package reader

import (
	"image"
	"os"
	"time"

	"github.com/foxis/EasyRobot/pkg/core/pipeline"
	"github.com/foxis/EasyRobot/pkg/core/pipeline/steps"
	"github.com/foxis/EasyRobot/pkg/core/plugin"
)

type defaultReader readerOpts

const DEFAULT_NAME = "rimg"

func init() {
	pipeline.Register(DEFAULT_NAME, NewDefault)
}

func NewDefault(opts ...plugin.Option) (pipeline.Step, error) {
	o := readerOpts{}
	plugin.ApplyOptions(&o, opts...)
	newOpts := opts
	newOpts = append(newOpts, WithDefaultReader(o.paths))
	return steps.NewReader(newOpts...)
}

func WithDefaultReader(paths []string) plugin.Option {
	return steps.WithSourceReader(&defaultReader{paths: paths})
}

func (s *defaultReader) Read(o steps.SourceOptions) (img interface{}, path string, index int, timestamp int64, err error) {
	if s.index >= len(s.paths) {
		if o.Repeat {
			s.index = 0
		} else {
			err = pipeline.ErrEOS
			return
		}
	}

	path = s.paths[s.index]

	f, err := os.Open(path)
	if err != nil {
		return
	}
	defer f.Close()
	im, _, err := image.Decode(f)
	if err != nil {
		return
	}
	img = &im
	timestamp = time.Now().UnixNano()

	index = s.index
	s.index++
	return
}

func (s *defaultReader) Reset() {
	s.index = 0
}

func (s *defaultReader) Open() error {
	s.index = 0
	return nil
}

func (s *defaultReader) Close() {
}

func (s *defaultReader) Name() string {
	return DEFAULT_NAME
}
