package reader

import (
	"encoding/binary"
	"os"
	"strings"
	"time"

	"github.com/foxis/EasyRobot/pkg/backend"
	"github.com/foxis/EasyRobot/pkg/pipeline"
	"github.com/foxis/EasyRobot/pkg/pipeline/steps"
	"github.com/foxis/EasyRobot/pkg/plugin"

	"gocv.io/x/gocv"
)

type readerGoCV readerOpts

const IMAGE_GOCV_NAME = "rmat"

func init() {
	pipeline.Register(IMAGE_GOCV_NAME, NewGoCV)
}

func NewGoCV(opts ...plugin.Option) (pipeline.Step, error) {
	o := readerOpts{}
	plugin.ApplyOptions(&o, opts...)
	newOpts := opts
	newOpts = append(newOpts, WithDefaultReader(o.paths))
	return steps.NewReader(newOpts...)
}

func WithReaderGoCV(paths []string) plugin.Option {
	return steps.WithSourceReader(&readerGoCV{paths: paths})
}

func (s *readerGoCV) Read(o steps.SourceOptions) (img interface{}, path string, index int, timestamp int64, err error) {
	if s.index >= len(s.paths) {
		if o.Repeat {
			s.index = 0
		} else {
			err = pipeline.ErrEOS
			return
		}
	}

	path = s.paths[s.index]

	mat := s.readMat(path)
	timestamp = time.Now().UnixNano()
	if !mat.Empty() {
		img = backend.FromGoCVMat(mat)
	}

	index = s.index
	s.index++
	return
}

func (s *readerGoCV) readMat(path string) gocv.Mat {
	if strings.HasSuffix(path, ".png") ||
		strings.HasSuffix(path, ".bmp") ||
		strings.HasSuffix(path, ".jpg") ||
		strings.HasSuffix(path, ".jpeg") {
		return gocv.IMRead(path, gocv.IMReadColor)
	}

	fp, err := os.Open(path)
	if err != nil {
		return gocv.Mat{}
	}
	defer fp.Close()

	var (
		magic    uint64
		lenSizes uint8
		sizes    []int
		matType  gocv.MatType
		lenBytes uint64
		bytes    []byte
	)
	err = binary.Read(fp, binary.LittleEndian, &magic)
	if err != nil {
		return gocv.Mat{}
	}

	if magic != 0xabcdef0012345678 {
		return gocv.Mat{}
	}

	err = binary.Read(fp, binary.LittleEndian, &lenSizes)
	if err != nil {
		return gocv.Mat{}
	}
	sizes = make([]int, lenSizes)
	err = binary.Read(fp, binary.LittleEndian, sizes)
	if err != nil {
		return gocv.Mat{}
	}

	err = binary.Read(fp, binary.LittleEndian, &matType)
	if err != nil {
		return gocv.Mat{}
	}

	err = binary.Read(fp, binary.LittleEndian, &lenBytes)
	if err != nil {
		return gocv.Mat{}
	}
	bytes = make([]byte, lenBytes)
	err = binary.Read(fp, binary.LittleEndian, bytes)
	if err != nil {
		return gocv.Mat{}
	}

	mat, err := gocv.NewMatWithSizesFromBytes(sizes, matType, bytes)
	if err != nil {
		return gocv.Mat{}
	}
	return mat
}

func (s *readerGoCV) Reset() {
	s.index = 0
}

func (s *readerGoCV) Open() error {
	s.index = 0
	return nil
}

func (s *readerGoCV) Close() {
}

func (s *readerGoCV) Name() string {
	return IMAGE_GOCV_NAME
}
