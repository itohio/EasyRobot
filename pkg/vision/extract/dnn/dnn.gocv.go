package dnn

import (
	"errors"
	"image"
	"path/filepath"

	"github.com/foxis/EasyRobot/pkg/backend"
	"github.com/foxis/EasyRobot/pkg/pipeline"
	"github.com/foxis/EasyRobot/pkg/pipeline/steps"
	"github.com/foxis/EasyRobot/pkg/plugin"
	"github.com/foxis/EasyRobot/pkg/store"

	"gocv.io/x/gocv"
)

type stepImpl struct {
	options Options
	step    pipeline.Step
	net     gocv.Net
	ratio   float64
	mean    gocv.Scalar
	swapRGB bool
}

func init() {
	pipeline.Register(NAME, NewGoCV)
}

func NewGoCV(opts ...plugin.Option) (pipeline.Step, error) {
	algorithm := &stepImpl{
		options: Options{
			backend: gocv.NetBackendDefault,
			target:  gocv.NetTargetCPU,
		},
	}

	plugin.ApplyOptions(&algorithm.options, opts...)

	data := store.New()
	data.Set(store.DNN_MODEL, algorithm.options.model)

	if algorithm.options.ratio == 0 {
		if filepath.Ext(algorithm.options.model) == ".caffemodel" {
			algorithm.ratio = 1.0
			algorithm.mean = gocv.NewScalar(104, 177, 123, 0)
			algorithm.swapRGB = false
		} else {
			algorithm.ratio = 1.0 / 127.5
			algorithm.mean = gocv.NewScalar(127.5, 127.5, 127.5, 0)
			algorithm.swapRGB = true
		}
	} else {
		algorithm.ratio = algorithm.options.ratio
		algorithm.mean = gocv.NewScalar(
			algorithm.options.mean[0],
			algorithm.options.mean[1],
			algorithm.options.mean[2],
			algorithm.options.mean[3],
		)
		algorithm.swapRGB = algorithm.options.swapRGB
	}

	newOpts := append([]plugin.Option{plugin.WithName(NAME)}, opts...)
	newOpts = append(
		newOpts,
		steps.WithProcessor(algorithm),
		steps.WithFields(data),
	)

	step, err := steps.NewProcessor(newOpts...)
	if err != nil {
		return nil, err
	}
	algorithm.step = step
	return step, nil
}

func (s *stepImpl) Init() error {
	s.net = gocv.ReadNet(s.options.model, s.options.config)
	if s.net.Empty() {
		return errors.New("could not load network")
	}

	s.net.SetPreferableBackend(gocv.NetBackendType(s.options.backend))
	s.net.SetPreferableTarget(gocv.NetTargetType(s.options.target))

	return nil
}

func (s *stepImpl) Reset() {
	return
}

func (s *stepImpl) Close() {
	s.net.Close()
	return
}

func (s *stepImpl) Process(src, dst pipeline.Data) error {
	imgVal, ok := src.Get(store.IMAGE)
	if !ok {
		return nil
	}
	img, ok := imgVal.(gocv.Mat)
	if !ok {
		return nil
	}

	blob := gocv.BlobFromImage(img, s.ratio, image.Pt(s.options.width, s.options.height), s.mean, s.swapRGB, false)

	// feed the blob into the detector
	s.net.SetInput(blob, "")

	// run a forward pass thru the network
	prob := s.net.Forward("")

	dst.Set(store.FEATURES, backend.FromGoCVMat(prob))

	return nil
}
