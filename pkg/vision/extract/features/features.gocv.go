package features

import (
	"github.com/foxis/EasyRobot/pkg/backend"
	"github.com/foxis/EasyRobot/pkg/core/pipeline"
	"github.com/foxis/EasyRobot/pkg/core/pipeline/steps"
	"github.com/foxis/EasyRobot/pkg/core/plugin"
	"github.com/foxis/EasyRobot/pkg/core/store"

	"gocv.io/x/gocv"
)

type GOCVDetector interface {
	Detect(gocv.Mat) []gocv.KeyPoint
	DetectAndCompute(src gocv.Mat, mask gocv.Mat) ([]gocv.KeyPoint, gocv.Mat)
}
type GOCVMatcher interface {
	KnnMatch(query, train gocv.Mat, k int) [][]gocv.DMatch
}

type stepImpl struct {
	options Options
	step    pipeline.Step

	orb   gocv.ORB
	kaze  gocv.KAZE
	akaze gocv.AKAZE
	brisk gocv.BRISK
	sift  gocv.SIFT
	fast  gocv.FastFeatureDetector
	gftt  gocv.GFTTDetector

	matcherFlann gocv.FlannBasedMatcher
	matcherBF    gocv.BFMatcher

	matcher  GOCVMatcher
	detector GOCVDetector
}

func init() {
	pipeline.Register(NAME, NewGoCV)
}

func NewGoCV(opts ...plugin.Option) (pipeline.Step, error) {
	algorithm := &stepImpl{
		options: Options{
			featureType: ORB,
		},
	}

	plugin.ApplyOptions(&algorithm.options, opts...)

	data := store.New()
	data.Set(store.FEATURES_TYPE, algorithm.options.featureType)
	data.Set(store.MATCHER, algorithm)

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
	return step, nil
}

func (s *stepImpl) Init() error {
	switch s.options.featureType {
	case ORB:
		s.orb = gocv.NewORB()
		s.matcherFlann = gocv.NewFlannBasedMatcher()
		s.matcher = &s.matcherFlann
		s.detector = &s.orb
	case SIFT:
		s.sift = gocv.NewSIFT()
		s.matcherBF = gocv.NewBFMatcher()
		s.matcher = &s.matcherBF
		s.detector = &s.sift
	case KAZE:
		s.kaze = gocv.NewKAZE()
		s.matcherFlann = gocv.NewFlannBasedMatcher()
		s.matcher = &s.matcherFlann
		s.detector = &s.kaze
	case AKAZE:
		s.akaze = gocv.NewAKAZE()
		s.matcherFlann = gocv.NewFlannBasedMatcher()
		s.matcher = &s.matcherFlann
		s.detector = &s.akaze
	case BRISK:
		s.brisk = gocv.NewBRISK()
		s.matcherFlann = gocv.NewFlannBasedMatcher()
		s.matcher = &s.matcherFlann
		s.detector = &s.brisk
	// case FAST:
	// 	s.fast = gocv.NewFastFeatureDetector()
	// 	s.detector = &s.fast
	default:
		return ErrNotSupported
	}

	return nil
}

func (s *stepImpl) Reset() {
	return
}
func (s *stepImpl) Close() {
	return
}

func (s *stepImpl) Process(src, dst pipeline.Data) error {
	imgVal, ok := src.Get(store.IMAGE)
	if !ok {
		return nil
	}
	img, ok := imgVal.(*gocv.Mat)
	if !ok {
		return nil
	}

	keyPoints, descriptors := s.detector.DetectAndCompute(*img, gocv.NewMat())
	dst.Set(store.KEY_POINTS, keyPoints)
	dst.Set(store.DESCRIPTORS, backend.FromGoCVMat(descriptors))

	return nil
}

func (s *stepImpl) Match(a, b interface{}) interface{} {
	query, ok := a.(*gocv.Mat)
	if !ok {
		return nil
	}
	train, ok := b.(*gocv.Mat)
	if !ok {
		return nil
	}

	return s.matcher.KnnMatch(*query, *train, 2)
}
