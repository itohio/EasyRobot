package features

import "errors"

const NAME = "features"

const (
	ORB = iota
	BRISK
	SURF
	SIFT
	KAZE
	AKAZE
	FREAK
	FAST
	GFTT
)

var (
	ErrNotSupported = errors.New("not supported")
)

type Matcher interface {
	Match(a, b interface{}) interface{}
}

type Options struct {
	featureType int
}
