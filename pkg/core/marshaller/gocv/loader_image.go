package gocv

import (
	"context"
	"fmt"
	"path/filepath"
	"time"

	cv "gocv.io/x/gocv"

	"github.com/itohio/EasyRobot/pkg/core/marshaller/types"
	tensorgocv "github.com/itohio/EasyRobot/pkg/core/math/tensor/gocv"
)

type imageLoader struct {
	path     string
	cfg      config
	consumed bool
}

func newImageLoader(path string, cfg config) sourceStream {
	return &imageLoader{
		path: filepath.Clean(path),
		cfg:  cfg,
	}
}

func (l *imageLoader) Next(ctx context.Context) (frameItem, bool, error) {
	if l.consumed {
		return frameItem{}, false, nil
	}
	l.consumed = true

	item, err := loadFrameFromFile(l.path, l.cfg)
	if err != nil {
		return frameItem{}, false, err
	}
	return item, true, nil
}

func (l *imageLoader) Close() error {
	return nil
}

func loadFrameFromFile(path string, cfg config) (frameItem, error) {
	ext := filepath.Ext(path)
	switch classifyPath(path) {
	case sourceKindSingle:
		return loadImageFile(path, cfg)
	case sourceKindVideoFile:
		return frameItem{}, fmt.Errorf("gocv: path %s is a video file, expected image", path)
	default:
		if ext == ".mat" {
			return loadMatFile(path, cfg)
		}
		return loadImageFile(path, cfg)
	}
}

func loadImageFile(path string, cfg config) (frameItem, error) {
	mat := cv.IMRead(path, cv.IMReadColor)
	if mat.Empty() {
		return frameItem{}, fmt.Errorf("gocv: failed to load image %s", path)
	}

	opts := append([]tensorgocv.Option{}, cfg.tensorOpts...)
	opts = append(opts, tensorgocv.WithAdoptedMat())

	tensor, err := tensorgocv.FromMat(mat, opts...)
	if err != nil {
		mat.Close()
		return frameItem{}, err
	}

	metadata := map[string]any{
		"path":      path,
		"timestamp": time.Now().UnixNano(),
		"source":    "image",
	}

	return frameItem{
		tensors:  []types.Tensor{tensor},
		metadata: metadata,
	}, nil
}

func loadMatFile(path string, cfg config) (frameItem, error) {
	mat, err := readCustomMat(path)
	if err != nil {
		return frameItem{}, err
	}

	tensor, err := matToTensor(mat, cfg, types.DT_UNKNOWN)
	if err != nil {
		return frameItem{}, err
	}

	metadata := map[string]any{
		"path":      path,
		"timestamp": time.Now().UnixNano(),
		"source":    "mat",
	}

	return frameItem{
		tensors:  []types.Tensor{tensor},
		metadata: metadata,
	}, nil
}
