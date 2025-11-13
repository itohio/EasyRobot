package gocv

import (
	"bufio"
	"errors"
	"fmt"
	"image"
	"io"
	"strings"

	cv "gocv.io/x/gocv"

	"github.com/itohio/EasyRobot/pkg/core/marshaller/types"
	tensorgocv "github.com/itohio/EasyRobot/pkg/core/math/tensor/gocv"
	tensortypes "github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// Unmarshaller decodes GoCV payloads.
type Unmarshaller struct {
	opts types.Options
	cfg  config
}

// NewUnmarshaller constructs a GoCV unmarshaller.
func NewUnmarshaller(opts ...types.Option) types.Unmarshaller {
	baseOpts, baseCfg := applyOptions(types.Options{}, defaultConfig(), opts)
	return &Unmarshaller{
		opts: baseOpts,
		cfg:  baseCfg,
	}
}

// Format identifies the unmarshaller.
func (u *Unmarshaller) Format() string {
	return "gocv"
}

// Unmarshal decodes into the provided destination.
func (u *Unmarshaller) Unmarshal(r io.Reader, dst any, opts ...types.Option) error {
	if dst == nil {
		return types.NewError("unmarshal", "gocv", "nil destination", nil)
	}

	localOpts, localCfg := applyOptions(u.opts, u.cfg, opts)

	switch out := dst.(type) {
	case *cv.Mat:
		return u.unmarshalMat(r, out)
	case **cv.Mat:
		if out == nil {
			return types.NewError("unmarshal", "gocv", "nil **gocv.Mat", nil)
		}
		var mat cv.Mat
		if err := u.unmarshalMat(r, &mat); err != nil {
			return err
		}
		*out = &mat
		return nil
	case *image.Image:
		return u.unmarshalImage(r, out)
	case **image.Image:
		if out == nil {
			return types.NewError("unmarshal", "gocv", "nil **image.Image", nil)
		}
		var img image.Image
		if err := u.unmarshalImage(r, &img); err != nil {
			return err
		}
		*out = &img
		return nil
	case *types.Tensor:
		tensor, err := u.unmarshalTensor(r, localCfg, localOpts.DestinationType)
		if err != nil {
			return err
		}
		*out = tensor
		return nil
	case *tensorgocv.Tensor:
		tensor, err := u.unmarshalTensor(r, localCfg, localOpts.DestinationType)
		if err != nil {
			return err
		}
		if tensor == nil {
			return types.NewError("unmarshal", "gocv", "tensor decode returned nil", nil)
		}
		gocvTensor, ok := tensor.(tensorgocv.Tensor)
		if !ok {
			return types.NewError("unmarshal", "gocv", "decoded tensor not GoCV-backed", nil)
		}
		*out = gocvTensor
		return nil
	case *types.FrameStream:
		stream, err := u.unmarshalStream(r, localCfg)
		if err != nil {
			return err
		}
		*out = stream
		return nil
	case *cv.Net:
		net, err := u.unmarshalNet(r, localCfg)
		if err != nil {
			return err
		}
		*out = net
		return nil
	case **cv.Net:
		if out == nil {
			return types.NewError("unmarshal", "gocv", "nil **gocv.Net", nil)
		}
		net, err := u.unmarshalNet(r, localCfg)
		if err != nil {
			return err
		}
		*out = &net
		return nil
	default:
		return types.NewError("unmarshal", "gocv", fmt.Sprintf("unsupported destination %T", dst), nil)
	}
}

func (u *Unmarshaller) unmarshalMat(r io.Reader, dst *cv.Mat) error {
	if dst == nil {
		return types.NewError("unmarshal", "gocv", "nil *gocv.Mat", nil)
	}
	data, err := io.ReadAll(r)
	if err != nil {
		return types.NewError("unmarshal", "gocv", "read mat bytes", err)
	}
	mat, err := decodeMatBytes(data, cv.IMReadColor)
	if err != nil {
		return types.NewError("unmarshal", "gocv", "decode mat", err)
	}
	*dst = mat
	return nil
}

func (u *Unmarshaller) unmarshalImage(r io.Reader, dst *image.Image) error {
	if dst == nil {
		return types.NewError("unmarshal", "gocv", "nil *image.Image", nil)
	}
	data, err := io.ReadAll(r)
	if err != nil {
		return types.NewError("unmarshal", "gocv", "read image bytes", err)
	}
	img, err := decodeImageBytes(data)
	if err != nil {
		return types.NewError("unmarshal", "gocv", "decode image", err)
	}
	*dst = img
	return nil
}

func (u *Unmarshaller) unmarshalTensor(r io.Reader, cfg config, dstType tensortypes.DataType) (types.Tensor, error) {
	mat := cv.Mat{}
	if err := u.unmarshalMat(r, &mat); err != nil {
		return nil, err
	}
	return matToTensor(mat, cfg, dstType)
}

func (u *Unmarshaller) unmarshalStream(r io.Reader, cfg config) (types.FrameStream, error) {
	if len(cfg.sources) == 0 && r != nil {
		paths, err := readPaths(r)
		if err != nil {
			return types.FrameStream{}, err
		}
		for _, p := range paths {
			cfg.sources = append(cfg.sources, sourceSpec{Path: p})
		}
	}

	streams, err := buildStreams(cfg)
	if err != nil {
		return types.FrameStream{}, err
	}
	return newFrameStream(cfg.ctx, streams, cfg.allowBestEffort, cfg.sequential)
}

func (u *Unmarshaller) unmarshalNet(r io.Reader, cfg config) (cv.Net, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return cv.Net{}, types.NewError("unmarshal", "gocv", "read DNN payload", err)
	}
	if len(data) == 0 {
		return cv.Net{}, types.NewError("unmarshal", "gocv", "empty DNN payload", errors.New("empty"))
	}
	net, err := loadNetFromBytes(data, cfg)
	if err != nil {
		return cv.Net{}, err
	}
	return net, nil
}

func readPaths(r io.Reader) ([]string, error) {
	var (
		scanner = bufio.NewScanner(r)
		paths   []string
	)
	for scanner.Scan() {
		path := strings.TrimSpace(scanner.Text())
		if path == "" {
			continue
		}
		paths = append(paths, path)
	}
	if err := scanner.Err(); err != nil {
		return nil, types.NewError("unmarshal", "gocv", "read paths", err)
	}
	return paths, nil
}
