package gocv

import (
	"fmt"
	"image"
	"io"

	cv "gocv.io/x/gocv"

	"github.com/itohio/EasyRobot/pkg/core/marshaller/types"
	tensorgocv "github.com/itohio/EasyRobot/pkg/core/math/tensor/gocv"
)

// Marshaller encodes GoCV-friendly values using gob envelopes.
type Marshaller struct {
	opts types.Options
	cfg  config
}

// NewMarshaller constructs a GoCV marshaller instance.
func NewMarshaller(opts ...types.Option) types.Marshaller {
	baseOpts, baseCfg := applyOptions(types.Options{}, defaultConfig(), opts)
	return &Marshaller{
		opts: baseOpts,
		cfg:  baseCfg,
	}
}

// Format returns the format identifier.
func (m *Marshaller) Format() string {
	return "gocv"
}

// Marshal serialises Mat, image, or tensor values.
func (m *Marshaller) Marshal(w io.Writer, value any, opts ...types.Option) error {
	if value == nil {
		return types.NewError("marshal", "gocv", "nil value", nil)
	}

	localOpts, localCfg := applyOptions(m.opts, m.cfg, opts)
	_ = localOpts // reserved for future use

	switch v := value.(type) {
	case cv.Mat:
		return m.writeMat(w, v)
	case *cv.Mat:
		if v == nil {
			return types.NewError("marshal", "gocv", "nil *gocv.Mat", nil)
		}
		return m.writeMat(w, *v)
	case image.Image:
		return m.writeImage(w, v, localCfg.imageEncoding)
	case *image.Image:
		if v == nil || *v == nil {
			return types.NewError("marshal", "gocv", "nil *image.Image", nil)
		}
		return m.writeImage(w, *v, localCfg.imageEncoding)
	case types.Tensor:
		return m.writeTensor(w, v)
	case []byte:
		// allow raw bytes pass-through for DNN weights.
		_, err := w.Write(v)
		return err
	default:
		return types.NewError("marshal", "gocv", fmt.Sprintf("unsupported value %T", value), nil)
	}
}

func (m *Marshaller) writeMat(w io.Writer, mat cv.Mat) error {
	data, err := encodeMatBytes(mat, m.cfg.imageEncoding)
	if err != nil {
		return types.NewError("marshal", "gocv", "encode mat", err)
	}
	if _, err := w.Write(data); err != nil {
		return types.NewError("marshal", "gocv", "write mat bytes", err)
	}
	return nil
}

func (m *Marshaller) writeImage(w io.Writer, img image.Image, format string) error {
	data, err := encodeImageBytes(img, format)
	if err != nil {
		return types.NewError("marshal", "gocv", "encode image", err)
	}
	if _, err := w.Write(data); err != nil {
		return types.NewError("marshal", "gocv", "write image bytes", err)
	}
	return nil
}

func (m *Marshaller) writeTensor(w io.Writer, tensor types.Tensor) error {
	accessor, ok := tensor.(tensorgocv.Accessor)
	if !ok {
		return types.NewError("marshal", "gocv", "tensor does not expose GoCV accessor", nil)
	}
	mat, err := accessor.MatClone()
	if err != nil {
		return types.NewError("marshal", "gocv", "clone mat from tensor", err)
	}
	defer mat.Close()
	return m.writeMat(w, mat)
}
