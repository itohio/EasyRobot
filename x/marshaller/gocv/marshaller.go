package gocv

import (
	"context"
	"errors"
	"fmt"
	"image"
	"io"

	cv "gocv.io/x/gocv"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor"
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
		return m.writeImage(w, v, localCfg.codec.imageEncoding)
	case *image.Image:
		if v == nil || *v == nil {
			return types.NewError("marshal", "gocv", "nil *image.Image", nil)
		}
		return m.writeImage(w, *v, localCfg.codec.imageEncoding)
	case types.Tensor:
		return m.writeTensor(w, v)
	case types.FrameStream:
		return m.writeFrameStream(w, v, localCfg)
	case *types.FrameStream:
		if v == nil {
			return types.NewError("marshal", "gocv", "nil *FrameStream", nil)
		}
		return m.writeFrameStream(w, *v, localCfg)
	case []byte:
		// allow raw bytes pass-through for DNN weights.
		_, err := w.Write(v)
		return err
	default:
		return types.NewError("marshal", "gocv", fmt.Sprintf("unsupported value %T", value), nil)
	}
}

func (m *Marshaller) writeMat(w io.Writer, mat cv.Mat) error {
	data, err := encodeMatBytes(mat, m.cfg.codec.imageEncoding)
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

func (m *Marshaller) writeFrameStream(w io.Writer, stream types.FrameStream, cfg config) error {
	// Write manifest if writer is provided (for serialization)
	// If w is nil, this is a side-effect operation (file/display only)
	if w != nil {
		manifest := configToManifest(cfg)
		if err := writeManifest(w, manifest); err != nil {
			return err
		}
	}

	targets, err := resolveOutputDirs(cfg)
	if err != nil {
		return err
	}

	defer stream.Close()

	frames := stream.C

	var (
		writers    []frameWriter
		fileWriter *fileWriter
	)

	if len(targets) > 0 {
		fw, err := newFileWriter(targets, cfg)
		if err != nil {
			return err
		}
		fileWriter = fw
		writers = append(writers, fw)
	}

	if cfg.display.enabled {
		dw, err := newDisplayWriter(cfg)
		if err != nil {
			return err
		}
		writers = append(writers, dw)
	}

	var (
		stop    bool
		stepErr error
	)

	loop := cfg.display.eventLoop
	if loop == nil {
		loop = defaultEventLoop
	}
	ctx := cfg.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	// Store cancel function in context if available
	var cancelFn context.CancelFunc
	if cancel, ok := ctx.Value("cancel").(context.CancelFunc); ok {
		cancelFn = cancel
		ctx = context.WithValue(ctx, "cancel", cancel)
	}

	step := func() bool {
		if stop {
			return false
		}
		
		// Check context cancellation before reading frame
		select {
		case <-ctx.Done():
			stop = true
			return false
		default:
		}
		
		frame, ok := <-frames
		if !ok {
			return false
		}

		// Use smart tensors with reference counting for fan-out to multiple writers
		// Wrap each tensor with reference counting and create views for additional writers
		numWriters := len(writers)
		if numWriters > 1 && len(frame.Tensors) > 0 {
			// Wrap all tensors with reference counting
			smartTensors := make([]*tensor.SmartTensor, len(frame.Tensors))
			for i, t := range frame.Tensors {
				smartTensors[i] = tensor.WithRefcount(t)
			}
			
			// Create views for each writer (N-1 views needed)
			viewFrames := make([]types.Frame, numWriters-1)
			for i := 0; i < numWriters-1; i++ {
				viewTensors := make([]types.Tensor, len(smartTensors))
				for j, st := range smartTensors {
					viewTensors[j] = st.View()
				}
				viewFrames[i] = types.Frame{
					Index:     frame.Index,
					Timestamp: frame.Timestamp,
					Metadata:  frame.Metadata,
					Tensors:   viewTensors,
				}
			}
			
			// Create frame with smart tensors for first writer
			smartFrame := types.Frame{
				Index:     frame.Index,
				Timestamp: frame.Timestamp,
				Metadata:  frame.Metadata,
				Tensors:   make([]types.Tensor, len(smartTensors)),
			}
			for i, st := range smartTensors {
				smartFrame.Tensors[i] = st
			}
			
			// Send to first writer with smart tensors
			if numWriters > 0 {
				if stop {
					return false
				}
				// Check context cancellation
				select {
				case <-ctx.Done():
					stop = true
					return false
				default:
				}
				if err := writers[0].WriteFrame(smartFrame); err != nil {
					if errors.Is(err, errStopLoop) {
						stop = true
						if cancelFn != nil {
							cancelFn()
						}
					} else {
						stepErr = err
						stop = true
					}
				}
			}
			
			// Send views to remaining writers
			for i := 1; i < numWriters; i++ {
				if stop {
					break
				}
				// Check context cancellation between writers
				select {
				case <-ctx.Done():
					stop = true
					break
				default:
				}
				if err := writers[i].WriteFrame(viewFrames[i-1]); err != nil {
					if errors.Is(err, errStopLoop) {
						stop = true
						if cancelFn != nil {
							cancelFn()
						}
					} else {
						stepErr = err
						stop = true
					}
				}
			}
		} else {
			// Single writer or no tensors - send frame as-is
			for _, writer := range writers {
				if stop {
					break
				}
				// Check context cancellation between writers
				select {
				case <-ctx.Done():
					stop = true
					break
				default:
				}
				if err := writer.WriteFrame(frame); err != nil {
					if errors.Is(err, errStopLoop) {
						stop = true
						// Cancel context to propagate stop signal
						if cancelFn != nil {
							cancelFn()
						}
					} else {
						stepErr = err
						stop = true
					}
				}
			}
			
			// Release tensors if AutoRelease is enabled or single writer
			// (smart tensors handle it automatically for multiple writers when AutoRelease is false)
			if localCfg.AutoRelease || numWriters == 1 {
				for _, t := range frame.Tensors {
					t.Release()
				}
			}
		}

		return !stop
	}

	loop(ctx, step)

	for _, writer := range writers {
		_ = writer.Close()
	}

	if stepErr != nil {
		return stepErr
	}

	if fileWriter != nil && w != nil {
		if summary := fileWriter.Summary(); len(summary) > 0 {
			if _, err := w.Write(summary); err != nil {
				return types.NewError("marshal", "gocv", "write summary", err)
			}
		}
	}

	return nil
}

func tensorToMat(tensor types.Tensor) (cv.Mat, error) {
	accessor, ok := tensor.(tensorgocv.Accessor)
	if !ok {
		return cv.Mat{}, fmt.Errorf("gocv: tensor does not expose GoCV accessor")
	}
	mat, err := accessor.MatClone()
	if err != nil {
		return cv.Mat{}, err
	}
	return mat, nil
}

func defaultEventLoop(ctx context.Context, step func() bool) {
	for {
		select {
		case <-ctx.Done():
			return
		default:
		}
		if !step() {
			return
		}
	}
}
