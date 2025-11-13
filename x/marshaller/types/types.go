package types

import (
	"context"
	"fmt"
	"io"

	"github.com/itohio/EasyRobot/pkg/core/math/nn/types"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	tensortypes "github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// Option configures marshaller/unmarshaller behavior.
type Option interface {
	Apply(*Options)
}

// Options holds marshaller/unmarshaller configuration.
type Options struct {
	FormatVersion   string
	Hint            string            // optional value hint (e.g. "tensor", "matrix", "model", "layer")
	Metadata        map[string]string // free-form, backend specific
	TensorFactory   func(DataType, Shape) Tensor
	DestinationType DataType // target data type for conversion during unmarshal
	Context         context.Context
}

// Marshaller encodes values to a format.
type Marshaller interface {
	Format() string
	Marshal(w io.Writer, value any, opts ...Option) error
}

// Unmarshaller decodes values from a format.
type Unmarshaller interface {
	Format() string
	Unmarshal(r io.Reader, dst any, opts ...Option) error
}

// Frame represents a collection of tensors and metadata produced by streaming
// sources such as image sequences or video captures.
type Frame struct {
	Index     int
	Timestamp int64
	Metadata  map[string]any
	Tensors   []Tensor
}

// FrameStream is a lazily produced stream of frames that can be cancelled by
// the consumer.
type FrameStream struct {
	C     <-chan Frame
	close func()
}

// NewFrameStream constructs a FrameStream from a channel and close callback.
func NewFrameStream(ch <-chan Frame, closer func()) FrameStream {
	return FrameStream{
		C:     ch,
		close: closer,
	}
}

// Close signals to the producer that the consumer no longer needs frames.
func (fs FrameStream) Close() {
	if fs.close != nil {
		fs.close()
	}
}

// Error wraps marshalling errors with context.
type Error struct {
	Op      string // operation (e.g., "marshal", "unmarshal")
	Format  string // format name (e.g., "gob", "text")
	Message string // additional context
	Err     error  // underlying error
}

func (e *Error) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("marshaller: %s(%s): %s: %v", e.Op, e.Format, e.Message, e.Err)
	}
	return fmt.Sprintf("marshaller: %s(%s): %v", e.Op, e.Format, e.Err)
}

func (e *Error) Unwrap() error {
	return e.Err
}

// NewError creates a new Error.
func NewError(op, format, message string, err error) error {
	return &Error{
		Op:      op,
		Format:  format,
		Message: message,
		Err:     err,
	}
}

// Option implementations

type withFormatVersion string

func (v withFormatVersion) Apply(opts *Options) {
	opts.FormatVersion = string(v)
}

// WithFormatVersion sets the format version.
func WithFormatVersion(version string) Option {
	return withFormatVersion(version)
}

type withHint string

func (h withHint) Apply(opts *Options) {
	opts.Hint = string(h)
}

// WithHint sets the type hint.
func WithHint(hint string) Option {
	return withHint(hint)
}

type withMetadata map[string]string

func (m withMetadata) Apply(opts *Options) {
	if opts.Metadata == nil {
		opts.Metadata = make(map[string]string)
	}
	for k, v := range m {
		opts.Metadata[k] = v
	}
}

// WithMetadata adds metadata key-value pairs.
func WithMetadata(metadata map[string]string) Option {
	return withMetadata(metadata)
}

type withTensorFactory func(DataType, Shape) Tensor

func (f withTensorFactory) Apply(opts *Options) {
	opts.TensorFactory = f
}

// WithTensorFactory sets the tensor constructor.
func WithTensorFactory(ctor func(DataType, Shape) Tensor) Option {
	return withTensorFactory(ctor)
}

type withDestinationType DataType

func (dt withDestinationType) Apply(opts *Options) {
	opts.DestinationType = DataType(dt)
}

// WithDestinationType sets the destination data type for type conversion during unmarshal.
func WithDestinationType(dtype DataType) Option {
	return withDestinationType(dtype)
}

type withContextOption struct {
	ctx context.Context
}

func (opt withContextOption) Apply(opts *Options) {
	if opts == nil {
		return
	}
	if opt.ctx == nil {
		opts.Context = context.Background()
		return
	}
	opts.Context = opt.ctx
}

// WithContext sets a context for cancellation and deadlines.
func WithContext(ctx context.Context) Option {
	return withContextOption{ctx: ctx}
}

// Domain type aliases for convenience
type (
	Tensor    = tensor.Tensor
	Shape     = tensor.Shape
	DataType  = tensor.DataType
	Model     = types.Model
	Layer     = types.Layer
	Parameter = types.Parameter
)

// DataType constants re-exported for convenience
const (
	DT_UNKNOWN DataType = tensortypes.DT_UNKNOWN
	INT64      DataType = tensortypes.INT64
	FP64       DataType = tensortypes.FP64
	INT32      DataType = tensortypes.INT32
	FP32       DataType = tensortypes.FP32
	INT        DataType = tensortypes.INT
	INT16      DataType = tensortypes.INT16
	FP16       DataType = tensortypes.FP16
	INT8       DataType = tensortypes.INT8
	INT48      DataType = tensortypes.INT48
	UINT8      DataType = tensortypes.UINT8
)
