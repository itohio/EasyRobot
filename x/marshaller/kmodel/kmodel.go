// Package kmodel marshals int8 networks into Kendryte kmodel v3 files for
// the K210's KPU (drivers/kpu), in pure Go.
//
// The Marshaller compiles a qnet graph, or an int8 TFLite flatbuffer read
// with qnet/tfl (the pure-Go TFLite reader; x/marshaller/tflite is the cgo
// runtime wrapper), through drivers/kpu/compile into one kmodel. Options
// resize the input (convolutional graphs are size-agnostic) and cut the
// graph after a layer. The Unmarshaller parses and checks a kmodel.
//
// Layers the KPU cannot run are refused with an error naming the first one;
// the kmodel's input and output conventions are in compile.Info.
package kmodel

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"

	"github.com/itohio/EasyRobot/drivers/kpu"
	"github.com/itohio/EasyRobot/drivers/kpu/compile"
	km "github.com/itohio/EasyRobot/drivers/kpu/kmodel"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/nn/qnet/tfl"
)

const format = "kmodel"

// Metadata keys of the options below (types.Options.Metadata).
const (
	keyInput = "kmodel.input" // "HxW"
	keyCut   = "kmodel.cut"   // last layer index
)

// WithInputSize resizes the graph's input to h×w before compiling.
func WithInputSize(h, w int) types.Option {
	return types.WithMetadata(map[string]string{keyInput: fmt.Sprintf("%dx%d", h, w)})
}

// WithCut keeps the graph's layers 0..last (qnet layer indices; see
// Layers) and compiles them.
func WithCut(last int) types.Option {
	return types.WithMetadata(map[string]string{keyCut: strconv.Itoa(last)})
}

// Marshaller writes kmodel files.
type Marshaller struct {
	opts types.Options
	// Info is the description of the last kmodel written.
	Info compile.Info
}

// NewMarshaller returns a kmodel Marshaller.
func NewMarshaller(opts ...types.Option) *Marshaller {
	m := &Marshaller{}
	apply(&m.opts, opts)
	return m
}

func apply(o *types.Options, opts []types.Option) {
	for _, opt := range opts {
		opt.Apply(o) // WithMetadata merges into Metadata
	}
}

// Format returns "kmodel".
func (m *Marshaller) Format() string { return format }

// Marshal compiles value into a kmodel and writes it to w. value is a
// *qnet.Graph or qnet.Graph, or an int8 TFLite flatbuffer ([]byte) that
// qnet/tfl reads.
func (m *Marshaller) Marshal(w io.Writer, value any, opts ...types.Option) error {
	o := m.opts
	if o.Metadata != nil {
		meta := make(map[string]string, len(o.Metadata))
		for k, v := range o.Metadata {
			meta[k] = v
		}
		o.Metadata = meta
	}
	apply(&o, opts)
	g, err := graphOf(value)
	if err != nil {
		return types.NewError("marshal", format, "input", err)
	}
	if g, err = Prepare(g, o.Metadata); err != nil {
		return types.NewError("marshal", format, "prepare", err)
	}
	data, info, err := compile.Kmodel(g)
	if err != nil {
		return types.NewError("marshal", format, "compile", err)
	}
	m.Info = info
	_, err = w.Write(data)
	return err
}

func graphOf(value any) (*qnet.Graph, error) {
	switch v := value.(type) {
	case *qnet.Graph:
		return v, nil
	case qnet.Graph:
		return &v, nil
	case []byte:
		g, err := tfl.Load(v)
		if err != nil {
			return nil, err
		}
		return &g, nil
	case nil:
		return nil, errors.New("nil value")
	}
	return nil, fmt.Errorf("unsupported value %T (want *qnet.Graph, qnet.Graph or TFLite []byte)", value)
}

// Prepare applies the cut (kmodel.cut) and then the input size
// (kmodel.input, "HxW") from meta to g.
func Prepare(g *qnet.Graph, meta map[string]string) (*qnet.Graph, error) {
	if s, ok := meta[keyCut]; ok {
		last, err := strconv.Atoi(s)
		if err != nil {
			return nil, fmt.Errorf("cut %q: %w", s, err)
		}
		if g, err = compile.Cut(g, last); err != nil {
			return nil, err
		}
	}
	if s, ok := meta[keyInput]; ok {
		hs, ws, found := strings.Cut(s, "x")
		h, err1 := strconv.Atoi(hs)
		w, err2 := strconv.Atoi(ws)
		if !found || err1 != nil || err2 != nil || h <= 0 || w <= 0 {
			return nil, fmt.Errorf("input size %q (want HxW)", s)
		}
		var err error
		if g, err = compile.Resize(g, h, w); err != nil {
			return nil, err
		}
	}
	return g, nil
}

// Unmarshaller reads kmodel files.
type Unmarshaller struct{}

// NewUnmarshaller returns a kmodel Unmarshaller.
func NewUnmarshaller(...types.Option) *Unmarshaller { return &Unmarshaller{} }

// Format returns "kmodel".
func (u *Unmarshaller) Format() string { return format }

// Unmarshal reads a kmodel into dst (*km.Model): the data is copied into a
// 128-byte-aligned buffer (as kpu.New needs), parsed and checked.
func (u *Unmarshaller) Unmarshal(r io.Reader, dst any, _ ...types.Option) error {
	out, ok := dst.(*km.Model)
	if !ok {
		return types.NewError("unmarshal", format, "destination", fmt.Errorf("want *kmodel.Model, got %T", dst))
	}
	var buf bytes.Buffer
	if _, err := io.Copy(&buf, r); err != nil {
		return types.NewError("unmarshal", format, "read", err)
	}
	m, err := km.Parse(kpu.AlignedCopy(buf.Bytes()))
	if err != nil {
		return types.NewError("unmarshal", format, "parse", err)
	}
	if err := m.Check(); err != nil {
		return types.NewError("unmarshal", format, "check", err)
	}
	*out = *m
	return nil
}

var (
	_ types.Marshaller   = (*Marshaller)(nil)
	_ types.Unmarshaller = (*Unmarshaller)(nil)
)
