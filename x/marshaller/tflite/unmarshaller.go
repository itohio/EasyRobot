package tflite

import (
	"fmt"
	"io"
	"os"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// Unmarshaller implements types.Unmarshaller for TFLite models.
type Unmarshaller struct {
	opts          types.Options
	numThreads    int
	errorReporter func(string)
}

// NewUnmarshaller creates a new TFLite unmarshaller.
func NewUnmarshaller(opts ...types.Option) *Unmarshaller {
	u := &Unmarshaller{
		opts:       types.Options{},
		numThreads: 1, // Default to single thread
	}
	// Apply options
	for _, opt := range opts {
		if opt != nil {
			opt.Apply(&u.opts)
		}
	}
	return u
}

// Format returns the format name.
func (u *Unmarshaller) Format() string {
	return "tflite"
}

// WithNumThreads sets the number of threads for inference.
func (u *Unmarshaller) WithNumThreads(numThreads int) *Unmarshaller {
	u.numThreads = numThreads
	return u
}

// WithErrorReporter sets a custom error reporter.
func (u *Unmarshaller) WithErrorReporter(reporter func(string)) *Unmarshaller {
	u.errorReporter = reporter
	return u
}

// Unmarshal reads a TFLite model from an io.Reader and unmarshals it into dst.
// dst must be a pointer to a *Model or types.Model variable.
func (u *Unmarshaller) Unmarshal(r io.Reader, dst any, opts ...types.Option) error {
	// Apply additional options
	for _, opt := range opts {
		if opt != nil {
			opt.Apply(&u.opts)
		}
	}

	// Read all data from reader
	data, err := io.ReadAll(r)
	if err != nil {
		return types.NewError("unmarshal", "tflite", "failed to read model data", err)
	}

	// Load model
	model, err := u.loadModel(data, "tflite_model")
	if err != nil {
		return types.NewError("unmarshal", "tflite", "failed to load model", err)
	}

	// Assign to destination
	switch d := dst.(type) {
	case **Model:
		*d = model
	case *types.Model:
		*d = model
	default:
		model.Close()
		return types.NewError("unmarshal", "tflite",
			fmt.Sprintf("unsupported destination type %T, expected **tflite.Model or *types.Model", dst), nil)
	}

	return nil
}

// LoadFromFile loads a TFLite model from a file and wraps it as nn.types.Model.
func (u *Unmarshaller) LoadFromFile(filename string) (*Model, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read TFLite model file: %w", err)
	}
	return u.loadModel(data, filename)
}

// LoadFromReader loads a TFLite model from an io.Reader.
func (u *Unmarshaller) LoadFromReader(r io.Reader) (*Model, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("failed to read TFLite model data: %w", err)
	}
	return u.loadModel(data, "tflite_model")
}

// loadModel loads a TFLite model from bytes and wraps it as nn.types.Model.
func (u *Unmarshaller) loadModel(data []byte, name string) (*Model, error) {
	// Build interpreter options
	opts := []InterpreterOption{
		WithNumThreads(u.numThreads),
	}

	if u.errorReporter != nil {
		opts = append(opts, WithErrorReporter(u.errorReporter))
	}

	// Create TFLite model wrapper
	model, err := NewModel(data, name, opts...)
	if err != nil {
		return nil, fmt.Errorf("tflite.Unmarshaller.loadModel: %w", err)
	}

	return model, nil
}

// Loader provides a convenient API for loading TFLite models.
type Loader struct {
	unmarshaller *Unmarshaller
}

// NewLoader creates a new Loader instance.
func NewLoader(opts ...types.Option) *Loader {
	return &Loader{
		unmarshaller: NewUnmarshaller(opts...),
	}
}

// WithNumThreads configures the number of threads used for inference.
func (l *Loader) WithNumThreads(numThreads int) *Loader {
	if l == nil {
		return nil
	}
	l.unmarshaller.WithNumThreads(numThreads)
	return l
}

// WithErrorReporter sets a custom error reporter.
func (l *Loader) WithErrorReporter(reporter func(string)) *Loader {
	if l == nil {
		return nil
	}
	l.unmarshaller.WithErrorReporter(reporter)
	return l
}

// LoadFromFile loads a TFLite model from disk.
func (l *Loader) LoadFromFile(filename string) (*Model, error) {
	if l == nil || l.unmarshaller == nil {
		return nil, fmt.Errorf("tflite.Loader.LoadFromFile: loader not initialized")
	}
	return l.unmarshaller.LoadFromFile(filename)
}

// LoadFromReader loads a TFLite model from an io.Reader.
func (l *Loader) LoadFromReader(r io.Reader) (*Model, error) {
	if l == nil || l.unmarshaller == nil {
		return nil, fmt.Errorf("tflite.Loader.LoadFromReader: loader not initialized")
	}
	return l.unmarshaller.LoadFromReader(r)
}
