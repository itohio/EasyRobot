package gocv

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"

	corepb "github.com/itohio/EasyRobot/types/core"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	cv "gocv.io/x/gocv"
	"google.golang.org/protobuf/proto"
)

// StreamSink is an interface for writing frames to various destinations.
// It separates serialization from consumption, allowing frames to be written
// to protobuf streams, files, displays, or composed sinks.
type StreamSink interface {
	WriteFrame(frame types.Frame) error
	Close() error
}

// ProtobufSink serializes frames to a protobuf stream using types.core.Frame.
type ProtobufSink struct {
	writer io.Writer
}

// NewProtobufSink creates a new ProtobufSink that writes to the given writer.
func NewProtobufSink(w io.Writer) *ProtobufSink {
	return &ProtobufSink{writer: w}
}

// WriteFrame serializes a frame to protobuf and writes it to the stream.
// Each frame is written as a length-prefixed protobuf message.
func (s *ProtobufSink) WriteFrame(frame types.Frame) error {
	pbFrame := frameToProto(frame)

	data, err := proto.Marshal(pbFrame)
	if err != nil {
		return types.NewError("marshal", "gocv", "protobuf encode frame", err)
	}

	// Write length-prefixed message (4-byte little-endian length, then data)
	length := uint32(len(data))
	if err := binary.Write(s.writer, binary.LittleEndian, length); err != nil {
		return types.NewError("marshal", "gocv", "write frame length", err)
	}

	if _, err := s.writer.Write(data); err != nil {
		return types.NewError("marshal", "gocv", "write frame data", err)
	}

	return nil
}

// Close closes the sink. For ProtobufSink, this is a no-op.
func (s *ProtobufSink) Close() error {
	return nil
}

// DirectorySink writes frames to disk as GoCV-encoded images.
// This wraps the existing fileWriter logic.
type DirectorySink struct {
	writer *fileWriter
}

// NewDirectorySink creates a new DirectorySink that writes frames to the given directories.
func NewDirectorySink(targets []string, cfg config) (*DirectorySink, error) {
	fw, err := newFileWriter(targets, cfg)
	if err != nil {
		return nil, err
	}
	return &DirectorySink{writer: fw}, nil
}

// WriteFrame writes a frame to disk as GoCV-encoded images.
func (s *DirectorySink) WriteFrame(frame types.Frame) error {
	return s.writer.Write(frame)
}

// Close closes the sink and returns the summary of written files.
func (s *DirectorySink) Close() error {
	return s.writer.Close()
}

// Summary returns the summary of written file paths.
func (s *DirectorySink) Summary() []byte {
	return s.writer.Summary()
}

// DisplaySink displays frames in a GoCV window.
// This wraps the existing displayWriter logic.
type DisplaySink struct {
	writer *displayWriter
}

// NewDisplaySink creates a new DisplaySink with the given display configuration.
func NewDisplaySink(cfg config) (*DisplaySink, error) {
	dw, err := newDisplayWriter(cfg)
	if err != nil {
		return nil, err
	}
	// Type assertion is safe because newDisplayWriter returns *displayWriter
	if dwWriter, ok := dw.(*displayWriter); ok {
		return &DisplaySink{writer: dwWriter}, nil
	}
	return nil, fmt.Errorf("gocv: unexpected display writer type")
}

// NewDisplaySinkFromOptions creates a new DisplaySink from marshaller options.
// This is a convenience function for creating a DisplaySink without needing direct access to config.
// Additional options (e.g., WithCancel, WithClose) can be passed to configure cancel behavior.
func NewDisplaySinkFromOptions(ctx context.Context, title string, width, height int, opts ...types.Option) (*DisplaySink, error) {
	allOpts := []types.Option{
		WithDisplay(ctx),
		WithTitle(title),
	}
	if width > 0 && height > 0 {
		allOpts = append(allOpts, WithWindowSize(width, height))
	}
	// Append any additional options (e.g., WithCancel, WithOnClose)
	allOpts = append(allOpts, opts...)

	// Apply options to get config
	_, cfg := applyOptions(types.Options{}, defaultConfig(), allOpts)
	return NewDisplaySink(cfg)
}

// WriteFrame displays a frame in the GoCV window.
func (s *DisplaySink) WriteFrame(frame types.Frame) error {
	return s.writer.Write(frame)
}

// RunEventLoop runs a continuous event loop that keeps the window responsive
// even when no frames are being written. This should be called in a goroutine.
// The loop will exit when ctx is cancelled or the window is closed.
func (s *DisplaySink) RunEventLoop(ctx context.Context) {
	s.writer.RunEventLoop(ctx)
}

// Window returns the underlying gocv window. This allows external code
// to access the window for additional operations if needed.
func (s *DisplaySink) Window() *cv.Window {
	return s.writer.Window()
}

// Close closes the display window.
func (s *DisplaySink) Close() error {
	return s.writer.Close()
}

// MultiSink composes multiple sinks and writes to all of them.
type MultiSink struct {
	sinks []StreamSink
}

// NewMultiSink creates a new MultiSink that writes to all provided sinks.
func NewMultiSink(sinks ...StreamSink) *MultiSink {
	return &MultiSink{sinks: sinks}
}

// WriteFrame writes the frame to all composed sinks.
// If any sink returns an error, WriteFrame stops and returns that error.
func (s *MultiSink) WriteFrame(frame types.Frame) error {
	for _, sink := range s.sinks {
		if err := sink.WriteFrame(frame); err != nil {
			return err
		}
	}
	return nil
}

// Close closes all composed sinks.
// Errors from individual sinks are collected but only the last error is returned.
func (s *MultiSink) Close() error {
	var lastErr error
	for _, sink := range s.sinks {
		if err := sink.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}

// frameToProto converts a types.Frame to a protobuf core.Frame.
// Note: Per GoCV-first policy, tensor pixel data is not embedded in protobuf.
// Only metadata is serialized. Actual frame pixels remain in GoCV-managed formats.
func frameToProto(frame types.Frame) *corepb.Frame {
	pbFrame := &corepb.Frame{
		Index:     int64(frame.Index),
		Timestamp: frame.Timestamp,
		Metadata:  make(map[string]string),
		Tensors:   nil, // Tensors not serialized per GoCV-first policy
	}

	// Convert metadata (map[string]any -> map[string]string)
	for k, v := range frame.Metadata {
		if str, ok := v.(string); ok {
			pbFrame.Metadata[k] = str
		} else {
			// Convert non-string values to string representation
			pbFrame.Metadata[k] = fmt.Sprintf("%v", v)
		}
	}

	// Note: Per GoCV-first policy, tensor pixel data is not serialized in protobuf.
	// Frames are serialized with metadata only. Actual pixels are handled separately
	// via GoCV image encoding when writing to files/displays.

	return pbFrame
}
