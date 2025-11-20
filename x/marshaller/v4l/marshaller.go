// go:build linux
package v4l

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// Marshaller implements the marshaller interface for V4L devices
type Marshaller struct {
	opts types.Options
	cfg  Options
}

// NewMarshaller creates a new V4L marshaller
func NewMarshaller(opts ...types.Option) *Marshaller {
	baseOpts, baseCfg := applyOptions(types.Options{}, Options{}, opts...)
	return &Marshaller{
		opts: baseOpts,
		cfg:  baseCfg,
	}
}

// Format returns the marshaller format identifier
func (m *Marshaller) Format() string {
	return "v4l"
}

// Marshal handles marshalling of V4L-related types
func (m *Marshaller) Marshal(w io.Writer, value any, opts ...types.Option) error {
	localOpts, localCfg := applyOptions(m.opts, m.cfg, opts...)

	switch v := value.(type) {
	case DeviceInfo:
		return m.marshalDeviceInfo(w, v)
	case []DeviceInfo:
		return m.marshalDeviceInfoList(w, v)
	case *Stream:
		return m.marshalStreamConfig(w, *v, localCfg)
	case Stream:
		return m.marshalStreamConfig(w, v, localCfg)
	case *MultiStream:
		return m.marshalMultiStreamConfig(w, *v, localCfg)
	case MultiStream:
		return m.marshalMultiStreamConfig(w, v, localCfg)
	default:
		return types.NewError("marshal", "v4l", fmt.Sprintf("unsupported type %T", value), nil)
	}
}

// marshalDeviceInfo marshals device information
func (m *Marshaller) marshalDeviceInfo(w io.Writer, info DeviceInfo) error {
	data, err := json.MarshalIndent(info, "", "  ")
	if err != nil {
		return types.NewError("marshal", "v4l", "failed to marshal device info", err)
	}
	_, err = w.Write(data)
	return err
}

// marshalDeviceInfoList marshals a list of device information
func (m *Marshaller) marshalDeviceInfoList(w io.Writer, infos []DeviceInfo) error {
	data, err := json.MarshalIndent(infos, "", "  ")
	if err != nil {
		return types.NewError("marshal", "v4l", "failed to marshal device info list", err)
	}
	_, err = w.Write(data)
	return err
}

// marshalStreamConfig marshals stream configuration
func (m *Marshaller) marshalStreamConfig(w io.Writer, stream Stream, cfg Options) error {
	config := map[string]any{
		"type": "stream",
		"device": func() string {
			if v4lStream, ok := stream.(*v4lStream); ok {
				return v4lStream.device.Path()
			}
			return "unknown"
		}(),
		"format":  stream.Format(),
		"options": cfg,
	}

	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return types.NewError("marshal", "v4l", "failed to marshal stream config", err)
	}
	_, err = w.Write(data)
	return err
}

// marshalMultiStreamConfig marshals multi-stream configuration
func (m *Marshaller) marshalMultiStreamConfig(w io.Writer, multiStream MultiStream, cfg Options) error {
	devices := make([]string, len(multiStream.streams))
	for i, stream := range multiStream.streams {
		if v4lStream, ok := stream.(*v4lStream); ok {
			devices[i] = v4lStream.device.Path()
		} else {
			devices[i] = "unknown"
		}
	}

	config := map[string]any{
		"type":    "multi_stream",
		"devices": devices,
		"options": cfg,
	}

	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return types.NewError("marshal", "v4l", "failed to marshal multi-stream config", err)
	}
	_, err = w.Write(data)
	return err
}
