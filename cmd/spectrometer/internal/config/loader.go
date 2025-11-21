package config

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"log/slog"

	"github.com/itohio/EasyRobot/types/spectrometer"
	jsonUnmarshaller "github.com/itohio/EasyRobot/x/marshaller/json"
	marshallerProto "github.com/itohio/EasyRobot/x/marshaller/proto"
	yamlUnmarshaller "github.com/itohio/EasyRobot/x/marshaller/yaml"
)

// Loader loads configuration from various formats.
type Loader struct {
	outputFormat string // Override format from --output flag
}

// NewLoader creates a new config loader.
func NewLoader(outputFormat string) *Loader {
	return &Loader{
		outputFormat: strings.ToLower(outputFormat),
	}
}

// Load loads configuration from a file.
// Format is auto-detected from file extension or overridden by outputFormat.
func (l *Loader) Load(ctx context.Context, path string) (*spectrometer.SpectrometerConfiguration, error) {
	format := l.detectFormat(path)
	slog.Debug("Loading config", "path", path, "format", format)

	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open config file: %w", err)
	}
	defer file.Close()

	return l.LoadFromReader(ctx, file, format)
}

// LoadFromReader loads configuration from an io.Reader.
func (l *Loader) LoadFromReader(ctx context.Context, r io.Reader, format string) (*spectrometer.SpectrometerConfiguration, error) {
	format = strings.ToLower(format)

	var config spectrometer.SpectrometerConfiguration
	var err error

	switch format {
	case "pb", "proto", "protobuf":
		unmarshaller := marshallerProto.NewUnmarshaller()
		err = unmarshaller.Unmarshal(r, &config)
	case "json":
		unmarshaller := jsonUnmarshaller.NewUnmarshaller()
		err = unmarshaller.Unmarshal(r, &config)
	case "yaml", "yml":
		unmarshaller := yamlUnmarshaller.NewUnmarshaller()
		err = unmarshaller.Unmarshal(r, &config)
	default:
		return nil, fmt.Errorf("unsupported format: %s (supported: pb, json, yaml)", format)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	return &config, nil
}

// detectFormat detects file format from extension or uses outputFormat override.
func (l *Loader) detectFormat(path string) string {
	// Override format if specified
	if l.outputFormat != "" {
		return l.outputFormat
	}

	// Auto-detect from extension
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".pb", ".proto":
		return "pb"
	case ".json":
		return "json"
	case ".yaml", ".yml":
		return "yaml"
	case ".csv":
		return "csv"
	default:
		return "yaml" // default
	}
}
