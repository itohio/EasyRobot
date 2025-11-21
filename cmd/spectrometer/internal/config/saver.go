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
	jsonMarshaller "github.com/itohio/EasyRobot/x/marshaller/json"
	marshallerProto "github.com/itohio/EasyRobot/x/marshaller/proto"
	yamlMarshaller "github.com/itohio/EasyRobot/x/marshaller/yaml"
)

// Saver saves configuration to various formats.
type Saver struct {
	outputFormat string // Override format from --output flag
}

// NewSaver creates a new config saver.
func NewSaver(outputFormat string) *Saver {
	return &Saver{
		outputFormat: strings.ToLower(outputFormat),
	}
}

// Save saves configuration to a file.
// Format is auto-detected from file extension or overridden by outputFormat.
func (s *Saver) Save(ctx context.Context, path string, config *spectrometer.SpectrometerConfiguration) error {
	format := s.detectFormat(path)
	slog.Debug("Saving config", "path", path, "format", format)

	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create config file: %w", err)
	}
	defer file.Close()

	return s.SaveToWriter(ctx, file, format, config)
}

// SaveToWriter saves configuration to an io.Writer.
func (s *Saver) SaveToWriter(ctx context.Context, w io.Writer, format string, config *spectrometer.SpectrometerConfiguration) error {
	if config == nil {
		return fmt.Errorf("config cannot be nil")
	}

	format = strings.ToLower(format)

	var err error

	switch format {
	case "pb", "proto", "protobuf":
		marshaller := marshallerProto.NewMarshaller()
		err = marshaller.Marshal(w, config)
	case "json":
		marshaller := jsonMarshaller.NewMarshaller()
		err = marshaller.Marshal(w, config)
	case "yaml", "yml":
		marshaller := yamlMarshaller.NewMarshaller()
		err = marshaller.Marshal(w, config)
	default:
		return fmt.Errorf("unsupported format: %s (supported: pb, json, yaml)", format)
	}

	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	return nil
}

// detectFormat detects file format from extension or uses outputFormat override.
func (s *Saver) detectFormat(path string) string {
	// Override format if specified
	if s.outputFormat != "" {
		return s.outputFormat
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
