package config

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"log/slog"

	"github.com/itohio/EasyRobot/types/math"
	"github.com/itohio/EasyRobot/types/spectrometer"
	"github.com/itohio/EasyRobot/x/math/mat"
	csvUnmarshaller "github.com/itohio/EasyRobot/x/marshaller/csv"
	jsonUnmarshaller "github.com/itohio/EasyRobot/x/marshaller/json"
	marshallerProto "github.com/itohio/EasyRobot/x/marshaller/proto"
	yamlUnmarshaller "github.com/itohio/EasyRobot/x/marshaller/yaml"
)

// ReferenceResolver resolves and loads reference spectrum paths.
type ReferenceResolver struct {
	cache map[string]mat.Matrix
	mu    sync.RWMutex
}

// NewReferenceResolver creates a new reference spectrum resolver.
func NewReferenceResolver() *ReferenceResolver {
	return &ReferenceResolver{
		cache: make(map[string]mat.Matrix),
	}
}

// Resolve resolves a single reference spectrum entry.
// Returns the spectrum matrix, either from inline data or loaded from file.
func (r *ReferenceResolver) Resolve(ctx context.Context, entry *spectrometer.ReferenceSpectrumEntry) (mat.Matrix, error) {
	if entry == nil {
		return nil, fmt.Errorf("entry cannot be nil")
	}

	// Check if entry has inline matrix
	if spectrum := entry.GetSpectrum(); spectrum != nil {
		return r.convertProtoMatrix(spectrum)
	}

	// Check if entry has path
	path := entry.GetPath()
	if path == "" {
		return nil, fmt.Errorf("entry '%s' has neither inline spectrum nor path", entry.Name)
	}

	// Check cache first
	r.mu.RLock()
	if cached, ok := r.cache[path]; ok {
		r.mu.RUnlock()
		return cached, nil
	}
	r.mu.RUnlock()

	// Load from file
	matrix, err := r.loadFromFile(ctx, path, entry.Format)
	if err != nil {
		return nil, fmt.Errorf("failed to load reference spectrum '%s' from %s: %w", entry.Name, path, err)
	}

	// Cache the loaded matrix
	r.mu.Lock()
	r.cache[path] = matrix
	r.mu.Unlock()

	return matrix, nil
}

// ResolveAll resolves all reference spectrum entries.
// Returns a map of entry names to loaded matrices.
func (r *ReferenceResolver) ResolveAll(ctx context.Context, entries []*spectrometer.ReferenceSpectrumEntry) (map[string]mat.Matrix, error) {
	result := make(map[string]mat.Matrix, len(entries))
	var errors []string

	for _, entry := range entries {
		if entry == nil {
			continue
		}

		name := entry.Name
		if name == "" {
			name = fmt.Sprintf("unnamed_%d", len(result))
		}

		matrix, err := r.Resolve(ctx, entry)
		if err != nil {
			errors = append(errors, fmt.Sprintf("%s: %v", name, err))
			continue
		}

		result[name] = matrix
	}

	if len(errors) > 0 {
		return result, fmt.Errorf("failed to resolve some reference spectrums: %s", strings.Join(errors, "; "))
	}

	return result, nil
}

// loadFromFile loads a matrix from a file using the appropriate marshaller.
func (r *ReferenceResolver) loadFromFile(ctx context.Context, path string, format string) (mat.Matrix, error) {
	// Resolve path (absolute or relative)
	resolvedPath := path
	if !filepath.IsAbs(path) {
		// For relative paths, resolve relative to current working directory
		var err error
		resolvedPath, err = filepath.Abs(path)
		if err != nil {
			return nil, fmt.Errorf("failed to resolve path: %w", err)
		}
	}

	// Check if file exists
	if _, err := os.Stat(resolvedPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("file does not exist: %s", resolvedPath)
	}

	// Detect format from extension if not provided
	if format == "" {
		format = r.detectFormat(resolvedPath)
	}
	format = strings.ToLower(format)

	// Open file
	file, err := os.Open(resolvedPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	slog.Debug("Loading reference spectrum", "path", resolvedPath, "format", format)

	// Load matrix based on format
	var matrix mat.Matrix
	switch format {
	case "csv":
		unmarshaller := csvUnmarshaller.NewUnmarshaller()
		err = unmarshaller.Unmarshal(file, &matrix)
	case "json":
		unmarshaller := jsonUnmarshaller.NewUnmarshaller()
		err = unmarshaller.Unmarshal(file, &matrix)
	case "pb", "proto", "protobuf":
		unmarshaller := marshallerProto.NewUnmarshaller()
		var protoMatrix math.Matrix
		err = unmarshaller.Unmarshal(file, &protoMatrix)
		if err == nil {
			matrix, err = r.convertProtoMatrix(&protoMatrix)
		}
	case "yaml", "yml":
		unmarshaller := yamlUnmarshaller.NewUnmarshaller()
		var protoMatrix math.Matrix
		err = unmarshaller.Unmarshal(file, &protoMatrix)
		if err == nil {
			matrix, err = r.convertProtoMatrix(&protoMatrix)
		}
	default:
		return nil, fmt.Errorf("unsupported format: %s (supported: csv, json, pb, yaml)", format)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal matrix: %w", err)
	}

	if matrix == nil {
		return nil, fmt.Errorf("loaded matrix is nil")
	}

	return matrix, nil
}

// convertProtoMatrix converts a protobuf Matrix to mat.Matrix.
func (r *ReferenceResolver) convertProtoMatrix(pbMatrix *math.Matrix) (mat.Matrix, error) {
	if pbMatrix == nil {
		return nil, fmt.Errorf("protobuf matrix is nil")
	}

	rows := int(pbMatrix.Rows)
	cols := int(pbMatrix.Cols)
	data := pbMatrix.Data

	if rows <= 0 || cols <= 0 {
		return nil, fmt.Errorf("invalid matrix dimensions: %dx%d", rows, cols)
	}

	if len(data) != rows*cols {
		return nil, fmt.Errorf("matrix data length (%d) does not match dimensions (%dx%d = %d)", len(data), rows, cols, rows*cols)
	}

	// Create matrix with backing data
	matrix := mat.New(rows, cols, data...)
	return matrix, nil
}

// detectFormat detects file format from extension.
func (r *ReferenceResolver) detectFormat(path string) string {
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
		return "csv" // Default to CSV for spectrum files
	}
}

