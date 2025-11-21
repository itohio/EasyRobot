package measure

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"log/slog"

	"github.com/itohio/EasyRobot/cmd/spectrometer/internal/render"
	"github.com/itohio/EasyRobot/types/spectrometer"
	jsonUnmarshaller "github.com/itohio/EasyRobot/x/marshaller/json"
	marshallerProto "github.com/itohio/EasyRobot/x/marshaller/proto"
	yamlUnmarshaller "github.com/itohio/EasyRobot/x/marshaller/yaml"
	"github.com/itohio/EasyRobot/x/math/colorscience"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// LoadPatches loads color patches from a file (YAML/JSON/Proto).
// Format is auto-detected from file extension.
func LoadPatches(ctx context.Context, path string) (*spectrometer.ColorPatches, error) {
	format := detectPatchFormat(path)
	slog.Debug("Loading patches", "path", path, "format", format)

	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open patch file: %w", err)
	}
	defer file.Close()

	var patches spectrometer.ColorPatches
	switch format {
	case "pb", "proto", "protobuf":
		unmarshaller := marshallerProto.NewUnmarshaller()
		if err := unmarshaller.Unmarshal(file, &patches); err != nil {
			return nil, fmt.Errorf("failed to unmarshal proto patches: %w", err)
		}
	case "json":
		unmarshaller := jsonUnmarshaller.NewUnmarshaller()
		if err := unmarshaller.Unmarshal(file, &patches); err != nil {
			return nil, fmt.Errorf("failed to unmarshal JSON patches: %w", err)
		}
	case "yaml", "yml":
		unmarshaller := yamlUnmarshaller.NewUnmarshaller()
		if err := unmarshaller.Unmarshal(file, &patches); err != nil {
			return nil, fmt.Errorf("failed to unmarshal YAML patches: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported patch format: %s (supported: pb, json, yaml)", format)
	}

	return &patches, nil
}

// detectPatchFormat detects file format from extension.
func detectPatchFormat(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".pb", ".proto":
		return "pb"
	case ".json":
		return "json"
	case ".yaml", ".yml":
		return "yaml"
	default:
		return "yaml" // default
	}
}

// ConvertProtoPatchToRenderPatch converts a proto ColorPatch to render.Patch.
func ConvertProtoPatchToRenderPatch(protoPatch *spectrometer.ColorPatch) (render.Patch, error) {
	patch := render.Patch{
		Name: protoPatch.GetName(),
	}

	// Convert XYZ
	if protoPatch.GetXyz() != nil {
		xyzData := protoPatch.GetXyz().GetData()
		if len(xyzData) != 3 {
			return patch, fmt.Errorf("XYZ vector must have 3 elements, got %d", len(xyzData))
		}
		patch.ReferenceXYZ = colorscience.XYZ(vec.Vector3D{xyzData[0], xyzData[1], xyzData[2]})
	}

	// Convert LAB
	if protoPatch.GetLab() != nil {
		labData := protoPatch.GetLab().GetData()
		if len(labData) != 3 {
			return patch, fmt.Errorf("LAB vector must have 3 elements, got %d", len(labData))
		}
		patch.Reference = colorscience.LAB(vec.Vector3D{labData[0], labData[1], labData[2]})
	}

	// Convert spectrum if provided
	if protoPatch.GetSpectrum() != nil {
		protoMatrix := protoPatch.GetSpectrum()
		rows := int(protoMatrix.GetRows())
		cols := int(protoMatrix.GetCols())
		data := protoMatrix.GetData()

		if rows != 2 {
			return patch, fmt.Errorf("spectrum matrix must have 2 rows (wavelengths, intensities), got %d", rows)
		}

		// Extract wavelengths (row 0) and values (row 1) from row-major flattened data
		wavelengths := vec.New(cols)
		values := vec.New(cols)

		for i := 0; i < cols; i++ {
			wavelengths[i] = data[i] // Row 0
			values[i] = data[cols+i] // Row 1
		}

		// Create SPD from spectrum
		spd := colorscience.NewSPD(wavelengths, values)
		// Note: We don't store the reference spectrum in render.Patch currently
		// If needed, we could add a ReferenceSPD field to render.Patch
		_ = spd
	}

	return patch, nil
}

// ConvertMeasuredSPDToRenderPatch converts a measured SPD to render.Patch with colorimetry.
// Computes XYZ, LAB, RGB from the measured spectrum and calculates Delta E against reference.
func ConvertMeasuredSPDToRenderPatch(measuredSPD colorscience.SPD, refPatch render.Patch, whitePoint colorscience.WhitePoint) (render.Patch, error) {
	// Compute colorimetry from measured spectrum
	cs, err := colorscience.New()
	if err != nil {
		return render.Patch{}, fmt.Errorf("failed to create colorscience instance: %w", err)
	}
	measuredXYZ, err := cs.ComputeXYZ(measuredSPD.Matrix)
	if err != nil {
		return render.Patch{}, fmt.Errorf("failed to compute XYZ from measured spectrum: %w", err)
	}

	measuredLAB := measuredXYZ.ToLAB(whitePoint)
	measuredRGB := measuredXYZ.ToRGB(true)

	// Create measured patch
	measuredPatch := render.Patch{
		Name:         refPatch.Name,
		Reference:    refPatch.Reference,
		Measured:     measuredLAB,
		ReferenceXYZ: refPatch.ReferenceXYZ,
		MeasuredXYZ:  measuredXYZ,
		RGB:          measuredRGB,
	}

	// Calculate Delta E
	measuredPatch.CalculateDeltaE()

	return measuredPatch, nil
}
