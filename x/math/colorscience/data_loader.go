package colorscience

import (
	"embed"
	"fmt"
	"strings"

	"github.com/itohio/EasyRobot/x/math/mat"
)

//go:embed data/*.csv
var dataFiles embed.FS

// ObserverType represents a standard observer type.
type ObserverType string

const (
	Observer2Deg  ObserverType = "2"  // CIE 1931 2 Degree Standard Observer
	Observer10Deg ObserverType = "10" // CIE 1964 10 Degree Standard Observer
)

// DataLoader loads and manages CIE observer and illuminant spectral datasets.
type DataLoader struct {
	observers   map[ObserverType]ObserverCMF
	illuminants map[string]SPD
}

// NewDataLoader creates a new DataLoader and loads embedded SPD data.
func NewDataLoader() (*DataLoader, error) {
	dl := &DataLoader{
		observers:   make(map[ObserverType]ObserverCMF),
		illuminants: make(map[string]SPD),
	}

	if err := dl.loadObservers(); err != nil {
		return nil, fmt.Errorf("failed to load observers: %w", err)
	}

	if err := dl.loadIlluminants(); err != nil {
		return nil, fmt.Errorf("failed to load illuminants: %w", err)
	}

	return dl, nil
}

// loadObservers loads observer CMF data from embedded files.
func (dl *DataLoader) loadObservers() error {
	// Load CIE 1931 2 Degree Standard Observer
	data2, err := dataFiles.ReadFile("data/CIE_xyz_1931_2deg.csv")
	if err != nil {
		return fmt.Errorf("failed to read CIE_xyz_1931_2deg.csv: %w", err)
	}

	wavelengths2, values2, err := parseCSV(string(data2))
	if err != nil {
		return fmt.Errorf("failed to parse CIE_xyz_1931_2deg.csv: %w", err)
	}

	if len(values2) < 3 {
		return fmt.Errorf("CIE_xyz_1931_2deg.csv must have at least 3 value columns")
	}

	// Create 4-row matrix: row0=wavelengths, row1=XBar, row2=YBar, row3=ZBar
	cmf2 := mat.New(4, wavelengths2.Len())
	cmf2.SetRow(0, wavelengths2)
	cmf2.SetRow(1, values2[0])
	cmf2.SetRow(2, values2[1])
	cmf2.SetRow(3, values2[2])

	dl.observers[Observer2Deg] = ObserverCMF{Matrix: cmf2}

	// Load CIE 1964 10 Degree Standard Observer
	data10, err := dataFiles.ReadFile("data/CIE_xyz_1964_10deg.csv")
	if err != nil {
		return fmt.Errorf("failed to read CIE_xyz_1964_10deg.csv: %w", err)
	}

	wavelengths10, values10, err := parseCSV(string(data10))
	if err != nil {
		return fmt.Errorf("failed to parse CIE_xyz_1964_10deg.csv: %w", err)
	}

	if len(values10) < 3 {
		return fmt.Errorf("CIE_xyz_1964_10deg.csv must have at least 3 value columns")
	}

	// Create 4-row matrix: row0=wavelengths, row1=XBar, row2=YBar, row3=ZBar
	cmf10 := mat.New(4, wavelengths10.Len())
	cmf10.SetRow(0, wavelengths10)
	cmf10.SetRow(1, values10[0])
	cmf10.SetRow(2, values10[1])
	cmf10.SetRow(3, values10[2])

	dl.observers[Observer10Deg] = ObserverCMF{Matrix: cmf10}

	return nil
}

// loadIlluminants loads illuminant SPD data from embedded files.
func (dl *DataLoader) loadIlluminants() error {
	illuminantFiles := []string{
		"data/CIE_std_illum_A_1nm.csv",
		"data/CIE_std_illum_D50.csv",
		"data/CIE_std_illum_D65.csv",
	}

	illuminantNames := []string{
		"A",
		"D50",
		"D65",
	}

	for i, filename := range illuminantFiles {
		data, err := dataFiles.ReadFile(filename)
		if err != nil {
			return fmt.Errorf("failed to read %s: %w", filename, err)
		}

		wavelengths, values, err := parseCSV(string(data))
		if err != nil {
			return fmt.Errorf("failed to parse %s: %w", filename, err)
		}

		if len(values) < 1 {
			return fmt.Errorf("%s must have at least 1 value column", filename)
		}

		dl.illuminants[illuminantNames[i]] = NewSPD(wavelengths, values[0])
	}

	return nil
}

// GetObserver returns the Color Matching Functions for the specified observer type.
// Returns full CMF data (no interpolation - CMF has higher resolution than measured SPD).
func (dl *DataLoader) GetObserver(observer ObserverType) (ObserverCMF, error) {
	cmf, ok := dl.observers[observer]
	if !ok {
		return ObserverCMF{}, fmt.Errorf("unknown observer type: %s (use '2' or '10')", observer)
	}

	// Return cloned CMF
	cloned := mat.New(4, cmf.Len())
	cloned.CopyFrom(cmf.Matrix)

	return ObserverCMF{Matrix: cloned}, nil
}

// GetIlluminant returns the SPD for the specified illuminant.
// Returns full SPD data (no interpolation - illuminant has higher resolution than measured SPD).
func (dl *DataLoader) GetIlluminant(illuminant string) (SPD, error) {
	illuminant = strings.ToUpper(illuminant)

	// Parse illuminant (e.g., "D65/10" -> "D65")
	if idx := strings.Index(illuminant, "/"); idx >= 0 {
		illuminant = illuminant[:idx]
	}

	spd, ok := dl.illuminants[illuminant]
	if !ok {
		available := make([]string, 0, len(dl.illuminants))
		for name := range dl.illuminants {
			available = append(available, name)
		}
		return SPD{}, fmt.Errorf("unknown illuminant '%s'. Available: %v", illuminant, available)
	}

	// Return cloned SPD
	cloned := mat.New(2, spd.Len())
	cloned.CopyFrom(spd.Matrix)

	return SPD{Matrix: cloned}, nil
}

// AvailableIlluminants returns a list of available illuminant names.
func (dl *DataLoader) AvailableIlluminants() []string {
	names := make([]string, 0, len(dl.illuminants))
	for name := range dl.illuminants {
		names = append(names, name)
	}
	return names
}
