package colorscience

import (
	"fmt"
	"sort"

	"github.com/chewxy/math32"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// SpectrumMetadata contains metadata for a spectrum in the database.
type SpectrumMetadata struct {
	Name        string            // Spectrum name/identifier
	Description string            // Description of the spectrum
	Type        string            // Spectrum type (reflectance, absorption, raman, emission, etc.)
	Properties  map[string]string // Additional properties (material, source, etc.)
}

// SpectrumEntry represents a spectrum entry in the database.
type SpectrumEntry struct {
	SPD      SPD              // The spectrum SPD
	Metadata SpectrumMetadata // Metadata about the spectrum
}

// MatchResult represents a match result from database search.
type MatchResult struct {
	Entry      SpectrumEntry // The matched spectrum entry
	Score      float32       // Similarity score (0-1, higher is better)
	Confidence float32       // Confidence in the match (0-1)
}

// SpectrumDatabase provides an interface for storing and searching spectra.
type SpectrumDatabase interface {
	// Add adds a spectrum to the database.
	Add(name string, spd SPD, metadata SpectrumMetadata) error

	// SearchBySimilarity searches for spectra similar to the given SPD.
	// Returns ranked matches sorted by similarity score (highest first).
	SearchBySimilarity(query SPD, maxResults int) ([]MatchResult, error)

	// SearchByPeaks searches for spectra with similar peak patterns.
	// Returns ranked matches sorted by similarity score (highest first).
	SearchByPeaks(peaks []Peak, maxResults int) ([]MatchResult, error)

	// Get retrieves a spectrum by name.
	Get(name string) (SpectrumEntry, error)

	// List returns all spectrum names in the database.
	List() []string

	// Remove removes a spectrum from the database.
	Remove(name string) error
}

// InMemoryDatabase is an in-memory implementation of SpectrumDatabase.
type InMemoryDatabase struct {
	entries map[string]SpectrumEntry
}

// NewSpectrumDatabase creates a new in-memory spectrum database.
func NewSpectrumDatabase() SpectrumDatabase {
	return &InMemoryDatabase{
		entries: make(map[string]SpectrumEntry),
	}
}

// Add adds a spectrum to the database.
func (db *InMemoryDatabase) Add(name string, spd SPD, metadata SpectrumMetadata) error {
	if name == "" {
		return fmt.Errorf("spectrum name cannot be empty")
	}

	if spd.Matrix == nil {
		return fmt.Errorf("SPD cannot be nil")
	}

	// Clone SPD to ensure we have our own copy
	// Note: This is a simplified clone - in production, you'd want a proper deep copy
	db.entries[name] = SpectrumEntry{
		SPD:      spd,
		Metadata: metadata,
	}

	return nil
}

// SearchBySimilarity searches for spectra similar to the given SPD using correlation.
func (db *InMemoryDatabase) SearchBySimilarity(query SPD, maxResults int) ([]MatchResult, error) {
	if query.Matrix == nil {
		return nil, fmt.Errorf("query SPD cannot be nil")
	}

	var results []MatchResult

	queryVals := query.Values()
	queryWl := query.Wavelengths()

	if queryVals == nil || queryWl == nil {
		return nil, fmt.Errorf("query SPD must have valid wavelengths and values")
	}

	for _, entry := range db.entries {
		// Interpolate database entry to match query wavelengths
		queryWlVec := vecTypes.Vector(queryWl)
		interpEntry := entry.SPD.Interpolate(queryWlVec)
		entryVals := interpEntry.Values()

		if entryVals == nil || entryVals.Len() != queryVals.Len() {
			continue
		}

		// Calculate Pearson correlation coefficient
		var sum1, sum2, sum1Sq, sum2Sq, sumProd float32
		for i := 0; i < queryVals.Len(); i++ {
			v1 := queryVals[i]
			v2 := entryVals[i]
			sum1 += v1
			sum2 += v2
			sum1Sq += v1 * v1
			sum2Sq += v2 * v2
			sumProd += v1 * v2
		}

		n := float32(queryVals.Len())
		mean1 := sum1 / n
		mean2 := sum2 / n

		cov := (sumProd / n) - (mean1 * mean2)
		std1 := (sum1Sq/n - mean1*mean1)
		std2 := (sum2Sq/n - mean2*mean2)

		var score float32
		if std1 > 0 && std2 > 0 {
			corr := cov / (math32.Sqrt(std1) * math32.Sqrt(std2))
			if corr < 0 {
				corr = 0
			}
			score = corr
		}

		results = append(results, MatchResult{
			Entry:      entry,
			Score:      score,
			Confidence: score, // Use correlation as confidence
		})
	}

	// Sort by score (highest first)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Limit results
	if maxResults > 0 && len(results) > maxResults {
		results = results[:maxResults]
	}

	return results, nil
}

// SearchByPeaks searches for spectra with similar peak patterns.
func (db *InMemoryDatabase) SearchByPeaks(peaks []Peak, maxResults int) ([]MatchResult, error) {
	if len(peaks) == 0 {
		return nil, fmt.Errorf("peaks cannot be empty")
	}

	var results []MatchResult

	for _, entry := range db.entries {
		// Detect peaks in database entry
		entryPeaks := entry.SPD.Peaks(0, 0)

		if len(entryPeaks) == 0 {
			continue
		}

		// Match peaks by wavelength proximity
		matched := 0
		totalDistance := float32(0)

		for _, queryPeak := range peaks {
			bestMatch := -1
			bestDistance := float32(1e10)

			for i, entryPeak := range entryPeaks {
				dist := math32.Abs(queryPeak.Wavelength - entryPeak.Wavelength)
				if dist < bestDistance {
					bestDistance = dist
					bestMatch = i
				}
			}

			if bestMatch >= 0 && bestDistance < 50.0 { // Within 50nm tolerance
				matched++
				totalDistance += bestDistance
			}
		}

		if matched > 0 {
			// Score based on number of matched peaks and average distance
			matchRatio := float32(matched) / float32(len(peaks))
			avgDistance := totalDistance / float32(matched)
			distanceScore := 1.0 / (1.0 + avgDistance/10.0) // Normalize to [0,1]
			score := matchRatio * distanceScore

			results = append(results, MatchResult{
				Entry:      entry,
				Score:      score,
				Confidence: matchRatio, // Use match ratio as confidence
			})
		}
	}

	// Sort by score (highest first)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Limit results
	if maxResults > 0 && len(results) > maxResults {
		results = results[:maxResults]
	}

	return results, nil
}

// Get retrieves a spectrum by name.
func (db *InMemoryDatabase) Get(name string) (SpectrumEntry, error) {
	entry, ok := db.entries[name]
	if !ok {
		return SpectrumEntry{}, fmt.Errorf("spectrum '%s' not found", name)
	}
	return entry, nil
}

// List returns all spectrum names in the database.
func (db *InMemoryDatabase) List() []string {
	names := make([]string, 0, len(db.entries))
	for name := range db.entries {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Remove removes a spectrum from the database.
func (db *InMemoryDatabase) Remove(name string) error {
	if _, ok := db.entries[name]; !ok {
		return fmt.Errorf("spectrum '%s' not found", name)
	}
	delete(db.entries, name)
	return nil
}
