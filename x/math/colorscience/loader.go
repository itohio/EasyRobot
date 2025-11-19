package colorscience

import (
	"fmt"
)

var (
	// defaultDataLoader is a singleton data loader for embedded data.
	defaultDataLoader *DataLoader
)

// init initializes the default data loader.
func init() {
	var err error
	defaultDataLoader, err = NewDataLoader()
	if err != nil {
		panic(fmt.Sprintf("failed to initialize default data loader: %v", err))
	}
}

// LoadCMF loads Color Matching Functions for the specified observer.
// Returns full CMF data (CMF has higher resolution than measured SPD).
func LoadCMF(observer ObserverType) (ObserverCMF, error) {
	return defaultDataLoader.GetObserver(observer)
}

// LoadIlluminantSPD loads the SPD for the specified illuminant.
// Returns full SPD data (illuminant has higher resolution than measured SPD).
func LoadIlluminantSPD(illuminantName string) (SPD, error) {
	return defaultDataLoader.GetIlluminant(illuminantName)
}

// AvailableIlluminants returns a list of available illuminant names.
func AvailableIlluminants() []string {
	return defaultDataLoader.AvailableIlluminants()
}
