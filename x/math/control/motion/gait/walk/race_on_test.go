//go:build race

package walk

// The race detector slows the walker ~35× (TestRandomCommandSequences took
// 105 s against 3 s), so the long randomised runs are shortened under -race.
// They still run, so the race gate keeps their coverage.
const (
	randomTicks = 1500 // TestRandomCommandSequences ticks per configuration
	stopSeeds   = 2    // TestStopLatency seeds per configuration (4 stops each)
)
