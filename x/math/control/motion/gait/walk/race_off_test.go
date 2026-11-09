//go:build !race

package walk

const (
	randomTicks = 5000 // TestRandomCommandSequences ticks per configuration
	stopSeeds   = 8    // TestStopLatency seeds per configuration (4 stops each)
)
