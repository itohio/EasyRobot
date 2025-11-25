package mask

import (
	"github.com/itohio/EasyRobot/x/math/dsp"
)

type amplitudeTable struct {
	levels   []float32
	envelope []float32
}

func newAmplitudeTable(cfg Config) amplitudeTable {
	return amplitudeTable{
		levels:   cfg.AmplitudeLevels,
		envelope: buildEnvelope(cfg),
	}
}

func buildEnvelope(cfg Config) []float32 {
	n := cfg.SamplesPerSymbol
	env := make([]float32, n)
	switch cfg.Envelope {
	case "raised_cosine", "root_raised_cosine", "":
		center := float32(n-1) / 2
		scale := 1 / float32(n)
		gen := dsp.NewRaisedCosineGenerator(center, scale, cfg.RaisedCosineBeta, 1.0)
		gen.Generate(env)
	default:
		// Rectangular
		for i := range n {
			env[i] = 1.0
		}
	}
	return env
}

