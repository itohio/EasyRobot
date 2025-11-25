package mpsk

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/dsp"
)

type constellationTable struct {
	phases         []float32
	sin            []float32
	cos            []float32
	envelope       []float32
	envelopeEnergy float32
}

func newConstellationTable(cfg Config) constellationTable {
	size := cfg.constellationSize()
	env := buildEnvelope(cfg)
	table := constellationTable{
		phases:   make([]float32, size),
		sin:      make([]float32, size),
		cos:      make([]float32, size),
		envelope: env,
	}
	step := twoPi / float32(size)
	for i := range size {
		phase := step * float32(i)
		table.phases[i] = phase
		table.sin[i] = math32.Sin(phase)
		table.cos[i] = math32.Cos(phase)
	}
	table.envelopeEnergy = computeEnvelopeEnergy(env)
	return table
}

func computeEnvelopeEnergy(env []float32) float32 {
	if len(env) == 0 {
		return 1
	}
	var sum float32
	for _, v := range env {
		sum += v * v
	}
	if sum == 0 {
		return 1
	}
	return sum
}

func buildEnvelope(cfg Config) []float32 {
	n := cfg.SamplesPerSymbol
	env := make([]float32, n)
	switch cfg.Envelope {
	case "raised_cosine", "root_raised_cosine", "":
		center := float32(n-1) / 2
		scale := 1 / float32(n)
		gen := dsp.NewRaisedCosineGenerator(center, scale, cfg.RaisedCosineBeta, cfg.Amplitude)
		gen.Generate(env)
	default:
		for i := range n {
			env[i] = cfg.Amplitude
		}
	}
	return env
}
