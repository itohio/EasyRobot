package mfsk

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/dsp"
)

type toneTable struct {
	phases   []float32
	incr     []float32
	freqs    []float32
	envelope []float32
}

func newToneTable(cfg Config) toneTable {
	count := cfg.toneCount()
	table := toneTable{
		phases: make([]float32, count),
		incr:   make([]float32, count),
		freqs:  make([]float32, count),
	}
	for i := range count {
		freq := cfg.ToneFrequency(i)
		table.freqs[i] = freq
		table.incr[i] = 2 * math32.Pi * freq / cfg.SampleRate
	}
	table.envelope = buildEnvelope(cfg)
	return table
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
