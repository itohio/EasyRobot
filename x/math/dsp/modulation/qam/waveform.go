package qam

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/dsp"
)

type iqPoint struct {
	i float32
	q float32
}

type constellationTable struct {
	points         []iqPoint
	axisLevels     []float32
	envelope       []float32
	envelopeEnergy float32
	axisBits       int
	axisMask       int
	avgSymbolPower float32
}

func newConstellationTable(cfg Config) constellationTable {
	size := cfg.constellationSize()
	axisBits := cfg.axisBits()
	levelsPerDim := cfg.levelsPerDimension()

	table := constellationTable{
		points:     make([]iqPoint, size),
		axisLevels: make([]float32, levelsPerDim),
		envelope:   buildEnvelope(cfg),
		axisBits:   axisBits,
		axisMask:   (1 << axisBits) - 1,
	}

	scale := cfg.ConstellationNormalization()
	for i := range levelsPerDim {
		raw := float32(-(levelsPerDim - 1) + i*2)
		binaryVal := raw * scale
		gray := binaryToGray(i)
		table.axisLevels[gray] = binaryVal
	}

	for idx := range size {
		iBits := idx >> axisBits
		qBits := idx & table.axisMask
		iGray := binaryToGray(iBits)
		qGray := binaryToGray(qBits)
		table.points[idx] = iqPoint{
			i: table.axisLevels[iGray],
			q: table.axisLevels[qGray],
		}
	}

	table.avgSymbolPower = computeAveragePower(table.points)
	table.envelopeEnergy = computeEnvelopeEnergy(table.envelope)
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

func computeAveragePower(points []iqPoint) float32 {
	if len(points) == 0 {
		return 0
	}
	var sum float32
	for _, p := range points {
		sum += p.i*p.i + p.q*p.q
	}
	return sum / float32(len(points))
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

func (t constellationTable) quantizeAxis(value float32) (idx int, level float32) {
	bestIdx := 0
	bestLevel := t.axisLevels[0]
	bestDist := math32.Abs(value - bestLevel)
	for i := range t.axisLevels {
		lvl := t.axisLevels[i]
		dist := math32.Abs(value - lvl)
		if dist < bestDist {
			bestDist = dist
			bestIdx = i
			bestLevel = lvl
		}
	}
	return bestIdx, bestLevel
}
