package ssd

import (
	"errors"
	"math"
)

// AnchorConfig describes the anchor grid of an SSD head as TF Object
// Detection API's ssd_anchor_generator builds it
// (multiple_grid_anchor_generator.create_ssd_anchors), for a square input.
//
// Layer k of n has scale MinScale + (MaxScale − MinScale)·k/(n−1); a box of
// scale s and aspect ratio r is s/√r high and s·√r wide, centred on every
// cell of the layer's feature map. Each cell gets one box per aspect ratio,
// then, when InterpolatedRatio > 0, one of that ratio at the scale
// √(s·s_next) (s_next is the next layer's scale, 1 after the last). With
// ReduceLowest the first layer has three boxes instead: scale 0.1 at ratio
// 1, then its own scale at ratios 2 and 0.5.
type AnchorConfig struct {
	// Maps are the head feature maps (H, W), finest first; at least two.
	Maps               [][2]int
	MinScale, MaxScale float32
	AspectRatios       []float32
	InterpolatedRatio  float32
	ReduceLowest       bool
}

// MobileNetV1Anchors is the generator setting of TF's SSD-MobileNetV1 COCO
// exports (scales 0.2 … 0.95; ratios 1, 2, 1/2, 3, 1/3; an interpolated
// square box; three boxes in the first layer) for the given head maps:
// 19, 10, 5, 3, 2, 1 squared at a 300×300 input, 1917 boxes.
func MobileNetV1Anchors(maps [][2]int) AnchorConfig {
	return AnchorConfig{
		Maps: maps, MinScale: 0.2, MaxScale: 0.95,
		AspectRatios:      []float32{1, 2, 0.5, 3, 1.0 / 3},
		InterpolatedRatio: 1, ReduceLowest: true,
	}
}

// ErrAnchorConfig reports an AnchorConfig that describes no anchor grid.
var ErrAnchorConfig = errors.New("ssd: invalid anchor config (two or more non-empty maps, 0 < MinScale ≤ MaxScale, positive aspect ratios)")

// Anchors returns cfg's boxes as [ycenter, xcenter, h, w] rows in
// normalised coordinates (Config.Anchors), ordered layer by layer, row by
// row, cell by cell, box by box, as the model's outputs are.
func Anchors(cfg AnchorConfig) ([]float32, error) {
	n := len(cfg.Maps)
	pos := func(v float32) bool { return v > 0 && !math.IsInf(float64(v), 0) }
	// The interpolated ratio is 0 (no such box) or positive; NaN fails both.
	if n < 2 || !pos(cfg.MinScale) || !pos(cfg.MaxScale) || cfg.MaxScale < cfg.MinScale ||
		len(cfg.AspectRatios) == 0 || !(cfg.InterpolatedRatio == 0 || pos(cfg.InterpolatedRatio)) {
		return nil, ErrAnchorConfig
	}
	for _, r := range cfg.AspectRatios {
		if !pos(r) {
			return nil, ErrAnchorConfig
		}
	}
	boxes := 0
	for k, m := range cfg.Maps {
		if m[0] <= 0 || m[1] <= 0 {
			return nil, ErrAnchorConfig
		}
		per := len(cfg.AspectRatios)
		if cfg.InterpolatedRatio > 0 {
			per++
		}
		if k == 0 && cfg.ReduceLowest {
			per = 3
		}
		boxes += m[0] * m[1] * per
	}
	scale := func(k int) float64 {
		if k == n {
			return 1
		}
		return float64(cfg.MinScale) + (float64(cfg.MaxScale)-float64(cfg.MinScale))*float64(k)/float64(n-1)
	}
	type box struct{ h, w float32 }
	specs := make([]box, 0, len(cfg.AspectRatios)+1)
	add := func(s, r float64) {
		q := math.Sqrt(r)
		specs = append(specs, box{float32(s / q), float32(s * q)})
	}
	out := make([]float32, 0, 4*boxes)
	for k, m := range cfg.Maps {
		specs = specs[:0]
		if k == 0 && cfg.ReduceLowest {
			add(0.1, 1)
			add(scale(0), 2)
			add(scale(0), 0.5)
		} else {
			for _, r := range cfg.AspectRatios {
				add(scale(k), float64(r))
			}
			if cfg.InterpolatedRatio > 0 {
				add(math.Sqrt(scale(k)*scale(k+1)), float64(cfg.InterpolatedRatio))
			}
		}
		for y := range m[0] {
			yc := float32((float64(y) + 0.5) / float64(m[0]))
			for x := range m[1] {
				xc := float32((float64(x) + 0.5) / float64(m[1]))
				for _, b := range specs {
					out = append(out, yc, xc, b.h, b.w)
				}
			}
		}
	}
	return out, nil
}
