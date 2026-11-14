// Package nms is greedy non-maximum suppression for axis-aligned boxes, the
// last step of single-shot detectors (YOLO, SSD, BlazeFace). It allocates
// nothing: the caller owns the candidate and result slices.
package nms

import "slices"

// Box is one detection candidate: corners X0,Y0 (top left) and X1,Y1
// (bottom right) in any pixel space, a score, a class and the index of the
// anchor it came from (for tracing back to the model output).
type Box struct {
	X0, Y0, X1, Y1 float32
	Score          float32
	Class          int32
	Index          int32
}

// IoU is the intersection over union of a and b; 0 when either is empty or
// they don't overlap.
func IoU(a, b Box) float32 {
	w := min(a.X1, b.X1) - max(a.X0, b.X0)
	h := min(a.Y1, b.Y1) - max(a.Y0, b.Y0)
	if w <= 0 || h <= 0 {
		return 0
	}
	inter := w * h
	union := (a.X1-a.X0)*(a.Y1-a.Y0) + (b.X1-b.X0)*(b.Y1-b.Y0) - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

// Greedy sorts cand in place by score (highest first; ties by lower Index),
// then appends to out each candidate whose IoU with every box already kept
// is at most iou, comparing only boxes of the same class when perClass is
// set. It stops when out is full (len == cap), so the best boxes win, and
// returns out. It allocates nothing.
func Greedy(cand []Box, iou float32, perClass bool, out []Box) []Box {
	slices.SortFunc(cand, func(a, b Box) int {
		switch {
		case a.Score > b.Score:
			return -1
		case a.Score < b.Score:
			return 1
		}
		return int(a.Index - b.Index)
	})
	for _, c := range cand {
		if len(out) == cap(out) {
			break
		}
		keep := true
		for _, k := range out {
			if (!perClass || k.Class == c.Class) && IoU(k, c) > iou {
				keep = false
				break
			}
		}
		if keep {
			out = append(out, c)
		}
	}
	return out
}
