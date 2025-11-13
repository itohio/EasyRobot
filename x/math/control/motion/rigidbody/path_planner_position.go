package rigidbody

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/vec"
)

type positionPlanner struct {
	positions []vec.Vector3D
	lengths   []float32
	total     float32
	signature uint64
}

func newPositionPlanner(points []vec.Vector3D) (pathPlanner, error) {
	if err := validateWaypoints(points); err != nil {
		return nil, err
	}
	lengths, total := computeLengths(points)
	planner := &positionPlanner{
		positions: points,
		lengths:   lengths,
		total:     total,
		signature: hashPositions(points),
	}
	return planner, nil
}

func (p *positionPlanner) Length() float32 {
	return p.total
}

func (p *positionPlanner) Signature() uint64 {
	return p.signature
}

func (p *positionPlanner) Sample(progress float32) samplePoint {
	if progress <= 0 {
		tangent := unitTangent(p.positions[0], p.positions[1])
		return samplePoint{
			position:       p.positions[0],
			tangent:        tangent,
			orientation:    quaternionFromTangent(tangent),
			hasOrientation: true,
		}
	}
	if progress >= p.total {
		n := len(p.positions)
		tangent := unitTangent(p.positions[n-2], p.positions[n-1])
		return samplePoint{
			position:       p.positions[n-1],
			tangent:        tangent,
			orientation:    quaternionFromTangent(tangent),
			hasOrientation: true,
		}
	}
	progress = clampProgress(progress, p.total)
	idx := searchSegment(p.lengths, progress)
	start := p.positions[idx]
	end := p.positions[idx+1]
	segLen := p.lengths[idx+1] - p.lengths[idx]
	ratio := float32(0)
	if segLen > 0 {
		ratio = (progress - p.lengths[idx]) / segLen
	}
	pos := interpolateVec(start, end, ratio)
	tangent := unitTangent(start, end)
	return samplePoint{
		position:       pos,
		tangent:        tangent,
		orientation:    quaternionFromTangent(tangent),
		hasOrientation: true,
	}
}

func (p *positionPlanner) Project(position vec.Vector3D) float32 {
	best := float32(0)
	minDist := float32(math32.MaxFloat32)
	for i := 0; i < len(p.positions)-1; i++ {
		a := p.positions[i]
		b := p.positions[i+1]
		proj := projectPoint(position, a, b)
		d := distance(position, proj)
		if d < minDist {
			minDist = d
			best = p.lengths[i] + distance(a, proj)
		}
	}
	return clampFloat(best, 0, p.total)
}

func (p *positionPlanner) Curvature(progress float32) float32 {
	if len(p.positions) < 3 {
		return 0
	}
	progress = clampProgress(progress, p.total)
	prev := clampFloat(progress-math32.Max(p.total*0.01, 0.05), 0, p.total)
	next := clampFloat(progress+math32.Max(p.total*0.01, 0.05), 0, p.total)
	p0 := p.Sample(prev).position
	p1 := p.Sample(progress).position
	p2 := p.Sample(next).position
	return curvature2D(p0, p1, p2)
}

func hashPositions(points []vec.Vector3D) uint64 {
	h := uint64(1469598103934665603)
	for _, p := range points {
		h = hashVector(h, p)
	}
	return h
}
