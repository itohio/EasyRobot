package graph

import (
	"github.com/chewxy/math32"
)

// KDNode represents a node in a k-d tree
type KDNode struct {
	Point     []float32 // K-dimensional point
	Dimension int       // Which dimension this node splits on (0 to k-1)
	Left      *KDNode   // Points where point[Dimension] <= splitting value
	Right     *KDNode   // Points where point[Dimension] > splitting value
	Parent    *KDNode
}

// KDTree represents a k-dimensional tree for spatial queries
type KDTree struct {
	Root   *KDNode
	K      int         // Number of dimensions
	Points [][]float32 // All points (for reference, optional)
}

// NewKDTree creates a new k-d tree from points
func NewKDTree(points [][]float32) *KDTree {
	if len(points) == 0 {
		return &KDTree{
			Root:   nil,
			K:      0,
			Points: points,
		}
	}

	k := len(points[0])
	kt := &KDTree{
		Root:   nil,
		K:      k,
		Points: points,
	}

	kt.Root = kt.buildTree(points, 0)

	return kt
}

// buildTree builds a k-d tree recursively
func (kt *KDTree) buildTree(points [][]float32, depth int) *KDNode {
	if len(points) == 0 {
		return nil
	}

	if len(points) == 1 {
		return &KDNode{
			Point:     points[0],
			Dimension: depth % kt.K,
			Left:      nil,
			Right:     nil,
			Parent:    nil,
		}
	}

	// Select dimension to split on (alternating)
	dimension := depth % kt.K

	// Find median point for this dimension
	medianIndex := kt.findMedian(points, dimension)
	medianPoint := points[medianIndex]

	// Split points
	leftPoints := make([][]float32, 0)
	rightPoints := make([][]float32, 0)

	for i, point := range points {
		if i == medianIndex {
			continue
		}

		if point[dimension] <= medianPoint[dimension] {
			leftPoints = append(leftPoints, point)
		} else {
			rightPoints = append(rightPoints, point)
		}
	}

	node := &KDNode{
		Point:     medianPoint,
		Dimension: dimension,
		Left:      kt.buildTree(leftPoints, depth+1),
		Right:     kt.buildTree(rightPoints, depth+1),
		Parent:    nil,
	}

	// Set parent pointers
	if node.Left != nil {
		node.Left.Parent = node
	}
	if node.Right != nil {
		node.Right.Parent = node
	}

	return node
}

// findMedian finds the median index for points sorted by dimension
func (kt *KDTree) findMedian(points [][]float32, dimension int) int {
	if len(points) == 0 {
		return 0
	}

	// Create indices and sort by dimension value
	indices := make([]int, len(points))
	for i := range indices {
		indices[i] = i
	}

	// Simple selection algorithm to find median (could optimize)
	medianPos := len(points) / 2

	for i := 0; i < len(points); i++ {
		for j := i + 1; j < len(points); j++ {
			if points[indices[i]][dimension] > points[indices[j]][dimension] {
				indices[i], indices[j] = indices[j], indices[i]
			}
		}
		if i == medianPos {
			break // Early exit when we reach median
		}
	}

	return indices[medianPos]
}

// NearestNeighbor finds the nearest neighbor to the query point
func (kt *KDTree) NearestNeighbor(query []float32) []float32 {
	if kt.Root == nil || len(query) != kt.K {
		return nil
	}

	bestPoint, _ := kt.nearestNeighborRecursive(kt.Root, query, nil, math32.MaxFloat32)
	return bestPoint
}

func (kt *KDTree) nearestNeighborRecursive(node *KDNode, query []float32, bestPoint []float32, bestDist float32) ([]float32, float32) {
	if node == nil {
		return bestPoint, bestDist
	}

	// Calculate distance to current node
	currentDist := kt.euclideanDistance(query, node.Point)
	if currentDist < bestDist {
		bestDist = currentDist
		bestPoint = node.Point
	}

	// Determine which side to search first
	dim := node.Dimension
	diff := query[dim] - node.Point[dim]

	var nearSubtree, farSubtree *KDNode
	if diff <= 0 {
		nearSubtree = node.Left
		farSubtree = node.Right
	} else {
		nearSubtree = node.Right
		farSubtree = node.Left
	}

	// Search near subtree
	bestPoint, bestDist = kt.nearestNeighborRecursive(nearSubtree, query, bestPoint, bestDist)

	// Check if we need to search far subtree
	if diff*diff < bestDist {
		bestPoint, bestDist = kt.nearestNeighborRecursive(farSubtree, query, bestPoint, bestDist)
	}

	return bestPoint, bestDist
}

// KNearestNeighbors finds k nearest neighbors to the query point
func (kt *KDTree) KNearestNeighbors(query []float32, k int) [][]float32 {
	if kt.Root == nil || len(query) != kt.K || k <= 0 {
		return nil
	}

	// Use a simple approach: collect all points, sort by distance
	neighbors := make([]neighborInfo, 0)
	kt.collectAllPoints(kt.Root, query, &neighbors)

	// Sort by distance (simple bubble sort)
	for i := 0; i < len(neighbors) && i < k; i++ {
		for j := i + 1; j < len(neighbors); j++ {
			if neighbors[i].distance > neighbors[j].distance {
				neighbors[i], neighbors[j] = neighbors[j], neighbors[i]
			}
		}
	}

	// Return k nearest
	result := make([][]float32, 0, k)
	for i := 0; i < k && i < len(neighbors); i++ {
		result = append(result, neighbors[i].point)
	}

	return result
}

type neighborInfo struct {
	point    []float32
	distance float32
}

func (kt *KDTree) collectAllPoints(node *KDNode, query []float32, neighbors *[]neighborInfo) {
	if node == nil {
		return
	}

	dist := kt.euclideanDistance(query, node.Point)
	*neighbors = append(*neighbors, neighborInfo{
		point:    node.Point,
		distance: dist,
	})

	kt.collectAllPoints(node.Left, query, neighbors)
	kt.collectAllPoints(node.Right, query, neighbors)
}

// RangeQuery finds all points within the specified range
func (kt *KDTree) RangeQuery(min, max []float32) [][]float32 {
	if kt.Root == nil || len(min) != kt.K || len(max) != kt.K {
		return nil
	}

	result := make([][]float32, 0)
	kt.rangeQueryRecursive(kt.Root, min, max, &result)
	return result
}

func (kt *KDTree) rangeQueryRecursive(node *KDNode, min, max []float32, result *[][]float32) {
	if node == nil {
		return
	}

	point := node.Point
	inRange := true

	for i := 0; i < kt.K; i++ {
		if point[i] < min[i] || point[i] > max[i] {
			inRange = false
			break
		}
	}

	if inRange {
		*result = append(*result, point)
	}

	dim := node.Dimension
	if node.Left != nil && min[dim] <= node.Point[dim] {
		kt.rangeQueryRecursive(node.Left, min, max, result)
	}
	if node.Right != nil && max[dim] >= node.Point[dim] {
		kt.rangeQueryRecursive(node.Right, min, max, result)
	}
}

// euclideanDistance calculates Euclidean distance between two points
func (kt *KDTree) euclideanDistance(p1, p2 []float32) float32 {
	if len(p1) != len(p2) {
		return math32.MaxFloat32
	}

	sum := float32(0.0)
	for i := 0; i < len(p1); i++ {
		diff := p1[i] - p2[i]
		sum += diff * diff
	}

	return math32.Sqrt(sum)
}

// Insert adds a new point to the tree
func (kt *KDTree) Insert(point []float32) {
	if len(point) != kt.K {
		return
	}

	if kt.Root == nil {
		kt.Root = &KDNode{
			Point:     point,
			Dimension: 0,
			Left:      nil,
			Right:     nil,
			Parent:    nil,
		}
		kt.Points = append(kt.Points, point)
		return
	}

	kt.insertRecursive(kt.Root, point, 0)
	kt.Points = append(kt.Points, point)
}

func (kt *KDTree) insertRecursive(node *KDNode, point []float32, depth int) {
	dim := depth % kt.K

	if point[dim] <= node.Point[dim] {
		if node.Left == nil {
			node.Left = &KDNode{
				Point:     point,
				Dimension: (depth + 1) % kt.K,
				Left:      nil,
				Right:     nil,
				Parent:    node,
			}
		} else {
			kt.insertRecursive(node.Left, point, depth+1)
		}
	} else {
		if node.Right == nil {
			node.Right = &KDNode{
				Point:     point,
				Dimension: (depth + 1) % kt.K,
				Left:      nil,
				Right:     nil,
				Parent:    node,
			}
		} else {
			kt.insertRecursive(node.Right, point, depth+1)
		}
	}
}

// GetHeight returns the height of the tree
func (kt *KDTree) GetHeight() int {
	if kt.Root == nil {
		return 0
	}

	return kt.getHeightRecursive(kt.Root)
}

func (kt *KDTree) getHeightRecursive(node *KDNode) int {
	if node == nil {
		return -1
	}

	leftHeight := kt.getHeightRecursive(node.Left)
	rightHeight := kt.getHeightRecursive(node.Right)

	if leftHeight > rightHeight {
		return leftHeight + 1
	}

	return rightHeight + 1
}
