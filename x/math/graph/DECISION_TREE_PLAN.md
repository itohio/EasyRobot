# Decision Tree and K-D Tree Implementation Plan

## Overview
Implement decision tree structures for classification/regression and k-d trees for spatial queries, building upon the existing tree structures.

## 1. Decision Tree

### Purpose
- Classification and regression tasks
- Built upon tree structure
- Nodes contain decision criteria and outcomes

### Structure
```go
type DecisionNode struct {
    FeatureIndex int           // Which feature to split on
    Threshold    float32       // Threshold value for split
    Left         *DecisionNode // Points with feature[FeatureIndex] <= Threshold
    Right        *DecisionNode // Points with feature[FeatureIndex] > Threshold
    Outcome      any           // Classification result (for leaf nodes)
    IsLeaf       bool          // True if this is a leaf node
}

type DecisionTree struct {
    Root        *DecisionNode
    MaxDepth    int
    MinSamples int
}
```

### Operations
- `BuildTree(data [][]float32, labels []any, maxDepth, minSamples int)` - Build tree from data
- `Predict(features []float32) any` - Make prediction for a feature vector
- `PredictProbabilities(features []float32) map[any]float32` - Return class probabilities
- `FindBestSplit(data [][]float32, labels []any) (featureIndex int, threshold float32)` - Find optimal split
- `CalculateImpurity(labels []any) float32` - Calculate Gini/entropy impurity
- `IsLeaf(node *DecisionNode) bool` - Check if node is leaf

### Splitting Criteria
- Gini impurity (for classification)
- Entropy/Information Gain
- Mean Squared Error (for regression)

### Features
- Support both classification and regression
- Configurable max depth and min samples per leaf
- Pruning support (optional)

## 2. K-D Tree (K-Dimensional Tree)

### Purpose
- Spatial data structure for organizing points in k-dimensional space
- Efficient nearest neighbor search
- Range queries

### Structure
```go
type KDNode struct {
    Point      []float32   // K-dimensional point
    Dimension  int         // Which dimension this node splits on
    Left       *KDNode     // Points where point[Dimension] <= splitting value
    Right      *KDNode     // Points where point[Dimension] > splitting value
    Parent     *KDNode
}

type KDTree struct {
    Root    *KDNode
    K       int         // Number of dimensions
    Points  [][]float32 // All points (for reference)
}
```

### Operations
- `BuildTree(points [][]float32)` - Build k-d tree from points
- `Insert(point []float32)` - Insert a point into tree
- `NearestNeighbor(query []float32) []float32` - Find nearest neighbor
- `KNearestNeighbors(query []float32, k int) [][]float32` - Find k nearest neighbors
- `RangeQuery(min, max []float32) [][]float32` - Find all points in range
- `Delete(point []float32) bool` - Remove a point from tree

### Building Strategy
- Alternating dimension splitting
- Median-based splitting for balanced trees
- Recursive construction

### Optimization
- Use for high-dimensional nearest neighbor searches
- Efficient range queries
- Support for duplicate points

## 3. Random Forest (Future)

### Structure
- Collection of DecisionTrees
- Each tree trained on bootstrap sample
- Random feature selection at each split
- Voting/aggregation for final prediction

## Implementation Order

### Phase 1: Decision Tree
1. Implement DecisionNode and DecisionTree structures
2. Implement splitting criteria (Gini, entropy)
3. Implement tree building algorithm
4. Implement prediction methods
5. Add pruning support (optional)

### Phase 2: K-D Tree
1. Implement KDNode and KDTree structures
2. Implement tree building from points
3. Implement nearest neighbor search
4. Implement range queries
5. Add insertion/deletion methods

### Phase 3: Integration
1. Ensure both work with existing graph structures
2. Add helper functions for conversion
3. Create comprehensive tests

## Files to Create

```
pkg/core/math/graph/
├── decision_tree.go       # DecisionTree, DecisionNode
├── decision_tree_test.go  # Tests for decision tree
├── kd_tree.go            # KDTree, KDNode
└── kd_tree_test.go       # Tests for k-d tree
```

## Design Principles

1. **Build on existing tree structures** - Reuse patterns from Tree and BinaryTree
2. **Separation of concerns** - Keep decision tree logic separate from k-d tree logic
3. **Type safety** - Use generics where appropriate, fall back to `any` for flexibility
4. **Performance** - Optimize for common operations (predictions, nearest neighbor searches)
5. **Extensibility** - Design for future random forest implementation

