package graph

// DecisionNode represents a node in a decision tree
type DecisionNode struct {
	FeatureIndex int           // Which feature to split on (-1 for leaf nodes)
	Threshold    float32       // Threshold value for split
	Left         *DecisionNode // Points with feature[FeatureIndex] <= Threshold
	Right        *DecisionNode // Points with feature[FeatureIndex] > Threshold
	Outcome      any           // Classification result (for leaf nodes)
	IsLeaf       bool          // True if this is a leaf node
	SampleCount  int           // Number of samples in this node
}

// DecisionTree represents a decision tree for classification/regression
type DecisionTree struct {
	Root       *DecisionNode
	MaxDepth   int
	MinSamples int
	K          int // Number of features
}

// NewDecisionTree creates a new decision tree
func NewDecisionTree(maxDepth, minSamples int) *DecisionTree {
	return &DecisionTree{
		Root:       nil,
		MaxDepth:   maxDepth,
		MinSamples: minSamples,
		K:          0,
	}
}

// BuildTree builds a decision tree from training data
// data: Feature vectors (each row is a sample, columns are features)
// labels: Classification labels (same length as data)
func (dt *DecisionTree) BuildTree(data [][]float32, labels []any) error {
	if len(data) == 0 || len(data) != len(labels) {
		return nil // Empty data
	}

	if len(data[0]) == 0 {
		return nil // No features
	}

	dt.K = len(data[0])
	dt.Root = dt.buildTreeRecursive(data, labels, 0)

	return nil
}

func (dt *DecisionTree) buildTreeRecursive(data [][]float32, labels []any, depth int) *DecisionNode {
	// Check stopping criteria
	if depth >= dt.MaxDepth || len(data) < dt.MinSamples {
		return dt.createLeafNode(labels)
	}

	// Check if all labels are the same (pure node)
	if dt.isPure(labels) {
		return dt.createLeafNode(labels)
	}

	// Find best split
	bestFeature, bestThreshold := dt.findBestSplit(data, labels)

	if bestFeature == -1 {
		// No good split found
		return dt.createLeafNode(labels)
	}

	// Split data
	leftData, leftLabels, rightData, rightLabels := dt.splitData(data, labels, bestFeature, bestThreshold)

	if len(leftData) == 0 || len(rightData) == 0 {
		// Split didn't work, create leaf
		return dt.createLeafNode(labels)
	}

	// Create decision node
	node := &DecisionNode{
		FeatureIndex: bestFeature,
		Threshold:    bestThreshold,
		IsLeaf:       false,
		SampleCount:  len(data),
		Left:         dt.buildTreeRecursive(leftData, leftLabels, depth+1),
		Right:        dt.buildTreeRecursive(rightData, rightLabels, depth+1),
	}

	return node
}

func (dt *DecisionTree) createLeafNode(labels []any) *DecisionNode {
	// Find most common label
	outcome := dt.mostCommonLabel(labels)

	return &DecisionNode{
		FeatureIndex: -1,
		Threshold:    0,
		IsLeaf:       true,
		Outcome:      outcome,
		SampleCount:  len(labels),
		Left:         nil,
		Right:        nil,
	}
}

func (dt *DecisionTree) isPure(labels []any) bool {
	if len(labels) == 0 {
		return true
	}

	firstLabel := labels[0]
	for i := 1; i < len(labels); i++ {
		if labels[i] != firstLabel {
			return false
		}
	}

	return true
}

func (dt *DecisionTree) mostCommonLabel(labels []any) any {
	if len(labels) == 0 {
		return nil
	}

	labelCount := make(map[any]int)
	for _, label := range labels {
		labelCount[label]++
	}

	maxCount := 0
	var mostCommon any
	for label, count := range labelCount {
		if count > maxCount {
			maxCount = count
			mostCommon = label
		}
	}

	return mostCommon
}

func (dt *DecisionTree) findBestSplit(data [][]float32, labels []any) (bestFeature int, bestThreshold float32) {
	bestFeature = -1
	bestThreshold = 0
	bestGini := float32(1.0) // Start with maximum impurity

	currentGini := dt.calculateGiniImpurity(labels)

	for featureIndex := 0; featureIndex < len(data[0]); featureIndex++ {
		// Get unique values for this feature
		values := make([]float32, 0, len(data))
		for _, sample := range data {
			values = append(values, sample[featureIndex])
		}

		// Sort and try thresholds at midpoints between unique values
		uniqueValues := dt.getUniqueSorted(values)

		for i := 0; i < len(uniqueValues)-1; i++ {
			threshold := (uniqueValues[i] + uniqueValues[i+1]) / 2.0

			leftLabels, rightLabels := dt.splitLabels(data, labels, featureIndex, threshold)

			if len(leftLabels) == 0 || len(rightLabels) == 0 {
				continue
			}

			// Calculate weighted Gini impurity
			leftGini := dt.calculateGiniImpurity(leftLabels)
			rightGini := dt.calculateGiniImpurity(rightLabels)

			leftWeight := float32(len(leftLabels)) / float32(len(labels))
			rightWeight := float32(len(rightLabels)) / float32(len(labels))

			weightedGini := leftWeight*leftGini + rightWeight*rightGini

			// Information gain = current - weighted
			infoGain := currentGini - weightedGini

			if infoGain > 0 && weightedGini < bestGini {
				bestGini = weightedGini
				bestFeature = featureIndex
				bestThreshold = threshold
			}
		}
	}

	return bestFeature, bestThreshold
}

func (dt *DecisionTree) calculateGiniImpurity(labels []any) float32 {
	if len(labels) == 0 {
		return 0
	}

	labelCount := make(map[any]int)
	for _, label := range labels {
		labelCount[label]++
	}

	impurity := float32(1.0)
	total := float32(len(labels))

	for _, count := range labelCount {
		prob := float32(count) / total
		impurity -= prob * prob
	}

	return impurity
}

func (dt *DecisionTree) getUniqueSorted(values []float32) []float32 {
	valueMap := make(map[float32]bool)
	for _, v := range values {
		valueMap[v] = true
	}

	unique := make([]float32, 0, len(valueMap))
	for v := range valueMap {
		unique = append(unique, v)
	}

	// Simple bubble sort (could optimize with better sort)
	for i := 0; i < len(unique); i++ {
		for j := i + 1; j < len(unique); j++ {
			if unique[i] > unique[j] {
				unique[i], unique[j] = unique[j], unique[i]
			}
		}
	}

	return unique
}

func (dt *DecisionTree) splitLabels(data [][]float32, labels []any, featureIndex int, threshold float32) ([]any, []any) {
	var leftLabels, rightLabels []any

	for i, sample := range data {
		if sample[featureIndex] <= threshold {
			leftLabels = append(leftLabels, labels[i])
		} else {
			rightLabels = append(rightLabels, labels[i])
		}
	}

	return leftLabels, rightLabels
}

func (dt *DecisionTree) splitData(data [][]float32, labels []any, featureIndex int, threshold float32) ([][]float32, []any, [][]float32, []any) {
	var leftData, rightData [][]float32
	var leftLabels, rightLabels []any

	for i, sample := range data {
		if sample[featureIndex] <= threshold {
			leftData = append(leftData, sample)
			leftLabels = append(leftLabels, labels[i])
		} else {
			rightData = append(rightData, sample)
			rightLabels = append(rightLabels, labels[i])
		}
	}

	return leftData, leftLabels, rightData, rightLabels
}

// Predict makes a prediction for a feature vector
func (dt *DecisionTree) Predict(features []float32) any {
	if dt.Root == nil {
		return nil
	}

	if len(features) != dt.K {
		return nil // Wrong number of features
	}

	return dt.predictRecursive(dt.Root, features)
}

func (dt *DecisionTree) predictRecursive(node *DecisionNode, features []float32) any {
	if node.IsLeaf {
		return node.Outcome
	}

	if features[node.FeatureIndex] <= node.Threshold {
		return dt.predictRecursive(node.Left, features)
	}

	return dt.predictRecursive(node.Right, features)
}

// PredictProbabilities returns class probabilities (if applicable)
func (dt *DecisionTree) PredictProbabilities(features []float32) map[any]float32 {
	if dt.Root == nil {
		return nil
	}

	if len(features) != dt.K {
		return nil
	}

	probabilities := make(map[any]float32)
	dt.predictProbabilitiesRecursive(dt.Root, features, probabilities, 1.0)

	return probabilities
}

func (dt *DecisionTree) predictProbabilitiesRecursive(node *DecisionNode, features []float32, probabilities map[any]float32, weight float32) {
	if node.IsLeaf {
		// Distribute probability based on sample counts at leaves
		// For simplicity, we assign full weight to the outcome
		probabilities[node.Outcome] = probabilities[node.Outcome] + weight
		return
	}

	if features[node.FeatureIndex] <= node.Threshold {
		dt.predictProbabilitiesRecursive(node.Left, features, probabilities, weight)
	} else {
		dt.predictProbabilitiesRecursive(node.Right, features, probabilities, weight)
	}
}

// GetDepth returns the depth of the tree
func (dt *DecisionTree) GetDepth() int {
	if dt.Root == nil {
		return 0
	}

	return dt.getDepthRecursive(dt.Root)
}

func (dt *DecisionTree) getDepthRecursive(node *DecisionNode) int {
	if node.IsLeaf {
		return 0
	}

	leftDepth := dt.getDepthRecursive(node.Left)
	rightDepth := dt.getDepthRecursive(node.Right)

	if leftDepth > rightDepth {
		return leftDepth + 1
	}

	return rightDepth + 1
}

// CountNodes returns the total number of nodes in the tree
func (dt *DecisionTree) CountNodes() int {
	if dt.Root == nil {
		return 0
	}

	return dt.countNodesRecursive(dt.Root)
}

func (dt *DecisionTree) countNodesRecursive(node *DecisionNode) int {
	if node == nil {
		return 0
	}

	if node.IsLeaf {
		return 1
	}

	return 1 + dt.countNodesRecursive(node.Left) + dt.countNodesRecursive(node.Right)
}
