package yaml

import (
	"fmt"
	"io"
	"iter"
	"reflect"

	"gopkg.in/yaml.v3"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/graph"
	"github.com/itohio/EasyRobot/x/math/tensor"
)

// Unmarshaller implements YAML-based unmarshalling.
type Unmarshaller struct {
	opts types.Options
}

// NewUnmarshaller creates a new YAML unmarshaller.
func NewUnmarshaller(opts ...types.Option) *Unmarshaller {
	u := &Unmarshaller{
		opts: types.Options{},
	}
	for _, opt := range opts {
		opt.Apply(&u.opts)
	}
	return u
}

// Format returns the format name.
func (u *Unmarshaller) Format() string {
	return "yaml"
}

// Unmarshal decodes value from YAML format.
func (u *Unmarshaller) Unmarshal(r io.Reader, dst any, opts ...types.Option) error {
	// Apply additional options
	localOpts := u.opts
	for _, opt := range opts {
		opt.Apply(&localOpts)
	}

	decoder := yaml.NewDecoder(r)

	// Decode yamlValue
	var yv yamlValue
	if err := decoder.Decode(&yv); err != nil {
		return types.NewError("unmarshal", "yaml", "decoding", err)
	}

	// Convert yamlValue to actual value
	value, err := u.yamlToValue(&yv, localOpts)
	if err != nil {
		return types.NewError("unmarshal", "yaml", "conversion", err)
	}

	// Assign to dst
	if err := u.assignToDst(dst, value); err != nil {
		return types.NewError("unmarshal", "yaml", "assignment", err)
	}

	return nil
}

func (u *Unmarshaller) yamlToValue(yv *yamlValue, opts types.Options) (any, error) {
	if yv == nil {
		return nil, fmt.Errorf("nil yamlValue")
	}

	// Get tensor factory from options, default to tensor.New wrapped
	tensorFactory := opts.TensorFactory
	if tensorFactory == nil {
		tensorFactory = func(dtype types.DataType, shape types.Shape) types.Tensor {
			return tensor.New(dtype, shape)
		}
	}

	switch yv.Kind {
	case "tensor":
		if yv.Tensor == nil {
			return nil, fmt.Errorf("nil tensor in yamlValue")
		}
		return yamlToTensor(yv.Tensor, opts.DestinationType, tensorFactory)

	case "layer":
		if yv.Layer == nil {
			return nil, fmt.Errorf("nil layer in yamlValue")
		}
		return *yv.Layer, nil

	case "model":
		if yv.Model == nil {
			return nil, fmt.Errorf("nil model in yamlValue")
		}
		return *yv.Model, nil

	case "graph", "tree", "decision_tree", "expression_graph", "generic":
		if yv.Graph == nil {
			return nil, fmt.Errorf("nil graph in yamlValue")
		}
		return yamlToGraph(yv.Graph, opts)

	case "slice":
		// For slices, YAML already decoded the data, but we need to convert types
		return convertSliceData(yv.SliceData, yv.SliceType)

	default:
		return nil, fmt.Errorf("unknown yamlValue kind: %s", yv.Kind)
	}
}

func yamlToTensor(yt *yamlTensor, destType types.DataType, tensorFactory func(types.DataType, types.Shape) types.Tensor) (types.Tensor, error) {
	if yt == nil || len(yt.Shape) == 0 {
		return nil, fmt.Errorf("invalid tensor data")
	}

	dtype := stringToDtype(yt.DataType)
	shape := types.Shape(yt.Shape)

	// Create tensor
	var t types.Tensor
	if destType != 0 {
		t = tensorFactory(destType, shape)
	} else {
		t = tensorFactory(dtype, shape)
	}

	// Copy data from YAML
	if dataSlice, ok := yt.Data.([]any); ok {
		for i, v := range dataSlice {
			if i >= t.Size() {
				break
			}
			var val float64
			switch v := v.(type) {
			case float64:
				val = v
			case float32:
				val = float64(v)
			case int:
				val = float64(v)
			case int64:
				val = float64(v)
			default:
				return nil, fmt.Errorf("unsupported data type in YAML: %T", v)
			}
			t.SetAt(val, i)
		}
	}

	return t, nil
}

func convertSliceData(data any, sliceType string) (any, error) {
	// YAML decodes numeric arrays as []any, we need to convert to proper type
	if data == nil {
		return nil, fmt.Errorf("nil slice data")
	}

	anySlice, ok := data.([]any)
	if !ok {
		// Data is already the correct type
		return data, nil
	}

	// Convert based on slice type
	switch sliceType {
	case "[]float32":
		result := make([]float32, len(anySlice))
		for i, v := range anySlice {
			switch v := v.(type) {
			case float64:
				result[i] = float32(v)
			case int:
				result[i] = float32(v)
			}
		}
		return result, nil
	case "[]float64":
		result := make([]float64, len(anySlice))
		for i, v := range anySlice {
			switch v := v.(type) {
			case float64:
				result[i] = v
			case int:
				result[i] = float64(v)
			}
		}
		return result, nil
	case "[]int":
		result := make([]int, len(anySlice))
		for i, v := range anySlice {
			switch v := v.(type) {
			case int:
				result[i] = v
			case float64:
				result[i] = int(v)
			}
		}
		return result, nil
	case "[]int64":
		result := make([]int64, len(anySlice))
		for i, v := range anySlice {
			switch v := v.(type) {
			case int64:
				result[i] = v
			case int:
				result[i] = int64(v)
			case float64:
				result[i] = int64(v)
			}
		}
		return result, nil
	case "[]int32":
		result := make([]int32, len(anySlice))
		for i, v := range anySlice {
			switch v := v.(type) {
			case int32:
				result[i] = v
			case int:
				result[i] = int32(v)
			case float64:
				result[i] = int32(v)
			}
		}
		return result, nil
	default:
		return data, nil
	}
}

func (u *Unmarshaller) assignToDst(dst any, value any) error {
	if dst == nil {
		return fmt.Errorf("dst is nil")
	}

	dstVal := reflect.ValueOf(dst)
	if dstVal.Kind() != reflect.Ptr {
		return fmt.Errorf("dst must be a pointer, got %s", dstVal.Kind())
	}

	dstElem := dstVal.Elem()
	if !dstElem.CanSet() {
		return fmt.Errorf("dst element cannot be set")
	}

	valueVal := reflect.ValueOf(value)
	if !valueVal.Type().AssignableTo(dstElem.Type()) {
		return fmt.Errorf("cannot assign %s to %s", valueVal.Type(), dstElem.Type())
	}

	dstElem.Set(valueVal)
	return nil
}

// yamlToGraph reconstructs a graph from YAML representation.
// Returns GenericGraph, GenericTree, GenericDecisionTree, or GenericExpressionGraph
// based on the metadata kind. Unmarshals into Generic* types with any for node and edge data.
func yamlToGraph(yg *yamlGraph, opts types.Options) (any, error) {
	if yg == nil {
		return nil, fmt.Errorf("nil yamlGraph")
	}

	kind := "graph"
	if yg.Metadata != nil {
		kind = yg.Metadata.Kind
	}

	switch kind {
	case "tree":
		return yamlToTree(yg)
	case "decision_tree":
		return yamlToDecisionTree(yg, opts)
	case "expression_graph":
		return yamlToExpressionGraph(yg, opts)
	default:
		return yamlToGenericGraph(yg)
	}
}

// tempYamlGraphNode is a temporary node implementation for graph reconstruction.
type tempYamlGraphNode[N any, E any] struct {
	id   int64
	data N
}

func (n *tempYamlGraphNode[N, E]) ID() int64 { return n.id }
func (n *tempYamlGraphNode[N, E]) Data() N   { return n.data }
func (n *tempYamlGraphNode[N, E]) Equal(other graph.Node[N, E]) bool {
	if other == nil {
		return false
	}
	return n.id == other.ID()
}
func (n *tempYamlGraphNode[N, E]) Compare(other graph.Node[N, E]) int {
	if other == nil {
		return 1
	}
	if n.id < other.ID() {
		return -1
	}
	if n.id > other.ID() {
		return 1
	}
	return 0
}
func (n *tempYamlGraphNode[N, E]) Neighbors() iter.Seq[graph.Node[N, E]] {
	return func(yield func(graph.Node[N, E]) bool) {}
}
func (n *tempYamlGraphNode[N, E]) Edges() iter.Seq[graph.Edge[N, E]] {
	return func(yield func(graph.Edge[N, E]) bool) {}
}
func (n *tempYamlGraphNode[N, E]) NumNeighbors() int                     { return 0 }
func (n *tempYamlGraphNode[N, E]) Cost(toOther graph.Node[N, E]) float32 { return 0 }

// tempYamlGraphEdge is a temporary edge implementation for graph reconstruction.
type tempYamlGraphEdge[N any, E any] struct {
	from graph.Node[N, E]
	to   graph.Node[N, E]
	data E
	id   int64
}

func (e *tempYamlGraphEdge[N, E]) ID() int64 {
	if e.id == 0 {
		e.id = e.from.ID()*1000000 + e.to.ID()
	}
	return e.id
}

func (e *tempYamlGraphEdge[N, E]) From() graph.Node[N, E] { return e.from }
func (e *tempYamlGraphEdge[N, E]) To() graph.Node[N, E]   { return e.to }
func (e *tempYamlGraphEdge[N, E]) Data() E                { return e.data }
func (e *tempYamlGraphEdge[N, E]) Cost() float32 {
	switch v := any(e.data).(type) {
	case float32:
		return v
	case float64:
		return float32(v)
	case int:
		return float32(v)
	case int64:
		return float32(v)
	default:
		return 0
	}
}

// yamlToGenericGraph reconstructs a GenericGraph[any, any].
func yamlToGenericGraph(yg *yamlGraph) (*graph.GenericGraph[any, any], error) {
	g := graph.NewGenericGraph[any, any]()

	// Create node map for quick lookup (stores the actual nodes from the graph)
	nodeMap := make(map[int64]graph.Node[any, any])

	// Add nodes
	for _, yn := range yg.Nodes {
		tempNode := &tempYamlGraphNode[any, any]{
			id:   yn.ID,
			data: yn.Data,
		}
		if err := g.AddNode(tempNode); err != nil {
			return nil, fmt.Errorf("failed to add node %d: %w", yn.ID, err)
		}
		// Retrieve the actual node from the graph
		for node := range g.Nodes() {
			if node.ID() == yn.ID {
				nodeMap[yn.ID] = node
				break
			}
		}
	}

	// Add edges
	for _, ye := range yg.Edges {
		fromNode, okFrom := nodeMap[ye.FromID]
		toNode, okTo := nodeMap[ye.ToID]
		if !okFrom || !okTo {
			continue // Skip invalid edges
		}

		// Create a temporary edge that implements Edge interface
		tempEdge := &tempYamlGraphEdge[any, any]{
			from: fromNode,
			to:   toNode,
			data: ye.Data,
		}
		if err := g.AddEdge(tempEdge); err != nil {
			return nil, fmt.Errorf("failed to add edge %d->%d: %w", ye.FromID, ye.ToID, err)
		}
	}

	return g, nil
}

// yamlToTree reconstructs a GenericTree[any, any].
func yamlToTree(yg *yamlGraph) (*graph.GenericTree[any, any], error) {
	if yg.Metadata == nil || yg.Metadata.RootID == 0 {
		return nil, fmt.Errorf("tree requires root ID in metadata")
	}

	// Find root node
	var rootData any
	for _, yn := range yg.Nodes {
		if yn.ID == yg.Metadata.RootID {
			rootData = yn.Data
			break
		}
	}
	if rootData == nil {
		return nil, fmt.Errorf("root node %d not found", yg.Metadata.RootID)
	}

	tree := graph.NewGenericTree[any, any](rootData)

	// Create node map
	nodeMap := make(map[int64]int) // ID -> index
	nodeMap[yg.Metadata.RootID] = tree.RootIdx()

	// Add child nodes (we need to build parent-child relationships from edges)
	// For trees, edges represent parent->child relationships
	for _, ye := range yg.Edges {
		parentIdx, ok := nodeMap[ye.FromID]
		if !ok {
			continue
		}

		// Find child data
		var childData any
		for _, yn := range yg.Nodes {
			if yn.ID == ye.ToID {
				childData = yn.Data
				break
			}
		}
		if childData == nil {
			continue
		}

		// Add child
		childIdx := tree.AddChild(parentIdx, childData)
		nodeMap[ye.ToID] = childIdx

		// Set edge cost if provided
		if ye.Data != nil {
			tree.SetCost(parentIdx, childIdx, ye.Data)
		}
	}

	return tree, nil
}

// yamlToDecisionTree reconstructs a GenericDecisionTree[any, any, any, any].
// Operations must be registered via options (similar to graph marshaller).
func yamlToDecisionTree(yg *yamlGraph, opts types.Options) (*graph.GenericDecisionTree[any, any, any, any], error) {
	if yg.Metadata == nil || yg.Metadata.RootID == 0 {
		return nil, fmt.Errorf("decision tree requires root ID in metadata")
	}

	// Find root node
	var rootData any
	for _, yn := range yg.Nodes {
		if yn.ID == yg.Metadata.RootID {
			rootData = yn.Data
			break
		}
	}
	if rootData == nil {
		return nil, fmt.Errorf("root node %d not found", yg.Metadata.RootID)
	}

	dt := graph.NewGenericDecisionTree[any, any, any, any](rootData)

	// Build tree structure first (same as yamlToTree)
	nodeMap := make(map[int64]int)
	nodeMap[yg.Metadata.RootID] = dt.RootIdx()

	for _, ye := range yg.Edges {
		parentIdx, ok := nodeMap[ye.FromID]
		if !ok {
			continue
		}

		var childData any
		for _, yn := range yg.Nodes {
			if yn.ID == ye.ToID {
				childData = yn.Data
				break
			}
		}
		if childData == nil {
			continue
		}

		childIdx := dt.AddChild(parentIdx, childData)
		nodeMap[ye.ToID] = childIdx

		if ye.Data != nil {
			dt.SetCost(parentIdx, childIdx, ye.Data)
		}
	}

	// Register operations from metadata (if available)
	// Operations should be registered via options similar to graph marshaller
	// For now, we'll leave this as a placeholder - operations need to be
	// registered via reflection-based options similar to graph marshaller

	return dt, nil
}

// yamlToExpressionGraph reconstructs a GenericExpressionGraph[any, any, any, any].
// Operations must be registered via options (similar to graph marshaller).
func yamlToExpressionGraph(yg *yamlGraph, opts types.Options) (*graph.GenericExpressionGraph[any, any, any, any], error) {
	eg := graph.NewGenericExpressionGraph[any, any, any, any]()

	// Create node map
	nodeMap := make(map[int64]graph.Node[any, any])

	// Add nodes
	for _, yn := range yg.Nodes {
		node, err := eg.AddNode(yn.Data, nil) // Operations registered separately
		if err != nil {
			return nil, fmt.Errorf("failed to add node %d: %w", yn.ID, err)
		}
		nodeMap[yn.ID] = node

		// Set as root if it's the expression root
		if yg.Metadata != nil && yg.Metadata.ExpressionRootID == yn.ID {
			eg.SetRoot(node)
		}
	}

	// Add edges
	for _, ye := range yg.Edges {
		fromNode, okFrom := nodeMap[ye.FromID]
		toNode, okTo := nodeMap[ye.ToID]
		if !okFrom || !okTo {
			continue
		}

		var edgeData any
		if ye.Data != nil {
			edgeData = ye.Data
		}
		if err := eg.AddEdge(fromNode, toNode, edgeData); err != nil {
			return nil, fmt.Errorf("failed to add edge %d->%d: %w", ye.FromID, ye.ToID, err)
		}
	}

	// Register operations from metadata (if available)
	// Similar to decision tree, operations need to be registered via options

	return eg, nil
}
