package json

import (
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"reflect"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/graph"
	"github.com/itohio/EasyRobot/x/math/tensor"
)

// Unmarshaller implements JSON-based unmarshalling.
type Unmarshaller struct {
	opts types.Options
}

// NewUnmarshaller creates a new JSON unmarshaller.
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
	return "json"
}

// Unmarshal decodes value from JSON format.
func (u *Unmarshaller) Unmarshal(r io.Reader, dst any, opts ...types.Option) error {
	// Apply additional options
	localOpts := u.opts
	for _, opt := range opts {
		opt.Apply(&localOpts)
	}

	decoder := json.NewDecoder(r)

	// Decode jsonValue
	var jv jsonValue
	if err := decoder.Decode(&jv); err != nil {
		return types.NewError("unmarshal", "json", "decoding", err)
	}

	// Convert jsonValue to actual value
	value, err := u.jsonToValue(&jv, localOpts)
	if err != nil {
		return types.NewError("unmarshal", "json", "conversion", err)
	}

	// Assign to dst
	if err := u.assignToDst(dst, value); err != nil {
		return types.NewError("unmarshal", "json", "assignment", err)
	}

	return nil
}

func (u *Unmarshaller) jsonToValue(jv *jsonValue, opts types.Options) (any, error) {
	if jv == nil {
		return nil, fmt.Errorf("nil jsonValue")
	}

	// Get tensor factory from options, default to tensor.New wrapped
	tensorFactory := opts.TensorFactory
	if tensorFactory == nil {
		tensorFactory = func(dtype types.DataType, shape types.Shape) types.Tensor {
			return tensor.New(dtype, shape)
		}
	}

	switch jv.Kind {
	case "tensor":
		if jv.Tensor == nil {
			return nil, fmt.Errorf("nil tensor in jsonValue")
		}
		return jsonToTensor(jv.Tensor, opts.DestinationType, tensorFactory)

	case "layer":
		if jv.Layer == nil {
			return nil, fmt.Errorf("nil layer in jsonValue")
		}
		// For layers, we cannot fully reconstruct them without type information
		return *jv.Layer, nil

	case "model":
		if jv.Model == nil {
			return nil, fmt.Errorf("nil model in jsonValue")
		}
		// For models, similar to layers, we cannot fully reconstruct them
		return *jv.Model, nil

	case "graph", "tree", "decision_tree", "expression_graph", "generic":
		if jv.Graph == nil {
			return nil, fmt.Errorf("nil graph in jsonValue")
		}
		return jsonToGraph(jv.Graph, opts)

	case "slice":
		// For slices, JSON already decoded the data, but we need to convert types
		return convertSliceData(jv.SliceData, jv.SliceType)

	default:
		return nil, fmt.Errorf("unknown jsonValue kind: %s", jv.Kind)
	}
}

func jsonToTensor(jt *jsonTensor, destType types.DataType, tensorFactory func(types.DataType, types.Shape) types.Tensor) (types.Tensor, error) {
	if jt == nil || len(jt.Shape) == 0 {
		return nil, fmt.Errorf("invalid tensor data")
	}

	dtype := stringToDtype(jt.DataType)
	shape := types.Shape(jt.Shape)

	// Create tensor
	var t types.Tensor
	if destType != 0 {
		// Create with destination type for conversion
		t = tensorFactory(destType, shape)
	} else {
		t = tensorFactory(dtype, shape)
	}

	// Copy data from JSON - need to handle type conversion
	// JSON decodes numbers as float64 by default
	if dataSlice, ok := jt.Data.([]any); ok {
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
				return nil, fmt.Errorf("unsupported data type in JSON: %T", v)
			}
			t.SetAt(val, i)
		}
	}

	return t, nil
}

func convertSliceData(data any, sliceType string) (any, error) {
	// JSON decodes numeric arrays as []any, we need to convert to proper type
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
			if f, ok := v.(float64); ok {
				result[i] = float32(f)
			}
		}
		return result, nil
	case "[]float64":
		result := make([]float64, len(anySlice))
		for i, v := range anySlice {
			if f, ok := v.(float64); ok {
				result[i] = f
			}
		}
		return result, nil
	case "[]int":
		result := make([]int, len(anySlice))
		for i, v := range anySlice {
			if f, ok := v.(float64); ok {
				result[i] = int(f)
			}
		}
		return result, nil
	case "[]int64":
		result := make([]int64, len(anySlice))
		for i, v := range anySlice {
			if f, ok := v.(float64); ok {
				result[i] = int64(f)
			}
		}
		return result, nil
	case "[]int32":
		result := make([]int32, len(anySlice))
		for i, v := range anySlice {
			if f, ok := v.(float64); ok {
				result[i] = int32(f)
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

	// dst should be a pointer
	dstVal := reflect.ValueOf(dst)
	if dstVal.Kind() != reflect.Ptr {
		return fmt.Errorf("dst must be a pointer, got %s", dstVal.Kind())
	}

	// Get the element the pointer points to
	dstElem := dstVal.Elem()
	if !dstElem.CanSet() {
		return fmt.Errorf("dst element cannot be set")
	}

	// Set the value
	valueVal := reflect.ValueOf(value)
	if !valueVal.Type().AssignableTo(dstElem.Type()) {
		return fmt.Errorf("cannot assign %s to %s", valueVal.Type(), dstElem.Type())
	}

	dstElem.Set(valueVal)
	return nil
}

// jsonToGraph reconstructs a graph from JSON representation.
// Returns GenericGraph, GenericTree, GenericDecisionTree, or GenericExpressionGraph
// based on the metadata kind. Unmarshals into Generic* types with any for node and edge data.
func jsonToGraph(jg *jsonGraph, opts types.Options) (any, error) {
	if jg == nil {
		return nil, fmt.Errorf("nil jsonGraph")
	}

	kind := "graph"
	if jg.Metadata != nil {
		kind = jg.Metadata.Kind
	}

	switch kind {
	case "tree":
		return jsonToTree(jg)
	case "decision_tree":
		return jsonToDecisionTree(jg, opts)
	case "expression_graph":
		return jsonToExpressionGraph(jg, opts)
	default:
		return jsonToGenericGraph(jg)
	}
}

// tempGraphNode is a temporary node implementation for graph reconstruction.
type tempGraphNode[N any, E any] struct {
	id   int64
	data N
}

func (n *tempGraphNode[N, E]) ID() int64 { return n.id }
func (n *tempGraphNode[N, E]) Data() N   { return n.data }
func (n *tempGraphNode[N, E]) Equal(other graph.Node[N, E]) bool {
	if other == nil {
		return false
	}
	return n.id == other.ID()
}
func (n *tempGraphNode[N, E]) Compare(other graph.Node[N, E]) int {
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
func (n *tempGraphNode[N, E]) Neighbors() iter.Seq[graph.Node[N, E]] {
	return func(yield func(graph.Node[N, E]) bool) {}
}
func (n *tempGraphNode[N, E]) Edges() iter.Seq[graph.Edge[N, E]] {
	return func(yield func(graph.Edge[N, E]) bool) {}
}
func (n *tempGraphNode[N, E]) NumNeighbors() int                     { return 0 }
func (n *tempGraphNode[N, E]) Cost(toOther graph.Node[N, E]) float32 { return 0 }

// jsonToGenericGraph reconstructs a GenericGraph[any, any].
func jsonToGenericGraph(jg *jsonGraph) (*graph.GenericGraph[any, any], error) {
	g := graph.NewGenericGraph[any, any]()

	// Create node map for quick lookup (stores the actual nodes from the graph)
	nodeMap := make(map[int64]graph.Node[any, any])

	// Add nodes
	for _, jn := range jg.Nodes {
		tempNode := &tempGraphNode[any, any]{
			id:   jn.ID,
			data: jn.Data,
		}
		if err := g.AddNode(tempNode); err != nil {
			return nil, fmt.Errorf("failed to add node %d: %w", jn.ID, err)
		}
		// Retrieve the actual node from the graph
		for node := range g.Nodes() {
			if node.ID() == jn.ID {
				nodeMap[jn.ID] = node
				break
			}
		}
	}

	// Add edges
	for _, je := range jg.Edges {
		fromNode, okFrom := nodeMap[je.FromID]
		toNode, okTo := nodeMap[je.ToID]
		if !okFrom || !okTo {
			continue // Skip invalid edges
		}

		// Create a temporary edge that implements Edge interface
		tempEdge := &tempGraphEdge[any, any]{
			from: fromNode,
			to:   toNode,
			data: je.Data,
		}
		if err := g.AddEdge(tempEdge); err != nil {
			return nil, fmt.Errorf("failed to add edge %d->%d: %w", je.FromID, je.ToID, err)
		}
	}

	return g, nil
}

// tempGraphEdge is a temporary edge implementation for graph reconstruction.
type tempGraphEdge[N any, E any] struct {
	from graph.Node[N, E]
	to   graph.Node[N, E]
	data E
	id   int64
}

func (e *tempGraphEdge[N, E]) ID() int64 {
	if e.id == 0 {
		e.id = e.from.ID()*1000000 + e.to.ID()
	}
	return e.id
}

func (e *tempGraphEdge[N, E]) From() graph.Node[N, E] { return e.from }
func (e *tempGraphEdge[N, E]) To() graph.Node[N, E]   { return e.to }
func (e *tempGraphEdge[N, E]) Data() E                { return e.data }
func (e *tempGraphEdge[N, E]) Cost() float32 {
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

// jsonToTree reconstructs a GenericTree[any, any].
func jsonToTree(jg *jsonGraph) (*graph.GenericTree[any, any], error) {
	if jg.Metadata == nil || jg.Metadata.RootID == 0 {
		return nil, fmt.Errorf("tree requires root ID in metadata")
	}

	// Find root node
	var rootData any
	for _, jn := range jg.Nodes {
		if jn.ID == jg.Metadata.RootID {
			rootData = jn.Data
			break
		}
	}
	if rootData == nil {
		return nil, fmt.Errorf("root node %d not found", jg.Metadata.RootID)
	}

	tree := graph.NewGenericTree[any, any](rootData)

	// Create node map
	nodeMap := make(map[int64]int) // ID -> index
	nodeMap[jg.Metadata.RootID] = tree.RootIdx()

	// Add child nodes (we need to build parent-child relationships from edges)
	// For trees, edges represent parent->child relationships
	for _, je := range jg.Edges {
		parentIdx, ok := nodeMap[je.FromID]
		if !ok {
			// Parent not yet added, find its data
			var parentData any
			for _, jn := range jg.Nodes {
				if jn.ID == je.FromID {
					parentData = jn.Data
					break
				}
			}
			if parentData == nil {
				continue
			}
			// This shouldn't happen in a valid tree, but handle it
			continue
		}

		// Find child data
		var childData any
		for _, jn := range jg.Nodes {
			if jn.ID == je.ToID {
				childData = jn.Data
				break
			}
		}
		if childData == nil {
			continue
		}

		// Add child
		childIdx := tree.AddChild(parentIdx, childData)
		nodeMap[je.ToID] = childIdx

		// Set edge cost if provided
		if je.Data != nil {
			tree.SetCost(parentIdx, childIdx, je.Data)
		}
	}

	return tree, nil
}

// jsonToDecisionTree reconstructs a GenericDecisionTree[any, any, any, any].
// Operations must be registered via options (similar to graph marshaller).
func jsonToDecisionTree(jg *jsonGraph, opts types.Options) (*graph.GenericDecisionTree[any, any, any, any], error) {
	if jg.Metadata == nil || jg.Metadata.RootID == 0 {
		return nil, fmt.Errorf("decision tree requires root ID in metadata")
	}

	// Find root node
	var rootData any
	for _, jn := range jg.Nodes {
		if jn.ID == jg.Metadata.RootID {
			rootData = jn.Data
			break
		}
	}
	if rootData == nil {
		return nil, fmt.Errorf("root node %d not found", jg.Metadata.RootID)
	}

	dt := graph.NewGenericDecisionTree[any, any, any, any](rootData)

	// Build tree structure first (same as jsonToTree)
	nodeMap := make(map[int64]int)
	nodeMap[jg.Metadata.RootID] = dt.RootIdx()

	for _, je := range jg.Edges {
		parentIdx, ok := nodeMap[je.FromID]
		if !ok {
			continue
		}

		var childData any
		for _, jn := range jg.Nodes {
			if jn.ID == je.ToID {
				childData = jn.Data
				break
			}
		}
		if childData == nil {
			continue
		}

		childIdx := dt.AddChild(parentIdx, childData)
		nodeMap[je.ToID] = childIdx

		if je.Data != nil {
			dt.SetCost(parentIdx, childIdx, je.Data)
		}
	}

	// Register operations from metadata (if available)
	// Operations should be registered via options similar to graph marshaller
	// For now, we'll leave this as a placeholder - operations need to be
	// registered via reflection-based options similar to graph marshaller

	return dt, nil
}

// jsonToExpressionGraph reconstructs a GenericExpressionGraph[any, any, any, any].
// Operations must be registered via options (similar to graph marshaller).
func jsonToExpressionGraph(jg *jsonGraph, opts types.Options) (*graph.GenericExpressionGraph[any, any, any, any], error) {
	eg := graph.NewGenericExpressionGraph[any, any, any, any]()

	// Create node map
	nodeMap := make(map[int64]graph.Node[any, any])

	// Add nodes
	for _, jn := range jg.Nodes {
		node, err := eg.AddNode(jn.Data, nil) // Operations registered separately
		if err != nil {
			return nil, fmt.Errorf("failed to add node %d: %w", jn.ID, err)
		}
		nodeMap[jn.ID] = node

		// Set as root if it's the expression root
		if jg.Metadata != nil && jg.Metadata.ExpressionRootID == jn.ID {
			eg.SetRoot(node)
		}
	}

	// Add edges
	for _, je := range jg.Edges {
		fromNode, okFrom := nodeMap[je.FromID]
		toNode, okTo := nodeMap[je.ToID]
		if !okFrom || !okTo {
			continue
		}

		var edgeData any
		if je.Data != nil {
			edgeData = je.Data
		}
		if err := eg.AddEdge(fromNode, toNode, edgeData); err != nil {
			return nil, fmt.Errorf("failed to add edge %d->%d: %w", je.FromID, je.ToID, err)
		}
	}

	// Register operations from metadata (if available)
	// Similar to decision tree, operations need to be registered via options

	return eg, nil
}
