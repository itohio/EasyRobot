package yaml

import (
	"fmt"
	"reflect"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/graph"
)

// Internal structs for YAML encoding/decoding.
// YAML uses same structure as JSON since YAML is a superset of JSON.

// yamlTensor represents a tensor for YAML encoding/decoding.
type yamlTensor struct {
	DataType string `yaml:"dtype"`
	Shape    []int  `yaml:"shape"`
	Data     any    `yaml:"data"`
}

// yamlParameter represents a parameter for YAML encoding/decoding.
type yamlParameter struct {
	Data         *yamlTensor `yaml:"data,omitempty"`
	Grad         *yamlTensor `yaml:"grad,omitempty"`
	RequiresGrad bool        `yaml:"requires_grad"`
}

// yamlLayer represents a layer for YAML encoding/decoding.
type yamlLayer struct {
	Name       string                   `yaml:"name"`
	Type       string                   `yaml:"type"`
	CanLearn   bool                     `yaml:"can_learn"`
	InputShape []int                    `yaml:"input_shape,omitempty"`
	Parameters map[string]yamlParameter `yaml:"parameters,omitempty"`
}

// yamlModel represents a model for YAML encoding/decoding.
type yamlModel struct {
	Name       string                   `yaml:"name"`
	Type       string                   `yaml:"type"`
	CanLearn   bool                     `yaml:"can_learn"`
	LayerCount int                      `yaml:"layer_count"`
	Layers     []yamlLayer              `yaml:"layers,omitempty"`
	Parameters map[string]yamlParameter `yaml:"parameters,omitempty"`
}

// yamlGraphNode represents a node in a graph for YAML encoding/decoding.
type yamlGraphNode struct {
	ID   int64 `yaml:"id"`
	Data any   `yaml:"data,omitempty"`
}

// yamlGraphEdge represents an edge in a graph for YAML encoding/decoding.
type yamlGraphEdge struct {
	FromID int64 `yaml:"from_id"`
	ToID   int64 `yaml:"to_id"`
	Data   any   `yaml:"data,omitempty"`
}

// yamlGraphMetadata represents graph metadata (kind, operations, root IDs).
type yamlGraphMetadata struct {
	Kind             string               `yaml:"kind"` // "graph", "tree", "decision_tree", "expression_graph"
	RootID           int64                `yaml:"root_id,omitempty"`
	TreeType         string               `yaml:"tree_type,omitempty"`
	NodeOps          map[int64]string     `yaml:"node_ops,omitempty"`           // For decision/expression trees
	EdgeOps          []yamlDecisionEdgeOp `yaml:"edge_ops,omitempty"`           // For decision trees
	ExpressionRootID int64                `yaml:"expression_root_id,omitempty"` // For expression graphs
}

// yamlDecisionEdgeOp represents a decision edge operation.
type yamlDecisionEdgeOp struct {
	ParentID int64  `yaml:"parent_id"`
	ChildID  int64  `yaml:"child_id"`
	OpName   string `yaml:"op_name"`
}

// yamlGraph represents a graph for YAML encoding/decoding.
type yamlGraph struct {
	Nodes    []yamlGraphNode    `yaml:"nodes"`
	Edges    []yamlGraphEdge    `yaml:"edges"`
	Metadata *yamlGraphMetadata `yaml:"metadata,omitempty"`
}

// yamlValue is a discriminated union for different value types.
type yamlValue struct {
	Kind string `yaml:"kind"` // "tensor", "layer", "model", "slice", "graph", "tree", "decision_tree", "expression_graph", "generic"

	// Type-specific fields
	Tensor *yamlTensor `yaml:"tensor,omitempty"`
	Layer  *yamlLayer  `yaml:"layer,omitempty"`
	Model  *yamlModel  `yaml:"model,omitempty"`
	Graph  *yamlGraph  `yaml:"graph,omitempty"`

	// For slices/arrays
	SliceType string `yaml:"slice_type,omitempty"`
	SliceData any    `yaml:"slice_data,omitempty"`
}

// Helper functions - reuse from common logic

func dtypeToString(dt types.DataType) string {
	switch dt {
	case types.FP32:
		return "fp32"
	case types.FP64:
		return "fp64"
	case types.INT8:
		return "int8"
	case types.INT16:
		return "int16"
	case types.INT32:
		return "int32"
	case types.INT64:
		return "int64"
	case types.INT:
		return "int"
	case types.FP16:
		return "fp16"
	case types.INT48:
		return "int48"
	default:
		return "unknown"
	}
}

func stringToDtype(s string) types.DataType {
	switch s {
	case "fp32":
		return types.FP32
	case "fp64":
		return types.FP64
	case "int8":
		return types.INT8
	case "int16":
		return types.INT16
	case "int32":
		return types.INT32
	case "int64":
		return types.INT64
	case "int":
		return types.INT
	case "fp16":
		return types.FP16
	case "int48":
		return types.INT48
	default:
		return types.DT_UNKNOWN
	}
}

func tensorToYAML(t types.Tensor) *yamlTensor {
	if t.Empty() {
		return nil
	}

	return &yamlTensor{
		DataType: dtypeToString(t.DataType()),
		Shape:    t.Shape(),
		Data:     t.Data(),
	}
}

func parameterToYAML(p types.Parameter) yamlParameter {
	result := yamlParameter{
		RequiresGrad: p.RequiresGrad,
	}

	if !p.Data.Empty() {
		result.Data = tensorToYAML(p.Data)
	}
	if !p.Grad.Empty() {
		result.Grad = tensorToYAML(p.Grad)
	}

	return result
}

func layerToYAML(layer types.Layer) yamlLayer {
	result := yamlLayer{
		Name:       layer.Name(),
		CanLearn:   layer.CanLearn(),
		Parameters: make(map[string]yamlParameter),
	}

	// Get input shape if available
	input := layer.Input()
	if !input.Empty() {
		result.InputShape = input.Shape()
	}

	// Convert parameters
	params := layer.Parameters()
	for idx, param := range params {
		result.Parameters[string(rune(idx))] = parameterToYAML(param)
	}

	return result
}

func modelToYAML(model types.Model) yamlModel {
	result := yamlModel{
		Name:       model.Name(),
		CanLearn:   model.CanLearn(),
		LayerCount: model.LayerCount(),
		Layers:     make([]yamlLayer, 0),
		Parameters: make(map[string]yamlParameter),
	}

	// Convert layers
	for i := 0; i < model.LayerCount(); i++ {
		layer := model.GetLayer(i)
		if layer != nil {
			result.Layers = append(result.Layers, layerToYAML(layer))
		}
	}

	// Convert parameters
	params := model.Parameters()
	for idx, param := range params {
		result.Parameters[string(rune(idx))] = parameterToYAML(param)
	}

	return result
}

// graphToYAML converts a graph to YAML representation.
func graphToYAML(value any) (*yamlGraph, error) {
	// Check if it's a graph interface
	var g graph.Graph[any, any]
	switch v := value.(type) {
	case graph.Graph[any, any]:
		g = v
	default:
		// Try to detect via reflection
		return captureGraphViaReflection(value)
	}

	// Capture nodes and edges
	nodes := make([]yamlGraphNode, 0)
	edges := make([]yamlGraphEdge, 0)

	// Iterate nodes
	for node := range g.Nodes() {
		if node == nil {
			continue
		}
		nodes = append(nodes, yamlGraphNode{
			ID:   node.ID(),
			Data: node.Data(),
		})
	}

	// Iterate edges
	for edge := range g.Edges() {
		if edge == nil {
			continue
		}
		from := edge.From()
		to := edge.To()
		if from == nil || to == nil {
			continue
		}
		edges = append(edges, yamlGraphEdge{
			FromID: from.ID(),
			ToID:   to.ID(),
			Data:   edge.Data(),
		})
	}

	// Capture metadata
	metadata, err := captureGraphMetadata(value)
	if err != nil {
		return nil, err
	}

	return &yamlGraph{
		Nodes:    nodes,
		Edges:    edges,
		Metadata: metadata,
	}, nil
}

func captureGraphViaReflection(value any) (*yamlGraph, error) {
	val := reflect.ValueOf(value)
	if val.Kind() == reflect.Ptr {
		val = val.Elem()
	}

	// Check if it has Nodes() and Edges() methods
	nodesMethod := val.MethodByName("Nodes")
	edgesMethod := val.MethodByName("Edges")
	if !nodesMethod.IsValid() || !edgesMethod.IsValid() {
		return nil, fmt.Errorf("value does not implement graph.Graph")
	}

	// Capture nodes
	nodes := make([]yamlGraphNode, 0)
	seqNodes := nodesMethod.Call(nil)
	if len(seqNodes) == 1 {
		// Iterate sequence
		seqVal := seqNodes[0]
		if seqVal.Kind() == reflect.Func {
			// This is an iter.Seq - we need to call it with a callback
			// For now, return error - we'll handle this properly in marshaller
			return nil, fmt.Errorf("reflection-based graph capture requires graph.Graph interface")
		}
	}

	// Capture edges
	edges := make([]yamlGraphEdge, 0)
	seqEdges := edgesMethod.Call(nil)
	if len(seqEdges) == 1 {
		// Similar to nodes
	}

	// Capture metadata
	metadata, err := captureGraphMetadata(value)
	if err != nil {
		return nil, err
	}

	return &yamlGraph{
		Nodes:    nodes,
		Edges:    edges,
		Metadata: metadata,
	}, nil
}

func captureGraphMetadata(value any) (*yamlGraphMetadata, error) {
	// Detect graph kind and capture metadata
	val := reflect.ValueOf(value)
	if val.Kind() == reflect.Ptr {
		val = val.Elem()
	}

	meta := &yamlGraphMetadata{}

	// Detect kind
	if hasMethod(val, "Decide") {
		meta.Kind = "decision_tree"
	} else if hasMethod(val, "Compute") {
		meta.Kind = "expression_graph"
	} else if hasMethod(val, "Root") {
		meta.Kind = "tree"
	} else {
		meta.Kind = "graph"
	}

	// Capture root ID if tree
	if meta.Kind == "tree" || meta.Kind == "decision_tree" {
		rootMethod := val.MethodByName("Root")
		if rootMethod.IsValid() {
			results := rootMethod.Call(nil)
			if len(results) == 1 && !results[0].IsNil() {
				rootVal := results[0]
				idMethod := rootVal.MethodByName("ID")
				if idMethod.IsValid() {
					idResults := idMethod.Call(nil)
					if len(idResults) == 1 {
						meta.RootID = idResults[0].Int()
					}
				}
			}
		}
	}

	// Capture operations for decision/expression trees
	// This would require more complex reflection similar to graph marshaller
	// For now, we'll leave it empty and let unmarshaller handle it via options

	return meta, nil
}

func hasMethod(val reflect.Value, name string) bool {
	return val.MethodByName(name).IsValid()
}
