package json

import (
	"reflect"
	"strings"

	"github.com/itohio/EasyRobot/x/math/graph"
	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// Internal structs for JSON encoding/decoding.
// These structs provide proper JSON representation for domain types.

// jsonTensor represents a tensor for JSON encoding/decoding.
type jsonTensor struct {
	DataType string `json:"dtype"`
	Shape    []int  `json:"shape"`
	Data     any    `json:"data"` // JSON will handle type-specific encoding
}

// jsonParameter represents a parameter for JSON encoding/decoding.
type jsonParameter struct {
	Data         *jsonTensor `json:"data,omitempty"`
	Grad         *jsonTensor `json:"grad,omitempty"`
	RequiresGrad bool        `json:"requires_grad"`
}

// jsonLayer represents a layer for JSON encoding/decoding.
type jsonLayer struct {
	Name       string                   `json:"name"`
	Type       string                   `json:"type"`
	CanLearn   bool                     `json:"can_learn"`
	InputShape []int                    `json:"input_shape,omitempty"`
	Parameters map[string]jsonParameter `json:"parameters,omitempty"`
}

// jsonModel represents a model for JSON encoding/decoding.
type jsonModel struct {
	Name       string                   `json:"name"`
	Type       string                   `json:"type"`
	CanLearn   bool                     `json:"can_learn"`
	LayerCount int                      `json:"layer_count"`
	Layers     []jsonLayer              `json:"layers,omitempty"`
	Parameters map[string]jsonParameter `json:"parameters,omitempty"`
}

// jsonGraphNode represents a node in a graph for JSON encoding/decoding.
type jsonGraphNode struct {
	ID   int64 `json:"id"`
	Data any   `json:"data,omitempty"`
}

// jsonGraphEdge represents an edge in a graph for JSON encoding/decoding.
type jsonGraphEdge struct {
	FromID int64 `json:"from_id"`
	ToID   int64 `json:"to_id"`
	Data   any   `json:"data,omitempty"`
}

// jsonGraphMetadata represents graph metadata (kind, operations, root IDs).
type jsonGraphMetadata struct {
	Kind             string               `json:"kind"` // "graph", "tree", "decision_tree", "expression_graph"
	RootID           int64                `json:"root_id,omitempty"`
	TreeType         string               `json:"tree_type,omitempty"`
	NodeOps          map[int64]string     `json:"node_ops,omitempty"`           // For decision/expression trees
	EdgeOps          []jsonDecisionEdgeOp `json:"edge_ops,omitempty"`           // For decision trees
	ExpressionRootID int64                `json:"expression_root_id,omitempty"` // For expression graphs
}

// jsonDecisionEdgeOp represents a decision edge operation.
type jsonDecisionEdgeOp struct {
	ParentID int64  `json:"parent_id"`
	ChildID  int64  `json:"child_id"`
	OpName   string `json:"op_name"`
}

// jsonGraph represents a graph for JSON encoding/decoding.
type jsonGraph struct {
	Nodes    []jsonGraphNode    `json:"nodes"`
	Edges    []jsonGraphEdge    `json:"edges"`
	Metadata *jsonGraphMetadata `json:"metadata,omitempty"`
}

// jsonValue is a discriminated union for different value types.
type jsonValue struct {
	Kind string `json:"kind"` // "tensor", "layer", "model", "slice", "graph", "tree", "decision_tree", "expression_graph", "generic"

	// Type-specific fields
	Tensor *jsonTensor `json:"tensor,omitempty"`
	Layer  *jsonLayer  `json:"layer,omitempty"`
	Model  *jsonModel  `json:"model,omitempty"`
	Graph  *jsonGraph  `json:"graph,omitempty"`

	// For slices/arrays
	SliceType string `json:"slice_type,omitempty"`
	SliceData any    `json:"slice_data,omitempty"`
}

// Helper functions to convert between domain types and JSON structs

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

func tensorToJSON(t types.Tensor) *jsonTensor {
	if t.Empty() {
		return nil
	}

	return &jsonTensor{
		DataType: dtypeToString(t.DataType()),
		Shape:    t.Shape(),
		Data:     t.Data(),
	}
}

func parameterToJSON(p types.Parameter) jsonParameter {
	result := jsonParameter{
		RequiresGrad: p.RequiresGrad,
	}

	if !p.Data.Empty() {
		result.Data = tensorToJSON(p.Data)
	}
	if !p.Grad.Empty() {
		result.Grad = tensorToJSON(p.Grad)
	}

	return result
}

func layerToJSON(layer types.Layer) jsonLayer {
	result := jsonLayer{
		Name:       layer.Name(),
		CanLearn:   layer.CanLearn(),
		Parameters: make(map[string]jsonParameter),
	}

	// Get input shape if available
	input := layer.Input()
	if !input.Empty() {
		result.InputShape = input.Shape()
	}

	// Convert parameters
	params := layer.Parameters()
	for idx, param := range params {
		result.Parameters[string(rune(idx))] = parameterToJSON(param)
	}

	return result
}

func modelToJSON(model types.Model) jsonModel {
	result := jsonModel{
		Name:       model.Name(),
		CanLearn:   model.CanLearn(),
		LayerCount: model.LayerCount(),
		Layers:     make([]jsonLayer, 0),
		Parameters: make(map[string]jsonParameter),
	}

	// Convert layers
	for i := 0; i < model.LayerCount(); i++ {
		layer := model.GetLayer(i)
		if layer != nil {
			result.Layers = append(result.Layers, layerToJSON(layer))
		}
	}

	// Convert parameters
	params := model.Parameters()
	for idx, param := range params {
		result.Parameters[string(rune(idx))] = parameterToJSON(param)
	}

	return result
}

// graphToJSON converts a graph to JSON representation.
func graphToJSON(value any) (*jsonGraph, error) {
	// Use reflection for all graph types since Go doesn't have covariance
	return captureGraphViaReflection(value)
}


func captureGraphViaReflection(value any) (*jsonGraph, error) {
	// Try various GenericGraph concrete types
	switch g := value.(type) {
	case *graph.GenericGraph[any, any]:
		return captureGenericGraphPublicAPI(g)
	case *graph.GenericGraph[string, float32]:
		return captureGenericGraphStringFloat32(g)
	}

	// Try to handle other GenericGraph types via reflection
	val := reflect.ValueOf(value)
	if val.Kind() == reflect.Ptr {
		val = val.Elem()
	}

	typeName := val.Type().String()
	if strings.Contains(typeName, "GenericGraph") {
		// This is some form of GenericGraph, try to extract data
		return captureGenericGraphViaReflection(value)
	}

	// Fallback: capture metadata only
	metadata, err := captureGraphMetadata(value)
	if err != nil {
		return nil, err
	}

	return &jsonGraph{
		Nodes:    []jsonGraphNode{},
		Edges:    []jsonGraphEdge{},
		Metadata: metadata,
	}, nil
}

func captureGenericGraphStringFloat32(g *graph.GenericGraph[string, float32]) (*jsonGraph, error) {
	nodes := make([]jsonGraphNode, 0)
	edges := make([]jsonGraphEdge, 0)

	// Capture nodes using public API
	for node := range g.Nodes() {
		if node != nil {
			nodes = append(nodes, jsonGraphNode{
				ID:   node.ID(),
				Data: node.Data(),
			})
		}
	}

	// Capture edges using public API
	for edge := range g.Edges() {
		if edge != nil {
			from := edge.From()
			to := edge.To()
			if from != nil && to != nil {
				edges = append(edges, jsonGraphEdge{
					FromID: from.ID(),
					ToID:   to.ID(),
					Data:   edge.Data(),
				})
			}
		}
	}

	metadata, err := captureGraphMetadata(g)
	if err != nil {
		return nil, err
	}

	return &jsonGraph{
		Nodes:    nodes,
		Edges:    edges,
		Metadata: metadata,
	}, nil
}

func captureGenericGraphPublicAPI(g *graph.GenericGraph[any, any]) (*jsonGraph, error) {
	nodes := make([]jsonGraphNode, 0)
	edges := make([]jsonGraphEdge, 0)

	// Capture nodes using public API
	for node := range g.Nodes() {
		if node != nil {
			nodes = append(nodes, jsonGraphNode{
				ID:   node.ID(),
				Data: node.Data(),
			})
		}
	}

	// Capture edges using public API
	for edge := range g.Edges() {
		if edge != nil {
			from := edge.From()
			to := edge.To()
			if from != nil && to != nil {
				edges = append(edges, jsonGraphEdge{
					FromID: from.ID(),
					ToID:   to.ID(),
					Data:   edge.Data(),
				})
			}
		}
	}

	metadata, err := captureGraphMetadata(g)
	if err != nil {
		return nil, err
	}

	return &jsonGraph{
		Nodes:    nodes,
		Edges:    edges,
		Metadata: metadata,
	}, nil
}

func captureGenericGraphViaReflection(value any) (*jsonGraph, error) {
	// For GenericGraph[T, U] where T, U are not any, we need reflection
	// This is complex and for now, we'll return empty data
	metadata, err := captureGraphMetadata(value)
	if err != nil {
		return nil, err
	}

	return &jsonGraph{
		Nodes:    []jsonGraphNode{},
		Edges:    []jsonGraphEdge{},
		Metadata: metadata,
	}, nil
}

func captureGraphMetadata(value any) (*jsonGraphMetadata, error) {
	// Detect graph kind and capture metadata
	val := reflect.ValueOf(value)
	if val.Kind() == reflect.Ptr {
		val = val.Elem()
	}

	meta := &jsonGraphMetadata{}

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
