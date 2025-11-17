package graph

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

type graphKind string

const (
	graphKindGeneric         graphKind = "graph"
	graphKindTree            graphKind = "tree"
	graphKindDecisionTree    graphKind = "decision_tree"
	graphKindExpressionGraph graphKind = "expression_graph"

	metadataTypeName = "__graph_metadata__"
)

func graphKindToByte(kind graphKind) byte {
	switch kind {
	case graphKindTree:
		return 1
	case graphKindDecisionTree:
		return 2
	case graphKindExpressionGraph:
		return 3
	default:
		return 0
	}
}

func graphKindFromByte(code byte) graphKind {
	switch code {
	case 1:
		return graphKindTree
	case 2:
		return graphKindDecisionTree
	case 3:
		return graphKindExpressionGraph
	default:
		return graphKindGeneric
	}
}

type graphMetadata struct {
	Kind       graphKind           `json:"kind"`
	RootID     int64               `json:"rootId,omitempty"`
	TreeType   string              `json:"treeType,omitempty"`
	Decision   *decisionMetadata   `json:"decision,omitempty"`
	Expression *expressionMetadata `json:"expression,omitempty"`
}

type decisionMetadata struct {
	NodeOps map[int64]string  `json:"nodeOps,omitempty"`
	EdgeOps map[string]string `json:"edgeOps,omitempty"`
}

type expressionMetadata struct {
	NodeOps map[int64]string `json:"nodeOps,omitempty"`
	RootID  int64            `json:"rootId,omitempty"`
}

func buildGraphMetadata(value any, nodes []capturedNode, edges []capturedEdge) (graphKind, *graphMetadata, error) {
	if value == nil {
		return graphKindGeneric, nil, fmt.Errorf("graph value is nil")
	}
	graphVal := reflect.ValueOf(value)
	kind := detectGraphKind(graphVal)
	meta := &graphMetadata{Kind: kind}

	switch kind {
	case graphKindDecisionTree:
		rootID, err := captureRootID(graphVal)
		if err != nil {
			return kind, nil, err
		}
		meta.RootID = rootID
		meta.TreeType = detectTreeType(graphVal)
		decisionMeta, err := captureDecisionMetadata(graphVal)
		if err != nil {
			return kind, nil, err
		}
		meta.Decision = decisionMeta
	case graphKindTree:
		rootID, err := captureRootID(graphVal)
		if err != nil {
			return kind, nil, err
		}
		meta.RootID = rootID
		meta.TreeType = detectTreeType(graphVal)
	case graphKindExpressionGraph:
		exprMeta, err := captureExpressionMetadata(graphVal, nodes, edges)
		if err != nil {
			return kind, nil, err
		}
		meta.Expression = exprMeta
		if exprMeta != nil {
			meta.RootID = exprMeta.RootID
		}
	default:
		// Generic graph needs no additional metadata beyond kind
	}

	return kind, meta, nil
}

func detectGraphKind(graphVal reflect.Value) graphKind {
	if hasMethod(graphVal, "Decide") {
		return graphKindDecisionTree
	}
	if hasMethod(graphVal, "Compute") {
		return graphKindExpressionGraph
	}
	if hasMethod(graphVal, "Root") {
		return graphKindTree
	}
	return graphKindGeneric
}

func concreteValue(val reflect.Value) reflect.Value {
	if val.Kind() == reflect.Interface && !val.IsNil() {
		return val.Elem()
	}
	return val
}
func hasMethod(val reflect.Value, name string) bool {
	return val.MethodByName(name).IsValid()
}

func captureRootID(graphVal reflect.Value) (int64, error) {
	rootMethod := graphVal.MethodByName("Root")
	if !rootMethod.IsValid() {
		return 0, fmt.Errorf("graph does not expose Root method")
	}
	results := rootMethod.Call(nil)
	if len(results) != 1 {
		return 0, fmt.Errorf("unexpected Root signature")
	}
	rootVal := results[0]
	if !rootVal.IsValid() || rootVal.IsNil() {
		return 0, nil
	}
	return callInt64Method(rootVal, "ID")
}

func detectTreeType(graphVal reflect.Value) string {
	t := graphVal.Type()
	if t == nil {
		return "generic"
	}
	name := t.String()
	if name == "" {
		return "generic"
	}
	if strings.Contains(strings.ToLower(name), "binary") {
		return "binary"
	}
	return "generic"
}

func captureDecisionMetadata(graphVal reflect.Value) (*decisionMetadata, error) {
	meta := &decisionMetadata{
		NodeOps: make(map[int64]string),
		EdgeOps: make(map[string]string),
	}
	if err := visitNodes(graphVal, func(nodeVal reflect.Value, id int64) error {
		nodeVal = concreteValue(nodeVal)
		method := nodeVal.MethodByName("DecisionFunction")
		if !method.IsValid() {
			return nil
		}
		fnResults := method.Call(nil)
		if len(fnResults) != 1 {
			return nil
		}
		fnVal := fnResults[0]
		if !fnVal.IsValid() || fnVal.IsNil() || fnVal.Kind() != reflect.Func {
			return nil
		}
		name, err := functionName(fnVal)
		if err != nil {
			return err
		}
		meta.NodeOps[id] = name
		return nil
	}); err != nil {
		return nil, err
	}

	if err := visitEdges(graphVal, func(edgeVal reflect.Value, fromID, toID int64, _ any) error {
		edgeVal = concreteValue(edgeVal)
		method := edgeVal.MethodByName("CriteriaFunction")
		if !method.IsValid() {
			return nil
		}
		fnResults := method.Call(nil)
		if len(fnResults) != 1 {
			return nil
		}
		fnVal := fnResults[0]
		if !fnVal.IsValid() || fnVal.IsNil() || fnVal.Kind() != reflect.Func {
			return nil
		}
		name, err := functionName(fnVal)
		if err != nil {
			return err
		}
		key := edgeKey(fromID, toID)
		meta.EdgeOps[key] = name
		return nil
	}); err != nil {
		return nil, err
	}

	if len(meta.NodeOps) == 0 && len(meta.EdgeOps) == 0 {
		return nil, nil
	}
	return meta, nil
}

func captureExpressionMetadata(graphVal reflect.Value, nodes []capturedNode, edges []capturedEdge) (*expressionMetadata, error) {
	meta := &expressionMetadata{
		NodeOps: make(map[int64]string),
	}
	if err := visitNodes(graphVal, func(nodeVal reflect.Value, id int64) error {
		nodeVal = concreteValue(nodeVal)
		method := nodeVal.MethodByName("ExpressionFunction")
		if !method.IsValid() {
			return nil
		}
		fnResults := method.Call(nil)
		if len(fnResults) != 1 {
			return nil
		}
		fnVal := fnResults[0]
		if !fnVal.IsValid() || fnVal.IsNil() || fnVal.Kind() != reflect.Func {
			return nil
		}
		name, err := functionName(fnVal)
		if err != nil {
			return err
		}
		meta.NodeOps[id] = name
		return nil
	}); err != nil {
		return nil, err
	}

	if rootID := deriveRootID(nodes, edges); rootID != 0 {
		meta.RootID = rootID
	}

	if len(meta.NodeOps) == 0 && meta.RootID == 0 {
		return nil, nil
	}
	return meta, nil
}

func deriveRootID(nodes []capturedNode, edges []capturedEdge) int64 {
	if len(nodes) == 0 {
		return 0
	}
	incoming := make(map[int64]int)
	for _, node := range nodes {
		if _, ok := incoming[node.id]; !ok {
			incoming[node.id] = 0
		}
	}
	for _, edge := range edges {
		incoming[edge.toID]++
		if _, ok := incoming[edge.fromID]; !ok {
			incoming[edge.fromID] = 0
		}
	}
	for _, node := range nodes {
		if incoming[node.id] == 0 {
			return node.id
		}
	}
	return nodes[0].id
}

func serializeGraphMetadata(meta *graphMetadata) ([]byte, error) {
	if meta == nil {
		return nil, nil
	}
	return json.Marshal(meta)
}

func readGraphMetadata(storage types.MappedStorage, offset uint64) (*graphMetadata, error) {
	if offset == 0 {
		return nil, nil
	}
	dataType, typeName, payload, err := readDataEntry(storage, offset)
	if err != nil {
		return nil, err
	}
	if typeName != metadataTypeName {
		return nil, fmt.Errorf("unexpected metadata entry %q", typeName)
	}
	if dataType != DataTypeBytes {
		return nil, fmt.Errorf("invalid metadata data type: %d", dataType)
	}
	return parseGraphMetadata(payload)
}

func parseGraphMetadata(data []byte) (*graphMetadata, error) {
	if len(data) == 0 {
		return nil, nil
	}
	var meta graphMetadata
	if err := json.Unmarshal(data, &meta); err != nil {
		return nil, err
	}
	return &meta, nil
}
