package graph

import (
	"fmt"
	"reflect"
	"strings"

	marshalpb "github.com/itohio/EasyRobot/types/marshaller"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"google.golang.org/protobuf/proto"
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

func protoKindFromGraphKind(kind graphKind) marshalpb.GraphKind {
	switch kind {
	case graphKindTree:
		return marshalpb.GraphKind_GRAPH_KIND_TREE
	case graphKindDecisionTree:
		return marshalpb.GraphKind_GRAPH_KIND_DECISION_TREE
	case graphKindExpressionGraph:
		return marshalpb.GraphKind_GRAPH_KIND_EXPRESSION_GRAPH
	default:
		return marshalpb.GraphKind_GRAPH_KIND_GENERIC
	}
}

func graphKindFromProtoKind(kind marshalpb.GraphKind) graphKind {
	switch kind {
	case marshalpb.GraphKind_GRAPH_KIND_TREE:
		return graphKindTree
	case marshalpb.GraphKind_GRAPH_KIND_DECISION_TREE:
		return graphKindDecisionTree
	case marshalpb.GraphKind_GRAPH_KIND_EXPRESSION_GRAPH:
		return graphKindExpressionGraph
	default:
		return graphKindGeneric
	}
}

func buildGraphMetadata(value any, nodes []capturedNode, edges []capturedEdge) (graphKind, *marshalpb.GraphMetadata, error) {
	if value == nil {
		return graphKindGeneric, nil, fmt.Errorf("graph value is nil")
	}
	graphVal := reflect.ValueOf(value)
	kind := detectGraphKind(graphVal)
	meta := &marshalpb.GraphMetadata{
		Kind: protoKindFromGraphKind(kind),
	}

	switch kind {
	case graphKindDecisionTree:
		rootID, err := captureRootID(graphVal)
		if err != nil {
			return kind, nil, err
		}
		meta.RootId = rootID
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
		meta.RootId = rootID
		meta.TreeType = detectTreeType(graphVal)
	case graphKindExpressionGraph:
		exprMeta, err := captureExpressionMetadata(graphVal, nodes, edges)
		if err != nil {
			return kind, nil, err
		}
		meta.Expression = exprMeta
		if exprMeta != nil {
			meta.RootId = exprMeta.RootId
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

func captureDecisionMetadata(graphVal reflect.Value) (*marshalpb.DecisionMetadata, error) {
	meta := &marshalpb.DecisionMetadata{
		NodeOps: make(map[int64]string),
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

	edgeOps := make([]*marshalpb.DecisionEdgeOp, 0)
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
		edgeOps = append(edgeOps, &marshalpb.DecisionEdgeOp{
			ParentId: fromID,
			ChildId:  toID,
			OpName:   name,
		})
		return nil
	}); err != nil {
		return nil, err
	}
	if len(edgeOps) > 0 {
		meta.EdgeOps = edgeOps
	}

	if len(meta.NodeOps) == 0 && len(meta.EdgeOps) == 0 {
		return nil, nil
	}
	return meta, nil
}

func captureExpressionMetadata(graphVal reflect.Value, nodes []capturedNode, edges []capturedEdge) (*marshalpb.ExpressionMetadata, error) {
	meta := &marshalpb.ExpressionMetadata{
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
		meta.RootId = rootID
	}

	if len(meta.NodeOps) == 0 && meta.RootId == 0 {
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

func serializeGraphMetadata(meta *marshalpb.GraphMetadata) ([]byte, error) {
	if meta == nil {
		return nil, nil
	}
	return proto.Marshal(meta)
}

func readGraphMetadata(storage types.MappedStorage, offset uint64) (*marshalpb.GraphMetadata, error) {
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
	if dataType != DataTypeProtobuf {
		return nil, fmt.Errorf("invalid metadata data type: %d", dataType)
	}
	return parseGraphMetadata(payload)
}

func parseGraphMetadata(data []byte) (*marshalpb.GraphMetadata, error) {
	if len(data) == 0 {
		return nil, nil
	}
	meta := &marshalpb.GraphMetadata{}
	if err := proto.Unmarshal(data, meta); err != nil {
		return nil, err
	}
	return meta, nil
}
