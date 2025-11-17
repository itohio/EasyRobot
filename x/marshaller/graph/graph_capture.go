package graph

import (
	"fmt"
	"reflect"
)

type capturedNode struct {
	id int64
}

type capturedEdge struct {
	fromID int64
	toID   int64
	data   any
}

func captureGraph(value any) ([]capturedNode, []capturedEdge, error) {
	if value == nil {
		return nil, nil, fmt.Errorf("graph value is nil")
	}
	val := reflect.ValueOf(value)

	nodes, err := captureNodes(val)
	if err != nil {
		return nil, nil, err
	}

	edges, err := captureEdges(val)
	if err != nil {
		return nil, nil, err
	}

	return nodes, edges, nil
}

func captureNodes(graphVal reflect.Value) ([]capturedNode, error) {
	var result []capturedNode
	err := visitNodes(graphVal, func(nodeVal reflect.Value, id int64) error {
		result = append(result, capturedNode{id: id})
		return nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

func captureEdges(graphVal reflect.Value) ([]capturedEdge, error) {
	var result []capturedEdge
	err := visitEdges(graphVal, func(edgeVal reflect.Value, fromID, toID int64, data any) error {
		result = append(result, capturedEdge{fromID: fromID, toID: toID, data: data})
		return nil
	})
	if err != nil {
		return nil, err
	}
	return result, nil
}

func visitNodes(graphVal reflect.Value, fn func(nodeVal reflect.Value, id int64) error) error {
	nodesMethod := graphVal.MethodByName("Nodes")
	if !nodesMethod.IsValid() {
		return fmt.Errorf("graph is missing Nodes method")
	}
	seqValues := nodesMethod.Call(nil)
	if len(seqValues) != 1 {
		return fmt.Errorf("unexpected Nodes signature")
	}

	var iterErr error
	seq := seqValues[0]
	err := iterateSeq(seq, func(nodeVal reflect.Value) bool {
		id, err := callInt64Method(nodeVal, "ID")
		if err != nil {
			iterErr = err
			return false
		}
		if err := fn(nodeVal, id); err != nil {
			iterErr = err
			return false
		}
		return true
	})
	if err != nil {
		return err
	}
	return iterErr
}

func visitEdges(graphVal reflect.Value, fn func(edgeVal reflect.Value, fromID, toID int64, data any) error) error {
	edgesMethod := graphVal.MethodByName("Edges")
	if !edgesMethod.IsValid() {
		return fmt.Errorf("graph is missing Edges method")
	}
	seqValues := edgesMethod.Call(nil)
	if len(seqValues) != 1 {
		return fmt.Errorf("unexpected Edges signature")
	}

	var iterErr error
	seq := seqValues[0]
	err := iterateSeq(seq, func(edgeVal reflect.Value) bool {
		fromNode, err := callValueMethod(edgeVal, "From")
		if err != nil {
			iterErr = err
			return false
		}
		if !fromNode.IsValid() || fromNode.IsNil() {
			return true
		}
		fromID, err := callInt64Method(fromNode, "ID")
		if err != nil {
			iterErr = err
			return false
		}

		toNode, err := callValueMethod(edgeVal, "To")
		if err != nil {
			iterErr = err
			return false
		}
		if !toNode.IsValid() || toNode.IsNil() {
			return true
		}
		toID, err := callInt64Method(toNode, "ID")
		if err != nil {
			iterErr = err
			return false
		}

		dataVal, err := callValueMethod(edgeVal, "Data")
		if err != nil {
			iterErr = err
			return false
		}
		var edgeData any
		if dataVal.IsValid() {
			edgeData = dataVal.Interface()
		}
		if err := fn(edgeVal, fromID, toID, edgeData); err != nil {
			iterErr = err
			return false
		}
		return true
	})
	if err != nil {
		return err
	}
	return iterErr
}

func iterateSeq(seq reflect.Value, fn func(reflect.Value) bool) error {
	if !seq.IsValid() || seq.Kind() != reflect.Func {
		return fmt.Errorf("sequence is not a function")
	}
	if seq.Type().NumIn() != 1 || seq.Type().NumOut() != 0 {
		return fmt.Errorf("unexpected sequence signature")
	}
	cbType := seq.Type().In(0)
	if cbType.Kind() != reflect.Func || cbType.NumIn() != 1 || cbType.NumOut() != 1 {
		return fmt.Errorf("unexpected callback signature")
	}
	callback := reflect.MakeFunc(cbType, func(args []reflect.Value) []reflect.Value {
		cont := fn(args[0])
		return []reflect.Value{reflect.ValueOf(cont)}
	})
	seq.Call([]reflect.Value{callback})
	return nil
}

func callInt64Method(val reflect.Value, name string) (int64, error) {
	methodVal, err := callValueMethod(val, name)
	if err != nil {
		return 0, err
	}
	if !methodVal.IsValid() {
		return 0, fmt.Errorf("method %s returned invalid value", name)
	}
	switch methodVal.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return methodVal.Int(), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return int64(methodVal.Uint()), nil
	default:
		return 0, fmt.Errorf("method %s did not return an integer", name)
	}
}

func callValueMethod(val reflect.Value, name string) (reflect.Value, error) {
	if !val.IsValid() {
		return reflect.Value{}, fmt.Errorf("invalid receiver for method %s", name)
	}
	if (val.Kind() == reflect.Interface || val.Kind() == reflect.Ptr) && val.IsNil() {
		return reflect.Value{}, fmt.Errorf("receiver for method %s is nil", name)
	}
	method := val.MethodByName(name)
	if !method.IsValid() {
		return reflect.Value{}, fmt.Errorf("missing method %s", name)
	}
	results := method.Call(nil)
	if len(results) != 1 {
		return reflect.Value{}, fmt.Errorf("method %s returned unexpected values", name)
	}
	return results[0], nil
}
