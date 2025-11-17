package graph

import (
	"fmt"
	"reflect"
	"runtime"
)

type decisionNodeOpInfo struct {
	fn         reflect.Value
	inputType  reflect.Type
	outputType reflect.Type
}

type decisionEdgeOpInfo struct {
	fn        reflect.Value
	inputType reflect.Type
}

type expressionOpInfo struct {
	fn             reflect.Value
	inputType      reflect.Type
	childMapType   reflect.Type
	childValueType reflect.Type
	outputType     reflect.Type
}

func newDecisionNodeOpInfo(fn any) (decisionNodeOpInfo, string, error) {
	value := reflect.ValueOf(fn)
	if !value.IsValid() || value.IsNil() {
		return decisionNodeOpInfo{}, "", fmt.Errorf("decision op function is nil")
	}
	if value.Kind() != reflect.Func {
		return decisionNodeOpInfo{}, "", fmt.Errorf("decision op must be a function")
	}
	typ := value.Type()
	if typ.NumIn() != 1 {
		return decisionNodeOpInfo{}, "", fmt.Errorf("decision op must accept 1 argument, got %d", typ.NumIn())
	}
	if typ.NumOut() != 2 {
		return decisionNodeOpInfo{}, "", fmt.Errorf("decision op must return (value, bool)")
	}
	if typ.Out(1).Kind() != reflect.Bool {
		return decisionNodeOpInfo{}, "", fmt.Errorf("decision op second return must be bool")
	}
	name, err := functionName(value)
	if err != nil {
		return decisionNodeOpInfo{}, "", err
	}
	return decisionNodeOpInfo{
		fn:         value,
		inputType:  typ.In(0),
		outputType: typ.Out(0),
	}, name, nil
}

func newDecisionEdgeOpInfo(fn any) (decisionEdgeOpInfo, string, error) {
	value := reflect.ValueOf(fn)
	if !value.IsValid() || value.IsNil() {
		return decisionEdgeOpInfo{}, "", fmt.Errorf("decision edge op is nil")
	}
	if value.Kind() != reflect.Func {
		return decisionEdgeOpInfo{}, "", fmt.Errorf("decision edge op must be a function")
	}
	typ := value.Type()
	if typ.NumIn() != 1 {
		return decisionEdgeOpInfo{}, "", fmt.Errorf("decision edge op must accept 1 argument")
	}
	if typ.NumOut() != 1 || typ.Out(0).Kind() != reflect.Bool {
		return decisionEdgeOpInfo{}, "", fmt.Errorf("decision edge op must return bool")
	}
	name, err := functionName(value)
	if err != nil {
		return decisionEdgeOpInfo{}, "", err
	}
	return decisionEdgeOpInfo{
		fn:        value,
		inputType: typ.In(0),
	}, name, nil
}

func newExpressionOpInfo(fn any) (expressionOpInfo, string, error) {
	value := reflect.ValueOf(fn)
	if !value.IsValid() || value.IsNil() {
		return expressionOpInfo{}, "", fmt.Errorf("expression op is nil")
	}
	if value.Kind() != reflect.Func {
		return expressionOpInfo{}, "", fmt.Errorf("expression op must be a function")
	}
	typ := value.Type()
	if typ.NumIn() != 2 {
		return expressionOpInfo{}, "", fmt.Errorf("expression op must accept 2 arguments")
	}
	if typ.In(1).Kind() != reflect.Map {
		return expressionOpInfo{}, "", fmt.Errorf("expression op second argument must be map")
	}
	if typ.In(1).Key().Kind() != reflect.Int64 {
		return expressionOpInfo{}, "", fmt.Errorf("expression op map key must be int64")
	}
	if typ.NumOut() != 2 || typ.Out(1).Kind() != reflect.Bool {
		return expressionOpInfo{}, "", fmt.Errorf("expression op must return (value, bool)")
	}
	name, err := functionName(value)
	if err != nil {
		return expressionOpInfo{}, "", err
	}
	return expressionOpInfo{
		fn:             value,
		inputType:      typ.In(0),
		childMapType:   typ.In(1),
		childValueType: typ.In(1).Elem(),
		outputType:     typ.Out(0),
	}, name, nil
}

func functionName(value reflect.Value) (string, error) {
	if !value.IsValid() || value.Kind() != reflect.Func {
		return "", fmt.Errorf("value is not a function")
	}
	pointer := value.Pointer()
	if pointer == 0 {
		return "", fmt.Errorf("function pointer is zero")
	}
	if fn := runtime.FuncForPC(pointer); fn != nil {
		return fn.Name(), nil
	}
	return "", fmt.Errorf("unable to resolve function name")
}

func callDecisionNodeOp(info decisionNodeOpInfo, input any) (any, bool, error) {
	if !info.fn.IsValid() {
		return nil, false, fmt.Errorf("decision operation not registered")
	}
	arg, err := convertToType(input, info.inputType)
	if err != nil {
		return nil, false, err
	}
	results := info.fn.Call([]reflect.Value{arg})
	if len(results) != 2 {
		return nil, false, fmt.Errorf("decision op returned unexpected values")
	}
	return results[0].Interface(), results[1].Bool(), nil
}

func callDecisionEdgeOp(info decisionEdgeOpInfo, input any) (bool, error) {
	if !info.fn.IsValid() {
		return true, nil
	}
	arg, err := convertToType(input, info.inputType)
	if err != nil {
		return false, err
	}
	results := info.fn.Call([]reflect.Value{arg})
	if len(results) != 1 {
		return false, fmt.Errorf("decision edge op returned unexpected values")
	}
	return results[0].Bool(), nil
}

func callExpressionOp(info expressionOpInfo, input any, childOutputs map[int64]any) (any, bool, error) {
	if !info.fn.IsValid() {
		return nil, false, fmt.Errorf("expression operation not registered")
	}
	arg, err := convertToType(input, info.inputType)
	if err != nil {
		return nil, false, err
	}
	childMap, err := convertChildOutputs(childOutputs, info.childMapType, info.childValueType)
	if err != nil {
		return nil, false, err
	}
	results := info.fn.Call([]reflect.Value{arg, childMap})
	if len(results) != 2 {
		return nil, false, fmt.Errorf("expression op returned unexpected values")
	}
	return results[0].Interface(), results[1].Bool(), nil
}

func convertChildOutputs(children map[int64]any, mapType, elemType reflect.Type) (reflect.Value, error) {
	if mapType.Kind() != reflect.Map {
		return reflect.Value{}, fmt.Errorf("child map type must be map")
	}
	m := reflect.MakeMapWithSize(mapType, len(children))
	keyType := mapType.Key()
	for id, value := range children {
		childVal, err := convertToType(value, elemType)
		if err != nil {
			return reflect.Value{}, err
		}
		key := reflect.ValueOf(id)
		if keyType.Kind() != reflect.Int64 {
			key = key.Convert(keyType)
		}
		m.SetMapIndex(key, childVal)
	}
	return m, nil
}

func convertToType(value any, target reflect.Type) (reflect.Value, error) {
	if target == nil {
		return reflect.Value{}, fmt.Errorf("target type is nil")
	}
	if value == nil {
		return reflect.Zero(target), nil
	}
	val := reflect.ValueOf(value)
	if val.Type().AssignableTo(target) {
		if val.Type() == target {
			return val, nil
		}
		return val.Convert(target), nil
	}
	if val.Type().ConvertibleTo(target) {
		return val.Convert(target), nil
	}
	return reflect.Value{}, fmt.Errorf("cannot convert %s to %s", val.Type(), target)
}
