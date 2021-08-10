package plugin

import (
	"errors"
	"fmt"
	"reflect"

	. "github.com/foxis/EasyRobot/pkg/core/logger"
)

var ErrNOP = errors.New("No op")

func MarshalOptions(opts interface{}) map[string]interface{} {
	config := make(map[string]interface{})
	marshalOptions(opts, config)
	return config
}

func marshalOptions(opts interface{}, config map[string]interface{}) {
	rType := reflect.TypeOf(opts)
	rValue := reflect.ValueOf(opts)
	for i := 0; i < rType.NumField(); i++ {
		field := rType.Field(i)
		fieldName, ok := field.Tag.Lookup("opts")
		if !ok {
			continue
		}

		if !rValue.Field(i).CanInterface() {
			continue
		}

		val := rValue.Field(i).Interface()

		if field.Name == "Base" {
			marshalOptions(val, config)
			continue
		}

		if fieldName == "" {
			fieldName = field.Name
		}

		if field.Type.Kind() == reflect.Struct {
			tmp := make(map[string]interface{})
			marshalOptions(val, tmp)
			config[fieldName] = tmp
			continue
		}

		config[fieldName] = val
	}
}

// Thanks to dave
// https://stackoverflow.com/questions/26744873/converting-map-to-struct/26746461

func setField(obj interface{}, name string, value interface{}) error {
	structValue := reflect.ValueOf(obj).Elem()
	structType := reflect.TypeOf(obj).Elem()
	structFieldValue := structValue.FieldByNameFunc(func(s string) bool {
		if structFieldType, ok := structType.FieldByName(s); ok {
			if fieldName, ok := structFieldType.Tag.Lookup("opts"); ok {
				if fieldName == name {
					return true
				}
			}
		}
		if name == s {
			return true
		}
		return false
	})

	if !structFieldValue.IsValid() {
		return fmt.Errorf("No such field: %s in obj", name)
	}
	if !structFieldValue.CanSet() {
		return fmt.Errorf("Cannot set %s field value", name)
	}

	structFieldType := structFieldValue.Type()

	if structFieldType, ok := structType.FieldByName(name); ok {
		_, ok := structFieldType.Tag.Lookup("opts")
		if !ok {
			return fmt.Errorf("No tag set: %s", name)
		}
	}

	val := reflect.ValueOf(value)
	if value == nil {
		return nil
	}

	if structFieldType != val.Type() {
		var err error
		val, err = convertFieldValue(name, structFieldValue, structFieldType, val)
		if err != nil {
			return err
		}
	}

	structFieldValue.Set(val)
	return nil
}

func convertFieldValue(name string, structFieldValue reflect.Value, structFieldType reflect.Type, val reflect.Value) (reflect.Value, error) {
	if val.Type().ConvertibleTo(structFieldType) {
		return val.Convert(structFieldType), nil
	}

	switch structFieldType.Kind() {
	case reflect.Map:
		return convertMap(name, structFieldValue, structFieldType, val)
	case reflect.Slice:
		return convertArray(name, structFieldValue, structFieldType, val)
	case reflect.Array:
		Log.Warn().Str("name", name).Msg("Array not supported")
	case reflect.Struct:
		return convertStruct(name, structFieldValue, structFieldType, val)
	}

	return reflect.Value{}, fmt.Errorf("Provided value type didn't match obj field type: %s: %s v.s. %s", name, fmt.Sprint(structFieldType), fmt.Sprint(val.Type()))
}

func convertMap(name string, structFieldValue reflect.Value, structFieldType reflect.Type, val reflect.Value) (reflect.Value, error) {
	m, ok := val.Interface().(map[string]interface{})
	if !ok {
		return reflect.Value{}, fmt.Errorf("Not a map data: %s", name)
	}

	if structFieldType.Key().Kind() != reflect.String {
		return reflect.Value{}, fmt.Errorf("Map keys must be string: %s", name)
	}

	mapVal := reflect.MakeMapWithSize(structFieldType, len(m))
	elemType := structFieldType.Elem()

	for k, v := range m {
		rv := reflect.ValueOf(v)
		if reflect.TypeOf(v) != elemType {
			if rv.Type().ConvertibleTo(elemType) {
				rv = rv.Convert(elemType)
			} else {
				Log.Warn().Str("name", name).Str("key", k).Str("Elem type", fmt.Sprint(structFieldType)).Str("Value type", fmt.Sprint(reflect.TypeOf(v))).Msg("Slice item")
			}
		}
		mapVal.SetMapIndex(reflect.ValueOf(k), rv)
	}

	return mapVal, nil
}

func convertArray(name string, structFieldValue reflect.Value, structFieldType reflect.Type, val reflect.Value) (reflect.Value, error) {
	arr, ok := val.Interface().([]interface{})
	if !ok {
		return reflect.Value{}, fmt.Errorf("Not an array data: %s", name)
	}
	arrVal := reflect.MakeSlice(structFieldType, len(arr), len(arr))
	elemType := structFieldType.Elem()
	for i, v := range arr {
		a := arrVal.Index(i)
		rv := reflect.ValueOf(v)
		if reflect.TypeOf(v) != a.Type() {
			var err error
			rv, err = convertFieldValue(name, a, elemType, rv)
			if err != nil {
				Log.Warn().Str("name", name).Int("index", i).Str("Elem type", fmt.Sprint(structFieldType)).Str("Value type", fmt.Sprint(reflect.TypeOf(v))).Msg("Slice item")
				continue
			}
		}
		a.Set(rv)
	}

	return arrVal, nil
}

func convertStruct(name string, structFieldValue reflect.Value, structFieldType reflect.Type, val reflect.Value) (reflect.Value, error) {
	if !val.CanInterface() {
		return reflect.Value{}, fmt.Errorf("Field cannot be interfaced: %s", name)
	}
	m, ok := val.Interface().(map[string]interface{})
	if !ok {
		return reflect.Value{}, fmt.Errorf("Not a map data: %s", name)
	}
	for k, v := range m {
		setField(structFieldValue.Addr().Interface(), k, v)
		// if err != nil {
		// 	Log.Error().Err(err).Str("key", k).Msg("setField")
		// }
	}
	return reflect.Value{}, ErrNOP
}

func fillStruct(s interface{}, m map[string]interface{}) error {
	rv := reflect.ValueOf(s)
	if rv.Kind() != reflect.Ptr || rv.IsNil() {
		return errors.New("Options must be a pointer")
	}

	for k, v := range m {
		setField(s, k, v)
		// if err != nil {
		// 	Log.Error().Err(err).Str("key", k).Msg("setField")
		// }
	}

	base := rv.Elem().FieldByName("Base")
	if !base.CanAddr() {
		return nil
	}
	for k, v := range m {
		setField(base.Addr().Interface(), k, v)
		// if err != nil {
		// 	Log.Error().Err(err).Str("key", k).Msg("setField")
		// }
	}

	return nil
}
