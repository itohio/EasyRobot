package marshaller

import (
	"fmt"
	"sync"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

var (
	muMarshallers   sync.RWMutex
	marshallers     = make(map[string]func(...types.Option) types.Marshaller)
	muUnmarshallers sync.RWMutex
	unmarshallers   = make(map[string]func(...types.Option) types.Unmarshaller)
)

// registerMarshaller registers a marshaller constructor.
// Called by register_<name>.go files in init().
func registerMarshaller(name string, ctor func(...types.Option) types.Marshaller) {
	muMarshallers.Lock()
	defer muMarshallers.Unlock()
	marshallers[name] = ctor
}

// registerUnmarshaller registers an unmarshaller constructor.
// Called by register_<name>.go files in init().
func registerUnmarshaller(name string, ctor func(...types.Option) types.Unmarshaller) {
	muUnmarshallers.Lock()
	defer muUnmarshallers.Unlock()
	unmarshallers[name] = ctor
}

// NewMarshaller creates a marshaller for the given format.
func NewMarshaller(name string, opts ...types.Option) (types.Marshaller, error) {
	muMarshallers.RLock()
	ctor, ok := marshallers[name]
	muMarshallers.RUnlock()

	if !ok {
		return nil, fmt.Errorf("marshaller: format %q not registered", name)
	}

	return ctor(opts...), nil
}

// NewUnmarshaller creates an unmarshaller for the given format.
func NewUnmarshaller(name string, opts ...types.Option) (types.Unmarshaller, error) {
	muUnmarshallers.RLock()
	ctor, ok := unmarshallers[name]
	muUnmarshallers.RUnlock()

	if !ok {
		return nil, fmt.Errorf("marshaller: format %q not registered", name)
	}

	return ctor(opts...), nil
}

// Marshallers returns names of registered marshallers.
func Marshallers() []string {
	muMarshallers.RLock()
	defer muMarshallers.RUnlock()

	names := make([]string, 0, len(marshallers))
	for name := range marshallers {
		names = append(names, name)
	}
	return names
}

// Unmarshallers returns names of registered unmarshallers.
func Unmarshallers() []string {
	muUnmarshallers.RLock()
	defer muUnmarshallers.RUnlock()

	names := make([]string, 0, len(unmarshallers))
	for name := range unmarshallers {
		names = append(names, name)
	}
	return names
}
