package plugin

import (
	"errors"
	"sync"

	"github.com/foxis/EasyRobot/pkg/core/options"
)

var (
	Global               = New()
	ErrExists            = errors.New("already exists")
	ErrNotFound          = errors.New("not found")
	ErrCorruptedRegistry = errors.New("registry seems to be corrupted")
)

type Plugin interface{}
type Builder func(...options.Option) (Plugin, error)

type Registry struct {
	mutex    sync.RWMutex
	registry map[string]Builder
}

func New() *Registry {
	return &Registry{
		registry: make(map[string]Builder),
	}
}

func (p *Registry) Register(name string, builder Builder) error {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	if _, ok := p.registry[name]; ok {
		return ErrExists
	}

	p.registry[name] = builder
	return nil
}

func (p *Registry) Unregister(name string) error {
	p.mutex.Lock()
	defer p.mutex.Unlock()
	delete(p.registry, name)
	return nil
}

func (p *Registry) New(name string, opts ...options.Option) (Plugin, error) {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	if build, ok := p.registry[name]; ok {
		return build(opts...)
	}

	return nil, ErrNotFound
}

func (p *Registry) ForEach(pattern string, f func(string, Builder)) {
	p.mutex.RLock()
	defer p.mutex.RUnlock()
	for k, v := range p.registry {
		f(k, v)
	}
}
