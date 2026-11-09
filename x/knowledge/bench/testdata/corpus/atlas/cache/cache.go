// Package cache is a TTL read-through cache with single-flight loading.
package cache

import (
	"sync"
	"time"
)

type entry struct {
	value   []byte
	expires time.Time
}

// Cache maps keys to values that expire after TTL.
type Cache struct {
	mu      sync.Mutex
	TTL     time.Duration
	entries map[string]entry
	loading map[string]*sync.WaitGroup
}

// New returns an empty cache.
func New(ttl time.Duration) *Cache {
	return &Cache{TTL: ttl, entries: map[string]entry{}, loading: map[string]*sync.WaitGroup{}}
}

// Get returns the cached value or calls load once for concurrent misses.
func (c *Cache) Get(key string, now time.Time, load func() []byte) []byte {
	c.mu.Lock()
	if e, ok := c.entries[key]; ok && now.Before(e.expires) {
		c.mu.Unlock()
		return e.value
	}
	if wg, ok := c.loading[key]; ok {
		c.mu.Unlock()
		wg.Wait()
		return c.Get(key, now, load)
	}
	wg := &sync.WaitGroup{}
	wg.Add(1)
	c.loading[key] = wg
	c.mu.Unlock()

	v := load()
	c.mu.Lock()
	c.entries[key] = entry{value: v, expires: now.Add(c.TTL)}
	delete(c.loading, key)
	c.mu.Unlock()
	wg.Done()
	return v
}

// Invalidate removes key.
func (c *Cache) Invalidate(key string) {
	c.mu.Lock()
	delete(c.entries, key)
	c.mu.Unlock()
}
