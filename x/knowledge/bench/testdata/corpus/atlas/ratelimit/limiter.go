// Package ratelimit provides per-tenant token buckets.
package ratelimit

import (
	"sync"
	"time"
)

// TokenBucket allows bursts up to Capacity and refills at Rate tokens/s.
type TokenBucket struct {
	mu       sync.Mutex
	Capacity float64
	Rate     float64
	tokens   float64
	last     time.Time
}

// Allow takes one token if available.
func (b *TokenBucket) Allow(now time.Time) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.refill(now)
	if b.tokens < 1 {
		return false
	}
	b.tokens--
	return true
}

// refill adds Rate tokens per elapsed second, capped at Capacity.
func (b *TokenBucket) refill(now time.Time) {
	if b.last.IsZero() {
		b.tokens, b.last = b.Capacity, now
		return
	}
	b.tokens = min(b.Capacity, b.tokens+now.Sub(b.last).Seconds()*b.Rate)
	b.last = now
}

// Limiter keeps one bucket per tenant.
type Limiter struct {
	mu      sync.Mutex
	buckets map[string]*TokenBucket
	cap     float64
	rate    float64
}

// NewLimiter returns a limiter with the given per-tenant capacity and rate.
func NewLimiter(capacity, rate float64) *Limiter {
	return &Limiter{buckets: map[string]*TokenBucket{}, cap: capacity, rate: rate}
}

// AllowTenant reports whether tenant may make one more request.
func (l *Limiter) AllowTenant(tenant string, now time.Time) bool {
	l.mu.Lock()
	b, ok := l.buckets[tenant]
	if !ok {
		b = &TokenBucket{Capacity: l.cap, Rate: l.rate}
		l.buckets[tenant] = b
	}
	l.mu.Unlock()
	return b.Allow(now)
}
