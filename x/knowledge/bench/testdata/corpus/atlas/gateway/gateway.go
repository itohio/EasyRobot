// Package gateway chains the Atlas HTTP middleware.
package gateway

import (
	"crypto/ed25519"
	"net/http"
	"strings"
	"time"

	"example.com/fixture/atlas/auth"
	"example.com/fixture/atlas/cache"
	"example.com/fixture/atlas/ratelimit"
)

// Gateway authenticates, rate-limits and caches requests before proxying.
type Gateway struct {
	Key      ed25519.PublicKey
	Limits   *ratelimit.Limiter
	Cache    *cache.Cache
	Upstream http.Handler
}

// ServeHTTP runs the middleware chain.
func (g *Gateway) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	now := time.Now()
	raw := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	tok, err := auth.Verify(raw, g.Key, now)
	if err != nil {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}
	if !g.Limits.AllowTenant(tok.Tenant, now) {
		http.Error(w, "quota exceeded", http.StatusTooManyRequests)
		return
	}
	g.Upstream.ServeHTTP(w, r)
}
