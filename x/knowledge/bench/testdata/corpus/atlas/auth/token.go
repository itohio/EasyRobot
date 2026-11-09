// Package auth issues and verifies signed bearer tokens.
package auth

import (
	"crypto/ed25519"
	"encoding/base64"
	"errors"
	"strings"
	"time"
)

// ErrInvalid is returned for malformed, expired or forged tokens.
var ErrInvalid = errors.New("auth: invalid token")

// Token is a decoded bearer token.
type Token struct {
	Tenant  string
	Expires time.Time
}

// Verify checks a "payload.signature" bearer token against the public key.
func Verify(raw string, key ed25519.PublicKey, now time.Time) (Token, error) {
	payload, sig, ok := strings.Cut(raw, ".")
	if !ok {
		return Token{}, ErrInvalid
	}
	s, err := base64.RawURLEncoding.DecodeString(sig)
	if err != nil || !ed25519.Verify(key, []byte(payload), s) {
		return Token{}, ErrInvalid
	}
	tenant, exp, ok := strings.Cut(payload, "|")
	if !ok {
		return Token{}, ErrInvalid
	}
	t, err := time.Parse(time.RFC3339, exp)
	if err != nil || now.After(t) {
		return Token{}, ErrInvalid
	}
	return Token{Tenant: tenant, Expires: t}, nil
}

// Refresh issues a new token for tenant valid for ttl.
func Refresh(tenant string, key ed25519.PrivateKey, now time.Time, ttl time.Duration) string {
	payload := tenant + "|" + now.Add(ttl).Format(time.RFC3339)
	return payload + "." + base64.RawURLEncoding.EncodeToString(ed25519.Sign(key, []byte(payload)))
}
