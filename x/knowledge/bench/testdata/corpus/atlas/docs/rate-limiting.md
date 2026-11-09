---
title: Rate limiting
type: design
tags: [atlas, quotas]
updated: 2026-08-20
---

# Rate limiting

Each tenant gets its own token bucket: capacity 100, refilled at 20 per second. When the bucket is empty the gateway answers 429 so that a single noisy customer cannot starve the others.

Code: `atlas/ratelimit/limiter.go`.
