---
title: Attitude limiter
type: design
tags: [rover, safety]
updated: 2026-09-20
---

# Attitude limiter

The limiter reads pitch and roll from the fused attitude and scales the throttle down linearly once either angle leaves the safe envelope (pitch 18°, roll 15°). Beyond 25° it commands a full stop and reverses gently for half a second.

Code: `rover/safety/limiter.go` (`AttitudeLimiter`).
