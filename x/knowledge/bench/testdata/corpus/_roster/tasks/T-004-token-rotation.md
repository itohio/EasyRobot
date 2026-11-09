---
id: T-004
title: Token rotation
type: task
status: open
objective: O-002
tags: [task, atlas, auth]
updated: 2026-10-02
---

# T-004 — Token rotation

Serves [[O-002-atlas-public-beta]]. Rotate signing keys every 24 hours without logging users out.
Depends on [[AT-008-token-refresh-race]]. Paths: `atlas/auth/token.go`, `atlas/docs/auth-tokens.md`.
