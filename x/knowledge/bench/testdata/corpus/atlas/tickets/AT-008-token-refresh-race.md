---
id: AT-008
title: Token refresh race
type: ticket
status: open
tags: [atlas, ticket]
updated: 2026-10-02
---

# AT-008 — Token refresh race

Two tabs refreshing at once invalidate each other's refresh token. Blocks [[T-004-token-rotation]]. See [[atlas/docs/auth-tokens]].
