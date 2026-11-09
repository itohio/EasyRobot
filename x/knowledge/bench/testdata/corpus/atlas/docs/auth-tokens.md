---
title: Auth tokens
type: design
tags: [atlas, auth]
updated: 2026-10-02
---

# Auth tokens

Clients send a bearer token signed with an Ed25519 key. Tokens live 15 minutes; refresh tokens live 30 days. Signing keys rotate per [[T-004-token-rotation]].
