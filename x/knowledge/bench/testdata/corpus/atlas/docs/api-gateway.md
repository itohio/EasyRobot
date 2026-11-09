---
title: API gateway
type: design
tags: [atlas, gateway]
updated: 2026-09-22
---

# API gateway

The gateway is a chain of middleware: authentication, quota check, cache lookup, then the upstream proxy. Requests time out after 3 s ([[AT-001-gateway-timeouts]]).

Code: [gateway.go](../gateway/gateway.go).
