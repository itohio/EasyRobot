---
title: Atlas architecture
type: design
tags: [atlas, design]
updated: 2026-05-02
---

# Atlas architecture

Gateway → services → PostgreSQL, with a read-through cache in front of the hot tables.
The gateway authenticates ([[auth-tokens]]), applies quotas ([[rate-limiting]]) and serves cached reads ([[cache-invalidation]]).
