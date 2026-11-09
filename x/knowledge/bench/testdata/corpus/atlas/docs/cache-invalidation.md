---
title: Cache invalidation
type: design
tags: [atlas, cache]
updated: 2026-10-05
---

# Cache invalidation

Entries expire after a TTL of 60 s. Writes publish an invalidation event that deletes the key on every replica. A deploy flushes nothing, so schema changes need a version prefix in the key.

Root cause of [[AT-005-stale-cache-after-deploy]]: the key had no version prefix. Incident: [[atlas/docs/incidents/2026-10-03-cache-stampede]].
