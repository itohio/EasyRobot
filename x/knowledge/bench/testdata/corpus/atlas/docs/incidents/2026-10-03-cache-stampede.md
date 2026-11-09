---
title: Incident: cache stampede
type: incident
tags: [atlas, incident, cache]
updated: 2026-10-04
---

# Incident 2026-10-03 — cache stampede

At 14:02 a popular key expired and four hundred concurrent requests missed at once, all hitting PostgreSQL. p99 latency reached 9 s for six minutes.

Fix: single-flight loading in `atlas/cache/cache.go` so only one request refills a key. Background: [[cache-invalidation]].
