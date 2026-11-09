---
title: Deploy runbook (2026)
type: runbook
tags: [atlas, deploy]
updated: 2026-09-18
supersedes: deploy-runbook-2025
---

# Deploy runbook

1. Merge to main; CI builds a container image.
2. Roll it out with a canary at 5 %, then 50 %, then 100 %.
3. Watch the error rate for ten minutes; roll back with one command.
