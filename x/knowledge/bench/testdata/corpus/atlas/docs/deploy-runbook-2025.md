---
title: Deploy runbook (2025)
type: runbook
tags: [atlas, deploy]
updated: 2025-12-05
---

# Deploy runbook

1. SSH into each VM.
2. Pull the new binary, restart the systemd unit.
3. Watch the error rate for ten minutes.
