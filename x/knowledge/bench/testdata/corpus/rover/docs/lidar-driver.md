---
title: Lidar driver
type: design
tags: [rover, lidar]
updated: 2026-10-08
---

# Lidar driver

The scanner streams 47-byte packets at 230400 baud. The driver resynchronises on the 0x54 header byte and assembles a full rotation before publishing a scan.

Open problem: [[RV-009-lidar-dropouts]]. Bring-up task: [[T-005-lidar-bringup]].
