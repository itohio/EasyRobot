---
type: algorithm
status: tested
tags: [easyrobot, mat, jacobian, kinematics]
workstream: W9
module: "[[mat-vec]]"
summary: "analytic geometric Jacobian column (not finite differences)"
source: [x/math/mat/jacobian.go]
tests: [x/math/mat/jacobian_test.go, x/math/mat/mat_ik_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 ./x/... (2026-11-09): TestCalculateJacobianColumn (5 subtests with expected Linear/Angular), TestCalculateJacobianColumn_Properties, TestCalculateJacobianColumn_PrismaticProperties, TestIKSolverOperations pass"
---
# Jacobian column (geometric, per joint)

**What it does:** Despite the note name, `x/math/mat` has **no numeric (finite-difference) Jacobian**. It provides the analytic geometric Jacobian column of a serial chain: revolute `Jv = zᵢ × (p_ee − pᵢ)`, `Jω = zᵢ`; prismatic `Jv = zᵢ`, `Jω = 0`. Callers assemble the 6×N Jacobian column by column. (Finite-difference gradient checking exists only in `x/math/nn/layers/gradient_test_helper.go`.)

**Where:** `x/math/mat/jacobian.go:21-58` `CalculateJacobianColumn(jointPos, jointAxis, eePos [3]float32, isRevolute bool) JacobianColumn{Linear, Angular [3]float32}`.

**Inputs → outputs:** world-frame joint position and **unit** axis (not normalised internally), end-effector position → linear/angular 3-vectors (units: m/rad, m per m for prismatic).

## Completeness review

- Complete: yes for the column formula.
- Tested: 5 table cases (revolute at origin with EE on x/y, offset joint, x-axis rotation, prismatic) with exact expected vectors; property tests (perpendicularity, prismatic angular = 0); `TestIKSolverOperations` exercises matrix ops used by an IK step.
- Correct: yes (values checked; formula matches standard geometric Jacobian).

**Embedded suitability:** pure value arithmetic on `[3]float32`, no allocation, no interfaces — ideal for ESP32.

## TODOs

- [ ] Helper to assemble full 6×N Jacobian (into caller buffer) from DH frames — proposal, consumer in [[kinematics]] #p3
- [ ] Optional finite-difference Jacobian for arbitrary FK functions (validation of analytic J) — proposal #p3

## Links

- Module: [[mat-vec]]
- Related: [[mat-pseudo-inverse]], [[homogeneous-transforms]], [[kinematics]]
