---
type: algorithm
status: tested
tags: [easyrobot, learn, nn]
workstream: [W9, W6]
module: "[[learn]]"
summary: "dense/sequential/training tests pass; probe exact"
source: [x/math/learn/optimizer.go]
tests: [x/math/learn/dense_layer_test.go, x/math/learn/training_test.go, x/math/learn/sequential_single_layer_test.go, x/math/learn/sequential_two_layers_test.go, x/math/learn/sequential_classification_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "probe: [1,1]-0.1*[1,-2] → [0.9,1.2] exact; dense/sequential tests pass (2026-11-09)"
---
# learn SGD optimizer

**What it does:** plain SGD, `θ ← θ − lr·∇θ`, in place. No momentum, weight decay, clipping or LR schedule.

**Where:** `x/math/learn/optimizer.go:12-57` (`SGD`, `NewSGD`, `(*SGD).Update`); implements `nn/types.Optimizer`.

**Inputs → outputs:** `NewSGD(lr float64)` (panics if `lr<=0`, :20-22). `Update(types.Parameter) error`: skips when `!RequiresGrad` or Grad empty (:34-40); errors on empty Data / shape mismatch; mutates `param.Data` via `Data.AddScaled(nil, Grad, -lr)` (:54). Zero allocations per step.

## Completeness review

- Complete: yes for vanilla SGD. Momentum/Nesterov/weight decay absent (not claimed).
- Tested: indirectly. `TestDenseLayer_SimpleLinearRegression` (dense_layer_test.go:16) fits y=2x+1 and asserts loss decreases, w≈2±0.5, b≈1±0.5, predictions ±0.5 (:132-153). `TestTrainStep_MultipleSteps` asserts final loss < 0.1·initial (training_test.go:213). Sequential single/two-layer regression same tolerances; classification tests assert loss decreases and accuracy ≥50% (sequential_classification_test.go:172). All pass (lead run 2026-11-09). No direct unit test of `Update`.
- Correct: yes — probe (`tmp/easyrobot/filters/learnnotes/main.go`, `go run .`) `[1,1] − 0.1·[1,−2]` → `[0.9000 1.2000]`.

## TODOs

- [ ] Direct unit test for `SGD.Update` (values, RequiresGrad=false skip, shape mismatch error) — x/math/learn/optimizer.go:27 #p2
- [ ] Momentum / weight decay options (cheap, alloc-once state) for on-device fine-tuning — x/math/learn/optimizer.go:14 #p3
- [ ] `RequiresGrad` is marked deprecated in favour of layer CanLearn (nn/types/parameter.go:33) but optimizer still gates on it — x/math/learn/optimizer.go:34 #p3

## Links

- Module: [[learn]] · Workstream: [[W9-go-nn-framework]]
