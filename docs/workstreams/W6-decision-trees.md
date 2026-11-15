---
type: workstream
status: proposed
tags: [easyrobot, workstream, decision-tree, behaviour, esp32]
workstream: W6
---
# W6 — Decision-tree self-learning behaviours (incl. gaits?) on ESP32

**Goal:** decision-tree self-learning behaviours (possibly gaits) on ESP32 — end goal: camera obstacle avoidance on a wheeled ESP32. The brief also points to the existing filters and decoders and wants module APIs reviewed for composability.

## What exists

| Piece | Where | Status | Evidence |
|---|---|---|---|
| [[decision-tree]] — `GenericDecisionTree`: n-ary tree, closure op per node, closure criterion per edge, first-match DFS with backtracking | `x/math/graph/decision_tree.go`, `decision_interfaces.go` | tested | `TestGenericDecisionTree_Decide` pass (threshold routing incl. boundary) |
| [[vector-decision-tree]] — binary threshold routing on `vecTypes.Vector` features; helpers `VectorDimensionThreshold`, `VectorNormThreshold`, `VectorDotProductThreshold` | `x/math/graph/vector_decision_tree.go` | partial | tests pass; `Balance()` swaps true/false branches (`tree.go:206-224`); >2 children ignored (:135-148) |
| Storage/replay — `StoredDecisionTree` executes from memory-mapped node/edge files; ops looked up by registered Go function name, called via reflection | `x/marshaller/graph/stored_decision_tree.go:12-122`, `ops_registry.go:116-128` | tested (round trip) | `TestDecisionTreeRoundTrip` pass (`go test -count=1 -json ./x/marshaller/graph/...` → 11 pass, 2026-11-09) — see [[marshallers]] |
| [[expression-graph]] — bottom-up dataflow (could host feature extraction → decision) | `x/math/graph/expression_graph.go` | partial | no cycle detection (:162-177), `AddNode` handles detached (probe `exprprobe`) |
| Generic tree substrate (`GenericTree`), reorder/balance ops | `x/math/graph/tree.go` | partial | see [[graph-tree]] |
| Training infrastructure for neural nets (SGD/Adam, train step, PTQ) | `x/math/learn` | see [[learn]] | no tree learners |
| Feature sources: filters ([[filters]]), lidar ([[lidar]]), camera/vision ([[vision]], [[camera-io]]), K230 feature tensor ([[W4-k230-feature-stream]]) | various | various | — |

**Not found anywhere in the repo** (grep for gini / information gain / CART / ID3 / random forest / Hoeffding / behaviour tree / reinforcement / Q-learning over `*.go` and `*.md`, 2026-11-09): any tree *learning* — split search, impurity measures, training-set types, pruning, ensembles, online/incremental learning, reward-driven adaptation. No decision-tree user outside `x/math/graph` and `x/marshaller/graph` (the PEG parser uses the *expression* graph, not decision trees). No gait code uses trees.

## Gaps

1. **Representation is code, not data.** Splits are Go closures (`DecisionOp`, `DecisionCriteria`). A learner needs nodes as plain data (`feature uint16, threshold float32 | int16, left/right uint16, leaf class/value`) so it can create, mutate, serialise and execute them. The marshaller stores ops by function *name*; closures from `VectorDimensionThreshold(dim, thr)` share one name and lose `dim`/`thr` [INFERENCE from `ops_registry.go:116-128`].
2. **No learning algorithms.** Needed (pick per scenario): offline CART (Gini/entropy for classification, variance for regression); online Hoeffding tree / VFDT or Mondrian trees for "self-learning" on the device; small ensembles (random forest / gradient-boosted stumps) if accuracy matters; pruning (cost-complexity or depth/min-samples limits) for MCU memory.
3. **No data path.** No dataset type for (feature vector, label/reward) samples, no logging/recording of behaviour episodes on ESP32, no import of logs into a host trainer, no evaluation (confusion matrix, holdout).
4. **Execution cost on ESP32.** Current evaluation goes through interfaces, maps (`nodeOps`, `edgeOps`), per-iteration wrapper allocations (`decision_tree.go:270-294`, :313-322), and in the stored variant `reflect.Value.Call`. An MCU executor should be a flat array walk: zero allocation, O(depth), int16/float32 comparisons [INFERENCE: TinyGo build not attempted].
5. **Semantic hazards.** `VectorDecisionTree` depends on child order, but generic `Balance`/`ReorderChildren` change it; `GenericDecisionTree` calls an internal node op twice on failure (`decision_tree.go:246-253`).
6. **Features for camera obstacle avoidance.** Nothing yet produces a compact per-frame feature vector on the ESP32 side (e.g. pooled K230 feature tensor, column-wise free-space histogram, optical-flow magnitude, lidar sector minima). Trees need tens of features, not a raw 7×7×C tensor.
7. **Gaits.** Decision trees are a poor fit for continuous gait generation (phase/timing/trajectories — see [[W3-legged-ik]], [[motion-and-gait]]) but a reasonable fit for *discrete choices*: gait type (tripod/ripple/wave), stride length/height class, recovery behaviours, from IMU/foot-contact/terrain features.

## TODOs

Algorithm defects are tracked in the algorithm and module notes and rolled up by ticket in [[todo]]; this list holds workstream-level items only.

- [ ] Plain-data binary split node + flat-array executor (no maps/closures/reflection; zero alloc) with tests vs `VectorDecisionTree` on the same splits — x/math/graph (new) #p2
- [ ] Host-side CART trainer (Gini/entropy/variance, max depth, min samples, cost-complexity pruning) writing the flat format; golden test on a small public dataset with known accuracy #p2
- [ ] Serialisation of data-driven trees through `x/marshaller/graph` (or a compact binary blob for flash) — stored_decision_tree.go #p2
- [ ] Episode logger on ESP32 (feature vector, action, outcome) + host import — new #p2
- [ ] Online learner prototype (Hoeffding tree with bounded memory) evaluated on host with recorded logs before porting #p2

## Inferred objectives (proposals)

- **proposal (A, first milestone):** *Learning from demonstration, offline training.* Drive the wheeled ESP32 robot by RC/phone, log (features, operator action) at 10–20 Hz; train a depth-limited CART on the host (Go, reusing `x/math/learn` data loading conventions); flash the flat tree; evaluate obstacle-avoidance success rate. Lowest risk, gives the data path and executor everyone else needs.
- **proposal (B):** *On-device self-learning.* Hoeffding tree (or Mondrian forest) updated online from a self-supervised label: "collision/bump/near-range lidar within N s" as negative, "progress without event" as positive; bounded node budget in SRAM; freeze/rollback if performance drops.
- **proposal (C):** *Trees over learned features.* Use the K230 feature tensor (W4/W5), pooled to a few dozen channels, as tree inputs — the tree stays tiny and interpretable while the CNN does perception.
- **proposal (D, gaits):** tree selects gait mode and parameters; a CPG/phase oscillator with IK ([[kinematics]], [[motion-and-gait]]) executes it; trees trained from logged IMU stability metrics.

## Research questions for ideation

1. Which online tree learner gives the best accuracy per KB of SRAM on ESP32 (Hoeffding tree, Hoeffding Adaptive Tree, Mondrian forest, streaming random patches) for ~20–64 features at ~10 Hz?
2. What self-supervised labels are reliable for obstacle avoidance without a human (bumper, lidar near-range, wheel slip / stall current, optical-flow expansion), and how noisy are they?
3. Are gradient-boosted stumps or a small random forest worth their memory over a single tree for camera features — what accuracy/latency do published TinyML benchmarks report?
4. What compact per-frame features from a K230 backbone (global-average pooled channels, spatial column histograms) suffice for free-space classification?
5. Can trees choose discrete gait modes robustly from IMU + foot-contact features, and how much data does that need?
6. Is a decision tree the right policy class at all versus a tiny MLP trained by the existing `x/math/learn` stack (int8 on ESP32) — interpretability vs accuracy?

## Decisions for the owner

- **Representation:** keep closure-based `GenericDecisionTree` for hand-written behaviours *and* add a separate data-driven tree type (recommended), or migrate everything to data-driven nodes (cleaner, breaks the closure API).
- **Where learning happens:** offline on host from logs (recommended first) vs online on the ESP32 (true "self-learning", harder to debug).
- **Scope of "gaits":** discrete gait selection only (recommended) vs trees generating timing/trajectories (not recommended).
