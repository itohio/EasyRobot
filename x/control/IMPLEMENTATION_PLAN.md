# Control Package Implementation Plan

## Vision

Establish `pkg/control` as the home for high-level robot control primitives, separating hardware-specific code (in `pkg/devices` and TinyGo `devices/`) from core algorithms (`pkg/math/control`) and planning logic (`pkg/planning`). This plan documents the target architecture, migration path, and integration requirements with the dndm transport ecosystem.

## Target Package Layout

| Path | Purpose | Key Contents | Notes |
| --- | --- | --- | --- |
| `pkg/control/locomotion` | Body-level coordination, supports, actuators | `body`, `support`, `actuator`, runtime utilities | Device-independent; relies on dndm topics for I/O |
| `pkg/math/control` | Control algorithms and math primitives | IK/FK solvers, filters, optimization | Pure math; no transport dependencies |
| `pkg/vision` | Perception, low-level vision ops | Tensor conversions, feature extraction | Will later feed `pkg/planning` |
| `pkg/planning` | High-level planning and decision making | Action/motion planners, behavior trees | Coordinates with control targets |
| `pkg/devices` | Go/TinyGo device wrappers | Sensors, motors, comms adapters | Device-agnostic APIs; compiles for host |
| `devices/` | TinyGo firmware runtime | MCU drivers, dndm bindings | Publishes/consumes intents over dndm |
| `cmd/vision` | Vision pipelines | dndm-backed CLI/services | Consumes device streams, emits vision outputs |
| `proto/types/<module>` | Protobuf schemas per module | Transport payload definitions | Single source of truth for dndm payloads |

Deprecated packages (`pkg/backend`, `pkg/pipeline`, `pkg/plugin`, `pkg/store`) retain `DEPRECATED.md` pointers and will be dismantled as functionality migrates.

## Guiding Principles

- **Domain-by-Feature**: Organize packages by robot capability (control, math, vision) rather than legacy layers.
- **Device Independence**: `pkg/control` operates on typed messages, never on hardware handles.
- **Explicit Integration**: All transport payloads flow through protobuf schemas in `proto/types/<module>`.
- **Options Pattern**: Constructors accept functional options for easy extension without interface churn.
- **Single Responsibility**: Each subpackage owns a clear control aspect (e.g., locomotion coordination vs kinematic math).
- **No Global State**: Prefer dependency injection and context propagation for cancellation/timeouts.

## Milestones

### M1 — Documentation & Scaffolding

- Add this implementation plan and companion `SPEC.md` updates summarizing new boundaries.
- Document deprecated packages with links to replacements (already satisfied via `DEPRECATED.md` files).
- Create README in `pkg/control` outlining subpackage responsibilities and cross-project integration (EasyRobot ↔ dndm ↔ TinyGo).

### M2 — Protobuf Consolidation

- Inventory existing payloads currently defined ad hoc (e.g., locomotion, sensor streams).
- Relocate schemas into `proto/types/<module>/...` directories with consistent naming (e.g., `proto/types/control/locomotion.proto`).
- Update code generation scripts and ensure downstream consumer repos (dndm, TinyGo firmware) reference the new paths.

### M3 — Package Realignment

- Move locomotion package from `pkg/robot/locomotion` to `pkg/control/locomotion`, updating imports and tests.
- Extract math-heavy components (IK, filters, trajectory planners) from `pkg/control` into `pkg/math/control`.
- Create `pkg/devices` for platform-agnostic device APIs; mirror implementations for TinyGo under `devices/`.
- Audit planning-related logic (currently scattered) and stage move into `pkg/planning`.

### M4 — Integration & Tooling

- Define dndm topic conventions for control/vision/planning pipelines; document in module READMEs.
- Update CI/lint rules to enforce new package boundaries (e.g., disallow `pkg/control` importing `pkg/devices`).
- Provide migration guides for dependent applications (`cmd/vision`, external robots) to adopt new paths.

### M5 — Deprecation Cleanup

- Remove legacy registries (`pkg/backend`, `pkg/pipeline`, `pkg/plugin`, `pkg/store`) after consumers switch.
- Delete deprecated directories once build/test suites confirm green state.
- Announce final structure in project documentation and release notes.

## Dependencies & Coordination

- **dndm Library**: Primary transport; ensure topic schemas and message contracts live in protobuf directories.
- **TinyGo Repos**: Requires synchronized updates for firmware topic names and config loaders.
- **Math Packages**: `pkg/math/vec`, `pkg/math/mat`, `pkg/math/control` serve as foundational dependencies.
- **Vision Pipeline**: `cmd/vision` must align with new protobuf payloads and planning inputs.

## Risks & Mitigations

| Risk | Mitigation |
| --- | --- |
| Import churn during package moves | Stage documentation updates first; use scripted `goimports` fixes; run `go test ./...` frequently |
| Out-of-sync protobuf consumers | Version schemas, regenerate bindings in lockstep across repos, add CI check for stale generated code |
| TinyGo parity lag | Coordinate releases; provide shared config templates to avoid divergence |
| Cross-package coupling | Automate dependency checks to prevent `pkg/control` from pulling in deprecated modules |

## Success Criteria

- `pkg/control` hosts only device-independent control logic with clear subpackage boundaries.
- All dndm payloads originate from `proto/types/<module>` definitions and compile across repos.
- Deprecated packages are removed without breaking builds or downstream dependencies.
- Documentation reflects new structure, and developers can navigate packages by capability.


