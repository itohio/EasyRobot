## Kinematics Abstraction Refactoring Plan

### Goals
- Standardize how kinematics modules (wheels, joints, thrusters, etc.) expose their capabilities so they can be used standalone or embedded in motion planners.
- Support destination-based call patterns (`Forward`, `Backward`) that operate on consistent mathematical representations (vectors/matrices/tensors) for states and controls.
- Demonstrate the abstraction with a concrete example: a two-wheel balancer driven by the differential kinematics model.

### Deliverables
1. Unified kinematics interfaces and supporting data types.
2. Updated kinematics packages (wheels, thrusters, joints) implementing the new abstractions.
3. Motion planner integration adapting the new kinematics interface.
4. Showcase implementation and documentation for the two-wheel balancer example.

### Step-by-Step Plan

#### 1. Analyze Existing Math Primitives
- Review `pkg/core/math/mat`, `pkg/core/math/vec`, and `pkg/core/math/tensor` to determine the most appropriate data structures for representing:
  - Body state vectors (position, orientation, velocities).
  - Control / actuator vectors (wheel speeds, joint angles, thruster forces).
  - Constraint matrices (e.g., Jacobians or mapping matrices).
- Document chosen conventions (row-major vs column-major, indexing semantics) to ensure consistency across modules.

#### 2. Define Core Abstraction Interfaces
- Introduce a `pkg/core/math/control/kinematics/types` package that owns the shared interfaces (e.g., `Model`, `ForwardKinematics`, `BackwardKinematics`) and foundational data structures:
  - `Configure(params Config, constraints Constraints) error`
  - `Forward(state StateMatrix, destination StateMatrix, controls ControlVector) (StateMatrix, error)`
  - `Backward(state StateMatrix, destination StateMatrix, controls ControlVector) (ControlVector, error)`
- Define associated struct types (`Config`, `Constraints`, `StateMatrix`, `ControlVector`) using the math primitives selected in Step 1.
- Ensure interfaces capture metadata about DoFs, holonomic constraints, available control axes, and actuator limits.

#### 3. Establish Destination-Based Call Pattern
- Clarify expectations for `Forward` and `Backward`:
  - Inputs: current state, destination state, current controls.
  - Outputs: next state prediction (`Forward`) or control command update (`Backward`).
- Specify how destination states are provided (e.g., as matrices with rows per joint and columns per quantity).
- Provide guidelines for handling absent data (e.g., missing velocities) via zero-filled rows or optional fields.

#### 4. Update Existing Kinematics Modules
- Refactor wheels, thrusters, joints, and related kinematics packages to implement the new interfaces. Constructors/factories must return concrete struct pointers (never interfaces) to keep allocations explicit:
  - Encapsulate static configuration (geometry, mass, limits) in constructors or dedicated `Configure` calls.
  - Adapt internal calculations to use the standardized matrices/vectors.
  - Expose Forward/Backward methods conforming to the destination-based pattern.
- Add unit tests verifying:
  - Interface compliance.
  - Correct mapping between state/control representations and actuator outputs.

#### 5. Integrate with Motion Planner
- Adjust motion planner to depend on the new kinematics interfaces instead of concrete wheel/thruster implementations.
- Modify planner `Forward`/`Backward` logic to:
  - Request required transformations from the kinematics model.
  - Operate purely on body-level representations (pose, twist) while delegating actuator mapping.
- Ensure constraint handling uses metadata provided by the kinematics model (e.g., allowable lateral motion, torque limits).

#### 6. Two-Wheel Balancer Showcase
- Create a demo package (e.g., `pkg/examples/twowheel_balancer`) illustrating:
  - Instantiation of the differential drive kinematics model with configuration parameters.
  - Supplying current body state (orientation, speed), wheel states (speeds, torques), and desired body destination.
  - Using planner `Forward` to compute predicted trajectory and `Backward` to generate balancing control commands.
- Include README or GoDoc example demonstrating the expected usage pattern and highlight how the abstraction simplifies switching models.

#### 7. Documentation & Migration Guidance
- Produce documentation detailing:
  - Interface descriptions and data structure conventions.
  - Migration steps for existing kinematics clients.
  - Examples of integrating alternative models (e.g., hexapod joints).
- Update package READMEs and SPEC/PLAN documents to reflect the new architecture.

### Risks & Mitigations
- **Inconsistent matrix conventions** → mitigate with clear documentation and helper constructors.
- **Breaking existing consumers** → provide adapters or transitional wrappers; update downstream packages in tandem.
- **Over-generalization** → keep interfaces minimal; extend with optional capability interfaces as needed.

### Validation
- Run unit and integration tests across kinematics modules and motion planner.
- Verify the two-wheel balancer example maintains stability in simulation/tests.
- Peer review the abstractions to ensure they accommodate planned future models (thrusters, hexapods, etc.).

