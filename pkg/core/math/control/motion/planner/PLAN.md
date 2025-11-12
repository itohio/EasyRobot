## Motion Planner Design

### Purpose
Provide a reusable motion planner that keeps a robot aligned with and close to a supplied 3D path while respecting constraints on turning, torque, speed, acceleration, and deceleration. The planner produces predictive trajectories (forward pass) and constraint-aware control commands (backward pass) that downstream controllers can track.

### Inputs and Outputs
- **Inputs**
  - `state` – current robot state (position vector, orientation/yaw, linear velocity, timestamp).
  - `controls` – current control signals (linear velocity command, angular velocity command, torque/effort vector).
  - `path` – ordered list of 3D waypoints (XYZ) representing the desired path for the forward planner.
  - `trajectory` – desired states/poses for the backward planner (future segment extracted from higher-level planner).
  - `Constraints` – max linear velocity, max acceleration, max deceleration, allowed jerk (VAJ), max angular velocity, max angular acceleration, max lateral acceleration (torque proxy).
  - `Parameters` – sample period, lookahead distance, path smoothing window, curvature averaging window, PID gains for lateral error, blend factors for feed-forward vs feedback.
- **Outputs**
  - `Forward` returns a `Trajectory` describing pose and velocity targets that adhere to constraints and follow the path.
  - `Backward` returns bounded `Controls` that best realize the provided trajectory segment from the current state under the same constraints.

### High-Level Flow
1. **Path Tracking (Forward)**
   - Find closest waypoint to current position; compute arclength progress.
  - Project along path by configurable lookahead to derive focal waypoint.
  - Determine desired orientation by pointing toward projected waypoint; smooth with previous orientation respecting angular limits.
2. **Speed Planning (Forward)**
   - Maintain scalar progress variable along the path.
   - Feed desired progress into VAJ1D to enforce speed/accel/decel/jerk limits.
   - Clamp resulting speed based on curvature-driven angular and torque constraints (`v^2 * curvature <= lateralLimit`).
3. **Lateral Error Correction (Forward)**
   - Compute cross-track and heading errors relative to current segment.
   - PID adjusts orientation and minor lateral offset to converge toward path while maintaining stability.
4. **Trajectory Composition (Forward)**
   - Combine corrected position/orientation with constrained speed to produce future state(s) over the planning horizon (single step or short horizon array).
   - Persist internal planner state (progress index, VAJ state, PID integrals).
5. **Control Synthesis (Backward)**
   - Compare desired trajectory segment with current state.
   - Use feed-forward controls derived from trajectory derivatives (desired velocity/acceleration/turn rate).
   - Apply PID corrections on pose and velocity error; clamp outputs to constraints.
   - Return control commands (e.g., desired linear velocity, angular velocity, torque vector).

### Data Structures
- `State`
  - `Position` (`vec.Vector3D`)
  - `Yaw` (float32 radians)
  - `Speed` (float32 linear velocity magnitude)
  - `Timestamp` (float32 seconds)
- `Controls`
  - `Linear` (float32)
  - `Angular` (float32)
  - `Effort` (`vec.Vector`) optional torque/effort representation
- `Trajectory`
  - Slice of `State` covering planning horizon (minimum length 1)
- `Constraints`
  - `MaxSpeed`, `MaxAcceleration`, `MaxDeceleration`, `MaxJerk`
  - `MaxTurnRate`, `MaxTurnAccel`
  - `MaxLateralAcceleration`
- `Parameters`
  - `SamplePeriod`
  - `LookaheadDistance`
  - `CurvatureWindow`
  - `LateralPID` gains (`PID1D`)
  - `OrientationPID` gains
  - `SpeedPID` gains (for Backward control refinement)

### Public API
- `NewMotion(constraints Constraints, params Parameters) *Motion`
  - Validates inputs, initializes VAJ profile and PID controllers.
  - Precomputes path smoothing buffers if required.
- `(*Motion) Forward(state State, controls Controls, path []vec.Vector3D) (Trajectory, error)`
  - Requires non-empty path.
  - Updates internal progress toward path using VAJ/PID to generate next trajectory states.
- `(*Motion) Backward(traj Trajectory, state State, controls Controls) (Controls, error)`
  - Requires non-empty trajectory.
  - Generates bounded controls to follow the provided trajectory from current state.

### Constraint Handling
- Turning: orientation delta limited by `MaxTurnRate * dt`; turn acceleration limited by `MaxTurnAccel`.
- Torque: approximate via lateral acceleration; enforce `speed^2 * curvature <= MaxLateralAcceleration`.
- Speed/Acceleration/Deceleration: VAJ ensures smooth scalar progress; Backward clamps commanded deltas to respect same bounds.
- Deceleration near goal: when remaining path is shorter than computed stopping distance, reduce VAJ target speed toward zero.

### Error Handling & Edge Cases
- Empty path or trajectory returns `ErrInvalidInput`.
- Degenerate waypoints (duplicates) are skipped while preserving continuity.
- Planner detects large jumps in path index and resets VAJ/PID integrators accordingly.
- If already at goal (position and speed within tolerance), Forward returns stationary trajectory, Backward returns zero controls.

### Testing Strategy
- Table-driven tests:
  - Straight path constant speed (Forward).
  - Curved path verifying speed reduction (Forward).
  - Sudden stop requirement (Forward/Backward interplay).
  - Backward control generation meeting bounds.
  - Invalid inputs produce expected errors.

### Future Extensions
- Extend Controls to wheel-level commands for specific drive kinematics.
- Integrate latency compensation by predicting future states before applying controls.
- Export diagnostic metrics (curvature, errors, constraint activations) for tuning.

