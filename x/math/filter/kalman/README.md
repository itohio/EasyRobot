# Kalman Filter - Sensor Fusion Examples

This README provides practical examples of using the Kalman filter for sensor fusion in robotics applications.

## Table of Contents

1. [Basic Sensor Fusion: Accelerometer + Gyroscope](#example-1-basic-sensor-fusion-accelerometer--gyroscope)
2. [With Compass: Position and Heading Integration](#example-2-with-compass-position-and-heading-integration)
3. [Fusing Sensor Fusion with Visual Odometry](#example-3-fusing-sensor-fusion-with-visual-odometry)

---

## Example 1: Basic Sensor Fusion - Accelerometer + Gyroscope

### Scenario
You have an accelerometer and gyroscope. You want to estimate linear velocity and angular velocity.

### State Vector
```
x = [vx, vy, vz, wx, wy, wz]
   - vx, vy, vz: linear velocities (m/s)
   - wx, wy, wz: angular velocities (rad/s)
```

### Measurements
```
z = [ax, ay, az, wx_gyro, wy_gyro, wz_gyro]
   - ax, ay, az: linear accelerations (m/s²) from accelerometer
   - wx_gyro, wy_gyro, wz_gyro: angular velocities (rad/s) from gyroscope
```

### Implementation

```go
package main

import (
    "github.com/itohio/EasyRobot/pkg/core/math/filter/kalman"
    "github.com/itohio/EasyRobot/pkg/core/math/mat"
    "github.com/itohio/EasyRobot/pkg/core/math/vec"
)

func setupAccelGyroFusion(dt float32) *kalman.Kalman {
    // State dimension: 6 (vx, vy, vz, wx, wy, wz)
    // Measurement dimension: 6 (ax, ay, az, wx, wy, wz)
    n, m := 6, 6

    // State transition matrix F
    // v_new = v + a*dt  (velocity integrates acceleration)
    // w_new = w         (angular velocity stays same if no control)
    F := mat.New(6, 6,
        // vx   vy   vz   wx   wy   wz
        1,    0,    0,    0,    0,    0,  // vx = vx
        0,    1,    0,    0,    0,    0,  // vy = vy
        0,    0,    1,    0,    0,    0,  // vz = vz
        0,    0,    0,    1,    0,    0,  // wx = wx
        0,    0,    0,    0,    1,    0,  // wy = wy
        0,    0,    0,    0,    0,    1,  // wz = wz
    )

    // Measurement matrix H
    // We directly observe velocity through acceleration integration
    // For velocity: we observe a = (v - v_prev)/dt, so we model this as
    // We actually observe acceleration and angular velocity directly
    // Since we integrate acceleration: a_measured = (v_current - v_prev)/dt
    // For Kalman filter: we can model acceleration as difference in velocity
    // But simpler: use acceleration as control input and measure velocity changes
    
    // Actually, let's model it differently: measure acceleration and integrate it
    // H extracts what we directly measure:
    H := mat.New(6, 6,
        // For acceleration measurements: a ≈ (v - v_prev)/dt
        // For first 3 measurements (acceleration), we relate to velocity
        // For last 3 (angular velocity), we directly measure
        1/dt, 0,    0,    0,    0,    0,  // ax relates to vx/dt
        0,    1/dt, 0,    0,    0,    0,  // ay relates to vy/dt
        0,    0,    1/dt, 0,    0,    0,  // az relates to vz/dt
        0,    0,    0,    1,    0,    0,  // wx directly measured
        0,    0,    0,    0,    1,    0,  // wy directly measured
        0,    0,    0,    0,    0,    1,  // wz directly measured
    )

    // Process noise covariance Q
    // Uncertainty in velocity integration and angular velocity drift
    processNoise := float32(0.1)
    Q := mat.New(6, 6,
        processNoise, 0,           0,           0,           0,           0,
        0,           processNoise, 0,           0,           0,           0,
        0,           0,           processNoise, 0,           0,           0,
        0,           0,           0,           processNoise, 0,           0,
        0,           0,           0,           0,           processNoise, 0,
        0,           0,           0,           0,           0,           processNoise,
    )

    // Measurement noise covariance R
    // Uncertainty in accelerometer and gyroscope measurements
    accelNoise := float32(0.5)  // m/s² noise
    gyroNoise := float32(0.01)  // rad/s noise
    R := mat.New(6, 6,
        accelNoise, 0,         0,         0,       0,       0,
        0,         accelNoise, 0,         0,       0,       0,
        0,         0,         accelNoise, 0,       0,       0,
        0,         0,         0,         gyroNoise, 0,       0,
        0,         0,         0,         0,       gyroNoise, 0,
        0,         0,         0,         0,       0,       gyroNoise,
    )

    filter := kalman.New(n, m, F, H, Q, R)

    // Set initial state (all velocities start at zero)
    filter.SetState(vec.NewFrom(0, 0, 0, 0, 0, 0))

    // Set initial covariance (uncertainty about initial velocities)
    initialP := mat.New(6, 6)
    initialP.Eye()
    initialP.MulC(10.0)  // Higher initial uncertainty
    filter.SetCovariance(initialP)

    return filter
}

// Usage example
func processAccelGyro(filter *kalman.Kalman, ax, ay, az, wx, wy, wz float32, dt float32) {
    // Update state transition matrix with current dt
    // (In practice, you'd rebuild F or use control input)
    
    // Prepare measurement
    measurement := vec.NewFrom(ax, ay, az, wx, wy, wz)
    
    // Update filter
    filter.Predict()
    filter.UpdateMeasurement(measurement)
    
    // Get estimated velocities
    state := filter.GetOutput()
    vx := state[0]
    vy := state[1]
    vz := state[2]
    wx_est := state[3]
    wy_est := state[4]
    wz_est := state[5]
    
    // Use estimated velocities...
}
```

### Alternative Approach: Using Control Input

A better approach is to use acceleration as a control input:

```go
func setupAccelGyroFusionWithControl(dt float32) *kalman.Kalman {
    n, m, k := 6, 3, 3  // 6 states, 3 measurements (gyro), 3 control (accel)

    // State transition: velocities integrate acceleration, angular velocities stay same
    F := mat.New(6, 6)
    F.Eye()  // Identity: v_new = v, w_new = w (without control)

    // Control matrix B: acceleration affects velocity
    B := mat.New(6, 3,
        dt, 0,  0,  // vx += ax*dt
        0,  dt, 0,  // vy += ay*dt
        0,  0,  dt, // vz += az*dt
        0,  0,  0,  // wx unchanged
        0,  0,  0,  // wy unchanged
        0,  0,  0,  // wz unchanged
    )

    // Measurement matrix: we directly measure angular velocity
    H := mat.New(3, 6,
        0, 0, 0, 1, 0, 0,  // measure wx
        0, 0, 0, 0, 1, 0,  // measure wy
        0, 0, 0, 0, 0, 1,  // measure wz
    )

    Q := mat.New(6, 6)
    Q.Eye()
    Q.MulC(0.1)

    R := mat.New(3, 3)
    R.Eye()
    R.MulC(0.01)  // gyro noise

    filter := kalman.NewWithControl(n, m, k, F, H, B, Q, R)
    filter.SetState(vec.NewFrom(0, 0, 0, 0, 0, 0))
    
    initialP := mat.New(6, 6)
    initialP.Eye()
    initialP.MulC(10.0)
    filter.SetCovariance(initialP)

    return filter
}

func processWithControl(filter *kalman.Kalman, ax, ay, az, wx, wy, wz float32) {
    // Control input: acceleration
    control := vec.NewFrom(ax, ay, az)
    
    // Measurement: angular velocity
    measurement := vec.NewFrom(wx, wy, wz)
    
    filter.PredictWithControl(control)
    filter.UpdateMeasurement(measurement)
    
    state := filter.GetOutput()
    // state[0:3] = velocities, state[3:6] = angular velocities
}
```

---

## Example 2: With Compass - Position and Heading Integration

### Scenario
You have accelerometer, gyroscope, and compass. You want to estimate:
- Linear velocity (from accelerometer integration)
- Angular velocity (from gyroscope)
- Position (integrated from velocity)
- Heading (integrated from angular velocity, corrected by compass)

### State Vector
```
x = [px, py, pz, vx, vy, vz, heading, wx, wy, wz]
   - px, py, pz: position (m)
   - vx, vy, vz: linear velocity (m/s)
   - heading: orientation angle (rad)
   - wx, wy, wz: angular velocities (rad/s)
```

### Measurements
```
z = [ax, ay, az, heading_compass, wx_gyro, wy_gyro, wz_gyro]
   - ax, ay, az: linear acceleration (m/s²)
   - heading_compass: heading angle from compass (rad)
   - wx_gyro, wy_gyro, wz_gyro: angular velocities (rad/s)
```

### Implementation

```go
func setupAccelGyroCompassFusion(dt float32) *kalman.Kalman {
    // State: [px, py, pz, vx, vy, vz, heading, wx, wy, wz]
    // Measurements: [ax, ay, az, heading_compass, wx, wy, wz]
    n, m, k := 10, 7, 3

    // State transition matrix F
    // p_new = p + v*dt
    // v_new = v + a*dt (via control)
    // heading_new = heading + wz*dt
    // w_new = w
    F := mat.New(10, 10,
        // px   py   pz   vx   vy   vz   heading wx   wy   wz
        1,    0,    0,    dt,  0,    0,    0,      0,   0,   0,  // px = px + vx*dt
        0,    1,    0,    0,   dt,  0,    0,      0,   0,   0,  // py = py + vy*dt
        0,    0,    1,    0,   0,    dt,   0,      0,   0,   0,  // pz = pz + vz*dt
        0,    0,    0,    1,   0,    0,    0,      0,   0,   0,  // vx = vx
        0,    0,    0,    0,   1,    0,    0,      0,   0,   0,  // vy = vy
        0,    0,    0,    0,   0,    1,    0,      0,   0,   0,  // vz = vz
        0,    0,    0,    0,   0,    0,    1,      0,   0,   dt, // heading = heading + wz*dt
        0,    0,    0,    0,   0,    0,    0,      1,   0,   0,  // wx = wx
        0,    0,    0,    0,   0,    0,    0,      0,   1,   0,  // wy = wy
        0,    0,    0,    0,   0,    0,    0,      0,   0,   1,  // wz = wz
    )

    // Control matrix B: acceleration affects velocity
    B := mat.New(10, 3,
        // ax   ay   az
        0,    0,    0,  // px unchanged
        0,    0,    0,  // py unchanged
        0,    0,    0,  // pz unchanged
        dt,   0,    0,  // vx += ax*dt
        0,    dt,   0,  // vy += ay*dt
        0,    0,    dt, // vz += az*dt
        0,    0,    0,  // heading unchanged
        0,    0,    0,  // wx unchanged
        0,    0,    0,  // wy unchanged
        0,    0,    0,  // wz unchanged
    )

    // Measurement matrix H
    // We measure: acceleration (indirectly via velocity changes), compass heading, angular velocity
    // For acceleration: we observe velocity changes, so we model as derivative
    // Actually simpler: we can't directly measure velocity from acceleration in one step
    // So we'll measure position changes, compass heading, and angular velocity
    
    // Better approach: Measure velocity (through integration history), compass, gyro
    // Or: use accelerometer as control, measure compass + gyro
    H := mat.New(7, 10,
        // px   py   pz   vx   vy   vz   heading wx   wy   wz
        0,    0,    0,    1,   0,    0,    0,      0,   0,   0,  // measure vx (via accel integration)
        0,    0,    0,    0,   1,    0,    0,      0,   0,   0,  // measure vy
        0,    0,    0,    0,   0,    1,    0,      0,   0,   0,  // measure vz
        0,    0,    0,    0,   0,    0,    1,      0,   0,   0,  // measure heading from compass
        0,    0,    0,    0,   0,    0,    0,      1,   0,   0,  // measure wx from gyro
        0,    0,    0,    0,   0,    0,    0,      0,   1,   0,  // measure wy from gyro
        0,    0,    0,    0,   0,    0,    0,      0,   0,   1,  // measure wz from gyro
    )

    // Process noise
    Q := mat.New(10, 10)
    Q.Eye()
    Q.MulC(0.01)

    // Measurement noise
    R := mat.New(7, 7)
    R.Eye()
    // Higher noise for velocity (integrated) and compass, lower for gyro
    R[0][0] = 0.1  // vx noise
    R[1][1] = 0.1  // vy noise
    R[2][2] = 0.1  // vz noise
    R[3][3] = 0.05 // compass heading noise
    R[4][4] = 0.01 // wx noise
    R[5][5] = 0.01 // wy noise
    R[6][6] = 0.01 // wz noise

    filter := kalman.NewWithControl(n, m, k, F, H, B, Q, R)
    
    // Initial state: start at origin
    filter.SetState(vec.NewFrom(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    
    // Initial covariance
    initialP := mat.New(10, 10)
    initialP.Eye()
    initialP.MulC(1.0)
    filter.SetCovariance(initialP)

    return filter
}

func processAccelGyroCompass(
    filter *kalman.Kalman,
    ax, ay, az float32,
    heading_compass float32,
    wx, wy, wz float32,
) {
    // Control: acceleration
    control := vec.NewFrom(ax, ay, az)
    
    // Measurement: velocity (from integration), compass heading, angular velocity
    // Note: In practice, velocity measurement comes from integrating acceleration
    // For this example, we estimate velocity from the filter state
    state := filter.GetOutput()
    measurement := vec.NewFrom(
        state[3],              // vx estimate
        state[4],              // vy estimate
        state[5],              // vz estimate
        heading_compass,       // compass heading
        wx, wy, wz,           // gyro measurements
    )
    
    filter.PredictWithControl(control)
    filter.UpdateMeasurement(measurement)
    
    // Get updated state
    state = filter.GetOutput()
    px := state[0]
    py := state[1]
    pz := state[2]
    vx := state[3]
    vy := state[4]
    vz := state[5]
    heading := state[6]
    wx_est := state[7]
    wy_est := state[8]
    wz_est := state[9]
    
    // Use integrated position, heading, velocities...
}
```

**Note**: The velocity measurement in the above example is a simplification. In practice, you might:
- Use a separate velocity integrator and feed that as measurement
- Or measure position from another source (GPS, visual odometry) and derive velocity
- Or use a different filter architecture

---

## Example 3: Fusing Sensor Fusion with Visual Odometry

### Scenario
You have two sources of velocity and angular velocity estimates:
1. **Sensor Fusion**: Velocity and angular velocity from IMU (accelerometer + gyroscope)
2. **Visual Odometry**: Velocity and angular velocity from camera/visual processing

You want to fuse these to get optimal estimates of:
- Position (integrated from fused velocity)
- Heading (integrated from fused angular velocity)
- Velocity (fused estimate)
- Angular velocity (fused estimate)

### State Vector
```
x = [px, py, pz, heading, vx, vy, vz, wx, wy, wz]
   - px, py, pz: position (m)
   - heading: orientation angle (rad)
   - vx, vy, vz: linear velocity (m/s)
   - wx, wy, wz: angular velocity (rad/s)
```

### Measurements
```
z = [vx_imu, vy_imu, vz_imu, wx_imu, wy_imu, wz_imu,
     vx_vo,  vy_vo,  vz_vo,  wx_vo,  wy_vo,  wz_vo]
   - First 6: velocity and angular velocity from IMU sensor fusion
   - Last 6: velocity and angular velocity from visual odometry
```

### Implementation

```go
func setupVOIMUFusion(dt float32) *kalman.Kalman {
    // State: [px, py, pz, heading, vx, vy, vz, wx, wy, wz]
    // Measurements: [vx_imu, vy_imu, vz_imu, wx_imu, wy_imu, wz_imu,
    //                vx_vo,  vy_vo,  vz_vo,  wx_vo,  wy_vo,  wz_vo]
    n, m := 10, 12

    // State transition matrix F
    // p_new = p + v*dt
    // heading_new = heading + wz*dt
    // v_new = v (velocity stays same, updated by measurements)
    // w_new = w (angular velocity stays same, updated by measurements)
    F := mat.New(10, 10,
        // px   py   pz   heading vx   vy   vz   wx   wy   wz
        1,    0,    0,    0,       dt,  0,    0,    0,   0,   0,  // px = px + vx*dt
        0,    1,    0,    0,       0,   dt,   0,    0,   0,   0,  // py = py + vy*dt
        0,    0,    1,    0,       0,   0,    dt,   0,   0,   0,  // pz = pz + vz*dt
        0,    0,    0,    1,       0,   0,    0,    0,   0,   dt, // heading = heading + wz*dt
        0,    0,    0,    0,       1,   0,    0,    0,   0,   0,  // vx = vx
        0,    0,    0,    0,       0,   1,    0,    0,   0,   0,  // vy = vy
        0,    0,    0,    0,       0,   0,    1,    0,   0,   0,  // vz = vz
        0,    0,    0,    0,       0,   0,    0,    1,   0,   0,  // wx = wx
        0,    0,    0,    0,       0,   0,    0,    0,   1,   0,  // wy = wy
        0,    0,    0,    0,       0,   0,    0,    0,   0,   1,  // wz = wz
    )

    // Measurement matrix H
    // We directly measure velocity and angular velocity from both sources
    H := mat.New(12, 10,
        // px   py   pz   heading vx   vy   vz   wx   wy   wz
        // IMU measurements
        0,    0,    0,    0,       1,   0,    0,    0,   0,   0,  // vx_imu
        0,    0,    0,    0,       0,   1,    0,    0,   0,   0,  // vy_imu
        0,    0,    0,    0,       0,   0,    1,    0,   0,   0,  // vz_imu
        0,    0,    0,    0,       0,   0,    0,    1,   0,   0,  // wx_imu
        0,    0,    0,    0,       0,   0,    0,    0,   1,   0,  // wy_imu
        0,    0,    0,    0,       0,   0,    0,    0,   0,   1,  // wz_imu
        // Visual odometry measurements
        0,    0,    0,    0,       1,   0,    0,    0,   0,   0,  // vx_vo
        0,    0,    0,    0,       0,   1,    0,    0,   0,   0,  // vy_vo
        0,    0,    0,    0,       0,   0,    1,    0,   0,   0,  // vz_vo
        0,    0,    0,    0,       0,   0,    0,    1,   0,   0,  // wx_vo
        0,    0,    0,    0,       0,   0,    0,    0,   1,   0,  // wy_vo
        0,    0,    0,    0,       0,   0,    0,    0,   0,   1,  // wz_vo
    )

    // Process noise covariance Q
    // Uncertainty in position integration and velocity/angular velocity estimation
    Q := mat.New(10, 10)
    Q.Eye()
    Q[0][0] = 0.01  // px process noise
    Q[1][1] = 0.01  // py process noise
    Q[2][2] = 0.01  // pz process noise
    Q[3][3] = 0.001 // heading process noise
    Q[4][4] = 0.05  // vx process noise
    Q[5][5] = 0.05  // vy process noise
    Q[6][6] = 0.05  // vz process noise
    Q[7][7] = 0.01  // wx process noise
    Q[8][8] = 0.01  // wy process noise
    Q[9][9] = 0.01  // wz process noise

    // Measurement noise covariance R
    // Different noise levels for IMU vs Visual Odometry
    // Typically: VO has lower noise for velocity, IMU has lower noise for angular velocity
    imu_vel_noise := float32(0.2)   // IMU velocity uncertainty (higher)
    imu_ang_noise := float32(0.05)  // IMU angular velocity uncertainty (lower)
    vo_vel_noise := float32(0.1)    // VO velocity uncertainty (lower)
    vo_ang_noise := float32(0.2)    // VO angular velocity uncertainty (higher)
    
    R := mat.New(12, 12)
    // IMU measurements (first 6)
    R[0][0] = imu_vel_noise
    R[1][1] = imu_vel_noise
    R[2][2] = imu_vel_noise
    R[3][3] = imu_ang_noise
    R[4][4] = imu_ang_noise
    R[5][5] = imu_ang_noise
    // Visual odometry measurements (last 6)
    R[6][6] = vo_vel_noise
    R[7][7] = vo_vel_noise
    R[8][8] = vo_vel_noise
    R[9][9] = vo_ang_noise
    R[10][10] = vo_ang_noise
    R[11][11] = vo_ang_noise
    
    // Cross-correlations between IMU and VO (typically small but can model)
    // For simplicity, we assume independent measurements

    filter := kalman.New(n, m, F, H, Q, R)
    
    // Initial state
    filter.SetState(vec.NewFrom(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    
    // Initial covariance
    initialP := mat.New(10, 10)
    initialP.Eye()
    initialP.MulC(1.0)
    filter.SetCovariance(initialP)

    return filter
}

func processVOIMU(
    filter *kalman.Kalman,
    vx_imu, vy_imu, vz_imu, wx_imu, wy_imu, wz_imu float32,
    vx_vo, vy_vo, vz_vo, wx_vo, wy_vo, wz_vo float32,
) {
    // Prepare measurement vector
    measurement := vec.NewFrom(
        vx_imu, vy_imu, vz_imu, wx_imu, wy_imu, wz_imu,  // IMU
        vx_vo, vy_vo, vz_vo, wx_vo, wy_vo, wz_vo,        // Visual Odometry
    )
    
    // Predict and update
    filter.Predict()
    filter.UpdateMeasurement(measurement)
    
    // Get fused estimates
    state := filter.GetOutput()
    px := state[0]
    py := state[1]
    pz := state[2]
    heading := state[3]
    vx_fused := state[4]
    vy_fused := state[5]
    vz_fused := state[6]
    wx_fused := state[7]
    wy_fused := state[8]
    wz_fused := state[9]
    
    // Use fused position, heading, velocities...
}
```

### Key Points

1. **Noise Tuning**: The measurement noise covariance `R` should reflect the relative reliability of each sensor:
   - Visual odometry typically has better velocity estimates (lower noise)
   - IMU typically has better angular velocity estimates (lower noise)
   - Adjust based on your specific sensors

2. **Cross-correlations**: You can add off-diagonal terms in `R` to model correlation between IMU and VO errors if they're not independent

3. **Update Rate**: Make sure `dt` matches your sensor update rate

4. **Initial Conditions**: Set appropriate initial covariance based on your starting conditions

---

## General Tips

1. **Tune Noise Matrices**: `Q` and `R` are critical for filter performance. Start with reasonable values and tune based on actual sensor characteristics.

2. **Handle Missing Measurements**: You can skip updates if a measurement source is unavailable, or use a high noise value in `R` for that measurement.

3. **Coordinate Frames**: Ensure all measurements are in the same coordinate frame (e.g., body frame, world frame).

4. **Angle Wrapping**: For heading angles, you may need to handle wrapping (2π → 0) outside the filter.

5. **Numerical Stability**: If the innovation covariance `S` becomes singular, the filter will handle it by using identity matrix fallback, but you should monitor this.

6. **Testing**: Validate your filter with simulated data first, then move to real sensor data.

