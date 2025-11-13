# Kalman Filter vs Mahony/Madgwick AHRS Comparison

This document explains the key differences between Kalman filter and Mahony/Madgwick methods for sensor fusion, particularly in the context of IMU (Inertial Measurement Unit) data fusion.

## Overview

### Kalman Filter
- **Type**: Bayesian state estimator (statistical approach)
- **Purpose**: General-purpose state estimation for linear dynamic systems
- **Algorithm**: Optimal in the minimum mean-squared error sense
- **Output**: Any state vector you define (position, velocity, orientation, etc.)

### Mahony & Madgwick AHRS
- **Type**: Complementary filter (geometric approach)
- **Purpose**: Specifically designed for attitude/heading estimation
- **Algorithm**: Gradient descent (Madgwick) or PI controller (Mahony) on orientation error
- **Output**: Quaternion orientation only

---

## Detailed Comparison

### 1. Algorithm Philosophy

#### Kalman Filter
- **Statistical approach**: Models system and measurement uncertainties probabilistically
- **Optimal estimation**: Minimizes expected squared error under Gaussian noise assumption
- **Explicit uncertainty**: Maintains covariance matrix representing state uncertainty
- **General framework**: Works for any linear (or linearized) system model

#### Mahony/Madgwick
- **Geometric approach**: Directly estimates orientation from sensor data
- **Complementary filtering**: Fuses high-frequency (gyro) and low-frequency (accel/mag) data
- **No explicit uncertainty**: Does not maintain covariance estimates
- **Specialized**: Designed specifically for quaternion-based orientation estimation

---

### 2. State Representation

#### Kalman Filter
```go
// You define your state vector
x = [px, py, pz, vx, vy, vz, heading, wx, wy, wz, ...]
   // Any variables you want to estimate
```
- Flexible: Define any state vector
- Can estimate position, velocity, orientation, and other quantities simultaneously
- Requires you to define state transition model

#### Mahony/Madgwick
```go
// Fixed state: quaternion orientation
q = [w, x, y, z]  // quaternion representation
```
- Fixed: Only estimates orientation quaternion
- Cannot directly estimate position or velocity
- Internally handles quaternion dynamics

---

### 3. Mathematical Model

#### Kalman Filter

**State Transition Model:**
```
x(k+1) = F * x(k) + B * u(k) + w(k)
```
- Explicit state transition matrix `F`
- Optional control input matrix `B`
- Process noise `w(k)` with covariance `Q`

**Measurement Model:**
```
z(k) = H * x(k) + v(k)
```
- Measurement matrix `H` relates measurements to state
- Measurement noise `v(k)` with covariance `R`

**Update Equations:**
1. **Predict**: `x_pred = F*x`, `P_pred = F*P*F^T + Q`
2. **Update**: `K = P*H^T*(H*P*H^T + R)^-1`, `x = x_pred + K*(z - H*x_pred)`

#### Mahony/Madgwick

**Madgwick (Gradient Descent):**
```
// Orientation error from accelerometer and magnetometer
error = f(quaternion, accel, mag)

// Update quaternion using gradient descent
q_dot = 0.5 * q * [0, wx, wy, wz] - beta * (error / |error|)
```

**Mahony (PI Controller):**
```
// Cross product error between estimated and measured directions
error = cross(estimated_gravity, measured_accel) + 
        cross(estimated_mag_field, measured_mag)

// Integral term
eInt += error

// Corrected angular velocity with PI feedback
gyro_corrected = gyro + Kp*error + Ki*eInt

// Integrate quaternion
q_dot = 0.5 * q * [0, gyro_corrected]
```

---

### 4. Computational Complexity

#### Kalman Filter
- **Matrix operations**: Matrix multiplication, inversion, transpose
- **Complexity**: O(n³) where n is state dimension (due to matrix inversion)
- **Memory**: Stores multiple n×n matrices (P, Q, R, F, H, K, S, etc.)
- **Typical state dimension**: 6-20 for IMU applications
- **Computation**: ~100-1000 floating point operations per update

#### Mahony/Madgwick
- **Vector operations**: Mostly quaternion and 3D vector operations
- **Complexity**: O(1) - constant time, independent of any state dimension
- **Memory**: Stores only quaternion and error integral (if Mahony)
- **Computation**: ~50-200 floating point operations per update

**Winner**: Mahony/Madgwick is significantly faster and uses less memory

---

### 5. Tuning Parameters

#### Kalman Filter
- **Process noise covariance Q**: Uncertainty in state dynamics
  - Higher Q = trust measurements more, less trust in predictions
  - Lower Q = trust predictions more, smoother but less responsive
- **Measurement noise covariance R**: Uncertainty in sensor measurements
  - Higher R = less trust in that measurement
  - Lower R = more trust in that measurement
- **State transition matrix F**: Model of how state evolves
- **Measurement matrix H**: Model of how sensors relate to state

**Tuning difficulty**: Medium-High
- Need to understand system dynamics
- Need to estimate noise characteristics
- Multiple parameters to tune
- Can be data-driven (expectation-maximization) or manual

#### Mahony
- **Kp (Proportional gain)**: How quickly to correct orientation errors
  - Higher Kp = faster response but more noise
  - Lower Kp = slower response but smoother
- **Ki (Integral gain)**: How much to accumulate error for drift correction
  - Higher Ki = better long-term drift correction
  - Lower Ki = less drift correction

#### Madgwick
- **Beta (convergence rate)**: Gradient descent step size
  - Higher Beta = faster convergence but potential overshoot
  - Lower Beta = slower but more stable

**Tuning difficulty**: Low
- Usually 1-2 parameters to tune
- Standard values often work well
- Less dependent on understanding system model

---

### 6. Sensor Requirements

#### Kalman Filter
- **Flexible**: Can work with any combination of sensors
- **Model required**: Must define how sensors relate to state
- **Example configurations**:
  - Accel + Gyro → estimate velocity, angular velocity
  - Accel + Gyro + Compass → estimate position, heading, velocities
  - GPS + IMU → estimate position, velocity, orientation
  - Visual Odometry + IMU → fused estimates

#### Mahony/Madgwick
- **Fixed**: Designed for Accel + Gyro (required), Magnetometer (optional)
- **Sensor model**: Built-in (assumes gravity and magnetic field reference)
- **Cannot easily add**: GPS, visual odometry, or other sensors
- **Works best**: With standard 9-DOF IMU (3-axis accel, 3-axis gyro, 3-axis mag)

---

### 7. Uncertainty and Confidence

#### Kalman Filter
- **Explicit uncertainty**: Covariance matrix `P` tracks uncertainty in each state element
- **Information fusion**: Automatically weights measurements by their uncertainty
- **Anomaly detection**: Can detect outliers using innovation covariance
- **Confidence intervals**: Can provide confidence bounds on estimates

#### Mahony/Madgwick
- **No explicit uncertainty**: Does not track uncertainty
- **Fixed weighting**: Sensor fusion is fixed (cannot adjust per-sensor confidence)
- **No anomaly detection**: Cannot automatically detect bad measurements
- **Binary confidence**: Either estimate is good or bad (user must decide)

---

### 8. Use Cases

#### When to Use Kalman Filter

✅ **Use Kalman filter when:**
- You need to estimate multiple quantities (position, velocity, orientation, etc.)
- You have multiple different sensor types to fuse (GPS, IMU, visual odometry, etc.)
- You need uncertainty estimates (covariance)
- You have a good system model
- You can invest time in tuning
- You need optimal performance (minimum error)
- You want to handle missing measurements elegantly
- You need to estimate quantities beyond orientation

**Examples:**
- Robot position/velocity estimation from multiple sensors
- Fusing IMU with visual odometry
- GPS + IMU navigation
- Sensor fusion with position, velocity, and orientation

#### When to Use Mahony/Madgwick

✅ **Use Mahony/Madgwick when:**
- You only need orientation (attitude/heading) estimation
- You have standard 9-DOF IMU (accel + gyro + mag)
- You need fast, low-memory computation
- You want simple tuning
- You're working on embedded systems with limited resources
- You don't need uncertainty estimates
- Orientation-only applications (drones, wearable devices, game controllers)

**Examples:**
- Drone attitude estimation
- Wearable device orientation tracking
- VR/AR headset orientation
- Camera gimbal stabilization
- Game controller orientation

---

### 9. Code Example Comparison

#### Kalman Filter Example
```go
// Complex setup with matrices
n, m := 10, 7  // state dim, measurement dim
F := mat.New(10, 10, /* state transition */)
H := mat.New(7, 10, /* measurement */)
Q := mat.New(10, 10, /* process noise */)
R := mat.New(7, 7, /* measurement noise */)

filter := kalman.NewWithControl(n, m, k, F, H, B, Q, R)
filter.SetState(initialState)
filter.SetCovariance(initialP)

// Update
filter.PredictWithControl(accel)
filter.UpdateMeasurement(measurements)

state := filter.GetOutput()
// state[0:3] = position
// state[3:6] = velocity
// state[6] = heading
// state[7:10] = angular velocity
```

#### Mahony/Madgwick Example
```go
// Simple setup
ahrs := ahrs.NewMahony(
    ahrs.WithKP(0.1),
    ahrs.WithKI(0.1),
    ahrs.WithMagnetometer(true),
)

// Update (set sensor values first)
ahrs.Acceleration().CopyFrom(0, accel_reading)
ahrs.Gyroscope().CopyFrom(0, gyro_reading)
ahrs.Magnetometer().CopyFrom(0, mag_reading)

ahrs.Update(dt).Calculate()

quaternion := ahrs.Orientation()
// Only orientation available
```

---

### 10. Handling Missing Measurements

#### Kalman Filter
- **Elegant**: Can skip measurement updates
- **Uncertainty growth**: State uncertainty increases naturally when measurements missing
- **Automatic weighting**: Less certain predictions when measurements unavailable
- **Flexible**: Can model sensor availability in measurement matrix

#### Mahony/Madgwick
- **Limited**: Requires at least accel + gyro
- **Degraded performance**: Without magnetometer, heading drift accumulates
- **No uncertainty model**: Cannot represent confidence in estimates

---

### 11. Drift and Long-Term Stability

#### Kalman Filter
- **Depends on model**: Good model = low drift
- **Measurement correction**: Regular measurements prevent drift
- **Explicit drift**: Process noise Q represents drift uncertainty
- **Can correct**: Position/velocity corrections from GPS or visual odometry prevent drift

#### Mahony
- **Integral term**: Ki parameter helps correct long-term drift
- **Gyro bias**: Can accumulate over time
- **Compass correction**: Magnetometer helps prevent heading drift
- **Accel-only**: Position drift cannot be corrected without external reference

#### Madgwick
- **No integral**: Beta parameter provides some drift correction
- **Gradient descent**: Converges toward true orientation
- **Long-term**: May accumulate drift without compass

---

### 12. Initial Conditions

#### Kalman Filter
- **Initial state**: Must provide initial state estimate
- **Initial covariance**: Must provide initial uncertainty
- **Convergence**: Can take time to converge depending on initial uncertainty
- **Sensor initialization**: Can use first measurements to initialize

#### Mahony/Madgwick
- **Initial quaternion**: Usually starts at identity (level, north-facing)
- **Fast startup**: Quickly converges from any initial orientation
- **No initial uncertainty**: Assumes perfect initial orientation (usually acceptable)

---

## Hybrid Approaches

You can combine both:

1. **Mahony/Madgwick for orientation + Kalman for position/velocity**:
   - Use AHRS to get quaternion
   - Use Kalman filter to estimate position, velocity using orientation from AHRS
   - Best of both worlds: fast orientation, flexible state estimation

2. **Kalman filter with orientation from AHRS as measurement**:
   - Use AHRS as a "sensor" providing orientation measurements
   - Kalman filter fuses AHRS orientation with other sensors (GPS, visual odometry)
   - AHRS provides high-frequency orientation updates

---

## Performance Summary

| Feature | Kalman Filter | Mahony | Madgwick |
|---------|---------------|--------|---------|
| **Orientation accuracy** | High | High | High |
| **Computational cost** | High | Low | Low |
| **Memory usage** | High | Low | Low |
| **Tuning difficulty** | Medium-High | Low | Low |
| **Flexibility** | Very High | Low | Low |
| **Uncertainty tracking** | Yes | No | No |
| **Multi-sensor fusion** | Easy | Difficult | Difficult |
| **Position estimation** | Yes | No | No |
| **Velocity estimation** | Yes | No | No |
| **Embedded suitability** | Medium | High | High |

---

## Recommendations

### For Your Use Cases

#### 1. **Accelerometer + Gyroscope (Basic Sensor Fusion)**
- **If you only need orientation**: Use **Madgwick** or **Mahony**
- **If you need velocity estimation**: Use **Kalman filter**

#### 2. **Accelerometer + Gyroscope + Compass (Position and Heading Integration)**
- **If you only need orientation + heading**: Use **Mahony** or **Madgwick** with magnetometer
- **If you need position, velocity, heading**: Use **Kalman filter**

#### 3. **Fusing Sensor Fusion with Visual Odometry**
- **Must use Kalman filter**: Only Kalman filter can fuse multiple velocity/angular velocity sources
- Mahony/Madgwick cannot handle visual odometry input

---

## Conclusion

**Choose Mahony/Madgwick when:**
- You only need orientation quaternion
- You have limited computational resources
- You want simple, fast implementation
- You have standard 9-DOF IMU

**Choose Kalman Filter when:**
- You need to estimate position, velocity, or other quantities beyond orientation
- You have multiple different sensor types
- You need uncertainty estimates
- You need to fuse custom sensor combinations (like visual odometry + IMU)
- You have sufficient computational resources

Both methods are excellent at what they do. Mahony/Madgwick are specialists for orientation, while Kalman filter is a generalist that can handle any linear state estimation problem.

