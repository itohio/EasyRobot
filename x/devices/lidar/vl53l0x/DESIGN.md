## VL53L0X Array-Based LiDAR - Design

### Overview

- Device: Array of VL53L0X time-of-flight distance sensors connected via TCA9548A I2C multiplexer
- Architecture: Static sensor array with fixed angular positions
- Reference: 
  - VL53L0X: https://www.st.com/resource/en/datasheet/vl53l0x.pdf
  - TCA9548A: https://www.ti.com/lit/ds/symlink/tca9548a.pdf

### Goals

- Provide a LiDAR-like interface using an array of fixed-position ToF sensors
- Compose existing TCA9548A router and VL53L0X device implementations
- Support configurable sensor angles and I2C addresses
- Emit scans as 2×N matrix (distance mm, angle deg) conforming to `lidar.Device` interface
- Be composable, testable, and work on both TinyGo and Linux

### Architecture

- **TCA9548A Router**: Manages I2C channel switching automatically
- **VL53L0X Sensors**: Each sensor provides distance measurements (0-2000mm typical range)
- **Fixed Angles**: Each sensor has a predefined angular position
- **Continuous Mode**: All sensors operate in continuous measurement mode

### Sensor Configuration

- **Angles**: Array of angles (degrees) specifying the angular position of each sensor
  - Can be any angles (e.g., `[-25, -15, -5, 5, 15, 25]` for a forward-facing array)
  - Angles determine the field of view: `GetMinAngle()` and `GetMaxAngle()` return min/max from this array
- **I2C Addresses**: Optional array of I2C addresses for each sensor
  - If not provided or 0, uses `vl53l0x.DefaultAddress` (0x29)
  - Allows multiple sensors per TCA9548A channel if addresses differ
- **Channel Mapping**: Sensors are mapped to TCA9548A channels 0, 1, 2, ... (modulo 8)
  - Supports up to 8 sensors per TCA9548A mux
  - Can be extended to support multiple muxes in the future

### Public API

- Package `x/devices/lidar/array`
  - `type Config struct` – configuration for array-based LiDAR
    - `Router *tca9548a.Router` – TCA9548A router managing I2C multiplexer
    - `Angles []float32` – angle (deg) for each sensor
    - `SensorAddresses []uint8` – optional I2C addresses (defaults to 0x29)
  - `func New(ctx context.Context, cfg Config) (*Device, error)` – creates new array LiDAR
  - `type Device struct` – implements `lidar.Device` interface
    - `Configure(init bool) error` – initializes router and sensors, starts continuous mode
    - `OnRead(fn func(matTypes.Matrix))` – callback on each scan
    - `Read(dst matTypes.Matrix) int` – copy latest scan to preallocated matrix
    - `GetMinAngle() float32` – minimum angle from sensor array
    - `GetMaxAngle() float32` – maximum angle from sensor array
    - `GetPointCount() int` – number of sensors (points per scan)
    - `Close()` – stops continuous mode and releases resources

### Data Format

- **Matrix Layout**: 2×N matrix where N = number of sensors
  - Row 0: Distances in millimeters (float32)
  - Row 1: Angles in degrees (float32)
- **Scan Rate**: Configurable via read loop ticker (default: ~20Hz, 50ms period)
- **Invalid Readings**: Sensors that fail to read return 0 distance (invalid marker)

### Usage Example

```go
ctx, cancel := context.WithCancel(context.Background())
defer cancel()

// Setup I2C bus and TCA9548A router
bus := devices.NewTinyGoI2C(machine.I2C0)
router := tca9548a.NewRouter(bus, 0x70)
router.Configure(true)

// Create array LiDAR with 6 sensors at fixed angles
cfg := array.Config{
    Router: router,
    Angles: []float32{-25, -15, -5, 5, 15, 25}, // degrees
    SensorAddresses: []uint8{0x29, 0x29, 0x29, 0x29, 0x29, 0x29}, // optional
}
lidar, _ := array.New(ctx, cfg)
lidar.Configure(true)

// Receive scans
lidar.OnRead(func(m matTypes.Matrix) {
    // m[0][:] distances (mm), m[1][:] angles (deg)
    minAngle := lidar.GetMinAngle() // -25.0
    maxAngle := lidar.GetMaxAngle() // 25.0
    pointCount := lidar.GetPointCount() // 6
})
```

### Implementation Details

- **Preallocated Storage**: 2×N matrix preallocated at construction time
- **Continuous Mode**: All sensors start continuous measurement during `Configure(true)`
- **Read Loop**: Background goroutine reads all sensors periodically (default 50ms)
- **Channel Switching**: TCA9548A router automatically handles I2C channel switching
- **Error Handling**: Failed sensor reads result in 0 distance (invalid marker)
- **Thread Safety**: Mutex protects shared state (count, onRead callback)

### Limitations

- **Static Array**: Sensors have fixed positions (no mechanical scanning)
- **Limited Resolution**: Resolution determined by number of sensors
- **Field of View**: Determined by sensor placement angles
- **Range**: Limited by VL53L0X range (typically 0-2000mm, up to 4000mm in long-range mode)
- **Single Mux**: Currently supports one TCA9548A (8 sensors max), can be extended

### Future Enhancements

- Support for multiple TCA9548A muxes (cascade multiple routers)
- Configurable read loop period
- Sensor health monitoring and reporting
- Automatic sensor discovery and calibration
- Support for different ToF sensors (not just VL53L0X)

