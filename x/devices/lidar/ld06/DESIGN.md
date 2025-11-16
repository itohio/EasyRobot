## LD06 LiDAR Serial Decoder - Design

### Overview

- Device: LDROBOT LD06 DTOF (Direct Time of Flight) spinning LiDAR
- Protocol: One-way UART TX from LiDAR to host at 230400 baud. No RX/commands.
- Reference: LD06 Datasheet - https://www.inno-maker.com/wp-content/uploads/2020/11/LDROBOT_LD06_Datasheet.pdf

### Goals

- Provide a streaming decoder that reads frames from a `devices.Serial` and emits fully assembled 360° scans as 2xN matrix (distance, angle).
- Validate frames via CRC (1-byte checksum).
- Support optional PWM motor control with PID to maintain target point count.
- Be composable, testable, and not tied to any specific runtime (TinyGo or Linux).

### Protocol Summary (LD06)

Based on LD06 Development Manual, each data packet structure:

| Byte # | Field | Size | Description |
|--------|-------|------|-------------|
| 0 | Start Character | 1 | 0x54 |
| 1 | Data Length | 1 | Number of data points in this packet |
| 2-3 | Radar Speed | 2 | Motor speed (little-endian, units: 0.01 Hz) |
| 4-5 | Start Angle | 2 | Starting angle (little-endian, units: 0.01°) |
| 6..(6+3N-1) | Data Points | 3×N | Each point: [distance_LSB, distance_MSB, intensity] |
| (6+3N)..(6+3N+1) | End Angle | 2 | Ending angle (little-endian, units: 0.01°) |
| (6+3N+2)..(6+3N+3) | Timestamp | 2 | Timestamp (little-endian, units: 0.1 ms) |
| (6+3N+4) | CRC | 1 | CRC8 checksum |

- Distance: uint16 in mm (little-endian: LSB first, then MSB)
- Intensity: uint8 (0-255)
- Angles: int16 in 0.01° units (little-endian), can be negative
- Coordinate system: Left-handed, front is 0° (x-axis), angle increases clockwise

### Public API

- Package `x/devices/lidar/ld06`
  - `type Device struct` – streaming LiDAR device with preallocated 2×N matrix
  - `func New(ctx context.Context, ser devices.Serial, motor devices.PWM, targetPoints, maxPoints int) *Device`
    - `motor` can be nil to disable PWM control
    - `targetPoints`: desired points per rotation (0 = auto-calibrate)
    - `maxPoints`: preallocated capacity
  - `func (d *Device) Configure(init bool) error` – starts read loop and motor control
  - `func (d *Device) OnRead(fn func(matTypes.Matrix))` – callback on full rotation
  - `func (d *Device) Read(dst matTypes.Matrix) int` – copy latest scan to preallocated matrix
  - `func (d *Device) Close()` – stop all loops

### Motor Control

- PWM frequency: 20-50 KHz (configured via PWMDevice, not PWM channel)
- PWM duty cycle: 0-100% (0.0-1.0)
- Typical: 40% duty = ~10 Hz scan rate
- PID control maintains target point count per rotation

### Assembly Strategy

- Each packet contains a slice of data points with start/end angles
- Accumulate points until a full 360° rotation is detected (end angle wraps or exceeds start angle from previous packet)
- Angle interpolation: linear between start and end angles for each point

### Error Handling

- Invalid header/CRC: resynchronize by searching for next 0x54 header
- Overflow: reset rotation if maxPoints exceeded

### Testing

- Unit tests for CRC8 validation
- Frame parsing with synthetic packets
- Full rotation assembly test

### Reference

- Protocol details: LD06 Development Manual
- Datasheet: https://www.inno-maker.com/wp-content/uploads/2020/11/LDROBOT_LD06_Datasheet.pdf

