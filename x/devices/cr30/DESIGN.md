# CR30 Colorimeter Device Design

## Overview

The `cr30` package provides a Go implementation for communicating with the CR30 colorimeter device. It handles serial communication, packet protocol, device information retrieval, and spectral measurements.

## Package Structure

```
cr30/
├── DESIGN.md      # This document
├── packet.go      # Packet building and parsing
└── device.go      # Device communication and protocol
```

## Dependencies

**External Dependencies:**
- `encoding/binary` - Binary data packing/unpacking
- `io` - I/O operations
- `time` - Timeouts and delays
- `errors` - Error handling

**Internal Dependencies:**
- `github.com/itohio/EasyRobot/x/devices` - Serial interface

## Core Components

### Packet Structure

CR30 uses 60-byte fixed-length packets:
- Byte 0: Start byte (0xAA or 0xBB)
- Byte 1: Command
- Byte 2: Subcommand
- Byte 3: Parameter
- Bytes 4-55: Payload (52 bytes)
- Byte 58: Marker (0xFF for 0xAA, 0x00/0xFF for 0xBB)
- Byte 59: Checksum

Checksum calculation:
- Sum bytes 0-57
- For 0xBB packets: subtract 1
- Modulo 256

### Device

**Purpose**: Main device interface for CR30 colorimeter communication

**Key Responsibilities:**
- Serial connection management
- Packet building and parsing
- Device handshake and initialization
- Device information retrieval
- Measurement triggering and reading
- Spectrum data extraction

**Interface:**
```go
type Device struct {
    serial devices.Serial
    // ... internal state
}

func New(serial devices.Serial) *Device
func (d *Device) Connect() error
func (d *Device) Disconnect() error
func (d *Device) Handshake() error
func (d *Device) DeviceInfo() Info
func (d *Device) Measure(ctx context.Context) (*Measurement, error)
func (d *Device) WaitMeasurement(ctx context.Context) (*Measurement, error)
```

### Measurement Data

**Purpose**: Container for measurement results

**Structure:**
```go
type Measurement struct {
    Spectrum []float32  // 31 spectral values (400-700nm, 10nm steps)
    Header   Header    // Measurement header packet info
    Chunks   []Chunk   // Individual data chunks
    Raw      []byte    // Raw SPD bytes
}
```

### Device Info

**Purpose**: Device identification and version information

**Structure:**
```go
type Info struct {
    Name      string
    Model     string
    Serial    string
    Firmware  string
    Build     string
}
```

## Design Patterns

### Synchronous Communication
- Uses blocking I/O with timeouts
- Reads exactly 60 bytes per packet
- Validates packets before processing

### Error Handling
- Timeout-based error detection
- Packet structure validation
- Graceful connection management
- Invalid packets are rejected

### Measurement Protocol
1. Trigger measurement: Send `0xBB 0x01 0x00 0x00`
2. Receive header packet (subcmd 0x09)
3. Request data chunks: Send `0xBB 0x01 0x10/0x11/0x12/0x13 0x00`
4. Accumulate SPD data from chunks
5. Parse 124 bytes (31 floats) into spectrum

### Handshake Sequence
1. Query device name: `0xAA 0x0A 0x00 0x00`
2. Query serial number: `0xAA 0x0A 0x01 0x00`
3. Query firmware: `0xAA 0x0A 0x02 0x00`
4. Query build info: `0xAA 0x0A 0x03 0x00` (optional)
5. Initialize device: `0xBB 0x17 0x00 0x00`
6. Check command: `0xBB 0x13 0x00 0x00` with "Check" payload
7. Query parameters: `0xBB 0x28 0x00 0x00/0x01/0x02/0x03/0xFF`

## Implementation Notes

### Spectrum Data
- 31 spectral bands from 400nm to 700nm (10nm steps)
- Data stored as float32 values
- SPD data spread across 4 chunks (0x10, 0x11, 0x12, 0x13)
- Chunk 0x10: SPD bytes start at offset 2
- Chunks 0x11, 0x12: SPD bytes start at offset 2
- Chunk 0x13: Final chunk (may have different structure)

### Color Science
- Color science conversions (XYZ, LAB, RGB) are NOT implemented in this package
- This package only provides raw spectrum data
- Color science will be implemented in a separate package

### Serial Configuration
- Default baud rate: 19200
- 8 data bits, 1 stop bit, no parity (standard serial config)
- No flow control

## Open Questions

1. **Connection Resilience**: How should we handle connection drops?
   - Fail gracefully, require manual reconnection

2. **Concurrent Access**: Multiple goroutines accessing the same device?
   - Not supported, device is not thread-safe

3. **Measurement Timeouts**: What are appropriate timeout values?
   - Use context.Context with context.WithTimeout() for timeouts
   - Example: ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)

4. **Calibration**: Should calibration be implemented?
   - Not in initial implementation, can be added later

