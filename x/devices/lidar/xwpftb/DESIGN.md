## XWPFTB LiDAR Serial Decoder - Design

Overview

- Device: Delta-2G class spinning LiDAR (vacuum-robot style) using wireless power and IR UART link.
- Protocol: One-way UART TX from LiDAR to host. No RX/commands.
- Reference: See protocol details and field scalings in `https://notblackmagic.com/bitsnpieces/lidar-modules/`.

Goals

- Provide a streaming decoder that reads frames from a `devices.Serial` and emits fully assembled 360° scans as `types.LIDARReading`.
- Validate frames via CRC (16-bit cumulative sum over all bytes from header to last payload byte).
- Be composable, testable, and not tied to any specific runtime (TinyGo or Linux).

Protocol Summary (Delta-2G)

- Byte fields (indices are within one frame):
  - 0: Header = 0xAA
  - 1-2: Length (uint16, little-endian) – total frame bytes including header and CRC
  - 3: Protocol Version = 0x01
  - 4: Frame Type = 0x61
  - 5: Command
    - 0xAE: Device Health (RPM byte scaled x3 to RPM)
    - 0xAD: Measurement Information (scan slice)
  - 6-7: Payload Length (uint16, little-endian)
  - 8..N-3: Payload
  - N-2..N-1: CRC16 cumulative sum (little-endian)

- Measurement payload (Command = 0xAD):
  - 0: RPM byte (scale x3 → RPM)
  - 1-2: Offset Angle (x0.01 deg) – can be ignored
  - 3-4: Start Angle (x0.01 deg)
  - Then M samples, each 3 bytes:
    - (5 + 3N): Signal Quality N (ignored for now)
    - (5 + 3N + 1..2): Distance N (uint16, scale x0.25 mm)
  - M = (PayloadLength - 5)/3
  - Delta-2G: 15 messages per rotation; each message covers 24°.
  - Angle for sample N within the slice: StartAngle + (24/M)*N

Public API

- Package `x/devices/lidar/xwpftb`
  - `type Decoder struct` – stateless parser helpers + rotation assembly state.
  - `func NewDecoder() *Decoder`
  - `func (d *Decoder) ReadLoop(ctx context.Context, ser devices.Serial, out chan<- *types.LIDARReading) error`
    - Reads from serial, parses frames, validates CRC, accumulates slice data until a full 360° rotation is assembled, then emits a `LIDARReading` with angles (deg) and distances (mm). Timestamp taken at completion.
  - `func (d *Decoder) ParseFrame(buf []byte) (cmd byte, payload []byte, ok bool)`
  - Internal helpers for CRC, numeric decoding, and measurement-slice parsing.

Assembly Strategy

- Maintain a buffer of angle-distance pairs during rotation. On detecting completion of 15 slices or a wrap in angle space, flush as one scan.
- Angle monotonicity: Sort is avoided during streaming; messages are assumed to be in order (as observed). If out-of-order is detected, start a new rotation.

Error Handling

- Invalid header/length/CRC: resynchronize by searching for next header 0xAA within the ring buffer.
- Timeouts are not handled in the decoder; caller controls lifecycle via `ctx`.

Testing

- Unit tests for CRC and angle math with synthetic payloads.

Notes

- No RX/control path in this device.
- Keep functions small and single-responsibility. Avoid package-level state.
- Provide guard rails but don’t over-abstract; accept interfaces, return concrete structs.

Reference

- Protocol and scaling details are derived from: `https://notblackmagic.com/bitsnpieces/lidar-modules/`.


