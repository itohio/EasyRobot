# Monitor Tool Specification

## Purpose
- Inspect and debug arbitrary binary protocols transported over a serial link.
- Provide an operator-friendly view of packet framing, decoded fields, and CRC health.
- Offer a raw capture mode for offline analysis when the protocol definition is unknown.

## Scope and Goals
- Single executable located in `cmd/monitor`.
- Works against any serial device supported by `x/devices` (`devio.Serial` interface).
- Pattern-driven packet framing with optional length/CRC awareness.
- Zero external dependencies beyond the EasyRobot repo and Go stdlib.

### Non-goals
- Acting as a protocol-specific decoder (no schema awareness beyond pattern definition).
- Acting as a transport multiplexer or router.
- Persisting decoded packets (only stdout logging plus optional raw capture).

## CLI Interface

| Flag | Description |
|------|-------------|
| `-port` | Required. Serial device path (`/dev/ttyUSB0`, `COM3`, etc.). |
| `-baud` | Baud rate (default `115200`). |
| `-header` | Packet pattern string (default `AA`). Drives parsing, length, and CRC extraction. |
| `-bytes-per-line` | Hex width per line for packet dumps (default `64`). |
| `-crc` | Strict mode. When set, packets with invalid CRC are discarded instead of logged with warnings. |
| `-alternative` | When enabled, prints “packets within packets” that match the pattern while scanning a larger buffer. |
| `-h` | Shows the pattern grammar guide and exits. |
| `-capture` | Enables raw capture mode (no parsing). |
| `-capture-file` | Output path for capture mode (default `capture.bin`). |
| `-precision` | Decimal digits used for floating/derived values (default `2`). |
| `-maxlen` | Hard ceiling for packet length in bytes (default `2048`). |

### Operating Modes
1. **Decode Mode (default)**: Parse packets according to the pattern, validate CRC, and print decoded values.
2. **Capture Mode** (`-capture`): Stream raw bytes to disk while periodically syncing and reporting written size.

## Pattern Grammar (excerpt)
- **Exact bytes**: `AA`, `1C`, etc.
- **Wildcard byte**: `*` (skips one byte).
- **Offset jump**: `#N` (absolute byte index from packet start).
- **Anchors**:
  - `^` – must appear at the beginning of the pattern string; indicates packets always start with the pattern (useful for discarding leading noise quickly).
  - `$` – marks the logical end of the packet. When followed by a number (`$128`), it sets a per-pattern maximum length. When placed immediately after an offset (`#64$`), the offset value becomes the maximum length hint.
- **Length fields**:
  - `L` – single-byte length.
  - `LL` – big-endian uint16.
  - `ll` – little-endian uint16.
- **Decode fields** (prefixed with `%%` in CLI help, parsed as `%` in code):
  - Signed ints: `%i`, `%ii`, `%iii`, `%iiii` (little-endian). `%II`, `%III`, `%IIII` (big-endian).
  - Unsigned: `%u`, `%uu`, `%uuu`, `%uuuu` (little-endian) and `%UU…` big-endian equivalents.
  - Floats: `%f` (float32, LE), `%F` (float64, LE).
  - CRC markers: `%c` (CRC8), `%cc` (CRC16, cumulative sum, LE). These positions drive validator offsets.
  - Arrays: `%5f` → five float32 values; `%10u` → ten bytes, etc.

The parser converts the string into a `PacketPattern` with ordered `FieldType` entries, minimum byte count, optional length metadata, and CRC descriptors. Anchors/length hints are stored separately and do not consume bytes.

### Length Priority
During decoding, the effective packet length limit is:

```
min(
  --maxlen flag (default 2048),
  $N constraint (if any),
  runtime length field value L (when pattern defines it)
)
```

Packet accumulation stops as soon as the pattern fully matches or the buffered length reaches this effective limit. This prevents unbounded growth even if the stream lacks a terminating header.

## Planned: Derived Fields & Structured Arrays

### Motivation
- Operators frequently need inline unit conversions (e.g., ticks → degrees) and need to interpret repeated structs (arrays of `{value, status}`).
- Encoding these transformations inside the pattern keeps workflows self-contained.

### Functional Requirements
1. **Arithmetic expressions**: Extend decode tokens with optional math, e.g. `%(uu/360.0)`.
2. Support `+ - * /` and parentheses; literals may be ints or floats; expressions evaluate as `float64`.
3. `-precision` flag controls decimal digits when printing derived values (default `2`).
4. **Structured arrays**: `%N{token1,token2,...}` iterates `N` times through a mini-structure, optional expressions allowed in each token.
5. Arrays without braces keep legacy behavior; expressions apply per element.

### Parsing Strategy
- Pattern parser detects `%(` to capture a base token plus arithmetic tail; tail is compiled into a simple AST once.
- For structured arrays, after `%N{`, the parser recursively parses the comma-separated decode tokens to build a per-element layout.

### Evaluation
- Decoder obtains the base primitive value, converts to `float64`, and passes it to the expression evaluator.
- For arrays, expressions run for each element; structured arrays emit tuples (e.g., `[val,status]`).

### Open Questions
- Whether to show both raw and derived values (MVP: derived only).
- Structured arrays are fixed-stride; variable-sized structs remain out of scope.

## Architecture

| Component | Responsibility |
|-----------|----------------|
| `Parser` (`pattern.go`) | Tokenizes the pattern string into `FieldType` definitions, tracks length/CRC offsets, and computes the minimum viable packet size. |
| `PacketPattern` | Immutable description used during matching and decoding. |
| `Decoder` (`decoder.go`) | Interprets typed fields (integers, floats, CRC values) using endianness metadata. |
| `Matcher` (`matcher.go`) | Sliding-window matcher that locates pattern-aligned offsets in the byte buffer. |
| `CRCValidator` (`crc.go`) | Optional CRC8 / CRC16 verification using simple cumulative-sum implementations. |
| `Printer` (`printer.go`) | Hex dump plus human-readable field annotations and CRC verdict. |
| `PacketProcessor` (`main.go`) | Coordinates buffer management, packet promotion, alternative packet handling, and ties printer/CRC/matcher together. |

Supporting utilities like `processSerialStream`, `runCaptureMode`, and the CLI bootstrap wire these components together.

## Data Flow
1. CLI arguments configure serial port, pattern, and behavior.
2. `devio.NewSerialWithConfig` establishes an asynchronous serial reader (read calls block until bytes arrive).
3. `processSerialStream` accumulates bytes into `packetBuffer`, searching for pattern matches:
   - When buffer start matches, optional length fields determine total packet size.
   - When pattern is found deeper in the buffer, it may be treated as an alternative packet (useful for resynchronizing mid-stream).
4. Completed packets are passed to `PacketProcessor.ProcessPacket`:
   - Optional CRC validation (warning vs discard depending on `-crc`).
   - `Printer` renders compact hex plus decoded field descriptions.
5. On shutdown (Ctrl+C), the current partial buffer is printed for debugging.

Capture mode follows a simpler flow: read bytes → append to file → report byte count until the context is canceled.

## Error Handling Strategy
- All serial read errors are logged; `io.EOF` is treated as transient with a short sleep.
- CRC failures emit warnings by default; strict mode enforces drop.
- Pattern parsing failures (invalid characters, multiple length fields, etc.) abort startup with descriptive slog output.
- Buffer overruns are handled by truncating old data while retaining enough slack to recover the next packet.

## Usage Examples

```powershell
# Basic decoding of 1-byte length packets starting with 0xAA
go run ./cmd/monitor `
  -port COM5 `
  -baud 230400 `
  -header "AA*L%uu%cc" `
  -crc

# Capture unknown protocol for later offline parsing
go run ./cmd/monitor `
  -port /dev/ttyUSB0 `
  -capture `
  -capture-file slam_dump.bin
```

## Future Enhancements (ideas)
- Support for additional CRC algorithms (CRC16-IBM, CRC32, etc.) with polynomial selection.
- Named presets for common device protocols to speed up ramp-up.
- Optional JSON output for easier downstream parsing or integration into log pipelines.
- Timestamping of packets and per-field annotations for time series analysis.
- Adaptive resynchronization heuristics (e.g., tolerate noise bytes before header realignment).

## Testing Considerations
- Unit coverage exists for shared packages; `cmd/monitor` currently relies on manual/serial-in-the-loop testing.
- Recommend creating synthetic serial sources (e.g., loopback generator) to produce deterministic packet streams for automated verification in Dockerized tests.


