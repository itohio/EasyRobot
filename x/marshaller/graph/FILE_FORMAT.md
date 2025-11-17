# Graph Marshaller File Format Specification

## Overview

The graph marshaller uses a three-file storage format optimized for:
- **Efficient I/O**: Memory-mapped files (mmap) for zero-copy access
- **Fast Traversal**: Fixed-size records for cache-friendly access patterns
- **Append-Only Growth**: New records appended without full file rewrite
- **Type Safety**: Magic numbers and versioning for format validation
- **Extensibility**: Reserved bytes in headers and records for future expansion

## File Structure

The marshaller uses three separate files:

1. **`nodes.graph`** - Node records and metadata (fixed-size records)
2. **`edges.graph`** - Edge records and metadata (fixed-size records)
3. **`data.graph`** - Variable-length node/edge data payloads

This separation allows:
- Independent file growth (nodes and edges can grow at different rates)
- Efficient memory mapping (map only needed files)
- Parallel access (read nodes while writing edges)
- Defragmentation (compact each file independently)

## File Format Details

### 1. Node File (`nodes.graph`)

#### File Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Node File Header (64 bytes)                                 │
├─────────────────────────────────────────────────────────────┤
│ Node Record 0 (32 bytes)                                    │
│ Node Record 1 (32 bytes)                                    │
│ ...                                                          │
│ Node Record N-1 (32 bytes)                                  │
└─────────────────────────────────────────────────────────────┘
```

#### Node File Header (64 bytes)

| Offset | Size | Field | Type | Description |
|--------|------|-------|------|-------------|
| 0 | 8 | Magic | `[8]byte` | Format identifier: `"GRAPHND1"` |
| 8 | 4 | Version | `uint32` | Format version (currently `1`) |
| 12 | 8 | MaxID | `int64` | Maximum node ID (for ID generation) |
| 20 | 8 | NodeCount | `uint64` | Number of active node records |
| 28 | 8 | DataFileOffset | `uint64` | Offset to metadata entry in `data.graph` |
| 36 | 8 | Checksum | `uint64` | Header checksum for corruption detection |
| 44 | 20 | Reserved | `[20]byte` | Reserved for future use |

**Reserved Field Usage:**
- `Reserved[0]`: Graph kind code (0=generic, 1=tree, 2=decision_tree, 3=expression_graph)
- `Reserved[1-19]`: Reserved for future expansion

**Rationale:**
- **64-byte header**: Aligns with common cache line sizes (64 bytes) for efficient access
- **Magic number**: Quick format identification without parsing
- **Version field**: Enables format evolution while maintaining backward compatibility
- **MaxID tracking**: Efficient ID generation without scanning all records
- **DataFileOffset**: Points to graph metadata entry (root IDs, operation mappings)
- **Checksum**: Detects corruption during I/O operations
- **Reserved bytes**: Allows format extension without breaking compatibility

#### Node Record (32 bytes)

| Offset | Size | Field | Type | Description |
|--------|------|-------|------|-------------|
| 0 | 8 | ID | `int64` | Unique node identifier |
| 8 | 8 | DataOffset | `uint64` | Offset into `data.graph` for node data (0 = no data) |
| 16 | 1 | Flags | `uint8` | Status flags (deleted, active, etc.) |
| 17 | 15 | Reserved | `[15]byte` | Reserved for future expansion |

**Flags:**
- `FlagDeleted (1 << 0)`: Node is marked as deleted
- `FlagActive (1 << 1)`: Node is active (default)

**Rationale:**
- **32-byte records**: Power-of-two size for efficient memory alignment and cache usage
- **Fixed-size**: Enables O(1) random access via `headerSize + index * 32`
- **ID in record**: Allows direct node lookup without separate index
- **DataOffset**: Separates structure (nodes/edges) from payload (data), enabling variable-length data
- **Flags byte**: Supports soft deletion and future status tracking
- **15 reserved bytes**: Room for future fields (parent index, child count, etc.) without format change

### 2. Edge File (`edges.graph`)

#### File Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Edge File Header (64 bytes)                                 │
├─────────────────────────────────────────────────────────────┤
│ Edge Record 0 (32 bytes)                                    │
│ Edge Record 1 (32 bytes)                                    │
│ ...                                                          │
│ Edge Record N-1 (32 bytes)                                  │
└─────────────────────────────────────────────────────────────┘
```

#### Edge File Header (64 bytes)

| Offset | Size | Field | Type | Description |
|--------|------|-------|------|-------------|
| 0 | 8 | Magic | `[8]byte` | Format identifier: `"GRAPHED1"` |
| 8 | 4 | Version | `uint32` | Format version (currently `1`) |
| 12 | 8 | MaxID | `int64` | Maximum edge ID (for ID generation) |
| 20 | 8 | EdgeCount | `uint64` | Number of active edge records |
| 28 | 8 | DataFileOffset | `uint64` | Offset to metadata entry in `data.graph` |
| 36 | 8 | Checksum | `uint64` | Header checksum for corruption detection |
| 44 | 20 | Reserved | `[20]byte` | Reserved for future use |

**Reserved Field Usage:**
- `Reserved[0]`: Graph kind code (matches node file)
- `Reserved[1-19]`: Reserved for future expansion

**Rationale:**
- Same structure as node header for consistency and code reuse
- Separate MaxID allows independent edge ID generation
- DataFileOffset typically matches node file (points to same metadata entry)

#### Edge Record (32 bytes)

| Offset | Size | Field | Type | Description |
|--------|------|-------|------|-------------|
| 0 | 8 | FromID | `int64` | Source node ID |
| 8 | 8 | ToID | `int64` | Target node ID |
| 16 | 4 | DataOffset | `uint32` | Offset into `data.graph` for edge data (0 = no data) |
| 20 | 1 | Flags | `uint8` | Status flags (deleted, active, etc.) |
| 21 | 11 | Reserved | `[11]byte` | Reserved for future expansion |

**Flags:**
- `FlagDeleted (1 << 0)`: Edge is marked as deleted
- `FlagActive (1 << 1)`: Edge is active (default)

**Rationale:**
- **32-byte alignment**: Matches node record size for consistent access patterns
- **FromID/ToID**: Direct node references without indirection
- **DataOffset as uint32**: Edges typically have smaller data payloads than nodes, saving 4 bytes per record
- **No cost field**: Cost is algorithm-dependent and stored in edge data (protobuf/raw bytes)
- **11 reserved bytes**: Room for future fields (weight, label, etc.)

### 3. Data File (`data.graph`)

#### File Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Data File Header (64 bytes)                                 │
├─────────────────────────────────────────────────────────────┤
│ Data Entry 0 (variable length)                              │
│ Data Entry 1 (variable length)                              │
│ ...                                                          │
│ Data Entry N-1 (variable length)                            │
│ [Metadata Entry] (variable length, if present)               │
└─────────────────────────────────────────────────────────────┘
```

#### Data File Header (64 bytes)

| Offset | Size | Field | Type | Description |
|--------|------|-------|------|-------------|
| 0 | 8 | Magic | `[8]byte` | Format identifier: `"GRAPHDT1"` |
| 8 | 4 | Version | `uint32` | Format version (currently `1`) |
| 12 | 8 | EntryCount | `uint64` | Total number of data entries (including metadata) |
| 20 | 44 | Reserved | `[44]byte` | Reserved for future use |

**Rationale:**
- **EntryCount**: Enables validation and iteration without scanning
- **No MaxID**: Data entries don't have IDs (referenced by offset)
- **Large reserved space**: Room for future metadata (compression info, indices, etc.)

#### Data Entry Format

Each data entry has a variable-length structure:

```
┌─────────────────────────────────────────────────────────────┐
│ Data Entry Header (7 bytes)                                 │
├─────────────────────────────────────────────────────────────┤
│ Type Name (variable, 0-65535 bytes)                         │
├─────────────────────────────────────────────────────────────┤
│ Payload (variable, 0-4294967295 bytes)                      │
└─────────────────────────────────────────────────────────────┘
```

**Data Entry Header (7 bytes):**

| Offset | Size | Field | Type | Description |
|--------|------|-------|------|-------------|
| 0 | 4 | PayloadLength | `uint32` | Length of payload in bytes |
| 4 | 1 | DataType | `uint8` | Type of data (see DataType enum) |
| 5 | 2 | TypeNameLength | `uint16` | Length of type name string in bytes |

**Data Entry Body:**

| Offset | Size | Field | Type | Description |
|--------|------|-------|------|-------------|
| 7 | TypeNameLength | TypeName | `[]byte` | Type name (for protobuf: message name) |
| 7+TypeNameLength | PayloadLength | Payload | `[]byte` | Serialized data payload |

**DataType Enum:**

| Value | Name | Description |
|-------|------|-------------|
| 0 | `DataTypeProtobuf` | Protocol Buffer message |
| 1 | `DataTypeBytes` | Raw byte array |
| 2 | `DataTypeString` | UTF-8 string |
| 3 | `DataTypeInt` | Signed integer (64-bit) |
| 4 | `DataTypeInt8` | Signed 8-bit integer |
| 5 | `DataTypeInt16` | Signed 16-bit integer |
| 6 | `DataTypeInt32` | Signed 32-bit integer |
| 7 | `DataTypeInt64` | Signed 64-bit integer |
| 8 | `DataTypeUint` | Unsigned integer (64-bit) |
| 9 | `DataTypeUint8` | Unsigned 8-bit integer |
| 10 | `DataTypeUint16` | Unsigned 16-bit integer |
| 11 | `DataTypeUint32` | Unsigned 32-bit integer |
| 12 | `DataTypeUint64` | Unsigned 64-bit integer |
| 13 | `DataTypeFloat32` | 32-bit floating point |
| 14 | `DataTypeFloat64` | 64-bit floating point |
| 15 | `DataTypeArray` | Fixed-size array |
| 16 | `DataTypeSlice` | Variable-size slice |

**Rationale:**
- **Variable-length entries**: Supports arbitrary data sizes without waste
- **Type name storage**: Enables proper deserialization (especially for protobuf)
- **7-byte header**: Minimal overhead while supporting large payloads (4GB max)
- **TypeNameLength as uint16**: Supports type names up to 65KB (sufficient for fully qualified names)
- **PayloadLength as uint32**: Supports payloads up to 4GB per entry
- **Separate type field**: Enables efficient type-specific deserialization

#### Graph Metadata Entry

The metadata entry is a special data entry with:
- **TypeName**: "__graph_metadata__" (reserved identifier)
- **DataType**: `DataTypeProtobuf`
- **Payload**: Serialized `types.marshaller.GraphMetadata` protobuf message (defined in `proto/types/marshaller/graph_metadata.proto`)

**Protobuf Schema:**

```proto
syntax = "proto3";

package types.marshaller;

enum GraphKind {
  GRAPH_KIND_GENERIC = 0;
  GRAPH_KIND_TREE = 1;
  GRAPH_KIND_DECISION_TREE = 2;
  GRAPH_KIND_EXPRESSION_GRAPH = 3;
}

message DecisionEdgeOp {
  int64 parent_id = 1;
  int64 child_id = 2;
  string op_name = 3;
}

message DecisionMetadata {
  map<int64, string> node_ops = 1;
  repeated DecisionEdgeOp edge_ops = 2;
}

message ExpressionMetadata {
  map<int64, string> node_ops = 1;
  int64 root_id = 2;
}

message GraphMetadata {
  GraphKind kind = 1;
  int64 root_id = 2;
  string tree_type = 3;
  DecisionMetadata decision = 4;
  ExpressionMetadata expression = 5;
}
```

**Rationale:**
- **Strong typing**: Protobuf schema shared between marshaller and runtime code
- **Function name storage**: Operations identified by reflection-derived names
- **Separate metadata entry**: Keeps structural data (nodes/edges) separate from metadata
- **Optional fields**: Only stores metadata relevant to graph kind

## Byte Ordering

All multi-byte integers are stored in **little-endian** format:
- Consistent with x86/x64 architectures (most common deployment)
- Efficient on little-endian systems (no byte swapping)
- Standard for most binary file formats

## Alignment Considerations

### Page Alignment

Memory-mapped files require page-aligned offsets for efficient access:
- **System page size**: Typically 4KB (4096 bytes) on Linux/Windows
- **Data entries**: Written at page-aligned offsets when possible
- **Record arrays**: Headers (64 bytes) + records (32 bytes) naturally align

### Cache Line Alignment

Records are sized for cache-friendly access:
- **32-byte records**: Fits in single cache line on most CPUs (64-byte cache lines)
- **64-byte headers**: Single cache line access
- **Sequential access**: Records stored contiguously for prefetching

## Versioning and Compatibility

### Format Version

Current format version: **1**

**Version Evolution Strategy:**
- **Increment version**: When breaking changes are introduced
- **Backward compatibility**: Older readers should handle newer formats when possible
- **Forward compatibility**: New readers must reject unsupported versions
- **Reserved bytes**: Allow format extension without version bump

### Magic Numbers

Magic numbers provide quick format identification:
- **`GRAPHND1`**: Node file (Graph Node Data v1)
- **`GRAPHED1`**: Edge file (Graph Edge Data v1)
- **`GRAPHDT1`**: Data file (Graph Data v1)

**Rationale:**
- **Human-readable**: Easy to identify in hex dumps
- **Version suffix**: `1` indicates format version
- **Unique per file**: Prevents accidental file mix-ups

## Checksum Algorithm

Current implementation uses a simple sum checksum:

```go
func CalculateChecksum(data []byte) uint64 {
    var sum uint64
    for _, b := range data {
        sum += uint64(b)
    }
    return sum
}
```

**Rationale:**
- **Fast**: O(n) computation, no cryptographic overhead
- **Simple**: Easy to implement and verify
- **Placeholder**: Can be upgraded to CRC32/CRC64 or cryptographic hash if needed
- **Header-only**: Checksums only header fields, not full file (performance)

## Defragmentation Support

The format supports defragmentation through:

1. **Soft Deletion**: Records marked with `FlagDeleted` instead of removed
2. **Hole Identification**: Defragmenter scans active records to find unused data regions
3. **Compaction**: Records and data can be moved to fill holes
4. **Offset Updates**: All references updated after compaction

**Design for Defragmentation:**
- **Flags field**: Enables soft deletion without format change
- **Offset-based references**: Can be updated after data movement
- **Separate files**: Each file can be defragmented independently

## Performance Considerations

### Read Performance

- **Fixed-size records**: O(1) random access via `headerSize + index * recordSize`
- **Sequential layout**: Records stored contiguously for efficient scanning
- **Cache-friendly**: 32-byte records fit in cache lines
- **Memory mapping**: Zero-copy access via mmap

### Write Performance

- **Append-only**: New records appended to end (no file rewrite)
- **Batch writes**: Multiple records written in single I/O operation
- **Separate data file**: Large payloads don't affect record array writes

### Space Efficiency

- **Fixed-size records**: Predictable space usage
- **Variable-length data**: No waste for small payloads
- **Separate files**: Nodes/edges can grow independently
- **Reserved bytes**: Trade-off between space and extensibility

## Example File Layout

### Small Graph (3 nodes, 2 edges)

**nodes.graph (160 bytes):**
```
Offset 0:   Header (64 bytes)
  - Magic: "GRAPHND1"
  - Version: 1
  - MaxID: 3
  - NodeCount: 3
  - DataFileOffset: 0 (no metadata)
Offset 64:  Node Record 0 (32 bytes)
  - ID: 1
  - DataOffset: 0
  - Flags: FlagActive
Offset 96:  Node Record 1 (32 bytes)
  - ID: 2
  - DataOffset: 100
  - Flags: FlagActive
Offset 128: Node Record 2 (32 bytes)
  - ID: 3
  - DataOffset: 0
  - Flags: FlagActive
```

**edges.graph (160 bytes):**
```
Offset 0:   Header (64 bytes)
  - Magic: "GRAPHED1"
  - Version: 1
  - MaxID: 2
  - EdgeCount: 2
  - DataFileOffset: 0
Offset 64:  Edge Record 0 (32 bytes)
  - FromID: 1
  - ToID: 2
  - DataOffset: 200
  - Flags: FlagActive
Offset 96:  Edge Record 1 (32 bytes)
  - FromID: 2
  - ToID: 3
  - DataOffset: 0
  - Flags: FlagActive
```

**data.graph (variable):**
```
Offset 0:   Header (64 bytes)
  - Magic: "GRAPHDT1"
  - Version: 1
  - EntryCount: 2
Offset 64:  Data Entry 0 (node 2 data)
  - PayloadLength: 36
  - DataType: DataTypeProtobuf
  - TypeNameLength: 15
  - TypeName: "MyNodeMessage"
  - Payload: [protobuf bytes...]
Offset 100: Data Entry 1 (edge 1->2 data)
  - PayloadLength: 8
  - DataType: DataTypeFloat32
  - TypeNameLength: 0
  - Payload: [4-byte float32 cost...]
```

## Future Extensions

The format is designed for extensibility:

1. **Reserved bytes**: Headers and records have reserved space for new fields
2. **Version field**: Enables format evolution
3. **Flags field**: Can add new status bits
4. **Metadata entry**: Can store additional graph properties
5. **Separate files**: New file types can be added without affecting existing files

## References

- **Neo4j Storage Format**: Inspiration for append-only, fixed-record design
- **Memory-Mapped Files**: Standard POSIX/Windows mmap for zero-copy access
- **Protocol Buffers**: Used for complex data serialization
- **Little-Endian**: Standard for x86/x64 architectures
