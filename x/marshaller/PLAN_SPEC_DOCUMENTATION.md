# Plan: SPEC.md Documentation Gaps

## Overview
This document identifies missing or outdated SPEC.md files across marshaller subpackages and outlines what each should contain.

## Current Status

### ✅ Existing SPEC.md Files

1. **`x/marshaller/SPEC.md`** - General marshaller specification
   - Status: ✅ Exists, needs review for accuracy
   - Last Updated: Recent (registry removed, API standardized)

2. **`x/marshaller/types/SPEC.md`** - Common types specification
   - Status: ✅ Exists, comprehensive
   - Content: Camera, display, input event types

3. **`x/marshaller/gocv/SPEC.md`** - GoCV marshaller specification
   - Status: ✅ Exists, comprehensive
   - Content: Usage examples, camera enumeration, display

4. **`x/marshaller/v4l/SPEC.md`** - V4L unmarshaller specification
   - Status: ✅ Exists, comprehensive
   - Content: V4L device usage, streaming, controls

5. **`x/marshaller/graph/SPEC.md`** - Graph marshaller specification
   - Status: ✅ Exists
   - Content: Graph serialization, storage format

6. **`x/marshaller/storage/SPEC.md`** - Storage specification
   - Status: ✅ Exists
   - Content: Storage backends, MappedStorage interface

### ❌ Missing SPEC.md Files

1. **`x/marshaller/gob/SPEC.md`** - Gob marshaller specification
2. **`x/marshaller/json/SPEC.md`** - JSON marshaller specification
3. **`x/marshaller/yaml/SPEC.md`** - YAML marshaller specification
4. **`x/marshaller/proto/SPEC.md`** - Protobuf marshaller specification (has README.md instead)
5. **`x/marshaller/tflite/SPEC.md`** - TFLite unmarshaller specification

## Required Content for Each SPEC.md

### Template Structure

Each SPEC.md should follow this structure:

```markdown
# <MarshallerName> Marshaller/Unmarshaller Specification

## Overview
Brief description of the marshaller/unmarshaller, its purpose, and use cases.

## Constructor

### NewMarshaller() / NewUnmarshaller()
Constructor signature, return type (concrete type), and description.

## Supported Types

### Domain Objects
List of domain objects supported (tensor, matrix, graph, etc.)

### Graph/Tree Support
Document graph and tree serialization if supported.

## Options

### Supported Options
List of `types.Option` implementations supported:
- `WithContext()` - Context for cancellation
- `WithRelease()` - Release resources after processing (sink marshallers only)
- Format-specific options

## Usage Examples

### Basic Usage
Simple examples of marshalling/unmarshalling common types.

### Advanced Usage
Complex examples, streaming, error handling.

## Format Details

### Encoding Details
Format-specific encoding details, limitations, compatibility notes.

### Performance Considerations
Performance characteristics, optimization tips.

## Implementation Notes

### Resource Management
How resources are managed, cleanup requirements.

### Thread Safety
Thread safety guarantees, concurrent usage notes.

### Error Handling
Error types, error handling patterns.
```

### 1. gob/SPEC.md

**Required Content:**

- **Overview:**
  - Go's native binary format
  - Use cases: fast serialization, Go-only communication
  - Limitations: Go-specific, not portable

- **Constructor:**
  - `NewMarshaller()` - returns `*Marshaller`
  - `NewUnmarshaller()` - returns `*Unmarshaller`

- **Supported Types:**
  - All Go types (with `encoding/gob` registration)
  - Tensor types
  - Matrix/Vector types
  - Custom types (if registered)

- **Options:**
  - `WithContext()` - Not applicable (synchronous)
  - `WithRelease()` - Not applicable (not a sink)
  - Other options?

- **Usage Examples:**
  - Basic tensor marshalling
  - Custom type marshalling
  - Round-trip examples

- **Format Details:**
  - Binary format (no human-readable option)
  - Go version compatibility
  - Size considerations

### 2. json/SPEC.md

**Required Content:**

- **Overview:**
  - Human-readable JSON format
  - Use cases: configuration, debugging, interop
  - Graph/tree support

- **Constructor:**
  - `NewMarshaller()` - returns `*Marshaller`
  - `NewUnmarshaller()` - returns `*Unmarshaller`

- **Supported Types:**
  - Basic Go types
  - Tensor types (how encoded?)
  - Graph types (document graph serialization)
  - Tree types
  - Decision tree support

- **Options:**
  - `WithContext()` - For streaming?
  - `WithRelease()` - Not applicable (not a sink)
  - Formatting options (indent, etc.)

- **Usage Examples:**
  - Basic marshalling
  - Graph marshalling (with ops/decisions)
  - Tree marshalling
  - Configuration marshalling

- **Format Details:**
  - JSON schema for graph/tree types
  - How ops/decisions are encoded
  - Round-trip guarantees

### 3. yaml/SPEC.md

**Required Content:**

- **Overview:**
  - Human-readable YAML format
  - Use cases: configuration, documentation, interop
  - Graph/tree support

- **Constructor:**
  - `NewMarshaller()` - returns `*Marshaller`
  - `NewUnmarshaller()` - returns `*Unmarshaller`

- **Supported Types:**
  - Basic Go types
  - Tensor types (how encoded?)
  - Graph types (document graph serialization)
  - Tree types
  - Decision tree support

- **Options:**
  - `WithContext()` - For streaming?
  - `WithRelease()` - Not applicable (not a sink)
  - Formatting options

- **Usage Examples:**
  - Basic marshalling
  - Graph marshalling (with ops/decisions)
  - Tree marshalling
  - Configuration marshalling

- **Format Details:**
  - YAML schema for graph/tree types
  - How ops/decisions are encoded
  - Round-trip guarantees

### 4. proto/SPEC.md

**Required Content:**

- **Overview:**
  - Protocol Buffer format
  - Use cases: inter-service communication, versioned APIs
  - Current status: has README.md, needs SPEC.md

- **Constructor:**
  - `NewMarshaller()` - returns `*Marshaller`
  - `NewUnmarshaller()` - returns `*Unmarshaller`

- **Supported Types:**
  - Protobuf message types
  - Tensor types (how encoded in proto?)
  - Frame stream metadata

- **Options:**
  - `WithContext()` - For streaming?
  - `WithRelease()` - Not applicable (not a sink)
  - Protobuf-specific options?

- **Usage Examples:**
  - Basic protobuf marshalling
  - Frame stream marshalling
  - Service communication examples

- **Format Details:**
  - Protobuf schema references
  - Version compatibility
  - Interop considerations

### 5. tflite/SPEC.md

**Required Content:**

- **Overview:**
  - TensorFlow Lite model loading
  - Use cases: inference, edge deployment
  - Only unmarshaller (models loaded, not saved)

- **Constructor:**
  - `NewUnmarshaller()` - returns `*Unmarshaller`

- **Supported Types:**
  - TFLite models
  - Model metadata
  - Tensor types from model

- **Options:**
  - `WithContext()` - For cancellation
  - `WithRelease()` - NOT APPLICABLE (unmarshaller, produces)
  - Model-specific options (threads, etc.)

- **Usage Examples:**
  - Loading TFLite model
  - Running inference
  - Accessing model metadata
  - Tensor I/O

- **Format Details:**
  - TFLite model format
  - Supported ops
  - Limitations
  - Performance notes

## Updates Needed for Existing SPEC.md

### x/marshaller/SPEC.md

**Updates Needed:**

1. **Option Support Matrix**
   - Add table showing which options are supported by which marshallers
   - Document `WithRelease()` semantics per marshaller type

2. **Constructor Naming Consistency**
   - Document that `v4l.New()` is acceptable (only unmarshaller)
   - Ensure all examples use correct constructor names

3. **Graph/Tree Support**
   - Add section documenting which marshallers support graph/tree
   - Link to graph/SPEC.md for details

4. **Resource Management**
   - Document `WithRelease()` usage clearly
   - Explain when users should call `Release()` manually
   - Add examples of proper resource cleanup

### x/marshaller/gocv/SPEC.md

**Updates Needed:**

1. **Display Event Handling**
   - Add examples of keyboard/mouse event handling
   - Document event loop lifecycle
   - Show window management examples

2. **WithRelease() Usage**
   - Document when to use `WithRelease()` option
   - Show examples of release patterns
   - Explain resource ownership

3. **Camera Control**
   - Verify camera enumeration examples are complete
   - Add runtime control examples
   - Document control parameter names

## Action Items

1. **Create Missing SPEC.md Files**
   - Priority 1: `json/SPEC.md`, `yaml/SPEC.md` (graph support recently added)
   - Priority 2: `gob/SPEC.md`, `proto/SPEC.md`
   - Priority 3: `tflite/SPEC.md`

2. **Update Existing SPEC.md Files**
   - Add option support matrix to `x/marshaller/SPEC.md`
   - Add event handling examples to `gocv/SPEC.md`
   - Verify all examples use correct constructors

3. **Review SPEC.md Completeness**
   - Ensure all marshallers have comprehensive examples
   - Verify format details are documented
   - Check that limitations are noted

4. **Add Cross-References**
   - Link between related SPEC.md files
   - Add "See Also" sections
   - Cross-reference graph/tree support

## Priority

- **High:** Create `json/SPEC.md` and `yaml/SPEC.md` (graph support needs documentation)
- **Medium:** Create `gob/SPEC.md`, `proto/SPEC.md`, `tflite/SPEC.md`
- **Medium:** Update existing SPEC.md files with missing content
- **Low:** Add cross-references and improve navigation
