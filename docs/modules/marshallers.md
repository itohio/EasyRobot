---
type: module
group: "IO, camera, vision"
status: partial
tags: [easyrobot, marshaller, serialization, storage]
workstream: [W1, W4, W6, W9]
packages: [x/marshaller, x/marshaller/types, x/marshaller/gob, x/marshaller/json, x/marshaller/yaml, x/marshaller/proto, x/marshaller/csv, x/marshaller/text, x/marshaller/graph, x/marshaller/storage]
test_result: pass
evidence: "go test -count=1 per package → all ok: root 6, csv 48, gob 4, graph 11, json 8, proto 4, storage 18, text 2, types 5, yaml 8 PASS (2026-11-09); probes in ~/src/tmp/easyrobot/io/ show dtype and graph defects"
---
# marshallers — `x/marshaller/**` (value formats and storage)

**Purpose:** one `Marshaller`/`Unmarshaller` contract for every way EasyRobot moves values in and out of bytes, files, devices and archives: tensors, NN models/layers, numeric slices, graphs/trees/decision trees, and (in the device packages) camera frames and audio. This note covers the value formats and the mapped-storage layer; camera back ends are in [[camera-io]], audio in [[audio-io]], TFLite in [[tflite]].

**Packages** (non-test LOC / test LOC; build tags):

| Package | LOC | Role |
|---|---|---|
| `x/marshaller` | 0 / 379 | test-only root: text + gob examples (`example_test.go`, `marshaller_test.go`); plan docs `SPEC.md`, `PLAN_*.md`, `ANALYSIS_SUMMARY.md` |
| `x/marshaller/types` | 973 / 76 | contract: `Marshaller`, `Unmarshaller`, `Option`/`Options`, `Frame`/`FrameStream`, `Error`, `MappedStorage`/`MappedRegion`, camera/display/audio device types (`camera.go`, `display.go`, `audio.go`) |
| `gob` | 537 / 300 | Go gob — [[marshaller-tensor-gob-conversion]] |
| `json`, `yaml` | 1551 / 708, 1442 / 708 | human-readable tensors, models, graphs — [[marshaller-tensor-json-yaml-conversion]] |
| `proto` | 686 / 297 | `types/core` protobuf — [[marshaller-tensor-proto-conversion]] |
| `csv` | 1338 / 850 | vectors/matrices/FP32 tensors — [[marshaller-csv-tensor-matrix]] |
| `text` | 287 / 109 | write-only human dump of tensors/layers/models (box drawing, truncation); no unmarshaller |
| `graph` | 3750 / 512 | mmap graph store + executable trees — [[graph-marshaller-mmap-format]], [[graph-marshaller-op-registry]] |
| `storage` | 1845 / 820 | `MappedStorage` back ends: file mmap (`!tinygo`, unix/windows/stub), memory, flash (`tinygo`), tar/tar.gz — [[mapped-storage-file-memory]], [[tar-archive-mapped-storage]] |

Out of scope here: `audio` ([[audio-io]]), `tflite` ([[tflite]]), `gocv` and `v4l` ([[camera-io]]).

## Public API

```go
type Marshaller   interface { Format() string; Marshal(w io.Writer, value any, opts ...Option) error }
type Unmarshaller interface { Format() string; Unmarshal(r io.Reader, dst any, opts ...Option) error }
type Option interface { Apply(*Options) }   // Options: FormatVersion, Hint, Metadata map, TensorFactory,
                                            // DestinationType, Context, MappedStorageFactory, ReleaseAfterProcessing
```
(`types/types.go:14-41`). Constructors: `gob|json|yaml|proto|csv|text.NewMarshaller(opts...)`, `…NewUnmarshaller(opts...)`; `graph.NewMarshaller(factory, opts...) (*GraphMarshaller, error)`; storage factories `storage.NewFileMap()`, `NewMemoryMap()`, `NewTarMap(path).Factory()`, `NewFlashMapFromData(b, ro)` (TinyGo). Common options: `WithHint`, `WithMetadata`, `WithTensorFactory`, `WithDestinationType`, `WithContext`, `WithMappedStorageFactory`, `WithRelease` (`types.go:105-238`). Graph options: `WithPath`, `WithEdgesPath`, `WithLabelsPath`, `WithType(proto.Message)`, `WithDecisionOp/Edge`, `WithExpressionOp`, `WithEqual/Compare/Cost`.

Composition: a value marshaller is chosen by the caller; tensors are created through `TensorFactory` (so pooled or GoCV-backed tensors can be injected); the graph marshaller writes through any `MappedStorageFactory`, so the same graph can live in a file, an archive, RAM or flash.

## Algorithms

<!-- vault:algorithms h=b9d427d69ac8 -->

Counts: 3 tested · 0 implemented · 5 partial · 0 stub · 0 unverified (8 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[graph-marshaller-op-registry]] | tested | pass | 2 e2e pass |
| [[marshaller-csv-tensor-matrix]] | tested | pass | 48 pass |
| [[marshaller-tensor-gob-conversion]] | tested | pass | 4 pass; probe all dtypes ok |
| [[graph-marshaller-mmap-format]] | partial | pass | 11 pass; probe shows nil node data |
| [[mapped-storage-file-memory]] | partial | pass | 18 pass (pkg) |
| [[marshaller-tensor-json-yaml-conversion]] | partial | pass | 8+8 pass; probe |
| [[marshaller-tensor-proto-conversion]] | partial | pass | partial (INT8/INT16 zeroed, UINT8 panic) |
| [[tar-archive-mapped-storage]] | partial | pass | 6 tar tests pass |

<!-- /vault:algorithms -->

Dtype round-trip matrix from probes (`graphprobe`, `fmtprobe`, 2×2 tensors, 2026-11-09):

| dtype | gob | json/yaml | proto | csv |
|---|---|---|---|---|
| FP32, FP64 | ok | ok | ok | FP32 ok, FP64 refused |
| INT32, INT64, INT | ok | **panic** (`SetAt`) | ok | refused |
| INT16, INT8 | ok | ok | **zeros** | refused |
| UINT8 (images) | ok | **panic** | **panic** | refused |

## Build targets and dependencies

- Pure Go, CGO-free: types, gob, json (`encoding/json`), yaml (`gopkg.in/yaml`), proto (`google.golang.org/protobuf`), csv, text, graph, storage (file back end uses `syscall.Mmap`, tag `!tinygo`; `flash.go` is the TinyGo path).
- Embedded suitability: heavy reflection (graph capture, json/yaml graph paths, `plugin`-style option mapping), `any`-typed values, maps in options, `fmt` errors → large TinyGo binaries; protobuf-go does not build on TinyGo without care [INFERENCE]. For ESP32 links only a hand-rolled fixed-layout codec (like the graph node/edge records) is realistic; `flash.go` shows the intended direction but is a placeholder.
- Allocations: every decode allocates a new tensor via the factory; `io.ReadAll` in proto; graph reads deserialize edge data per access.

## Tests and evidence

`GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 -v <pkg>` from `.`, 2026-11-09: `x/marshaller` 6 PASS, `csv` 48, `gob` 4, `graph` 11, `json` 8, `proto` 4, `storage` 18, `text` 2, `types` 5, `yaml` 8 — 0 failures (matches baseline-2026-11-08 §5). Probes (outside repo, `GOFLAGS=-mod=mod` in the probe module only): `~/src/tmp/easyrobot/io/graphprobe` (graph node data, memory map, proto dtypes), `fmtprobe` (gob/json/yaml/csv dtypes).

## API notes (ease of use, composability)

- **Good:** two tiny interfaces; every format is swappable behind them; `TensorFactory` and `MappedStorageFactory` are clean dependency-injection points; `types.Error` carries op/format; `FrameStream` (channel + `Close`) is simple to consume; `WithRelease` makes sink ownership explicit.
- **`any` everywhere:** `Marshal(value any)` / `Unmarshal(dst any)` dispatch by runtime type switch; unsupported combinations surface as runtime errors or panics (dtype table above). A typed layer (`MarshalTensor(w, t)`, generic `Unmarshal[T]`) on top would catch most misuse at compile time.
- **No format registry:** constructors differ (`graph.NewMarshaller(factory, …) (m, error)`, v4l `New`, others `NewMarshaller`), and nothing maps `"json"` → constructor, although every type has `Format()`. A `marshaller.Register/Get(format)` would make CLIs and config-driven pipelines trivial.
- **Option leakage:** options from one package are accepted by all (`types.Option`), but package-specific ones are either no-ops elsewhere or smuggled through `Metadata` strings (graph paths as `"graph.nodePath"`, `graph/options.go:33-39`) or hidden interfaces (graph callback options have empty `Apply`, `options.go:165,195,225`). `proto` applies per-call options to the marshaller's own struct (`proto/marshaller.go:37-39`, `unmarshaller.go:40-42`); the others copy per call, but `Metadata` maps are still shared (shallow copy).
- **Reader/writer ignored:** graph, gocv and v4l marshallers ignore the `io.Reader`/`io.Writer` argument and take paths/devices from options. That keeps the interface uniform but makes `Marshal(nil, v)` legal and misleading; a separate `Source`/`Sink`/`Store` interface would be more honest.
- **Duplication:** json and yaml are line-for-line copies; graph capture is implemented three times (json, yaml, graph). One reflection walker shared by all would fix bugs once.
- **Streams vs chains:** `FrameStream` (push, channel) and `x/chain` processors (pull, `buffer.Reader`) have no adapter; every app writes the glue loop ([[pipeline-and-chain]]).

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Table test: every dtype × every value marshaller — x/marshaller/marshaller_test.go #p2
- [ ] Format registry (`Register`, `Get(format)`) — x/marshaller/types/types.go #p3
- [ ] Merge json/yaml implementations — x/marshaller/json/internal.go, x/marshaller/yaml/internal.go #p3

## Inferred objectives (proposals)

- **Proposal:** make `proto` (or a fixed-layout binary codec) the canonical tensor wire format for the K230→ESP32 feature stream ([[W4-k230-feature-stream]]) and fix its UINT8/INT8 coverage first — feature tensors from quantised models are INT8/UINT8.
- **Proposal:** finish the graph store (node data, transactions) so learned decision trees ([[W6-decision-trees]]) can be trained on a PC and memory-mapped from flash on ESP32 via `flash.go`.
- **Proposal:** add a small typed façade over the `any`-based API for the common cases (tensor, model, graph) to improve ease of use without breaking existing callers.
