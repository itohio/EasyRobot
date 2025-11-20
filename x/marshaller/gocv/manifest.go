package gocv

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"

	marshallerpb "github.com/itohio/EasyRobot/types/marshaller"
	"github.com/itohio/EasyRobot/x/marshaller/types"
	"google.golang.org/protobuf/proto"
)

// sourceKindToProto converts internal sourceKind to protobuf SourceKind.
func sourceKindToProto(kind sourceKind) marshallerpb.SourceKind {
	switch kind {
	case sourceKindSingle:
		return marshallerpb.SourceKind_SOURCE_KIND_SINGLE
	case sourceKindVideoFile:
		return marshallerpb.SourceKind_SOURCE_KIND_VIDEO_FILE
	case sourceKindVideoDevice:
		return marshallerpb.SourceKind_SOURCE_KIND_VIDEO_DEVICE
	case sourceKindFileList:
		return marshallerpb.SourceKind_SOURCE_KIND_FILE_LIST
	default:
		return marshallerpb.SourceKind_SOURCE_KIND_UNKNOWN
	}
}

// protoToSourceKind converts protobuf SourceKind to internal sourceKind.
func protoToSourceKind(kind marshallerpb.SourceKind) sourceKind {
	switch kind {
	case marshallerpb.SourceKind_SOURCE_KIND_SINGLE:
		return sourceKindSingle
	case marshallerpb.SourceKind_SOURCE_KIND_VIDEO_FILE:
		return sourceKindVideoFile
	case marshallerpb.SourceKind_SOURCE_KIND_VIDEO_DEVICE:
		return sourceKindVideoDevice
	case marshallerpb.SourceKind_SOURCE_KIND_FILE_LIST:
		return sourceKindFileList
	default:
		return sourceKindUnknown
	}
}

// sourceSpecToProto converts internal sourceSpec to protobuf SourceSpec.
func sourceSpecToProto(spec sourceSpec) *marshallerpb.SourceSpec {
	pbSpec := &marshallerpb.SourceSpec{
		Kind: sourceKindToProto(spec.Kind),
		Path: spec.Path,
	}
	
	if spec.Device != nil {
		pbSpec.Device = &marshallerpb.DeviceSpec{
			Id:         int32(spec.Device.ID),
			Width:      int32(spec.Device.Width),
			Height:     int32(spec.Device.Height),
			FrameRate:  int32(spec.Device.FrameRate),
			PixelFormat: spec.Device.PixelFormat,
		}
		if len(spec.Device.Controls) > 0 {
			pbSpec.Device.Controls = make(map[string]int32)
			for k, v := range spec.Device.Controls {
				pbSpec.Device.Controls[k] = v
			}
		}
	}
	
	if len(spec.Files) > 0 {
		pbSpec.Files = append([]string(nil), spec.Files...)
	}
	
	return pbSpec
}

// protoToSourceSpec converts protobuf SourceSpec to internal sourceSpec.
func protoToSourceSpec(pbSpec *marshallerpb.SourceSpec) sourceSpec {
	spec := sourceSpec{
		Kind: protoToSourceKind(pbSpec.Kind),
		Path: pbSpec.Path,
	}
	
	if pbSpec.Device != nil {
		spec.Device = &deviceSpec{
			ID:          int(pbSpec.Device.Id),
			Width:       int(pbSpec.Device.Width),
			Height:      int(pbSpec.Device.Height),
			FrameRate:   int(pbSpec.Device.FrameRate),
			PixelFormat: pbSpec.Device.PixelFormat,
		}
		if len(pbSpec.Device.Controls) > 0 {
			spec.Device.Controls = make(map[string]int32)
			for k, v := range pbSpec.Device.Controls {
				spec.Device.Controls[k] = v
			}
		}
	}
	
	if len(pbSpec.Files) > 0 {
		spec.Files = append([]string(nil), pbSpec.Files...)
	}
	
	return spec
}

// configToManifest converts config to protobuf FrameStreamManifest.
func configToManifest(cfg config) *marshallerpb.FrameStreamManifest {
	manifest := &marshallerpb.FrameStreamManifest{
		Sources:    make([]*marshallerpb.SourceSpec, 0, len(cfg.stream.sources)),
		SyncMode:   syncModeToString(cfg.stream.sequential),
		BestEffort: cfg.stream.allowBestEffort,
		Metadata:   make(map[string]string),
	}
	
	// Convert sources
	for _, spec := range cfg.stream.sources {
		manifest.Sources = append(manifest.Sources, sourceSpecToProto(spec))
	}
	
	// Add any additional metadata from config if needed
	// (currently config doesn't have extra metadata, but this is extensible)
	
	return manifest
}

// manifestToConfig updates config with values from protobuf FrameStreamManifest.
func manifestToConfig(manifest *marshallerpb.FrameStreamManifest, cfg *config) {
	// Convert sources
	cfg.stream.sources = make([]sourceSpec, 0, len(manifest.Sources))
	for _, pbSpec := range manifest.Sources {
		cfg.stream.sources = append(cfg.stream.sources, protoToSourceSpec(pbSpec))
	}
	
	// Set sync mode
	cfg.stream.sequential = syncModeFromString(manifest.SyncMode)
	
	// Set best effort
	cfg.stream.allowBestEffort = manifest.BestEffort
}

// syncModeToString converts sequential bool to sync mode string.
func syncModeToString(sequential bool) string {
	if sequential {
		return "sequential"
	}
	return "parallel"
}

// syncModeFromString converts sync mode string to sequential bool.
func syncModeFromString(mode string) bool {
	return mode == "sequential"
}

// writeManifest writes a protobuf FrameStreamManifest to the writer.
// The manifest is written as a length-prefixed protobuf message.
func writeManifest(w io.Writer, manifest *marshallerpb.FrameStreamManifest) error {
	data, err := proto.Marshal(manifest)
	if err != nil {
		return types.NewError("marshal", "gocv", "protobuf encode manifest", err)
	}
	
	// Write length-prefixed message (4-byte little-endian length, then data)
	length := uint32(len(data))
	if err := binary.Write(w, binary.LittleEndian, length); err != nil {
		return types.NewError("marshal", "gocv", "write manifest length", err)
	}
	
	if _, err := w.Write(data); err != nil {
		return types.NewError("marshal", "gocv", "write manifest data", err)
	}
	
	return nil
}

// readManifest reads a protobuf FrameStreamManifest from the reader.
// Returns nil, nil if no manifest is present (for backward compatibility with text paths).
// Uses a buffered reader to peek at the first bytes to detect format.
func readManifest(r io.Reader) (*marshallerpb.FrameStreamManifest, error) {
	// Use a buffered reader to peek at the first bytes
	// Protobuf messages start with varint field tags, which are typically small values
	// Text paths start with printable characters or newlines
	bufReader, ok := r.(*bufio.Reader)
	if !ok {
		bufReader = bufio.NewReader(r)
	}
	
	// Peek at first 4 bytes to check if it looks like a length prefix
	peek, err := bufReader.Peek(4)
	if err != nil {
		if err == io.EOF {
			// Empty reader, no manifest
			return nil, nil
		}
		return nil, types.NewError("unmarshal", "gocv", "peek manifest", err)
	}
	
	// Check if first 4 bytes look like a reasonable length prefix (little-endian uint32)
	// Protobuf manifests should have a reasonable size (say, < 1MB)
	// Text paths typically start with '/' or letters
	length := binary.LittleEndian.Uint32(peek)
	if length > 1024*1024 || length == 0 {
		// Doesn't look like a protobuf length prefix, probably text format
		return nil, nil
	}
	
	// Read the length (we already peeked, so this should succeed)
	var actualLength uint32
	if err := binary.Read(bufReader, binary.LittleEndian, &actualLength); err != nil {
		// If we can't read, assume it's not a protobuf manifest
		return nil, nil
	}
	
	// Sanity check on length
	if actualLength > 10*1024*1024 { // 10MB limit
		return nil, types.NewError("unmarshal", "gocv", "manifest too large", fmt.Errorf("length %d exceeds limit", actualLength))
	}
	
	// Read manifest data
	data := make([]byte, actualLength)
	if _, err := io.ReadFull(bufReader, data); err != nil {
		return nil, types.NewError("unmarshal", "gocv", "read manifest data", err)
	}
	
	// Unmarshal protobuf
	manifest := &marshallerpb.FrameStreamManifest{}
	if err := proto.Unmarshal(data, manifest); err != nil {
		// If unmarshal fails, it's probably not a protobuf manifest
		return nil, nil
	}
	
	return manifest, nil
}

