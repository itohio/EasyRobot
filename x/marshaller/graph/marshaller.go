package graph

import (
	"fmt"
	"io"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// GraphMarshaller implements types.Marshaller for graph storage
type GraphMarshaller struct {
	storageFactory types.MappedStorageFactory
	opts           types.Options
	cfg            config
}

// NewMarshaller creates a new graph marshaller
func NewMarshaller(storageFactory types.MappedStorageFactory, opts ...types.Option) (*GraphMarshaller, error) {
	m := &GraphMarshaller{
		storageFactory: storageFactory,
		opts:           types.Options{},
		cfg:            config{mirror: true},
	}

	// Apply options
	for _, opt := range opts {
		opt.Apply(&m.opts)
	}

	// Extract config
	_, m.cfg = applyOptions(m.opts, m.cfg, opts)

	return m, nil
}

// Format returns the format name
func (m *GraphMarshaller) Format() string {
	return "graph"
}

// Marshal encodes a graph to storage.
// The writer parameter is ignored (similar to gocv marshaller).
// Storage paths are specified via WithPath, WithEdgesPath, WithLabelsPath options.
func (m *GraphMarshaller) Marshal(w io.Writer, value any, opts ...types.Option) error {
	// Apply options
	_, localCfg := applyOptions(m.opts, m.cfg, opts)

	// Create storage instances
	nodeStorage, err := m.storageFactory(localCfg.nodePath, false)
	if err != nil {
		return types.NewError("marshal", "graph", fmt.Sprintf("failed to create node storage: %v", err), err)
	}
	defer nodeStorage.Close()

	edgeStorage, err := m.storageFactory(localCfg.edgePath, false)
	if err != nil {
		return types.NewError("marshal", "graph", fmt.Sprintf("failed to create edge storage: %v", err), err)
	}
	defer edgeStorage.Close()

	dataStorage, err := m.storageFactory(localCfg.dataPath, false)
	if err != nil {
		return types.NewError("marshal", "graph", fmt.Sprintf("failed to create data storage: %v", err), err)
	}
	defer dataStorage.Close()

	// Marshal the graph
	return m.marshalGraph(value, nodeStorage, edgeStorage, dataStorage)
}

// marshalGraph writes the graph to storage
func (m *GraphMarshaller) marshalGraph(value any, nodeStorage, edgeStorage, dataStorage types.MappedStorage) error {
	nodes, capturedEdges, err := captureGraph(value)
	if err != nil {
		return types.NewError("marshal", "graph", err.Error(), err)
	}

	kind, meta, err := buildGraphMetadata(value, nodes, capturedEdges)
	if err != nil {
		return types.NewError("marshal", "graph", fmt.Sprintf("failed to capture metadata: %v", err), err)
	}

	var metadataEntry *serializedDataEntry
	if metaBytes, err := serializeGraphMetadata(meta); err != nil {
		return types.NewError("marshal", "graph", "failed to encode graph metadata", err)
	} else if len(metaBytes) > 0 {
		entry, err := newSerializedDataEntry(DataTypeBytes, metadataTypeName, metaBytes)
		if err != nil {
			return types.NewError("marshal", "graph", "invalid metadata payload", err)
		}
		metadataEntry = &entry
	}

	edges := make([]edgeInfo, 0, len(capturedEdges))
	edgeEntries := make([]serializedDataEntry, 0, len(capturedEdges))
	for _, edge := range capturedEdges {
		edges = append(edges, edgeInfo{fromID: edge.fromID, toID: edge.toID})

		payload, dataType, typeName, err := serializeData(edge.data)
		if err != nil {
			return types.NewError("marshal", "graph", fmt.Sprintf("failed to serialize edge %d->%d data", edge.fromID, edge.toID), err)
		}
		entry, err := newSerializedDataEntry(dataType, typeName, payload)
		if err != nil {
			return types.NewError("marshal", "graph", fmt.Sprintf("invalid edge %d->%d data", edge.fromID, edge.toID), err)
		}
		edgeEntries = append(edgeEntries, entry)
	}

	if err := writeNodeFile(nodeStorage, nodes, kind); err != nil {
		return types.NewError("marshal", "graph", "failed to write node file", err)
	}

	if err := writeEdgeFile(edgeStorage, edges, kind); err != nil {
		return types.NewError("marshal", "graph", "failed to write edge file", err)
	}

	nodeOffsets, edgeOffsets, metadataOffset, err := writeDataFile(dataStorage, nil, edgeEntries, metadataEntry)
	if err != nil {
		return types.NewError("marshal", "graph", "failed to write data file", err)
	}

	for idx, offset := range nodeOffsets {
		if err := updateNodeRecordDataOffset(nodeStorage, idx, offset); err != nil {
			return types.NewError("marshal", "graph", "failed to update node record", err)
		}
	}
	for idx, offset := range edgeOffsets {
		if err := updateEdgeRecordDataOffset(edgeStorage, idx, offset); err != nil {
			return types.NewError("marshal", "graph", "failed to update edge record", err)
		}
	}

	if err := updateNodeHeaderDataOffset(nodeStorage, metadataOffset); err != nil {
		return types.NewError("marshal", "graph", "failed to update node header", err)
	}
	if err := updateEdgeHeaderDataOffset(edgeStorage, metadataOffset); err != nil {
		return types.NewError("marshal", "graph", "failed to update edge header", err)
	}

	return nil
}
