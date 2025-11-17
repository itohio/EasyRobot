package graph

import (
	"fmt"
	"io"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// GraphUnmarshaller implements types.Unmarshaller for graph storage.
type GraphUnmarshaller struct {
	storageFactory types.MappedStorageFactory
	opts           types.Options
	cfg            config
}

// NewUnmarshaller creates a new graph unmarshaller.
func NewUnmarshaller(storageFactory types.MappedStorageFactory, opts ...types.Option) (*GraphUnmarshaller, error) {
	u := &GraphUnmarshaller{
		storageFactory: storageFactory,
		opts:           types.Options{},
		cfg:            config{mirror: true},
	}

	for _, opt := range opts {
		opt.Apply(&u.opts)
	}

	_, u.cfg = applyOptions(u.opts, u.cfg, opts)
	u.cfg.readOnly = true

	return u, nil
}

// Format returns the format name.
func (u *GraphUnmarshaller) Format() string {
	return "graph"
}

// Unmarshal loads a graph from storage.
func (u *GraphUnmarshaller) Unmarshal(r io.Reader, dst any, opts ...types.Option) error {
	if dst == nil {
		return types.NewError("unmarshal", "graph", "dst cannot be nil", nil)
	}

	_, localCfg := applyOptions(u.opts, u.cfg, opts)

	nodeStorage, err := u.storageFactory(localCfg.nodePath, true)
	if err != nil {
		return types.NewError("unmarshal", "graph", fmt.Sprintf("failed to open node storage: %v", err), err)
	}
	storages := []types.MappedStorage{nodeStorage}

	edgeStorage, err := u.storageFactory(localCfg.edgePath, true)
	if err != nil {
		closeStorages(storages)
		return types.NewError("unmarshal", "graph", fmt.Sprintf("failed to open edge storage: %v", err), err)
	}
	storages = append(storages, edgeStorage)

	dataStorage, err := u.storageFactory(localCfg.dataPath, true)
	if err != nil {
		closeStorages(storages)
		return types.NewError("unmarshal", "graph", fmt.Sprintf("failed to open data storage: %v", err), err)
	}
	storages = append(storages, dataStorage)

	nodeHeader, err := readNodeFileHeader(nodeStorage)
	if err != nil {
		closeStorages(storages)
		return types.NewError("unmarshal", "graph", err.Error(), err)
	}
	edgeHeader, err := readEdgeFileHeader(edgeStorage)
	if err != nil {
		closeStorages(storages)
		return types.NewError("unmarshal", "graph", err.Error(), err)
	}

	graphKind := graphKindFromByte(nodeHeader.Reserved[0])
	metadataOffset := nodeHeader.DataFileOffset
	if metadataOffset == 0 {
		metadataOffset = edgeHeader.DataFileOffset
	}
	meta, err := readGraphMetadata(dataStorage, metadataOffset)
	if err != nil {
		closeStorages(storages)
		return types.NewError("unmarshal", "graph", err.Error(), err)
	}
	if meta != nil && meta.Kind != "" {
		graphKind = meta.Kind
	}

	if err := validateProtobufTypes(dataStorage, localCfg.registeredTypes); err != nil {
		closeStorages(storages)
		return types.NewError("unmarshal", "graph", err.Error(), err)
	}

	storedGraph, err := newStoredGraph(nodeStorage, edgeStorage, dataStorage, localCfg.registeredTypes, graphKind)
	if err != nil {
		closeStorages(storages)
		return types.NewError("unmarshal", "graph", fmt.Sprintf("failed to load stored graph: %v", err), err)
	}

	switch target := dst.(type) {
	case *StoredGraph:
		*target = *storedGraph
	case *StoredTree:
		tree, err := newStoredTree(storedGraph, meta)
		if err != nil {
			storedGraph.Close()
			return types.NewError("unmarshal", "graph", err.Error(), err)
		}
		*target = *tree
	case *StoredDecisionTree:
		dTree, err := newStoredDecisionTree(storedGraph, meta, localCfg)
		if err != nil {
			storedGraph.Close()
			return types.NewError("unmarshal", "graph", err.Error(), err)
		}
		*target = *dTree
	case *StoredExpressionGraph:
		expr, err := newStoredExpressionGraph(storedGraph, meta, localCfg)
		if err != nil {
			storedGraph.Close()
			return types.NewError("unmarshal", "graph", err.Error(), err)
		}
		*target = *expr
	default:
		storedGraph.Close()
		return types.NewError("unmarshal", "graph", "unsupported destination type", nil)
	}

	return nil
}

func closeStorages(storages []types.MappedStorage) {
	for _, s := range storages {
		if s != nil {
			_ = s.Close()
		}
	}
}
