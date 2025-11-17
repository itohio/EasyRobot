package graph

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// edgeInfo holds edge information during marshalling.
// Note: cost is NOT stored - it's calculated via Graph.Cost() method when needed.
// Edge records only store FromID, ToID, DataOffset, and Flags.
type edgeInfo struct {
	fromID int64
	toID   int64
}

const dataEntryHeaderSize = 7 // uint32 payload length + uint8 type + uint16 type name length

type serializedDataEntry struct {
	dataType DataType
	typeName []byte
	payload  []byte
}

// writeNodeFile writes the node file to storage
func writeNodeFile(storage types.MappedStorage, nodes []capturedNode, kind graphKind) error {
	nodeCount := uint64(len(nodes))
	maxID := int64(0)
	for _, node := range nodes {
		if node.id > maxID {
			maxID = node.id
		}
	}

	// Calculate size needed
	headerSize := int64(NodeHeaderSize)
	recordsSize := int64(len(nodes)) * NodeRecordSize
	totalSize := headerSize + recordsSize

	// Grow storage if needed
	currentSize, err := storage.Size()
	if err != nil {
		return err
	}
	if totalSize > currentSize {
		if err := storage.Grow(totalSize); err != nil {
			return err
		}
	}

	// Map the entire file
	region, err := storage.Map(0, totalSize)
	if err != nil {
		return err
	}
	defer region.Unmap()

	data := region.Bytes()

	// Write header
	header := &NodeFileHeader{
		Magic:     [8]byte{'G', 'R', 'A', 'P', 'H', 'N', 'D', '1'},
		Version:   FormatVersion,
		MaxID:     maxID,
		NodeCount: nodeCount,
		// DataFileOffset and Checksum will be set later
	}
	header.Reserved[0] = graphKindToByte(kind)
	if err := WriteNodeHeader(data[0:NodeHeaderSize], header); err != nil {
		return err
	}

	// Write records
	offset := NodeHeaderSize
	for _, node := range nodes {
		record := &NodeRecord{
			ID:    node.id,
			Flags: FlagActive,
		}
		if err := WriteNodeRecord(data[offset:offset+NodeRecordSize], record); err != nil {
			return err
		}
		offset += NodeRecordSize
	}

	// Sync
	if err := region.Sync(); err != nil {
		return err
	}

	return nil
}

// writeEdgeFile writes the edge file to storage
func writeEdgeFile(storage types.MappedStorage, edges []edgeInfo, kind graphKind) error {
	edgeCount := uint64(len(edges))
	maxID := int64(0)
	for _, edge := range edges {
		if edge.fromID > maxID {
			maxID = edge.fromID
		}
		if edge.toID > maxID {
			maxID = edge.toID
		}
	}

	// Calculate size needed
	headerSize := int64(EdgeHeaderSize)
	recordsSize := int64(len(edges)) * EdgeRecordSize
	totalSize := headerSize + recordsSize

	// Grow storage if needed
	currentSize, err := storage.Size()
	if err != nil {
		return err
	}
	if totalSize > currentSize {
		if err := storage.Grow(totalSize); err != nil {
			return err
		}
	}

	// Map the entire file
	region, err := storage.Map(0, totalSize)
	if err != nil {
		return err
	}
	defer region.Unmap()

	data := region.Bytes()

	// Write header
	header := &EdgeFileHeader{
		Magic:     [8]byte{'G', 'R', 'A', 'P', 'H', 'E', 'D', '1'},
		Version:   FormatVersion,
		MaxID:     maxID,
		EdgeCount: edgeCount,
	}
	header.Reserved[0] = graphKindToByte(kind)
	if err := WriteEdgeHeader(data[0:EdgeHeaderSize], header); err != nil {
		return err
	}

	// Write records
	offset := EdgeHeaderSize
	for _, edge := range edges {
		record := &EdgeRecord{
			FromID: edge.fromID,
			ToID:   edge.toID,
			Flags:  FlagActive,
			// DataOffset will be set when writing data
		}
		if err := WriteEdgeRecord(data[offset:offset+EdgeRecordSize], record); err != nil {
			return err
		}
		offset += EdgeRecordSize
	}

	// Sync
	if err := region.Sync(); err != nil {
		return err
	}

	return nil
}

func newSerializedDataEntry(dataType DataType, typeName string, payload []byte) (serializedDataEntry, error) {
	if len(payload) > math.MaxUint32 {
		return serializedDataEntry{}, fmt.Errorf("data payload too large: %d bytes", len(payload))
	}
	entry := serializedDataEntry{
		dataType: dataType,
		payload:  make([]byte, len(payload)),
	}
	copy(entry.payload, payload)
	if typeName != "" {
		entry.typeName = []byte(typeName)
		if len(entry.typeName) > math.MaxUint16 {
			return serializedDataEntry{}, fmt.Errorf("type name too long: %d bytes", len(entry.typeName))
		}
	}
	return entry, nil
}

func writeDataFile(storage types.MappedStorage, nodeEntries, edgeEntries []serializedDataEntry, metadataEntry *serializedDataEntry) ([]uint64, []uint64, uint64, error) {
	nodeOffsets := make([]uint64, len(nodeEntries))
	edgeOffsets := make([]uint64, len(edgeEntries))
	var metadataOffset uint64

	totalEntries := len(nodeEntries) + len(edgeEntries)
	if metadataEntry != nil {
		totalEntries++
	}
	totalSize := int64(DataHeaderSize)
	requiredEntries := append(append([]serializedDataEntry{}, nodeEntries...), edgeEntries...)
	if metadataEntry != nil {
		requiredEntries = append(requiredEntries, *metadataEntry)
	}
	for _, entry := range requiredEntries {
		size, err := dataEntrySize(entry)
		if err != nil {
			return nil, nil, 0, err
		}
		totalSize += size
	}

	currentSize, err := storage.Size()
	if err != nil {
		return nil, nil, 0, err
	}
	if totalSize > currentSize {
		if err := storage.Grow(totalSize); err != nil {
			return nil, nil, 0, err
		}
	}

	region, err := storage.Map(0, totalSize)
	if err != nil {
		return nil, nil, 0, err
	}
	defer region.Unmap()

	data := region.Bytes()
	header := &DataFileHeader{
		Magic:      [8]byte{'G', 'R', 'A', 'P', 'H', 'D', 'T', '1'},
		Version:    FormatVersion,
		EntryCount: uint64(totalEntries),
	}
	if err := WriteDataHeader(data[:DataHeaderSize], header); err != nil {
		return nil, nil, 0, err
	}

	offset := int64(DataHeaderSize)
	writeEntries := func(entries []serializedDataEntry, offsets []uint64) error {
		for i, entry := range entries {
			entryOffset := offset
			payloadLen := len(entry.payload)
			typeNameLen := len(entry.typeName)

			start := int(offset)
			binary.LittleEndian.PutUint32(data[start:start+4], uint32(payloadLen))
			offset += 4
			start = int(offset)
			data[start] = byte(entry.dataType)
			offset++
			start = int(offset)
			binary.LittleEndian.PutUint16(data[start:start+2], uint16(typeNameLen))
			offset += 2

			start = int(offset)
			copy(data[start:start+typeNameLen], entry.typeName)
			offset += int64(typeNameLen)

			start = int(offset)
			copy(data[start:start+payloadLen], entry.payload)
			offset += int64(payloadLen)

			offsets[i] = uint64(entryOffset)
		}
		return nil
	}

	if err := writeEntries(nodeEntries, nodeOffsets); err != nil {
		return nil, nil, 0, err
	}
	if err := writeEntries(edgeEntries, edgeOffsets); err != nil {
		return nil, nil, 0, err
	}
	if metadataEntry != nil {
		entry := *metadataEntry
		entryOffset := offset
		payloadLen := len(entry.payload)
		typeNameLen := len(entry.typeName)

		start := int(offset)
		binary.LittleEndian.PutUint32(data[start:start+4], uint32(payloadLen))
		offset += 4
		start = int(offset)
		data[start] = byte(entry.dataType)
		offset++
		start = int(offset)
		binary.LittleEndian.PutUint16(data[start:start+2], uint16(typeNameLen))
		offset += 2

		start = int(offset)
		copy(data[start:start+typeNameLen], entry.typeName)
		offset += int64(typeNameLen)

		start = int(offset)
		copy(data[start:start+payloadLen], entry.payload)
		offset += int64(payloadLen)

		metadataOffset = uint64(entryOffset)
	}

	if err := region.Sync(); err != nil {
		return nil, nil, 0, err
	}

	return nodeOffsets, edgeOffsets, metadataOffset, nil
}
func dataEntrySize(entry serializedDataEntry) (int64, error) {
	if len(entry.payload) > math.MaxUint32 {
		return 0, fmt.Errorf("data payload too large: %d bytes", len(entry.payload))
	}
	if len(entry.typeName) > math.MaxUint16 {
		return 0, fmt.Errorf("type name too long: %d bytes", len(entry.typeName))
	}
	return int64(dataEntryHeaderSize + len(entry.typeName) + len(entry.payload)), nil
}

func updateNodeRecordDataOffset(storage types.MappedStorage, index int, dataOffset uint64) error {
	region, err := storage.Map(0, 0)
	if err != nil {
		return err
	}
	defer region.Unmap()

	data := region.Bytes()
	recordOffset := int64(NodeHeaderSize) + int64(index)*NodeRecordSize
	recordBytes := data[recordOffset : recordOffset+NodeRecordSize]
	record, err := ReadNodeRecord(recordBytes)
	if err != nil {
		return err
	}
	record.DataOffset = dataOffset
	if err := WriteNodeRecord(recordBytes, record); err != nil {
		return err
	}
	return region.Sync()
}

func updateNodeHeaderDataOffset(storage types.MappedStorage, dataOffset uint64) error {
	region, err := storage.Map(0, NodeHeaderSize)
	if err != nil {
		return err
	}
	defer region.Unmap()

	header, err := ReadNodeHeader(region.Bytes())
	if err != nil {
		return err
	}
	header.DataFileOffset = dataOffset
	if err := WriteNodeHeader(region.Bytes(), header); err != nil {
		return err
	}
	return region.Sync()
}

func updateEdgeHeaderDataOffset(storage types.MappedStorage, dataOffset uint64) error {
	region, err := storage.Map(0, EdgeHeaderSize)
	if err != nil {
		return err
	}
	defer region.Unmap()

	header, err := ReadEdgeHeader(region.Bytes())
	if err != nil {
		return err
	}
	header.DataFileOffset = dataOffset
	if err := WriteEdgeHeader(region.Bytes(), header); err != nil {
		return err
	}
	return region.Sync()
}

func updateEdgeRecordDataOffset(storage types.MappedStorage, index int, dataOffset uint64) error {
	if dataOffset > math.MaxUint32 {
		return fmt.Errorf("edge data offset %d exceeds uint32 capacity", dataOffset)
	}
	region, err := storage.Map(0, 0)
	if err != nil {
		return err
	}
	defer region.Unmap()

	data := region.Bytes()
	recordOffset := int64(EdgeHeaderSize) + int64(index)*EdgeRecordSize
	recordBytes := data[recordOffset : recordOffset+EdgeRecordSize]
	record, err := ReadEdgeRecord(recordBytes)
	if err != nil {
		return err
	}
	record.DataOffset = uint32(dataOffset)
	if err := WriteEdgeRecord(recordBytes, record); err != nil {
		return err
	}
	return region.Sync()
}
