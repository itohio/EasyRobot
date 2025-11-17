package graph

import (
	"encoding/binary"
	"fmt"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"google.golang.org/protobuf/proto"
)

// readNodeFile reads the node file header and validates it
func readNodeFileHeader(storage types.MappedStorage) (*NodeFileHeader, error) {
	// Map header region
	region, err := storage.Map(0, NodeHeaderSize)
	if err != nil {
		return nil, fmt.Errorf("failed to map node header: %w", err)
	}
	defer region.Unmap()

	data := region.Bytes()
	header, err := ReadNodeHeader(data)
	if err != nil {
		return nil, fmt.Errorf("failed to read node header: %w", err)
	}

	return header, nil
}

// readEdgeFileHeader reads the edge file header and validates it
func readEdgeFileHeader(storage types.MappedStorage) (*EdgeFileHeader, error) {
	// Map header region
	region, err := storage.Map(0, EdgeHeaderSize)
	if err != nil {
		return nil, fmt.Errorf("failed to map edge header: %w", err)
	}
	defer region.Unmap()

	data := region.Bytes()
	header, err := ReadEdgeHeader(data)
	if err != nil {
		return nil, fmt.Errorf("failed to read edge header: %w", err)
	}

	return header, nil
}

// readNodeRecord reads a node record at the given index
func readNodeRecord(storage types.MappedStorage, index int) (*NodeRecord, error) {
	region, err := storage.Map(0, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to map node storage: %w", err)
	}
	defer region.Unmap()

	data := region.Bytes()
	recordOffset := NodeHeaderSize + int64(index)*NodeRecordSize
	if recordOffset+NodeRecordSize > int64(len(data)) {
		return nil, fmt.Errorf("node record %d out of range", index)
	}
	recordBytes := data[recordOffset : recordOffset+NodeRecordSize]
	record, err := ReadNodeRecord(recordBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to read node record: %w", err)
	}

	return record, nil
}

// readEdgeRecord reads an edge record at the given index
func readEdgeRecord(storage types.MappedStorage, index int) (*EdgeRecord, error) {
	region, err := storage.Map(0, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to map edge storage: %w", err)
	}
	defer region.Unmap()

	data := region.Bytes()
	recordOffset := EdgeHeaderSize + int64(index)*EdgeRecordSize
	if recordOffset+EdgeRecordSize > int64(len(data)) {
		return nil, fmt.Errorf("edge record %d out of range", index)
	}
	recordBytes := data[recordOffset : recordOffset+EdgeRecordSize]
	record, err := ReadEdgeRecord(recordBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to read edge record: %w", err)
	}

	return record, nil
}

func readDataFileHeader(storage types.MappedStorage) (*DataFileHeader, error) {
	region, err := storage.Map(0, DataHeaderSize)
	if err != nil {
		return nil, fmt.Errorf("failed to map data header: %w", err)
	}
	defer region.Unmap()

	header, err := ReadDataHeader(region.Bytes())
	if err != nil {
		return nil, fmt.Errorf("failed to read data header: %w", err)
	}
	return header, nil
}

func forEachDataEntryMeta(storage types.MappedStorage, fn func(DataType, string) error) error {
	region, err := storage.Map(0, 0)
	if err != nil {
		return fmt.Errorf("failed to map data storage: %w", err)
	}
	defer region.Unmap()

	data := region.Bytes()
	if len(data) < DataHeaderSize {
		return fmt.Errorf("data file truncated")
	}
	header, err := ReadDataHeader(data[:DataHeaderSize])
	if err != nil {
		return fmt.Errorf("failed to read data header: %w", err)
	}

	offset := int64(DataHeaderSize)
	for i := uint64(0); i < header.EntryCount; i++ {
		if offset+dataEntryHeaderSize > int64(len(data)) {
			return fmt.Errorf("data entry header truncated")
		}
		headerBytes := data[offset : offset+dataEntryHeaderSize]
		payloadLen := binary.LittleEndian.Uint32(headerBytes[0:4])
		entryType := DataType(headerBytes[4])
		typeNameLen := binary.LittleEndian.Uint16(headerBytes[5:7])

		totalSize := dataEntryHeaderSize + int64(typeNameLen) + int64(payloadLen)
		if offset+totalSize > int64(len(data)) {
			return fmt.Errorf("data entry truncated")
		}
		entryBytes := data[offset : offset+totalSize]
		typeName := string(entryBytes[7 : 7+typeNameLen])
		if err := fn(entryType, typeName); err != nil {
			return err
		}
		offset += totalSize
	}
	return nil
}

func validateProtobufTypes(storage types.MappedStorage, registry map[string]proto.Message) error {
	return forEachDataEntryMeta(storage, func(dataType DataType, typeName string) error {
		if dataType != DataTypeProtobuf {
			return nil
		}
		if typeName == "" {
			return fmt.Errorf("protobuf data entry missing type name")
		}
		if typeName == metadataTypeName {
			return nil
		}
		if registry == nil {
			return fmt.Errorf("protobuf type %q not registered", typeName)
		}
		if proto := registry[typeName]; proto == nil {
			return fmt.Errorf("protobuf type %q not registered", typeName)
		}
		return nil
	})
}
