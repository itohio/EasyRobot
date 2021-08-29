package transport

import (
	"context"
	"errors"
	"io"
	"unsafe"
)

//go:generate protoc -I=./ -I=${GOPATH}/pkg/mod/ -I=${GOPATH}/src --gogofaster_out=./ types.proto
//go:generate go run ../../../cmd/codegen -i types.pb.go -c ../proto/proto.json -m re

var (
	Magic         uint32 = 0xBADAB00A
	maxHeaderSize        = int(unsafe.Sizeof(PacketHeader{}))
	BufferSize           = maxHeaderSize + 64
	errCtxDone           = errors.New("context done")
)

func handlePanic(place string) {
	if r := recover(); r != nil {
		println("Recovering from panic:", r, "@", place)
	}
}

// Reliable transport
// Packets always start with header
func ReadPackets(ctx context.Context, id uint32, reader io.Reader) <-chan PacketData {
	defer handlePanic("read packets")

	channel := make(chan PacketData, 1)
	go ReliableStreamReader(ctx, id, reader, channel)

	return channel
}

func WritePacket(id, dataType uint32, writer io.Writer, data []byte) error {
	packetData := PacketData{
		Type: dataType,
		Data: data,
	}
	dataSize := packetData.Size()

	packetHeader := PacketHeader{
		Magic:    Magic,
		ID:       id,
		TailSize: uint32(dataSize),
		CRC:      0,
	}
	headerSize := packetHeader.Size()
	dataBuffer := make([]byte, 1+headerSize+dataSize)

	dataBuffer[0] = byte(headerSize)
	_, err := packetHeader.MarshalTo(dataBuffer[1 : 1+headerSize])
	if err != nil {
		return err
	}
	_, err = packetData.MarshalTo(dataBuffer[1+headerSize:])
	if err != nil {
		return err
	}

	for {
		n, err := writer.Write(dataBuffer)
		if err != nil {
			return err
		}
		if n == len(dataBuffer) {
			return nil
		}
		dataBuffer = dataBuffer[n:]
	}
}

func ReliableStreamReader(ctx context.Context, id uint32, reader io.Reader, ch chan PacketData) {
	defer handlePanic("stream reader")
	var (
		packet PacketData
		err    error
		buffer = make([]byte, BufferSize)
		n      int
	)

	defer close(ch)

	for {
		buffer, n, packet, err = ReadPacketFromReliableStream(ctx, id, reader, buffer)
		if err != nil {
			println("read error", err)
			return
		}

		if n == 0 {
			continue
		}

		select {
		case ch <- packet:
			println("got packet", packet.Type)
		default:
			println("drop packet", packet.Type)
		}
	}
}

func ReadPacketFromReliableStream(ctx context.Context, id uint32, reader io.Reader, inbuf []byte) (buffer []byte, np int, tail PacketData, err error) {
	defer handlePanic("read packet")
	buffer = inbuf
	np = 0

	n, err := reader.Read(buffer[:1])
	if err != nil {
		return
	}
	if n == 0 {
		return
	}

	headerSize := uint32(buffer[0])
	if headerSize < 4 || int(headerSize) > maxHeaderSize {
		return
	}

	err = readAll(ctx, reader, buffer[:headerSize])
	if err != nil {
		return
	}

	var header PacketHeader
	err = header.Unmarshal(buffer[:headerSize])
	if err != nil {
		return
	}

	if header.Magic != Magic {
		return
	}

	if header.ID != id {
		err = skipAll(ctx, reader, buffer, int(header.TailSize))
		return
	}

	if header.TailSize > uint32(len(buffer)) {
		buffer = make([]byte, header.TailSize)
	}

	tmp := buffer[:header.TailSize]
	err = readAll(ctx, reader, tmp)
	if err != nil {
		return
	}

	err = tail.Unmarshal(tmp)

	np = int(header.TailSize)

	return
}

func skipAll(ctx context.Context, reader io.Reader, buf []byte, bytes int) error {
	defer handlePanic("skip all")
	for {
		if bytes <= 0 {
			return nil
		}
		if len(buf) > bytes {
			buf = buf[:bytes]
		}

		select {
		case <-ctx.Done():
			return errCtxDone
		default:
		}

		n, err := reader.Read(buf)
		if err != nil {
			return err
		}

		bytes -= n
	}
}

func readAll(ctx context.Context, reader io.Reader, buf []byte) error {
	defer handlePanic("read all")
	for {
		select {
		case <-ctx.Done():
			return errCtxDone
		default:
		}

		n, err := reader.Read(buf)
		if err != nil {
			return err
		}

		if n == len(buf) {
			return nil
		}
		if n == 0 {
			continue
		}

		buf = buf[n:]
	}
}
