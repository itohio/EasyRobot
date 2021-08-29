package transport

import (
	"context"
	io "io"
	"reflect"
	"testing"
)

type TestStream struct {
	data  []byte
	index int
	cap   int
}

func (w *TestStream) Write(data []byte) (int, error) {
	if w.data == nil {
		return len(data), nil
	}
	if w.cap+len(data) >= len(w.data) {
		return len(data), nil
	}

	copy(w.data[w.cap:], data)
	w.cap += len(data)
	return len(data), nil
}
func (w *TestStream) Read(data []byte) (int, error) {
	for i := range data {
		data[i] = w.data[w.index]
		w.index = (w.index + 1) % w.cap
	}
	return len(data), nil
}
func (w *TestStream) Close() error {
	return nil
}
func (w *TestStream) Reset() error {
	w.cap = 0
	w.index = 0
	return nil
}

func NewTestStream() *TestStream {
	return &TestStream{
		data: make([]byte, 10000),
		cap:  0,
	}
}

func TestReadPackets(t *testing.T) {
	type args struct {
		ctx    context.Context
		id     uint32
		reader io.Reader
	}
	tests := []struct {
		name string
		args func(t *testing.T) args

		want1 <-chan PacketData
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			got1 := ReadPackets(tArgs.ctx, tArgs.id, tArgs.reader)

			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("ReadPackets got1 = %v, want1: %v", got1, tt.want1)
			}
		})
	}
}

func TestWritePacket(t *testing.T) {
	stream := NewTestStream()
	dataString := "HelloWorldHelloWorld"
	err := WritePacket(123456, 321, stream, []byte(dataString))
	if err != nil {
		t.Fatalf("write packet failed: %v", err)
	}

	if stream.cap != 37 {
		t.Fatalf("wrong written length: %v", stream.cap)
	}

	var header PacketHeader
	var data PacketData

	var wantedHeader = PacketHeader{
		Magic:    Magic,
		ID:       123456,
		TailSize: 25,
		CRC:      0,
	}
	var wantedData = PacketData{
		Type: 321,
		Data: []byte(dataString),
	}

	headerSize := int(stream.data[0])

	err = header.Unmarshal(stream.data[1 : 1+headerSize])
	if err != nil {
		t.Fatalf("Unmarshal failed: %v, size %v, data %v", err, headerSize, stream.data[:30])
	}
	if !reflect.DeepEqual(header, wantedHeader) {
		t.Fatalf("Wrong header: wanted %v, got %v, size %v", wantedHeader, header, headerSize)
	}

	err = data.Unmarshal(stream.data[1+headerSize : 1+uint32(headerSize)+header.TailSize])
	if err != nil {
		t.Fatalf("Unmarshal failed: %v, offset %v, data %v", err, 1+headerSize, stream.data[:30])
	}

	if !reflect.DeepEqual(data, wantedData) {
		t.Fatalf("Wrong data: wanted %v, got %v", wantedData, data)
	}

	buf, n, packet, err := readPacket(context.Background(), 123456, stream, make([]byte, maxHeaderSize))
	if err != nil {
		t.Fatalf("read packet failed: %v", err)
	}

	if len(buf) == maxHeaderSize {
		t.Fatalf("buffer too small: %v vs %v", len(buf), maxHeaderSize)
	}

	if n != int(header.TailSize) {
		t.Fatalf("tail size too small: %v", n)
	}

	if !reflect.DeepEqual(packet, wantedData) {
		t.Fatalf("Wrong packet: wanted %v, got %v", wantedData, packet)
	}
}

func Test_reliableStreamReader(t *testing.T) {
	type args struct {
		ctx    context.Context
		id     uint32
		reader io.Reader
		ch     chan PacketData
	}
	tests := []struct {
		name string
		args func(t *testing.T) args
	}{
		//TODO: Add test cases
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tArgs := tt.args(t)

			reliableStreamReader(tArgs.ctx, tArgs.id, tArgs.reader, tArgs.ch)

		})
	}
}

func Test_skipAll(t *testing.T) {
	tests := []struct {
		name    string
		bufSize int
	}{
		{"smaller", 5},
		{"exact", 10},
		{"larger", 50},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := make([]byte, tt.bufSize)
			stream := NewTestStream()
			stream.Write([]byte("abcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcdabcd"))

			err := skipAll(context.Background(), stream, buf, 10)
			if err != nil {
				t.Fatalf("skip all failed: %v", err)
			}

			if stream.index != 10 {
				t.Fatalf("wanted %v, got %v", 10, stream.index)
			}
		})
	}
}

func Test_readAll(t *testing.T) {
	tests := []struct {
		name    string
		bufSize int
		result  string
	}{
		{"smaller", 5, "12345"},
		{"exact", 10, "1234567890"},
		{"larger", 15, "1234567890abcde"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			buf := make([]byte, tt.bufSize)
			stream := NewTestStream()
			stream.Write([]byte("1234567890abcdefg"))

			err := readAll(context.Background(), stream, buf)
			if err != nil {
				t.Fatalf("skip all failed: %v", err)
			}

			if stream.index != tt.bufSize {
				t.Fatalf("wanted %v, got %v", tt.bufSize, stream.index)
			}

			if string(buf) != tt.result {
				t.Fatalf("wanted %v, got %v", tt.result, string(buf))
			}
		})
	}
}
