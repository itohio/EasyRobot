package store

import (
	"errors"
	"fmt"
	"image"
	"io"

	. "github.com/foxis/EasyRobot/pkg/logger"
)

var (
	ErrExists    = errors.New("already exists")
	ErrNotFound  = errors.New("not found")
	ErrWrongType = errors.New("wrong type")
)

type FQDNType uint16
type Value interface{}

type ValueMarshaller interface {
	Value
	Size() int
	MarshalToSizedBuffer([]byte) (int, error)
	Unmarshal([]byte) (Value, error)
}

type ValueCloser interface {
	Close()
}
type ValueCloserWithError interface {
	Close() error
}
type ValueCloner interface {
	Clone() (Value, error)
}
type ValueWriter interface {
	Write(wr io.Writer, path string) error
}
type ImageGetter interface {
	Image() image.Image
	Value() interface{}
}

type Store interface {
	Name() string
	SetName(string)
	Set(fqdn FQDNType, value Value) error
	Get(fqdn FQDNType) (Value, bool)
	Del(fqdn FQDNType) error
	CopyFrom(src Store)
	Clone(fqdn FQDNType) Store
	Close(fqdn FQDNType)
	ForEach(fqdnPattern FQDNType, callback func(fqdn FQDNType, val Value))

	Index() (int64, error)
	Timestamp() (int64, error)
	DropCount() (int64, error)
	FPS() (float32, error)
	Image(fqdn FQDNType) (ImageGetter, error)
	SetIndex(idx int64)
	SetDropCount(int64)
	SetFPS(float32)
	SetTimestamp(ts int64)
	SetImage(fqdn FQDNType, img ImageGetter)

	Byte(fqdn FQDNType) (byte, error)

	Int(fqdn FQDNType) (int, error)
	Int8(fqdn FQDNType) (int8, error)
	Int16(fqdn FQDNType) (int16, error)
	Int32(fqdn FQDNType) (int32, error)
	Int64(fqdn FQDNType) (int64, error)

	UInt(fqdn FQDNType) (uint, error)
	UInt8(fqdn FQDNType) (uint8, error)
	UInt16(fqdn FQDNType) (uint16, error)
	UInt32(fqdn FQDNType) (uint32, error)
	UInt64(fqdn FQDNType) (uint64, error)

	Float32(fqdn FQDNType) (float32, error)
	Float64(fqdn FQDNType) (float64, error)

	Bytes(fqdn FQDNType) ([]byte, error)

	Ints(fqdn FQDNType) ([]int, error)
	Int8s(fqdn FQDNType) ([]int8, error)
	Int16s(fqdn FQDNType) ([]int16, error)
	Int32s(fqdn FQDNType) ([]int32, error)
	Int64s(fqdn FQDNType) ([]int64, error)

	UInts(fqdn FQDNType) ([]uint, error)
	UInt8s(fqdn FQDNType) ([]uint8, error)
	UInt16s(fqdn FQDNType) ([]uint16, error)
	UInt32s(fqdn FQDNType) ([]uint32, error)
	UInt64s(fqdn FQDNType) ([]uint64, error)

	Float32s(fqdn FQDNType) ([]float32, error)
	Float64s(fqdn FQDNType) ([]float64, error)

	String(fqdn FQDNType) (string, error)
	Strings(fqdn FQDNType) ([]string, error)
	StringMap(fqdn FQDNType) (map[string]string, error)

	StoreList(fqdn FQDNType) ([]Store, error)
	StoreMap(fqdn FQDNType) (map[string]Store, error)

	Dump(fqdn FQDNType)
}

type store struct {
	name string
	data map[FQDNType]interface{}
}

func New() Store {
	return &store{
		data: make(map[FQDNType]interface{}),
	}
}

func NewWithName(name string) Store {
	return &store{
		data: make(map[FQDNType]interface{}),
		name: name,
	}
}

func (s *store) Name() string {
	return s.name
}

func (s *store) SetName(name string) {
	s.name = name
}

func (s *store) Set(fqdn FQDNType, val Value) error {
	s.data[fqdn] = val
	return nil
}

func (s *store) Del(fqdn FQDNType) error {
	if _, ok := s.data[fqdn]; !ok {
		return ErrNotFound
	}

	delete(s.data, fqdn)
	return nil
}

func (s *store) Get(fqdn FQDNType) (Value, bool) {
	v, ok := s.data[fqdn]
	return v, ok
}

func (s *store) CopyFrom(src Store) {
	src.ForEach(ANY, func(name FQDNType, val Value) {
		s.Set(name, val)
	})
}

func (s *store) Clone(fqdnPattern FQDNType) Store {
	result := NewWithName(s.Name())
	s.ForEach(fqdnPattern, func(name FQDNType, val Value) {
		if obj, ok := val.(ValueCloner); ok {
			var err error
			val, err = obj.Clone()
			if err != nil {
				Log.Error().Err(err).Str("name", s.Name()).Msg("Clone")
				return
			}
		}

		result.Set(name, val)
	})
	return result
}

func (s *store) Close(fqdnPattern FQDNType) {
	s.ForEach(fqdnPattern, func(name FQDNType, val Value) {
		if obj, ok := val.(ValueCloser); ok {
			obj.Close()
		} else if obj, ok := val.(ValueCloserWithError); ok {
			obj.Close()
		}
		delete(s.data, name)
	})
}

func (s *store) ForEach(fqdnPattern FQDNType, callback func(name FQDNType, val Value)) {
	for name, val := range s.data {
		callback(name, val)
	}
}

func (s *store) Dump(fqdnPattern FQDNType) {
	log := Log.Debug()
	s.ForEach(fqdnPattern, func(name FQDNType, val Value) {
		log = log.Str(fmt.Sprint(name), fmt.Sprint(val))
	})
	log.Msg(s.name)
}
