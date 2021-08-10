package store

func (s *store) Index() (int64, error) {
	val, ok := s.Get(INDEX)
	if !ok {
		return 0, ErrNotFound
	}
	typedVal, ok := val.(int64)
	if !ok {
		return 0, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) FPS() (float32, error) {
	val, ok := s.Get(FPS)
	if !ok {
		return 0, ErrNotFound
	}
	typedVal, ok := val.(float32)
	if !ok {
		return 0, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) DropCount() (int64, error) {
	val, ok := s.Get(DROPPED_FRAMES)
	if !ok {
		return 0, ErrNotFound
	}
	typedVal, ok := val.(int64)
	if !ok {
		return 0, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) Timestamp() (int64, error) {
	val, ok := s.Get(TIMESTAMP)
	if !ok {
		return 0, ErrNotFound
	}
	typedVal, ok := val.(int64)
	if !ok {
		return 0, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) Image(fqdn FQDNType) (ImageGetter, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return nil, ErrNotFound
	}
	typedVal, ok := val.(ImageGetter)
	if !ok {
		return nil, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) SetIndex(idx int64) {
	s.Set(INDEX, idx)
}
func (s *store) SetDropCount(cnt int64) {
	s.Set(DROPPED_FRAMES, cnt)
}
func (s *store) SetFPS(fps float32) {
	s.Set(FPS, fps)
}
func (s *store) SetTimestamp(ts int64) {
	s.Set(TIMESTAMP, ts)
}
func (s *store) SetImage(fqdn FQDNType, img ImageGetter) {
	s.Set(fqdn, img)
}

func (s *store) Byte(fqdn FQDNType) (byte, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return 0, ErrNotFound
	}
	typedVal, ok := val.(byte)
	if !ok {
		return 0, ErrWrongType
	}
	return typedVal, nil
}

func (s *store) Int(fqdn FQDNType) (int, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return 0, ErrNotFound
	}
	typedVal, ok := val.(int)
	if !ok {
		return 0, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) Int8(fqdn FQDNType) (int8, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return 0, ErrNotFound
	}
	typedVal, ok := val.(int8)
	if !ok {
		return 0, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) Int16(fqdn FQDNType) (int16, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return 0, ErrNotFound
	}
	typedVal, ok := val.(int16)
	if !ok {
		return 0, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) Int32(fqdn FQDNType) (int32, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return 0, ErrNotFound
	}
	typedVal, ok := val.(int32)
	if !ok {
		return 0, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) Int64(fqdn FQDNType) (int64, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return 0, ErrNotFound
	}
	typedVal, ok := val.(int64)
	if !ok {
		return 0, ErrWrongType
	}
	return typedVal, nil
}

func (s *store) UInt(fqdn FQDNType) (uint, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return 0, ErrNotFound
	}
	typedVal, ok := val.(uint)
	if !ok {
		return 0, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) UInt8(fqdn FQDNType) (uint8, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return 0, ErrNotFound
	}
	typedVal, ok := val.(uint8)
	if !ok {
		return 0, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) UInt16(fqdn FQDNType) (uint16, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return 0, ErrNotFound
	}
	typedVal, ok := val.(uint16)
	if !ok {
		return 0, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) UInt32(fqdn FQDNType) (uint32, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return 0, ErrNotFound
	}
	typedVal, ok := val.(uint32)
	if !ok {
		return 0, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) UInt64(fqdn FQDNType) (uint64, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return 0, ErrNotFound
	}
	typedVal, ok := val.(uint64)
	if !ok {
		return 0, ErrWrongType
	}
	return typedVal, nil
}

func (s *store) Float32(fqdn FQDNType) (float32, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return 0, ErrNotFound
	}
	typedVal, ok := val.(float32)
	if !ok {
		return 0, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) Float64(fqdn FQDNType) (float64, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return 0, ErrNotFound
	}
	typedVal, ok := val.(float64)
	if !ok {
		return 0, ErrWrongType
	}
	return typedVal, nil
}

func (s *store) Bytes(fqdn FQDNType) ([]byte, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return nil, ErrNotFound
	}
	typedVal, ok := val.([]byte)
	if !ok {
		return nil, ErrWrongType
	}
	return typedVal, nil
}

func (s *store) Ints(fqdn FQDNType) ([]int, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return nil, ErrNotFound
	}
	typedVal, ok := val.([]int)
	if !ok {
		return nil, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) Int8s(fqdn FQDNType) ([]int8, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return nil, ErrNotFound
	}
	typedVal, ok := val.([]int8)
	if !ok {
		return nil, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) Int16s(fqdn FQDNType) ([]int16, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return nil, ErrNotFound
	}
	typedVal, ok := val.([]int16)
	if !ok {
		return nil, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) Int32s(fqdn FQDNType) ([]int32, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return nil, ErrNotFound
	}
	typedVal, ok := val.([]int32)
	if !ok {
		return nil, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) Int64s(fqdn FQDNType) ([]int64, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return nil, ErrNotFound
	}
	typedVal, ok := val.([]int64)
	if !ok {
		return nil, ErrWrongType
	}
	return typedVal, nil
}

func (s *store) UInts(fqdn FQDNType) ([]uint, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return nil, ErrNotFound
	}
	typedVal, ok := val.([]uint)
	if !ok {
		return nil, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) UInt8s(fqdn FQDNType) ([]uint8, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return nil, ErrNotFound
	}
	typedVal, ok := val.([]uint8)
	if !ok {
		return nil, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) UInt16s(fqdn FQDNType) ([]uint16, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return nil, ErrNotFound
	}
	typedVal, ok := val.([]uint16)
	if !ok {
		return nil, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) UInt32s(fqdn FQDNType) ([]uint32, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return nil, ErrNotFound
	}
	typedVal, ok := val.([]uint32)
	if !ok {
		return nil, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) UInt64s(fqdn FQDNType) ([]uint64, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return nil, ErrNotFound
	}
	typedVal, ok := val.([]uint64)
	if !ok {
		return nil, ErrWrongType
	}
	return typedVal, nil
}

func (s *store) Float32s(fqdn FQDNType) ([]float32, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return nil, ErrNotFound
	}
	typedVal, ok := val.([]float32)
	if !ok {
		return nil, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) Float64s(fqdn FQDNType) ([]float64, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return nil, ErrNotFound
	}
	typedVal, ok := val.([]float64)
	if !ok {
		return nil, ErrWrongType
	}
	return typedVal, nil
}

func (s *store) String(fqdn FQDNType) (string, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return "", ErrNotFound
	}
	typedVal, ok := val.(string)
	if !ok {
		return "", ErrWrongType
	}
	return typedVal, nil
}

func (s *store) Strings(fqdn FQDNType) ([]string, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return nil, ErrNotFound
	}
	typedVal, ok := val.([]string)
	if !ok {
		return nil, ErrWrongType
	}
	return typedVal, nil
}
func (s *store) StringMap(fqdn FQDNType) (map[string]string, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return nil, ErrNotFound
	}
	typedVal, ok := val.(map[string]string)
	if !ok {
		return nil, ErrWrongType
	}
	return typedVal, nil
}

func (s *store) StoreList(fqdn FQDNType) ([]Store, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return nil, ErrNotFound
	}
	typedVal, ok := val.([]Store)
	if !ok {
		return nil, ErrWrongType
	}
	return typedVal, nil
}

func (s *store) StoreMap(fqdn FQDNType) (map[string]Store, error) {
	val, ok := s.Get(fqdn)
	if !ok {
		return nil, ErrNotFound
	}
	typedVal, ok := val.(map[string]Store)
	if !ok {
		return nil, ErrWrongType
	}
	return typedVal, nil
}
