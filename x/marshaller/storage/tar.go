//go:build !tinygo

package storage

import (
	"archive/tar"
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"math"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

type tarStorage struct {
	mu        sync.RWMutex
	size      int64
	readOnly  bool
	file      *os.File
	offset    int64
	regions   map[*fileRegion]struct{}
	data      []byte
	closed    bool
	entryName string
}

type TarFactory struct {
	archivePath string

	mu        sync.Mutex
	dirty     map[string]*writableEntry // keyed by normalized entry name
	pathIndex map[string]string         // logical path -> entry name
	tempDir   string
}

type writableEntry struct {
	entryName   string
	logicalPath string
	buffer      *bufferStorage
}

func NewTarMap(archivePath string) *TarFactory {
	return &TarFactory{
		archivePath: strings.TrimSpace(archivePath),
		dirty:       make(map[string]*writableEntry),
		pathIndex:   make(map[string]string),
		tempDir:     filepath.Dir(archivePath),
	}
}

func (f *TarFactory) Factory() types.MappedStorageFactory {
	return f.create
}

func (f *TarFactory) create(path string, readOnly bool) (types.MappedStorage, error) {
	if entry := f.getCachedEntry(path); entry != nil {
		return entry.openStorage(readOnly)
	}

	if path == "" {
		return nil, errors.New("tar storage: path cannot be empty")
	}

	if storage, err := openExistingFile(path, readOnly); err == nil {
		return storage, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("tar storage: failed to open %s: %w", path, err)
	}

	candidates := entryCandidates(path)
	if len(candidates) == 0 {
		return nil, fmt.Errorf("tar storage: unable to derive archive entry for %s", path)
	}

	if readOnly {
		return f.openArchiveReadOnly(candidates)
	}

	entry, err := f.prepareWritableEntry(path, candidates)
	if err != nil {
		return nil, err
	}

	return entry.openStorage(false)
}

func (f *TarFactory) openArchiveReadOnly(candidates []string) (types.MappedStorage, error) {
	if f.archivePath == "" {
		return nil, fmt.Errorf("tar storage: archive not configured")
	}

	file, err := os.Open(f.archivePath)
	if err != nil {
		return nil, fmt.Errorf("tar storage: failed to open archive %s: %w", f.archivePath, err)
	}

	isGzip, err := detectGzip(file)
	if err != nil {
		file.Close()
		return nil, err
	}

	if isGzip {
		data, matchedEntry, err := loadGzipEntry(file, candidates)
		file.Close()
		if err != nil {
			return nil, err
		}
		return &tarStorage{
			size:      int64(len(data)),
			readOnly:  true,
			data:      data,
			entryName: matchedEntry,
			regions:   make(map[*fileRegion]struct{}),
		}, nil
	}

	offset, size, matchedEntry, err := locateTarEntry(file, candidates)
	if err != nil {
		file.Close()
		return nil, err
	}

	return &tarStorage{
		size:      size,
		readOnly:  true,
		file:      file,
		offset:    offset,
		entryName: matchedEntry,
		regions:   make(map[*fileRegion]struct{}),
	}, nil
}

func (f *TarFactory) prepareWritableEntry(path string, candidates []string) (*writableEntry, error) {
	entryName := candidates[0]
	var initial []byte
	if data, matched, err := f.readArchiveEntry(candidates); err == nil {
		entryName = matched
		initial = data
	} else if err != nil && !errors.Is(err, fs.ErrNotExist) && !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}

	entry := &writableEntry{
		entryName:   entryName,
		logicalPath: path,
		buffer:      newBufferStorage(initial),
	}

	f.mu.Lock()
	defer f.mu.Unlock()
	f.dirty[entry.entryName] = entry
	f.pathIndex[path] = entry.entryName

	return entry, nil
}

func (f *TarFactory) getCachedEntry(path string) *writableEntry {
	f.mu.Lock()
	defer f.mu.Unlock()
	if entryName, ok := f.pathIndex[path]; ok {
		return f.dirty[entryName]
	}
	return nil
}

func (entry *writableEntry) openStorage(readOnly bool) (types.MappedStorage, error) {
	if readOnly {
		return entry.buffer, nil
	}
	return entry.buffer, nil
}

func (f *TarFactory) Commit() error {
	f.mu.Lock()
	if len(f.dirty) == 0 {
		f.mu.Unlock()
		return nil
	}
	if f.archivePath == "" {
		f.mu.Unlock()
		return fmt.Errorf("tar storage: archive path is required for commit")
	}
	dirtyEntries := make([]*writableEntry, 0, len(f.dirty))
	dirtyMap := make(map[string]*writableEntry, len(f.dirty))
	for name, entry := range f.dirty {
		dirtyEntries = append(dirtyEntries, entry)
		dirtyMap[name] = entry
	}
	f.mu.Unlock()

	tempFile, err := os.CreateTemp(f.tempDir, "tar-archive-*")
	if err != nil {
		return fmt.Errorf("tar storage: failed to create temp archive: %w", err)
	}
	tempPath := tempFile.Name()

	isGzip, err := f.currentArchiveIsGzip()
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		tempFile.Close()
		os.Remove(tempPath)
		return err
	}

	var writer io.WriteCloser = tempFile
	var gzipWriter *gzip.Writer
	if isGzip {
		gzipWriter = gzip.NewWriter(tempFile)
		writer = gzipWriter
	}
	tw := tar.NewWriter(writer)

	if err := f.copyExistingEntries(tw, dirtyMap); err != nil {
		tw.Close()
		if gzipWriter != nil {
			gzipWriter.Close()
		}
		tempFile.Close()
		os.Remove(tempPath)
		return err
	}

	for _, entry := range dirtyEntries {
		if err := entry.buffer.writeToTar(tw, entry.entryName); err != nil {
			tw.Close()
			if gzipWriter != nil {
				gzipWriter.Close()
			}
			tempFile.Close()
			os.Remove(tempPath)
			return err
		}
	}

	if err := tw.Close(); err != nil {
		if gzipWriter != nil {
			gzipWriter.Close()
		}
		tempFile.Close()
		os.Remove(tempPath)
		return err
	}
	if gzipWriter != nil {
		if err := gzipWriter.Close(); err != nil {
			tempFile.Close()
			os.Remove(tempPath)
			return err
		}
	}
	if err := tempFile.Close(); err != nil {
		os.Remove(tempPath)
		return err
	}

	if err := os.Rename(tempPath, f.archivePath); err != nil {
		os.Remove(tempPath)
		return fmt.Errorf("tar storage: failed to finalize archive: %w", err)
	}

	f.mu.Lock()
	for _, entry := range dirtyEntries {
		delete(f.pathIndex, entry.logicalPath)
		delete(f.dirty, entry.entryName)
	}
	f.mu.Unlock()

	return nil
}

func (f *TarFactory) currentArchiveIsGzip() (bool, error) {
	if f.archivePath == "" {
		return false, os.ErrNotExist
	}

	file, err := os.Open(f.archivePath)
	if err != nil {
		return false, err
	}
	defer file.Close()

	return detectGzip(file)
}

func (f *TarFactory) copyExistingEntries(tw *tar.Writer, dirty map[string]*writableEntry) error {
	if f.archivePath == "" {
		return nil
	}

	src, err := os.Open(f.archivePath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	defer src.Close()

	isGzip, err := detectGzip(src)
	if err != nil {
		return err
	}

	if _, err := src.Seek(0, io.SeekStart); err != nil {
		return err
	}

	var reader io.Reader = src
	var gzipReader *gzip.Reader
	if isGzip {
		gzipReader, err = gzip.NewReader(src)
		if err != nil {
			return err
		}
		defer gzipReader.Close()
		reader = gzipReader
	}

	tr := tar.NewReader(reader)
	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("tar storage: failed to read archive: %w", err)
		}

		normalized := normalizeTarEntry(header.Name)
		if _, replaced := dirty[normalized]; replaced {
			if _, err := io.Copy(io.Discard, tr); err != nil {
				return err
			}
			continue
		}

		if err := tw.WriteHeader(header); err != nil {
			return fmt.Errorf("tar storage: failed to write header: %w", err)
		}
		if _, err := io.Copy(tw, tr); err != nil {
			return fmt.Errorf("tar storage: failed to copy entry: %w", err)
		}
	}

	return nil
}

func (f *TarFactory) readArchiveEntry(entryNames []string) ([]byte, string, error) {
	if f.archivePath == "" {
		return nil, "", os.ErrNotExist
	}

	file, err := os.Open(f.archivePath)
	if err != nil {
		return nil, "", err
	}
	defer file.Close()

	isGzip, err := detectGzip(file)
	if err != nil {
		return nil, "", err
	}

	if isGzip {
		return loadGzipEntry(file, entryNames)
	}

	offset, size, matchedEntry, err := locateTarEntry(file, entryNames)
	if err != nil {
		return nil, "", err
	}

	data := make([]byte, size)
	if _, err := file.ReadAt(data, offset); err != nil {
		return nil, "", err
	}

	return data, matchedEntry, nil
}

func (s *tarStorage) Map(offset, length int64) (types.MappedRegion, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil, errors.New("tar storage: storage is closed")
	}

	if offset < 0 || offset > s.size {
		return nil, fmt.Errorf("tar storage: invalid offset %d (size: %d)", offset, s.size)
	}

	if length == 0 {
		length = s.size - offset
	}

	if length <= 0 {
		return nil, errors.New("tar storage: cannot map empty region")
	}

	if length < 0 {
		return nil, fmt.Errorf("tar storage: invalid length %d", length)
	}

	if offset+length > s.size {
		return nil, fmt.Errorf("tar storage: region extends beyond entry (offset: %d, length: %d, size: %d)", offset, length, s.size)
	}

	if s.data != nil {
		region := &memoryRegion{
			data: s.data[offset : offset+length],
		}
		return region, nil
	}

	if s.file == nil {
		return nil, errors.New("tar storage: no backing file")
	}

	region, err := newFileRegion(s.file, s.offset+offset, length, true)
	if err != nil {
		return nil, err
	}

	if s.regions == nil {
		s.regions = make(map[*fileRegion]struct{})
	}
	s.regions[region] = struct{}{}

	return region, nil
}

func (s *tarStorage) Size() (int64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return 0, errors.New("tar storage: storage is closed")
	}
	return s.size, nil
}

func (s *tarStorage) Grow(size int64) error {
	return errors.New("tar storage: grow is not supported for archive entries")
}

func (s *tarStorage) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}

	var closeErr error

	for region := range s.regions {
		if region == nil {
			continue
		}
		if err := region.Unmap(); err != nil && closeErr == nil {
			closeErr = err
		}
	}
	s.regions = nil

	if s.file != nil {
		if err := s.file.Close(); err != nil && closeErr == nil {
			closeErr = err
		}
		s.file = nil
	}

	s.data = nil
	s.closed = true
	return closeErr
}

func (s *tarStorage) ReaderWriterSeeker() (io.ReadWriteSeeker, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return nil, errors.New("tar storage: storage is closed")
	}
	return &tarView{storage: s}, nil
}

type tarView struct {
	storage *tarStorage
	pos     int64
	mu      sync.Mutex
}

func (v *tarView) Read(p []byte) (int, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	s := v.storage
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return 0, io.EOF
	}

	if v.pos >= s.size {
		return 0, io.EOF
	}

	remaining := s.size - v.pos
	toRead := int64(len(p))
	if toRead > remaining {
		toRead = remaining
	}

	if toRead == 0 {
		return 0, nil
	}

	if s.data != nil {
		copy(p, s.data[v.pos:v.pos+toRead])
	} else {
		if s.file == nil {
			return 0, fmt.Errorf("tar storage: backing file missing")
		}
		_, err := s.file.ReadAt(p[:toRead], s.offset+v.pos)
		if err != nil && !errors.Is(err, io.EOF) {
			return 0, err
		}
	}

	v.pos += toRead
	return int(toRead), nil
}

func (v *tarView) Write([]byte) (int, error) {
	return 0, errors.New("tar storage: archive entries are read-only")
}

func (v *tarView) Seek(offset int64, whence int) (int64, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	s := v.storage
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.closed {
		return 0, fmt.Errorf("tar storage: storage is closed")
	}

	var newPos int64
	switch whence {
	case io.SeekStart:
		newPos = offset
	case io.SeekCurrent:
		newPos = v.pos + offset
	case io.SeekEnd:
		newPos = s.size + offset
	default:
		return 0, fmt.Errorf("tar storage: invalid whence: %d", whence)
	}

	if newPos < 0 {
		return 0, fmt.Errorf("tar storage: negative position: %d", newPos)
	}

	if newPos > s.size {
		return 0, fmt.Errorf("tar storage: position beyond end: %d", newPos)
	}

	v.pos = newPos
	return newPos, nil
}

func normalizeTarEntry(name string) string {
	clean := path.Clean(strings.TrimLeft(strings.TrimSpace(name), "/"))
	clean = strings.TrimPrefix(clean, "./")
	if clean == "." {
		return ""
	}
	return clean
}

func detectGzip(file *os.File) (bool, error) {
	var magic [2]byte
	if _, err := file.ReadAt(magic[:], 0); err != nil {
		if errors.Is(err, io.EOF) {
			return false, nil
		}
		return false, fmt.Errorf("tar storage: failed to read archive header: %w", err)
	}
	return magic[0] == 0x1f && magic[1] == 0x8b, nil
}

func loadGzipEntry(file *os.File, entryNames []string) ([]byte, string, error) {
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return nil, "", fmt.Errorf("tar storage: failed to rewind archive: %w", err)
	}

	gzipReader, err := gzip.NewReader(file)
	if err != nil {
		return nil, "", fmt.Errorf("tar storage: failed to open gzip stream: %w", err)
	}
	defer gzipReader.Close()

	tarReader := tar.NewReader(gzipReader)
	candidates := makeCandidateSet(entryNames)

	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			return nil, "", fs.ErrNotExist
		}
		if err != nil {
			return nil, "", fmt.Errorf("tar storage: failed to read tar header: %w", err)
		}

		if !header.FileInfo().Mode().IsRegular() {
			continue
		}

		normalized := normalizeTarEntry(header.Name)
		if _, ok := candidates[normalized]; !ok {
			continue
		}

		if header.Size > int64(math.MaxInt) {
			return nil, "", fmt.Errorf("tar storage: entry %s too large (%d bytes) to materialize in memory", header.Name, header.Size)
		}

		data := make([]byte, header.Size)
		if _, err := io.ReadFull(tarReader, data); err != nil {
			return nil, "", fmt.Errorf("tar storage: failed to read entry %s: %w", header.Name, err)
		}
		return data, normalized, nil
	}
}

type countingReader struct {
	r io.Reader
	n int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	return n, err
}

func locateTarEntry(file *os.File, entryNames []string) (int64, int64, string, error) {
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return 0, 0, "", fmt.Errorf("tar storage: failed to rewind archive: %w", err)
	}

	counter := &countingReader{r: file}
	tarReader := tar.NewReader(counter)
	candidates := makeCandidateSet(entryNames)

	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			return 0, 0, "", fs.ErrNotExist
		}
		if err != nil {
			return 0, 0, "", fmt.Errorf("tar storage: failed to read tar header: %w", err)
		}

		if !header.FileInfo().Mode().IsRegular() {
			continue
		}

		dataOffset := counter.n
		normalized := normalizeTarEntry(header.Name)
		if _, ok := candidates[normalized]; ok {
			return dataOffset, header.Size, normalized, nil
		}

		// Skip entry payload to advance to the next header.
		if _, err := io.Copy(io.Discard, tarReader); err != nil {
			return 0, 0, "", fmt.Errorf("tar storage: failed to skip entry: %w", err)
		}
	}
}

func openExistingFile(path string, readOnly bool) (types.MappedStorage, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, err
	}

	var flags int
	if readOnly {
		flags = os.O_RDONLY
	} else {
		flags = os.O_RDWR
	}

	file, err := os.OpenFile(path, flags, 0644)
	if err != nil {
		return nil, err
	}

	storage, err := newFileStorage(file, readOnly)
	if err != nil {
		file.Close()
		return nil, err
	}
	return storage, nil
}

func entryCandidates(path string) []string {
	var candidates []string

	if normalized := normalizeTarEntry(filepath.ToSlash(path)); normalized != "" {
		candidates = append(candidates, normalized)
	}

	if base := normalizeTarEntry(filepath.Base(path)); base != "" {
		duplicate := len(candidates) > 0 && candidates[len(candidates)-1] == base
		if !duplicate {
			candidates = append(candidates, base)
		}
	}

	return candidates
}

func makeCandidateSet(entries []string) map[string]struct{} {
	set := make(map[string]struct{}, len(entries))
	for _, entry := range entries {
		if entry == "" {
			continue
		}
		set[entry] = struct{}{}
	}
	return set
}
