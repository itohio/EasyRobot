package helpers

import (
	"errors"
	"iter"
	"runtime"
	"sync"
	"sync/atomic"
)

var (
	// ErrPoolClosed is returned when submitting work to a closed worker pool.
	ErrPoolClosed = errors.New("generics: worker pool closed")
	// ErrWorkerCallbackNil is returned when the provided callback is nil.
	ErrWorkerCallbackNil = errors.New("generics: worker callback cannot be nil")
	// ErrPoolAlreadyInitialized is returned when Init is called twice without Close.
	ErrPoolAlreadyInitialized = errors.New("generics: worker pool already initialized")
	// ErrPoolNotInitialized is returned when operations are attempted before Init.
	ErrPoolNotInitialized = errors.New("generics: worker pool not initialized")
)

// WorkerCallback defines the shape of the function invoked for each processed chunk.
// Implementations must be concurrency-safe and return an error to signal failure.
type WorkerCallback func(start, end int) error

// ChunkSizer controls how many elements belong to a chunk for a given workload.
type ChunkSizer func(total, workers int) int

// WorkerPoolOption configures WorkerPool construction.
type WorkerPoolOption func(*poolConfig)

type poolConfig struct {
	workers int
	sizer   ChunkSizer
}

// WorkerPool coordinates chunked parallel execution with bounded backpressure.
type WorkerPool struct {
	// Optional configuration fields set before Init.
	Size       int
	ChunkSizer ChunkSizer

	workers     int
	tasks       chan *poolJob
	stopCh      chan struct{}
	chunkSizer  ChunkSizer
	workerGroup sync.WaitGroup
	jobPool     sync.Pool
	closed      atomic.Bool
	initialized atomic.Bool
}

type poolJob struct {
	start int
	end   int
	state *executionState
}

func (j *poolJob) reset() {
	j.start = 0
	j.end = 0
	j.state = nil
}

type executionState struct {
	cb      WorkerCallback
	wg      sync.WaitGroup
	failure atomic.Uint32
	err     error
}

func newExecutionState(cb WorkerCallback) *executionState {
	return &executionState{cb: cb}
}

func (s *executionState) add(delta int) {
	s.wg.Add(delta)
}

func (s *executionState) done() {
	s.wg.Done()
}

func (s *executionState) wait() {
	s.wg.Wait()
}

func (s *executionState) setErr(err error) {
	if err == nil {
		return
	}
	if s.failure.CompareAndSwap(0, 1) {
		s.err = err
	}
}

func (s *executionState) shouldSkip() bool {
	return s.failure.Load() == 1
}

func (s *executionState) Err() error {
	return s.err
}

// Init prepares the worker pool for use. Init must be called before Execute or Iterator.
// The receiver must not be copied after initialization.
func (p *WorkerPool) Init(opts ...WorkerPoolOption) error {
	if p == nil {
		return errors.New("generics: nil worker pool")
	}
	if p.initialized.Load() {
		return ErrPoolAlreadyInitialized
	}

	cfg := poolConfig{
		workers: p.Size,
		sizer:   p.ChunkSizer,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	normalisePoolConfig(&cfg)

	p.workers = cfg.workers
	p.chunkSizer = cfg.sizer
	p.tasks = make(chan *poolJob, cfg.workers)
	p.stopCh = make(chan struct{})
	p.jobPool = sync.Pool{
		New: func() any { return &poolJob{} },
	}
	p.closed.Store(false)
	p.workerGroup = sync.WaitGroup{}

	p.workerGroup.Add(p.workers)
	for i := 0; i < p.workers; i++ {
		go p.worker()
	}

	p.initialized.Store(true)
	return nil
}

// Execute splits the range [0,total) into chunks and processes them in parallel.
// The submission blocks when all workers are busy, guaranteeing backpressure.
func (p *WorkerPool) Execute(total int, fn WorkerCallback) error {
	if fn == nil {
		return ErrWorkerCallbackNil
	}
	if total <= 0 {
		return nil
	}
	if p.closed.Load() {
		return ErrPoolClosed
	}
	if !p.initialized.Load() {
		return ErrPoolNotInitialized
	}

	state := newExecutionState(fn)
	chunkSize := p.chunkSize(total)

	for start := 0; start < total; start += chunkSize {
		end := start + chunkSize
		if end > total {
			end = total
		}
		if err := p.dispatch(state, start, end); err != nil {
			state.wait()
			return err
		}
	}

	state.wait()
	if err := state.Err(); err != nil {
		return err
	}
	if p.closed.Load() {
		return ErrPoolClosed
	}
	return nil
}

// Iterator returns a Go 1.22+ sequence compatible with range-over-func syntax.
func (p *WorkerPool) Iterator(total int) iter.Seq2[int, int] {
	if total <= 0 {
		return func(func(int, int) bool) {}
	}

	if !p.initialized.Load() {
		return func(func(int, int) bool) {}
	}

	chunkSize := p.chunkSize(total)
	if chunkSize <= 0 {
		chunkSize = 1
	}

	return func(yield func(int, int) bool) {
		for start := 0; start < total; start += chunkSize {
			end := start + chunkSize
			if end > total {
				end = total
			}
			if !yield(start, end) {
				return
			}
		}
	}
}

// Close gracefully shuts down the worker pool, waiting for all workers to exit.
// Outstanding tasks complete before shutdown.
func (p *WorkerPool) Close() {
	if p == nil {
		return
	}
	if !p.initialized.Load() {
		return
	}
	if !p.closed.CompareAndSwap(false, true) {
		return
	}
	close(p.stopCh)
	p.workerGroup.Wait()
	p.initialized.Store(false)
}

func (p *WorkerPool) worker() {
	defer p.workerGroup.Done()
	for {
		select {
		case <-p.stopCh:
			return
		case job := <-p.tasks:
			if job == nil {
				continue
			}
			state := job.state
			if state == nil {
				job.reset()
				p.jobPool.Put(job)
				continue
			}
			if state.shouldSkip() {
				state.done()
				job.reset()
				p.jobPool.Put(job)
				continue
			}
			err := state.cb(job.start, job.end)
			if err != nil {
				state.setErr(err)
			}
			state.done()
			job.reset()
			p.jobPool.Put(job)
		}
	}
}

func (p *WorkerPool) dispatch(state *executionState, start, end int) error {
	state.add(1)
	job := p.jobPool.Get().(*poolJob)
	job.start = start
	job.end = end
	job.state = state

	if err := p.submit(job); err != nil {
		state.done()
		job.reset()
		p.jobPool.Put(job)
		return err
	}
	return nil
}

func (p *WorkerPool) submit(job *poolJob) error {
	select {
	case <-p.stopCh:
		return ErrPoolClosed
	case p.tasks <- job:
		return nil
	}
}

func (p *WorkerPool) chunkSize(total int) int {
	if total <= 0 {
		return 0
	}
	size := p.chunkSizer(total, p.workers)
	if size <= 0 {
		return 1
	}
	return size
}

func defaultPoolConfig() poolConfig {
	return poolConfig{
		workers: runtime.NumCPU(),
		sizer:   defaultChunkSizer,
	}
}

func normalisePoolConfig(cfg *poolConfig) {
	if cfg.workers <= 0 {
		cfg.workers = runtime.GOMAXPROCS(0)
		if cfg.workers <= 0 {
			cfg.workers = 1
		}
	}
	if cfg.sizer == nil {
		cfg.sizer = defaultChunkSizer
	}
}

func defaultChunkSizer(total, workers int) int {
	if total <= 0 {
		return 0
	}
	if workers <= 0 {
		workers = 1
	}
	size := (total + workers - 1) / workers
	if size <= 0 {
		return 1
	}
	return size
}

// WithWorkers overrides the worker count used by the pool.
func WithWorkers(workers int) WorkerPoolOption {
	return func(cfg *poolConfig) {
		if workers > 0 {
			cfg.workers = workers
		}
	}
}

// WithChunkSizer provides a custom chunk sizing strategy.
func WithChunkSizer(sizer ChunkSizer) WorkerPoolOption {
	return func(cfg *poolConfig) {
		if sizer != nil {
			cfg.sizer = sizer
		}
	}
}

// WithTargetChunkSize caps chunk size to the provided maximum while ensuring minimum size of 1.
func WithTargetChunkSize(size int) WorkerPoolOption {
	return func(cfg *poolConfig) {
		if size <= 0 {
			return
		}
		cfg.sizer = func(total, _ int) int {
			if total <= 0 {
				return 0
			}
			if total < size {
				return total
			}
			return size
		}
	}
}
