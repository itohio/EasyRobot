package pipeline

import (
	"context"
	"errors"

	"github.com/foxis/EasyRobot/internal/concurrency"
	. "github.com/foxis/EasyRobot/pkg/core/logger"
	"github.com/foxis/EasyRobot/pkg/core/store"
)

var (
	ErrNilOutput = errors.New("source step output is nil")
	ErrTooFew    = errors.New("too few steps")
)

type stepChain []int

type Pipeline struct {
	steps map[int]Step
	chain []stepChain
	maxId int
}

func New() Pipeline {
	return Pipeline{
		steps: make(map[int]Step),
		chain: make([]stepChain, 0),
		maxId: 123,
	}
}

func FromJSON(json string) Pipeline {
	return New()
}

func (p *Pipeline) AddStep(step Step) (int, error) {
	id := p.maxId
	p.steps[id] = step
	p.maxId++
	return id, nil
}

func (p *Pipeline) AddOrFindStep(step Step) (int, error) {
	id, err := p.FindStep(step)
	if err == nil {
		return id, nil
	}

	return p.AddStep(step)
}

func (p *Pipeline) GetStep(id int) (Step, bool) {
	step, ok := p.steps[id]
	return step, ok
}

func (p *Pipeline) FindStep(step Step) (int, error) {
	for id, s := range p.steps {
		if s == step {
			return id, nil
		}
	}
	return 0, store.ErrNotFound
}

func (p *Pipeline) ConnectStepsById(ids ...int) (<-chan Data, error) {
	if len(ids) <= 1 {
		return nil, ErrTooFew
	}
	for _, id := range ids {
		if _, ok := p.steps[id]; !ok {
			return nil, store.ErrNotFound
		}
	}

	ch := p.connectChain(ids)

	p.chain = append(p.chain, ids)

	return ch, nil
}

func (p *Pipeline) connectChain(ids stepChain) (ch <-chan Data) {
	Log.Debug().Ints("ids", ids).Msg("chain")
	for _, id := range ids {
		if step, ok := p.steps[id]; ok {
			if ch != nil {
				step.In(ch)
			}
			ch = step.Out()
		}
	}
	return
}

func (p *Pipeline) ConnectSteps(steps ...Step) (<-chan Data, error) {
	if len(steps) <= 1 {
		return nil, ErrTooFew
	}
	ids := make(stepChain, len(steps))
	for i, step := range steps {
		if step == nil {
			continue
		}
		id, err := p.AddOrFindStep(step)
		if err != nil {
			Log.Error().Err(err)
			return nil, err
		}
		ids[i] = id
	}

	ch := p.connectChain(ids)

	p.chain = append(p.chain, ids)

	return ch, nil
}

func (p *Pipeline) Run(ctx context.Context) {
	for _, step := range p.steps {
		func(s Step) {
			concurrency.Submit(func() { s.Run(ctx) })
		}(step)
	}
}

func (p *Pipeline) Reset() ([]<-chan Data, error) {
	for _, step := range p.steps {
		step.Reset()
	}

	chains := make([]<-chan Data, len(p.chain))
	for i, chain := range p.chain {
		ch := p.connectChain(chain)
		chains[i] = ch
	}

	return chains, nil
}
