package adaptive

import (
	"fmt"
	"sort"

	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/chain"
)

// Factory creates a Processor instance for a given tier.
type Factory[Dst, Src chain.Sample] func() chain.Processor[Dst, Src]

// Tier defines a quality threshold and its corresponding factory.
type Tier[Dst, Src chain.Sample] struct {
	MinQuality float32
	Factory    Factory[Dst, Src]
}

// Processor routes Process/Reset to the best supported tier.
type Processor[Dst, Src chain.Sample] struct {
	tiers    []Tier[Dst, Src]
	current  int
	quality  float32
	delegate chain.Processor[Dst, Src]
}

// New builds an adaptive processor from tier configurations.
// Tier configs must be ordered high→low quality. The highest tier meeting
// initialQuality is selected, otherwise the lowest tier is used.
func New[Dst, Src chain.Sample](tiers []Tier[Dst, Src], initialQuality float32) (*Processor[Dst, Src], error) {
	if len(tiers) == 0 {
		return nil, fmt.Errorf("adaptive: at least one tier is required")
	}
	for i, t := range tiers {
		if t.Factory == nil {
			return nil, fmt.Errorf("adaptive: tier %d has nil factory", i)
		}
	}
	// Make a defensive copy to prevent external mutation.
	copied := append([]Tier[Dst, Src](nil), tiers...)
	// Ensure tiers are sorted descending by MinQuality.
	sort.SliceStable(copied, func(i, j int) bool {
		return copied[i].MinQuality > copied[j].MinQuality
	})

	ap := &Processor[Dst, Src]{
		tiers: copied,
	}
	ap.setQuality(initialQuality, true)
	return ap, nil
}

// Process delegates to the current tier implementation.
func (a *Processor[Dst, Src]) Process(dst buffer.Writer[Dst], src buffer.Reader[Src]) int {
	if a.delegate == nil {
		return 0
	}
	return a.delegate.Process(dst, src)
}

// Reset resets the current delegate (if any).
func (a *Processor[Dst, Src]) Reset() {
	if a.delegate != nil {
		a.delegate.Reset()
	}
}

// SetQuality updates the channel quality and triggers tier switch if needed.
func (a *Processor[Dst, Src]) SetQuality(q float32) {
	a.setQuality(q, false)
}

func (a *Processor[Dst, Src]) setQuality(q float32, force bool) {
	newIdx := a.selectTier(q)
	if !force && newIdx == a.current && a.delegate != nil {
		a.quality = q
		return
	}
	a.current = newIdx
	a.quality = q
	a.delegate = a.tiers[newIdx].Factory()
	if a.delegate == nil {
		// Should never happen; fall back to lowest tier.
		a.current = len(a.tiers) - 1
		a.delegate = a.tiers[a.current].Factory()
	}
}

func (a *Processor[Dst, Src]) selectTier(q float32) int {
	for idx, tier := range a.tiers {
		if q >= tier.MinQuality {
			return idx
		}
	}
	return len(a.tiers) - 1
}
