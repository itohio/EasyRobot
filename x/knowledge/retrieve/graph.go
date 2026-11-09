package retrieve

import (
	"cmp"
	"fmt"
	"math"
	"slices"

	"github.com/itohio/EasyRobot/x/knowledge"
)

// seed is a document graph expansion starts from.
type seed struct {
	id    knowledge.DocID
	score float64
}

// proposal is a neighbour reached from one seed, with its reason.
type proposal struct {
	doc    knowledge.DocID
	reason string
	weight float32 // weight of the edge that proposed it
}

// expand boosts graph neighbours of the top fused documents and of
// Query.Near: one hop over LinksTo in both directions, and two hops through
// id:/sym: nodes (seed Mentions X ↔ neighbour Defines X, and vice versa).
// A neighbour gains β · seedScore / log2(2 + Degree(neighbour)); at most
// Neighbors are kept per seed, chosen by edge weight (highest first; links
// before 2-hop on ties). Edge weights pick which neighbours are kept but do
// not scale the boost. pkg: nodes (import ↔ package) are not followed.
func (x *run) expand() error {
	if x.r.beta == 0 || x.r.neighbors <= 0 {
		return nil
	}
	seeds := x.seeds()
	if len(seeds) == 0 {
		return nil
	}
	isSeed := make(map[knowledge.DocID]bool, len(seeds))
	nodes := make([]knowledge.NodeID, len(seeds))
	for i, s := range seeds {
		isSeed[s.id] = true
		nodes[i] = knowledge.DocNode(s.id)
	}
	props := map[knowledge.DocID][]proposal{}
	add := func(from, to knowledge.DocID, reason string, weight float32) {
		if to == "" || to == from || x.near[to] {
			return
		}
		for _, p := range props[from] {
			if p.doc == to {
				return
			}
		}
		props[from] = append(props[from], proposal{doc: to, reason: reason, weight: weight})
	}

	links, err := x.r.store.Neighbors(x.ctx, nodes, []knowledge.EdgeKind{knowledge.LinksTo}, knowledge.Both, 4*x.r.neighbors)
	if err != nil {
		return fmt.Errorf("retrieve: link neighbours: %w", err)
	}
	for _, e := range links {
		from, to := docOf(e.From), docOf(e.To)
		if isSeed[from] {
			add(from, to, "linked from "+string(from), e.Weight)
		}
		if isSeed[to] {
			add(to, from, "links to "+string(to), e.Weight)
		}
	}

	out, err := x.r.store.Neighbors(x.ctx, nodes, []knowledge.EdgeKind{knowledge.Mentions, knowledge.Defines}, knowledge.Out, 0)
	if err != nil {
		return fmt.Errorf("retrieve: seed ids: %w", err)
	}
	mentioned := map[knowledge.NodeID][]knowledge.DocID{} // id/sym → seeds mentioning it
	defined := map[knowledge.NodeID][]knowledge.DocID{}   // id/sym → seeds defining it
	for _, e := range out {
		s := docOf(e.From)
		if !isSeed[s] || (e.To.Kind() != knowledge.NodeIdent && e.To.Kind() != knowledge.NodeSym) {
			continue
		}
		switch e.Kind {
		case knowledge.Mentions:
			mentioned[e.To] = appendUnique(mentioned[e.To], s)
		case knowledge.Defines:
			defined[e.To] = appendUnique(defined[e.To], s)
		}
	}
	hop := func(via map[knowledge.NodeID][]knowledge.DocID, kind knowledge.EdgeKind, why func(id string, seed knowledge.DocID) string) error {
		if len(via) == 0 {
			return nil
		}
		ns := make([]knowledge.NodeID, 0, len(via))
		for n := range via {
			ns = append(ns, n)
		}
		slices.Sort(ns)
		es, err := x.r.store.Neighbors(x.ctx, ns, []knowledge.EdgeKind{kind}, knowledge.In, 2*x.r.neighbors)
		if err != nil {
			return fmt.Errorf("retrieve: %s neighbours: %w", kind, err)
		}
		for _, e := range es {
			nb := docOf(e.From)
			if nb == "" {
				nb = e.Source
			}
			for _, s := range via[e.To] {
				add(s, nb, why(e.To.Value(), s), e.Weight)
			}
		}
		return nil
	}
	if err := hop(mentioned, knowledge.Defines, func(id string, s knowledge.DocID) string {
		return "defines " + id + " mentioned in " + string(s)
	}); err != nil {
		return err
	}
	if err := hop(defined, knowledge.Mentions, func(id string, s knowledge.DocID) string {
		return "mentions " + id + " defined in " + string(s)
	}); err != nil {
		return err
	}

	type boosted struct {
		proposal
		boost float64
	}
	for _, s := range seeds {
		// Keep the first Neighbors admitted proposals by edge weight (links
		// before 2-hop on ties, each in store order), so a hub seed costs
		// O(Neighbors) GetDoc/Degree calls, not one per edge.
		ps := props[s.id]
		slices.SortStableFunc(ps, func(a, b proposal) int { return cmp.Compare(b.weight, a.weight) })
		var bs []boosted
		for _, p := range ps {
			if len(bs) == x.r.neighbors {
				break
			}
			ok, err := x.admit(p.doc)
			if err != nil {
				return err
			}
			if !ok {
				continue
			}
			deg, err := x.degree(p.doc)
			if err != nil {
				return err
			}
			bs = append(bs, boosted{p, x.r.beta * s.score / math.Log2(2+float64(deg))})
		}
		for _, b := range bs {
			c := x.cand(b.doc)
			c.boost += b.boost
			if !slices.Contains(c.graphWhy, b.reason) {
				c.graphWhy = append(c.graphWhy, b.reason)
			}
		}
	}
	return nil
}

// seeds returns the top GraphSeeds fused documents plus Query.Near (scored
// as the top seed, or as a rank-1 hit when nothing matched lexically).
func (x *run) seeds() []seed {
	var out []seed
	for _, c := range x.sortedCands(func(c *cand) float64 { return c.fused }) {
		if len(out) >= x.r.graphSeeds || c.fused <= 0 {
			break
		}
		out = append(out, seed{c.id, c.fused})
	}
	top := 1.0 // a rank-1 lexical hit
	if len(out) > 0 {
		top = out[0].score
	}
	for _, id := range x.q.Near {
		if !slices.ContainsFunc(out, func(s seed) bool { return s.id == id }) {
			out = append(out, seed{id, top})
		}
	}
	return out
}

func (x *run) degree(id knowledge.DocID) (int, error) {
	if d, ok := x.degrees[id]; ok {
		return d, nil
	}
	d, err := x.r.store.Degree(x.ctx, knowledge.DocNode(id))
	if err != nil {
		return 0, fmt.Errorf("retrieve: degree of %s: %w", id, err)
	}
	x.degrees[id] = d
	return d, nil
}

// docOf returns the document of a doc: node, or "" for other nodes.
func docOf(n knowledge.NodeID) knowledge.DocID {
	if n.Kind() != knowledge.NodeDoc {
		return ""
	}
	return knowledge.DocID(n.Value())
}

func appendUnique[T comparable](s []T, v T) []T {
	if slices.Contains(s, v) {
		return s
	}
	return append(s, v)
}
