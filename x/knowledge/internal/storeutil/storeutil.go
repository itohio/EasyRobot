// Package storeutil holds what every knowledge.Store backend shares:
// update validation, lexical field extraction, query-term normalisation
// and the deterministic edge order of Neighbors.
package storeutil

import (
	"cmp"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
	"unicode"

	"github.com/itohio/EasyRobot/x/knowledge"
)

// Lexical fields of a chunk, in BM25 column order.
const (
	FieldHeading = iota
	FieldSymbol
	FieldPath
	FieldBody
	NumFields
)

// Weights are the BM25 field weights: heading 3, symbol 3, path 2, body 1.
var Weights = [NumFields]float64{3, 3, 2, 1}

// HeadingSep joins a chunk's heading breadcrumb into its heading field.
const HeadingSep = " › "

// ErrNotFound reports a write naming a document that is not indexed.
var ErrNotFound = errors.New("document not found")

// FieldTexts returns the lexical fields of c: heading breadcrumb, symbol,
// path (the DocID) and body.
func FieldTexts(c knowledge.Chunk) [NumFields]string {
	return [NumFields]string{
		FieldHeading: strings.Join(c.Heading, HeadingSep),
		FieldSymbol:  c.Symbol,
		FieldPath:    string(c.Doc),
		FieldBody:    c.Text,
	}
}

// CloneDoc returns d with its own Props map.
func CloneDoc(d knowledge.Doc) knowledge.Doc {
	d.Props = maps.Clone(d.Props)
	return d
}

// CloneChunk returns c with its own Heading slice.
func CloneChunk(c knowledge.Chunk) knowledge.Chunk {
	c.Heading = slices.Clone(c.Heading)
	return c
}

// Normalize validates u and returns a deep copy ready to store:
//   - Doc.ID must be non-empty;
//   - an empty Chunk.Doc or Edge.Source is set to Doc.ID, any other value
//     must equal it;
//   - a zero Chunk.ID is derived with knowledge.NewChunkID; chunk ids must
//     be unique within the update;
//   - edges identical in (From, To, Kind) merge into the first occurrence,
//     keeping the highest weight.
func Normalize(u knowledge.DocUpdate) (knowledge.DocUpdate, error) {
	id := u.Doc.ID
	if id == "" {
		return knowledge.DocUpdate{}, errors.New("empty document id")
	}
	out := knowledge.DocUpdate{Doc: CloneDoc(u.Doc)}
	if len(u.Chunks) > 0 {
		out.Chunks = make([]knowledge.Chunk, len(u.Chunks))
		seen := make(map[knowledge.ChunkID]struct{}, len(u.Chunks))
		for i, c := range u.Chunks {
			switch c.Doc {
			case "":
				c.Doc = id
			case id:
			default:
				return knowledge.DocUpdate{}, fmt.Errorf("chunk %d belongs to %q, not %q", c.Ord, c.Doc, id)
			}
			if c.ID == (knowledge.ChunkID{}) {
				c.ID = knowledge.NewChunkID(id, c.Ord, c.Text)
			}
			if _, dup := seen[c.ID]; dup {
				return knowledge.DocUpdate{}, fmt.Errorf("duplicate chunk id %s in %q", c.ID, id)
			}
			seen[c.ID] = struct{}{}
			out.Chunks[i] = CloneChunk(c)
		}
	}
	if len(u.Edges) > 0 {
		type key struct {
			from, to knowledge.NodeID
			kind     knowledge.EdgeKind
		}
		at := make(map[key]int, len(u.Edges))
		out.Edges = make([]knowledge.Edge, 0, len(u.Edges))
		for _, e := range u.Edges {
			switch e.Source {
			case "":
				e.Source = id
			case id:
			default:
				return knowledge.DocUpdate{}, fmt.Errorf("edge %s→%s asserted by %q, not %q", e.From, e.To, e.Source, id)
			}
			k := key{e.From, e.To, e.Kind}
			if i, dup := at[k]; dup {
				out.Edges[i].Weight = max(out.Edges[i].Weight, e.Weight)
				continue
			}
			at[k] = len(out.Edges)
			out.Edges = append(out.Edges, e)
		}
	}
	return out, nil
}

// RekeyChunk moves c to document to, re-deriving its id from (to, Ord,
// Text) so chunk ids stay content-derived after a rename.
func RekeyChunk(c knowledge.Chunk, to knowledge.DocID) knowledge.Chunk {
	c.Doc = to
	c.ID = knowledge.NewChunkID(to, c.Ord, c.Text)
	return c
}

// RekeyNode maps from's document node to to's, leaving other nodes as is.
func RekeyNode(n, from, to knowledge.NodeID) knowledge.NodeID {
	if n == from {
		return to
	}
	return n
}

// ValidTerm reports whether t can occur in the index: a non-empty run of
// letters, digits and the connectors _ - . (what package token emits).
func ValidTerm(t string) bool {
	if t == "" {
		return false
	}
	for _, r := range t {
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_' && r != '-' && r != '.' {
			return false
		}
	}
	return true
}

// QueryTerms lower-cases terms and drops duplicates and terms that can
// never match (see ValidTerm), keeping first-seen order.
func QueryTerms(terms []string) []string {
	out := make([]string, 0, len(terms))
	for _, t := range terms {
		t = strings.ToLower(t)
		if ValidTerm(t) && !slices.Contains(out, t) {
			out = append(out, t)
		}
	}
	return out
}

// Other returns the endpoint of e that is not seed (seed for a self-loop).
func Other(e knowledge.Edge, seed knowledge.NodeID) knowledge.NodeID {
	if e.From == seed {
		return e.To
	}
	return e.From
}

// CompareEdges orders edges touching seed for Neighbors: Weight desc,
// then the other endpoint, Kind, Source, From and To ascending.
func CompareEdges(seed knowledge.NodeID) func(a, b knowledge.Edge) int {
	return func(a, b knowledge.Edge) int {
		if c := cmp.Compare(b.Weight, a.Weight); c != 0 {
			return c
		}
		if c := cmp.Compare(Other(a, seed), Other(b, seed)); c != 0 {
			return c
		}
		if c := cmp.Compare(a.Kind, b.Kind); c != 0 {
			return c
		}
		if c := cmp.Compare(a.Source, b.Source); c != 0 {
			return c
		}
		if c := cmp.Compare(a.From, b.From); c != 0 {
			return c
		}
		return cmp.Compare(a.To, b.To)
	}
}

// KindIn reports whether k is in kinds; an empty list admits all kinds.
func KindIn(kinds []knowledge.EdgeKind, k knowledge.EdgeKind) bool {
	return len(kinds) == 0 || slices.Contains(kinds, k)
}

// UniqueNodes drops repeated seeds, keeping first-seen order.
func UniqueNodes(ns []knowledge.NodeID) []knowledge.NodeID {
	out := make([]knowledge.NodeID, 0, len(ns))
	for _, n := range ns {
		if !slices.Contains(out, n) {
			out = append(out, n)
		}
	}
	return out
}
