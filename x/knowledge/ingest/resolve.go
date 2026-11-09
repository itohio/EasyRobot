package ingest

import (
	"path"
	"slices"
	"strings"

	"github.com/itohio/EasyRobot/x/knowledge"
)

// resolver maps raw links to documents using every accepted file of the
// current walk (not only changed ones). Former paths of moved documents
// (PropMovedFrom) act as aliases, consulted only when no real file
// matches, so links written against an old name keep following the move.
type resolver struct {
	files       map[string]*entry   // rel → entry
	byName      map[string][]*entry // lower-case base name, with and without extension
	aliasFiles  map[string]*entry   // former rel → current entry
	aliasByName map[string][]*entry
	accepts     func(rel string) bool
}

// newResolver indexes entries; movedFrom maps a DocID to its former
// root-relative paths.
func newResolver(entries []entry, movedFrom map[knowledge.DocID][]string, accepts func(string) bool) *resolver {
	r := &resolver{
		files:       make(map[string]*entry, len(entries)),
		byName:      make(map[string][]*entry, 2*len(entries)),
		aliasFiles:  make(map[string]*entry),
		aliasByName: make(map[string][]*entry),
		accepts:     accepts,
	}
	for k := range entries {
		e := &entries[k]
		r.files[e.rel] = e
		addName(r.byName, e.rel, e)
	}
	// Aliases are registered even where a real file now exists: lookups
	// always try real files first, and namesake re-resolution needs them.
	for k := range entries {
		e := &entries[k]
		for _, old := range movedFrom[e.id] {
			if r.aliasFiles[old] != nil {
				continue
			}
			r.aliasFiles[old] = e
			addName(r.aliasByName, old, e)
		}
	}
	return r
}

func addName(m map[string][]*entry, rel string, e *entry) {
	base := strings.ToLower(path.Base(rel))
	m[base] = append(m[base], e)
	if stem := strings.TrimSuffix(base, path.Ext(base)); stem != base && stem != "" {
		m[stem] = append(m[stem], e)
	}
}

// file looks rel up among real files, then aliases.
func (r *resolver) file(rel string) *entry {
	if e := r.files[rel]; e != nil {
		return e
	}
	return r.aliasFiles[rel]
}

// linkOutcome is the result of resolving one link.
type linkOutcome uint8

const (
	linkDrop    linkOutcome = iota // external, outside the root, a non-document file, or a self-link
	linkFound                      // resolved to a document
	linkFuzzy                      // Markdown link resolved only by the base-name fallback
	linkMissing                    // a document link that did not resolve
)

// Edge weights of resolved links.
const (
	weightExact = 1
	weightFuzzy = 0.5
)

// resolve maps l, found in from, to a target entry.
func (r *resolver) resolve(from *entry, l knowledge.Link) (*entry, linkOutcome) {
	t := strings.TrimSpace(l.Target)
	if t == "" || strings.Contains(t, "://") || strings.HasPrefix(strings.ToLower(t), "mailto:") {
		return nil, linkDrop
	}
	var (
		to    *entry
		fuzzy bool
	)
	switch l.Kind {
	case knowledge.WikiLink:
		to = r.wiki(from, t)
	case knowledge.MarkdownLink:
		var drop bool
		if to, fuzzy, drop = r.markdown(from, t); drop {
			return nil, linkDrop
		}
	default:
		return nil, linkDrop
	}
	switch {
	case to == from:
		return nil, linkDrop
	case to != nil && fuzzy:
		return to, linkFuzzy
	case to != nil:
		return to, linkFound
	case path.Ext(t) != "" && !r.accepts(t):
		return nil, linkDrop // e.g. an image
	}
	return nil, linkMissing
}

func (r *resolver) wiki(from *entry, t string) *entry {
	if !strings.Contains(t, "/") {
		key := strings.ToLower(t)
		if e := best(from, r.byName[key]); e != nil {
			return e
		}
		return best(from, r.aliasByName[key])
	}
	clean := path.Clean(strings.TrimPrefix(t, "/"))
	rel := path.Join(path.Dir(from.rel), t)
	cands := []string{clean, clean + ".md", rel, rel + ".md"}
	for _, c := range cands {
		if inside(c) && r.files[c] != nil {
			return r.files[c]
		}
	}
	// Partial path ("folder/Note"): real files by base name whose path ends
	// with the written path.
	suffix := "/" + strings.ToLower(clean)
	var partial []*entry
	for _, e := range r.byName[strings.ToLower(path.Base(clean))] {
		lc := "/" + strings.ToLower(e.rel)
		if strings.HasSuffix(lc, suffix) || strings.HasSuffix(lc, suffix+".md") {
			partial = append(partial, e)
		}
	}
	if e := best(from, partial); e != nil {
		return e
	}
	for _, c := range cands {
		if inside(c) && r.aliasFiles[c] != nil {
			return r.aliasFiles[c]
		}
	}
	return nil
}

// markdown resolves a relative (or root-relative "/x") Markdown link; drop
// reports links escaping the root. When the path does not exist and its
// base name has an extension a chunker accepts, the link falls back to
// base-name resolution (fuzzy), as vault tools do for moved files.
func (r *resolver) markdown(from *entry, t string) (to *entry, fuzzy, drop bool) {
	var p string
	if strings.HasPrefix(t, "/") {
		p = path.Clean(strings.TrimLeft(t, "/"))
	} else {
		p = path.Join(path.Dir(from.rel), t)
	}
	if !inside(p) {
		return nil, false, true
	}
	if p == "." {
		p = ""
	}
	cands := []string{p}
	if path.Ext(p) == "" {
		cands = append(cands, p+".md")
	}
	cands = append(cands, path.Join(p, "index.md"), path.Join(p, "README.md"))
	for _, c := range cands {
		if c != "" && r.files[c] != nil {
			return r.files[c], false, false
		}
	}
	for _, c := range cands {
		if c != "" && r.aliasFiles[c] != nil {
			return r.aliasFiles[c], false, false
		}
	}
	base := path.Base(p)
	if p == "" || path.Ext(base) == "" || !r.accepts(base) {
		return nil, false, false
	}
	// Never fall back to the linking document itself: the link stays
	// missing (and tracked) until another document with that name exists.
	key := strings.ToLower(base)
	if e := best(from, others(r.byName[key], from)); e != nil {
		return e, true, false
	}
	if e := best(from, others(r.aliasByName[key], from)); e != nil {
		return e, true, false
	}
	return nil, false, false
}

// others returns cands without from (cands itself when it is absent).
func others(cands []*entry, from *entry) []*entry {
	if !slices.Contains(cands, from) {
		return cands
	}
	return slices.DeleteFunc(slices.Clone(cands), func(e *entry) bool { return e == from })
}

// best picks among candidates: same scope as from, then fewest path
// segments, then lexical order; nil when there are none.
func best(from *entry, cands []*entry) *entry {
	if len(cands) == 0 {
		return nil
	}
	return slices.MinFunc(cands, func(a, b *entry) int {
		if c := boolCmp(a.scope == from.scope, b.scope == from.scope); c != 0 {
			return c
		}
		if d := strings.Count(a.rel, "/") - strings.Count(b.rel, "/"); d != 0 {
			return d
		}
		return strings.Compare(a.rel, b.rel)
	})
}

// boolCmp orders true before false.
func boolCmp(a, b bool) int {
	switch {
	case a == b:
		return 0
	case a:
		return -1
	}
	return 1
}

// inside reports whether a cleaned relative path stays within the root.
func inside(p string) bool {
	return p != ".." && !strings.HasPrefix(p, "../") && !path.IsAbs(p)
}

// edges returns the deduplicated edges of a document: the chunker's edges
// plus one LinksTo edge per resolved or missing link (weight 1, or 0.5 for
// base-name fallbacks; a duplicate keeps the higher weight), and the
// numbers of missing and fallback-resolved link targets.
func (r *resolver) edges(from *entry, cr knowledge.ChunkResult) (edges []knowledge.Edge, missing, fuzzy int) {
	type key struct {
		from, to knowledge.NodeID
		kind     knowledge.EdgeKind
	}
	seen := make(map[key]int, len(cr.Edges)+len(cr.Links))
	out := make([]knowledge.Edge, 0, len(cr.Edges)+len(cr.Links))
	add := func(e knowledge.Edge) {
		k := key{e.From, e.To, e.Kind}
		if i, ok := seen[k]; ok {
			out[i].Weight = max(out[i].Weight, e.Weight)
			return
		}
		seen[k] = len(out)
		out = append(out, e)
	}
	for _, e := range cr.Edges {
		add(e)
	}
	self := knowledge.DocNode(from.id)
	for _, l := range cr.Links {
		target, o := r.resolve(from, l)
		e := knowledge.Edge{From: self, Kind: knowledge.LinksTo, Weight: weightExact, Source: from.id}
		switch o {
		case linkFound:
			e.To = knowledge.DocNode(target.id)
		case linkFuzzy:
			e.To, e.Weight = knowledge.DocNode(target.id), weightFuzzy
		case linkMissing:
			e.To = knowledge.MissingNode(strings.TrimSpace(l.Target))
		default:
			continue
		}
		add(e)
	}
	for _, e := range out {
		if e.Kind != knowledge.LinksTo || e.From != self {
			continue
		}
		switch {
		case e.To.Kind() == knowledge.NodeMissing:
			missing++
		case e.Weight == weightFuzzy:
			fuzzy++
		}
	}
	return out, missing, fuzzy
}
