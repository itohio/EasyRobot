package main

import (
	"fmt"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Issue is one check finding.
type Issue struct {
	Class string
	Path  string
	Line  int
	Msg   string
}

func (i Issue) String() string {
	loc := i.Path
	if i.Line > 0 {
		loc = fmt.Sprintf("%s:%d", i.Path, i.Line)
	}
	return fmt.Sprintf("%s: [%s] %s", loc, i.Class, i.Msg)
}

// Report is the outcome of check.
type Report struct {
	Errors   []Issue
	Warnings []Issue
	Foreign  []string
	Stale    int
}

func (r *Report) err(class string, n *Note, line int, format string, a ...any) {
	r.Errors = append(r.Errors, Issue{class, n.Rel, line, fmt.Sprintf(format, a...)})
}

func (r *Report) warn(class string, n *Note, line int, format string, a ...any) {
	r.Warnings = append(r.Warnings, Issue{class, n.Rel, line, fmt.Sprintf(format, a...)})
}

// linkIssue reports a link problem: an error, or a warning in a foreign note
// (owned by another writer, so it cannot block easyrobot's gate).
func (r *Report) linkIssue(class string, n *Note, line int, format string, a ...any) {
	if n.Foreign() {
		r.warn(class, n, line, format, a...)
		return
	}
	r.err(class, n, line, format, a...)
}

var (
	wikiRe   = regexp.MustCompile(`!?\[\[([^\[\]\n]*)\]\]`)
	mdLinkRe = regexp.MustCompile(`\[[^\]\n]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)`)
	schemeRe = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9+.-]*:`)
)

var (
	vocabStatus = map[string]bool{"tested": true, "implemented": true, "partial": true, "stub": true, "unverified": true}
	algFields   = []string{"module", "test_result", "complete", "correct", "evidence"}
)

// Check validates the vault; st is the stale-note result (warnings only).
func Check(v *Vault, st StaleResult) *Report {
	r := &Report{}
	for _, n := range v.Notes {
		if n.template() {
			continue
		}
		checkLinks(v, r, n)
		if n.Foreign() {
			r.Foreign = append(r.Foreign, n.Rel)
			continue
		}
		checkFrontmatter(v, r, n)
	}
	checkTodos(v, r)
	checkGenerated(v, r)

	for _, s := range st.Notes {
		rev := s.Note.Str("source_revision")
		if rev == "" {
			rev = v.VaultRevision
		}
		r.warn("stale", s.Note, 0, "source changed since %s: %s", short(rev), strings.Join(s.Files, ", "))
	}
	r.Stale = len(st.Notes)
	for _, b := range st.Bad {
		r.err("revision", b.Note, 0, "revision %q: %s; stale check skipped for it", b.Rev, b.Err)
	}
	if st.Err != nil {
		r.Warnings = append(r.Warnings, Issue{"stale", "git", 0, "stale check unavailable: " + st.Err.Error()})
	}
	for _, name := range sortedIDs(v.byName) {
		ns := v.byName[name]
		if len(ns) < 2 || name == "index" {
			continue
		}
		var paths []string
		for _, n := range ns {
			paths = append(paths, n.Rel)
		}
		r.warn("dup-name", ns[0], 0, "note name %q used by %s", name, strings.Join(paths, ", "))
	}
	sortIssues(r.Errors)
	sortIssues(r.Warnings)
	return r
}

func short(rev string) string {
	if len(rev) > 12 {
		return rev[:12]
	}
	return rev
}

func sortIssues(is []Issue) {
	sort.SliceStable(is, func(i, j int) bool {
		if is[i].Path != is[j].Path {
			return is[i].Path < is[j].Path
		}
		return is[i].Line < is[j].Line
	})
}

func checkLinks(v *Vault, r *Report, n *Note) {
	dir := path.Dir(n.Rel)
	for i, line := range n.Masked {
		ln := n.FileLine(i)
		for _, m := range wikiRe.FindAllStringSubmatch(line, -1) {
			checkWikilink(v, r, n, ln, m[0], m[1])
		}
		for _, m := range mdLinkRe.FindAllStringSubmatch(line, -1) {
			target := m[1]
			if schemeRe.MatchString(target) || strings.HasPrefix(target, "#") || strings.HasPrefix(target, "<") {
				continue
			}
			if k := strings.IndexAny(target, "#?"); k >= 0 {
				target = target[:k]
			}
			if u, err := url.PathUnescape(target); err == nil {
				target = u
			}
			if !strings.HasSuffix(target, ".md") {
				continue
			}
			rel := path.Clean(path.Join(dir, target))
			if strings.HasPrefix(target, "/") {
				rel = path.Clean(strings.TrimPrefix(target, "/"))
			}
			if strings.HasPrefix(rel, "../") {
				if _, err := os.Stat(filepath.Join(v.Dir, filepath.FromSlash(rel))); err != nil {
					r.linkIssue("mdlink", n, ln, "relative link %s does not resolve", m[1])
				}
				continue
			}
			if !v.files[rel] {
				r.linkIssue("mdlink", n, ln, "relative link %s does not resolve", m[1])
				continue
			}
			if !n.Foreign() {
				r.warn("mdlink-style", n, ln, "relative Markdown link %s; use [[%s]]", m[1], strings.TrimSuffix(path.Base(rel), ".md"))
			}
		}
	}
}

func checkWikilink(v *Vault, r *Report, n *Note, ln int, whole, inner string) {
	target := inner
	if k := strings.Index(target, "|"); k >= 0 {
		target = target[:k]
	}
	target = strings.TrimSpace(strings.TrimSuffix(target, `\`))
	parts := strings.Split(target, "#")
	name := strings.TrimSpace(parts[0])
	var dest *Note
	switch {
	case name == "" && len(parts) > 1:
		dest = n
	case name == "":
		r.linkIssue("link", n, ln, "empty wikilink %s", whole)
		return
	default:
		dest = v.Resolve(name)
	}
	if dest == nil {
		if ext := path.Ext(name); ext != "" && ext != ".md" && v.fileExists(name) {
			return
		}
		r.linkIssue("link", n, ln, "unresolved wikilink %s", whole)
		return
	}
	if name != "" {
		if amb := v.Ambiguous(name); amb != nil {
			r.linkIssue("link", n, ln, "ambiguous wikilink %s: %s; use a path", whole, strings.Join(amb, ", "))
		}
	}
	for _, h := range parts[1:] {
		h = strings.TrimSpace(h)
		if h == "" {
			continue
		}
		if strings.HasPrefix(h, "^") {
			if !contains(dest.BlockIDs, strings.ToLower(h[1:])) {
				r.linkIssue("link", n, ln, "%s: no block %s in %s", whole, h, dest.Rel)
			}
			continue
		}
		if !contains(dest.Headings, normHeading(h)) {
			r.linkIssue("link", n, ln, "%s: no heading %q in %s", whole, h, dest.Rel)
		}
	}
}

func checkFrontmatter(v *Vault, r *Report, n *Note) {
	if n.Folder == "research" {
		return
	}
	if n.FMErr != nil {
		r.err("frontmatter", n, 1, "frontmatter does not parse: %v", n.FMErr)
		return
	}
	var missing []string
	for _, k := range []string{"type", "status", "tags"} {
		if !n.Has(k) {
			missing = append(missing, k)
		}
	}
	if len(missing) > 0 {
		r.err("frontmatter", n, 1, "missing %s", strings.Join(missing, ", "))
	}
	if n.Type() == "algorithm" {
		checkAlgorithm(v, r, n)
	}
}

func checkAlgorithm(v *Vault, r *Report, n *Note) {
	status := n.Str("status")
	if n.Has("status") && !vocabStatus[status] {
		r.err("algorithm", n, 1, "status %q outside tested|implemented|partial|stub|unverified", status)
	}
	var missing []string
	for _, k := range algFields {
		if !n.Has(k) {
			missing = append(missing, k)
		}
	}
	if len(missing) > 0 {
		r.err("algorithm", n, 1, "missing %s", strings.Join(missing, ", "))
	}
	if status == "tested" && n.Str("test_result") != "pass" {
		r.err("algorithm", n, 1, "status tested needs test_result: pass (have %q)", n.Str("test_result"))
	}
	if status == "implemented" && (n.Str("correct") == "no" || n.Str("complete") == "no") {
		r.err("algorithm", n, 1, "status implemented with complete: %s, correct: %s (means partial)", n.Str("complete"), n.Str("correct"))
	}
	if n.Has("module") {
		name, m := v.ModuleOf(n)
		switch {
		case m == nil:
			r.err("module", n, 1, "module note %q does not exist", name)
		case m.Type() != "module":
			r.err("module", n, 1, "module %q is %s, not a module note", name, m.Rel)
		}
		if amb := v.Ambiguous(linkTarget(n.Str("module"))); amb != nil {
			r.err("module", n, 1, "module link is ambiguous: %s; use a path", strings.Join(amb, ", "))
		}
	}
}

func checkTodos(v *Vault, r *Report) {
	todos := collectTodos(v)
	tks := tickets(v)
	for _, t := range todos {
		if t.Prio == 1 && len(t.Tickets) == 0 && !t.Backlog {
			r.err("p1-tag", t.Note, t.Line, "#p1 without #ER-0xx or #backlog: %s", t.Text)
		}
		for _, id := range t.Tickets {
			if tks[id] == nil {
				r.err("ticket", t.Note, t.Line, "TODO names %s, which has no ticket note", id)
			}
		}
	}
	_, groups, _ := dedupe(v, todos)
	for _, g := range groups {
		notes := map[*Note]bool{}
		var locs []string
		for _, t := range g {
			notes[t.Note] = true
			locs = append(locs, fmt.Sprintf("%s:%d", t.Note.Rel, t.Line))
		}
		if len(notes) < 2 {
			continue
		}
		what := g[0].Anchor
		if what == "" {
			what = fmt.Sprintf("%q", g[0].Text)
		}
		r.err("duplicate", g[0].Note, g[0].Line, "TODO %s repeated across notes: %s", what, strings.Join(locs, ", "))
	}
}

// checkGenerated reports files gen skips (missing or unclosed markers,
// hand-edited blocks) and generated files that differ from what gen writes.
func checkGenerated(v *Vault, r *Report) {
	g := Generate(v, false)
	r.Errors = append(r.Errors, g.Skipped...)
	for _, p := range sortedIDs(g.Files) {
		cur := ""
		if n := v.byPath[p]; n != nil {
			cur = n.Raw
		}
		if cur != g.Files[p] {
			r.Errors = append(r.Errors, Issue{"generated", p, 0, "out of date with the notes; run make docs"})
		}
	}
}
