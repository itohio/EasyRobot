package main

import (
	"path"
	"regexp"
	"sort"
	"strings"
)

// Todo is one open TODO line.
type Todo struct {
	Note    *Note
	Line    int // file line
	Prio    int
	Text    string // line text after "- [ ] ", tags kept
	Tickets []string
	Backlog bool
	Module  string // module of the note ("" when the note has none)
	Anchor  string // first file.ext:N anchor as basename:N, or ""
	key     string
}

var (
	todoRe    = regexp.MustCompile(`^\s*[-*+] \[ \] (.*)$`)
	prioRe    = regexp.MustCompile(`(?:^|\s)#p([123])\b`)
	ticketRe  = regexp.MustCompile(`(?:^|\s)#(ER-\d{3})\b`)
	backlogRe = regexp.MustCompile(`(?:^|\s)#backlog\b`)
	anchorRe  = regexp.MustCompile(`([A-Za-z0-9_./-]*[A-Za-z0-9_-]\.(?:go|s|c|h|cc|cpp|hpp|py|ino|proto|rs|sh|js|ts|mod)):(\d+)`)
	tagRe     = regexp.MustCompile(`#[\w-]+`)
	nonWordRe = regexp.MustCompile(`[^\p{L}\p{N}]+`)
)

// noteModule is the module a note's TODOs belong to.
func (v *Vault) noteModule(n *Note) string {
	switch n.Type() {
	case "module":
		return n.Name
	case "algorithm":
		m, _ := v.ModuleOf(n)
		return m
	}
	if n.Has("module") {
		m, _ := v.ModuleOf(n)
		return m
	}
	return ""
}

// collectsTodos reports notes whose TODO lines count.
func (n *Note) collectsTodos() bool {
	return n.Rel != "todo.md" && !n.template() && !n.Foreign()
}

// collectTodos returns every unchecked TODO line: any checkbox line under a
// heading starting with "TODO", plus any checkbox line carrying a #pN tag.
// Code fences and inline code are ignored.
func collectTodos(v *Vault) []*Todo {
	var out []*Todo
	for _, n := range v.Notes {
		if !n.collectsTodos() {
			continue
		}
		mod := v.noteModule(n)
		inTodo := false
		for i, line := range n.Lines {
			masked := n.Masked[i]
			if strings.TrimSpace(masked) == "" {
				continue
			}
			if h := headingRe.FindStringSubmatch(masked); h != nil {
				inTodo = strings.HasPrefix(strings.ToLower(strings.TrimSpace(h[1])), "todo")
				continue
			}
			if !todoRe.MatchString(masked) {
				continue
			}
			p := prioRe.FindStringSubmatch(masked)
			if !inTodo && p == nil {
				continue
			}
			text := strings.TrimSpace(todoRe.FindStringSubmatch(line)[1])
			td := &Todo{Note: n, Line: n.FileLine(i), Prio: 3, Text: text, Module: mod,
				Backlog: backlogRe.MatchString(masked)}
			if p != nil {
				td.Prio = int(p[1][0] - '0')
			}
			seen := map[string]bool{}
			for _, m := range ticketRe.FindAllStringSubmatch(masked, -1) {
				if !seen[m[1]] {
					seen[m[1]] = true
					td.Tickets = append(td.Tickets, m[1])
				}
			}
			td.Anchor = anchorOf(text)
			out = append(out, td)
		}
	}
	return out
}

func normText(s string) string {
	s = strings.NewReplacer("[[", " ", "]]", " ").Replace(s)
	s = tagRe.ReplaceAllString(s, " ")
	s = nonWordRe.ReplaceAllString(strings.ToLower(s), " ")
	return strings.TrimSpace(s)
}

// anchorOf returns the defect anchor as basename:line: the first file:line
// after the last " — " (the TODO grammar's evidence part), else the first in
// the text, else "".
func anchorOf(text string) string {
	if i := strings.LastIndex(text, "—"); i >= 0 {
		if a := anchorRe.FindStringSubmatch(text[i:]); a != nil {
			return path.Base(a[1]) + ":" + a[2]
		}
	}
	if a := anchorRe.FindStringSubmatch(text); a != nil {
		return path.Base(a[1]) + ":" + a[2]
	}
	return ""
}

// dedupe groups TODOs by key: the defect anchor (basename:line, see anchorOf) plus the
// note module, else the normalised text plus the module. A note without a
// module joins an anchor key when exactly one module uses that anchor. Within
// a group the algorithm note's line wins (then higher priority, then path).
// It returns the kept items (in input order), the groups with more than one
// item (kept item first) and the number of dropped lines.
func dedupe(v *Vault, todos []*Todo) ([]*Todo, [][]*Todo, int) {
	anchorMods := map[string]map[string]bool{}
	for _, t := range todos {
		if t.Anchor != "" && t.Module != "" {
			if anchorMods[t.Anchor] == nil {
				anchorMods[t.Anchor] = map[string]bool{}
			}
			anchorMods[t.Anchor][t.Module] = true
		}
	}
	groups := map[string][]*Todo{}
	var order []string
	for _, t := range todos {
		mod := t.Module
		if t.Anchor != "" {
			if mod == "" && len(anchorMods[t.Anchor]) == 1 {
				for m := range anchorMods[t.Anchor] {
					mod = m
				}
			}
			t.key = "a|" + mod + "|" + t.Anchor
		} else {
			t.key = "t|" + mod + "|" + normText(t.Text)
		}
		if _, ok := groups[t.key]; !ok {
			order = append(order, t.key)
		}
		groups[t.key] = append(groups[t.key], t)
	}
	keep := map[*Todo]bool{}
	var dups [][]*Todo
	dropped := 0
	for _, k := range order {
		g := groups[k]
		sort.SliceStable(g, func(i, j int) bool {
			ai, aj := g[i].Note.Type() == "algorithm", g[j].Note.Type() == "algorithm"
			if ai != aj {
				return ai
			}
			if g[i].Prio != g[j].Prio {
				return g[i].Prio < g[j].Prio
			}
			if g[i].Note.Rel != g[j].Note.Rel {
				return g[i].Note.Rel < g[j].Note.Rel
			}
			return g[i].Line < g[j].Line
		})
		keep[g[0]] = true
		if len(g) > 1 {
			dups = append(dups, g)
			dropped += len(g) - 1
		}
	}
	var kept []*Todo
	for _, t := range todos {
		if keep[t] {
			kept = append(kept, t)
		}
	}
	return kept, dups, dropped
}
