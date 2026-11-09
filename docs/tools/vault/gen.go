package main

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// groupOrder fixes the order of module groups in the index; other groups
// follow alphabetically, modules without a group come last under "Other".
var groupOrder = []string{
	"Control and planning",
	"Signal, codecs, audio",
	"Tensor and NN",
	"Math and maps",
	"Devices and apps",
	"IO, camera, vision",
	"Knowledge engine",
}

const otherGroup = "Other"

// Ticket is an ER-xxx ticket note.
type Ticket struct {
	ID, Title, Status, Workstream string
	Note                          *Note
}

var ticketIDRe = regexp.MustCompile(`^ER-\d{3}`)

func tickets(v *Vault) map[string]*Ticket {
	out := map[string]*Ticket{}
	for _, n := range v.Notes {
		if n.Folder != "tickets" {
			continue
		}
		id := n.Str("id")
		if id == "" {
			id = ticketIDRe.FindString(n.Name)
		}
		if !ticketIDRe.MatchString(id) {
			continue
		}
		out[id] = &Ticket{ID: id, Title: n.Str("title"), Status: n.Str("status"), Workstream: n.Str("workstream"), Note: n}
	}
	return out
}

func sortedIDs[T any](m map[string]T) []string {
	ids := make([]string, 0, len(m))
	for k := range m {
		ids = append(ids, k)
	}
	sort.Strings(ids)
	return ids
}

// counts per status; index len(statusOrder) is the total.
type counts [6]int

func (c *counts) add(status string) {
	for i, s := range statusOrder {
		if s == status {
			c[i]++
		}
	}
	c[5]++
}

func (c counts) line() string {
	parts := make([]string, len(statusOrder))
	for i, s := range statusOrder {
		parts[i] = fmt.Sprintf("%d %s", c[i], s)
	}
	return fmt.Sprintf("%s (%d algorithms)", strings.Join(parts, " · "), c[5])
}

func statusRank(s string) int {
	for i, x := range statusOrder {
		if x == s {
			return i
		}
	}
	return len(statusOrder)
}

func cell(s string) string {
	s = strings.ReplaceAll(strings.TrimSpace(s), "\n", " ")
	return strings.ReplaceAll(s, "|", `\|`)
}

func algorithms(v *Vault) []*Note {
	var out []*Note
	for _, n := range v.Notes {
		if n.Type() == "algorithm" && !n.template() && !n.Foreign() {
			out = append(out, n)
		}
	}
	return out
}

func modules(v *Vault) []*Note {
	var out []*Note
	for _, n := range v.Notes {
		if n.Type() == "module" && !n.template() && !n.Foreign() {
			out = append(out, n)
		}
	}
	return out
}

func moduleBlock(v *Vault, mod *Note) string {
	var algs []*Note
	var c counts
	for _, a := range algorithms(v) {
		if _, m := v.ModuleOf(a); m == mod {
			algs = append(algs, a)
			c.add(a.Str("status"))
		}
	}
	sort.SliceStable(algs, func(i, j int) bool {
		ri, rj := statusRank(algs[i].Str("status")), statusRank(algs[j].Str("status"))
		if ri != rj {
			return ri < rj
		}
		return algs[i].Name < algs[j].Name
	})
	if len(algs) == 0 {
		return "No algorithm notes belong to this module."
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Counts: %s\n\n| Algorithm | Status | Tests | Summary |\n|---|---|---|---|\n", c.line())
	for _, a := range algs {
		fmt.Fprintf(&b, "| [[%s]] | %s | %s | %s |\n", a.Name, cell(a.Str("status")), cell(a.Str("test_result")), cell(a.Str("summary")))
	}
	return strings.TrimSuffix(b.String(), "\n")
}

func statusBlock(v *Vault) string {
	var c counts
	for _, a := range algorithms(v) {
		c.add(a.Str("status"))
	}
	return "State: " + c.line()
}

func modulesBlock(v *Vault) string {
	per := map[*Note]*counts{}
	var all counts
	for _, a := range algorithms(v) {
		all.add(a.Str("status"))
		if _, m := v.ModuleOf(a); m != nil {
			if per[m] == nil {
				per[m] = &counts{}
			}
			per[m].add(a.Str("status"))
		}
	}
	byGroup := map[string][]*Note{}
	for _, m := range modules(v) {
		g := m.Str("group")
		if g == "" {
			g = otherGroup
		}
		byGroup[g] = append(byGroup[g], m)
	}
	var groups []string
	for _, g := range groupOrder {
		if byGroup[g] != nil {
			groups = append(groups, g)
		}
	}
	var extra []string
	for g := range byGroup {
		if g != otherGroup && !contains(groupOrder, g) {
			extra = append(extra, g)
		}
	}
	sort.Strings(extra)
	groups = append(groups, extra...)
	if byGroup[otherGroup] != nil {
		groups = append(groups, otherGroup)
	}
	row := func(label string, c counts) string {
		return fmt.Sprintf("| %s | %d | %d | %d | %d | %d | %d |\n", label, c[0], c[1], c[2], c[3], c[4], c[5])
	}
	var b strings.Builder
	b.WriteString("| Module | tested | implemented | partial | stub | unverified | total |\n|---|---|---|---|---|---|---|\n")
	for _, g := range groups {
		fmt.Fprintf(&b, "| **%s** | | | | | | |\n", cell(g))
		ms := byGroup[g]
		sort.Slice(ms, func(i, j int) bool { return ms[i].Name < ms[j].Name })
		for _, m := range ms {
			var c counts
			if per[m] != nil {
				c = *per[m]
			}
			b.WriteString(row("[["+m.Name+"]]", c))
		}
	}
	b.WriteString(row("**All**", all))
	return strings.TrimSuffix(b.String(), "\n")
}

func contains(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}

// openP1 counts kept p1 TODOs per ticket id.
func openP1(kept []*Todo) map[string]int {
	out := map[string]int{}
	for _, t := range kept {
		if t.Prio == 1 {
			for _, id := range t.Tickets {
				out[id]++
			}
		}
	}
	return out
}

func ticketsBlock(tks map[string]*Ticket, p1 map[string]int) string {
	var b strings.Builder
	b.WriteString("| Ticket | Title | Status | Workstream | Open p1 |\n|---|---|---|---|---|\n")
	n := 0
	for _, id := range sortedIDs(tks) {
		t := tks[id]
		if t.Status == "done" {
			continue
		}
		n++
		fmt.Fprintf(&b, "| [[%s\\|%s]] | %s | %s | %s | %d |\n", t.Note.Name, id, cell(t.Title), cell(t.Status), cell(t.Workstream), p1[id])
	}
	if n == 0 {
		return "No open tickets."
	}
	return strings.TrimSuffix(b.String(), "\n")
}

func todoLine(t *Todo) string { return fmt.Sprintf("- [ ] %s ([[%s]])\n", t.Text, t.Note.Name) }

// todoGroup is the module label a TODO is listed under.
func todoGroup(t *Todo) string {
	if t.Module != "" {
		return t.Module
	}
	return t.Note.Name
}

func todoFile(tks map[string]*Ticket, kept []*Todo, dropped int) string {
	var tot [4]int
	for _, t := range kept {
		tot[t.Prio]++
	}
	var b strings.Builder
	b.WriteString("---\ntype: meta\nstatus: generated\ntags: [easyrobot, todo]\n---\n# TODO roll-up\n\n")
	b.WriteString("Generated by `make docs` (`docs/tools/vault`) from the open TODO lines of the vault notes. " +
		"Do not edit this file: edit the TODO lines in the source notes (algorithm notes first) and regenerate. " +
		"Priorities and ticket tags per [[conventions#TODO lines]]; when a fix lands follow [[conventions#After a fix]].\n\n")
	fmt.Fprintf(&b, "**Totals:** %d open after dedupe — p1 %d, p2 %d, p3 %d. %d duplicate lines dropped.\n\n",
		len(kept), tot[1], tot[2], tot[3], dropped)

	byTicket := map[string][]*Todo{}
	byModule := map[string][]*Todo{}
	var untagged []*Todo
	p23 := map[string]*[4]int{}
	for _, t := range kept {
		if t.Prio != 1 {
			g := todoGroup(t)
			if p23[g] == nil {
				p23[g] = &[4]int{}
			}
			p23[g][t.Prio]++
			continue
		}
		for _, id := range t.Tickets {
			byTicket[id] = append(byTicket[id], t)
		}
		switch {
		case len(t.Tickets) > 0:
		case t.Backlog:
			byModule[todoGroup(t)] = append(byModule[todoGroup(t)], t)
		default:
			untagged = append(untagged, t)
		}
	}

	b.WriteString("## p1 by ticket\n\n")
	if len(byTicket) == 0 {
		b.WriteString("None.\n\n")
	}
	for _, id := range sortedIDs(byTicket) {
		if t := tks[id]; t != nil {
			fmt.Fprintf(&b, "### [[%s|%s — %s]]\n\n", t.Note.Name, id, t.Title)
		} else {
			fmt.Fprintf(&b, "### %s (no ticket note)\n\n", id)
		}
		for _, t := range byTicket[id] {
			b.WriteString(todoLine(t))
		}
		b.WriteString("\n")
	}

	b.WriteString("## p1 backlog by module\n\n")
	if len(byModule) == 0 {
		b.WriteString("None.\n\n")
	}
	for _, m := range sortedIDs(byModule) {
		fmt.Fprintf(&b, "### [[%s]]\n\n", m)
		for _, t := range byModule[m] {
			b.WriteString(todoLine(t))
		}
		b.WriteString("\n")
	}

	b.WriteString("## Untagged p1\n\nEvery `#p1` names a ticket or `#backlog`; `make docs-check` fails while this list is not empty.\n\n")
	if len(untagged) == 0 {
		b.WriteString("None.\n\n")
	}
	for _, t := range untagged {
		b.WriteString(todoLine(t))
	}
	if len(untagged) > 0 {
		b.WriteString("\n")
	}

	b.WriteString("## p2 and p3 by module\n\nListed in each note's TODO lines; counts only.\n\n")
	if len(p23) == 0 {
		b.WriteString("None.\n")
		return b.String()
	}
	b.WriteString("| Module | p2 | p3 |\n|---|---|---|\n")
	for _, m := range sortedIDs(p23) {
		fmt.Fprintf(&b, "| [[%s]] | %d | %d |\n", m, p23[m][2], p23[m][3])
	}
	return b.String()
}

// block is a generated block in a note: offsets into Raw.
type block struct {
	openStart, contentStart, contentEnd int
	hash                                string // h= recorded in the opening marker ("" if none)
}

// findBlock locates <!-- vault:NAME [h=HASH] --> … <!-- /vault:NAME --> outside
// code. ok is false when either marker is missing.
func findBlock(n *Note, name string) (block, bool) {
	body := n.MaskedBody()
	m := regexp.MustCompile(`<!-- vault:` + regexp.QuoteMeta(name) + `(?: h=([0-9a-f]+))? -->`).FindStringSubmatchIndex(body)
	if m == nil {
		return block{}, false
	}
	j := strings.Index(body[m[1]:], "<!-- /vault:"+name+" -->")
	if j < 0 {
		return block{}, false
	}
	b := block{openStart: n.BodyStart + m[0], contentStart: n.BodyStart + m[1], contentEnd: n.BodyStart + m[1] + j}
	if m[2] >= 0 {
		b.hash = body[m[2]:m[3]]
	}
	return b, true
}

func contentHash(s string) string {
	sum := sha256.Sum256([]byte(strings.TrimSpace(s)))
	return hex.EncodeToString(sum[:6])
}

// handEdited reports a block whose content no longer matches the hash gen
// recorded in its opening marker.
func (b block) handEdited(n *Note) bool {
	return b.hash != "" && contentHash(n.Raw[b.contentStart:b.contentEnd]) != b.hash
}

// GenResult is what gen would write: Files holds the full new content of each
// generated file (module notes and index.md with complete markers, todo.md);
// Skipped lists the files left alone (no markers, hand-edited block).
type GenResult struct {
	Files   map[string]string
	Skipped []Issue
}

// fill rewrites the named blocks of n; it skips n (recording why) when a block
// is missing or, unless force, was edited by hand.
func (g *GenResult) fill(n *Note, force bool, blocks [][2]string) {
	cur := n
	for _, bl := range blocks {
		b, ok := findBlock(cur, bl[0])
		if !ok {
			g.Skipped = append(g.Skipped, Issue{"markers", n.Rel, 0,
				fmt.Sprintf("no markers (<!-- vault:%s --> … <!-- /vault:%s --> missing or unclosed)", bl[0], bl[0])})
			return
		}
		if !force && b.handEdited(cur) {
			g.Skipped = append(g.Skipped, Issue{"hand-edited", n.Rel, strings.Count(cur.Raw[:b.openStart], "\n") + 1,
				fmt.Sprintf("block %s was edited by hand; move the text out of the markers (gen -force overwrites it)", bl[0])})
			return
		}
		marker := fmt.Sprintf("<!-- vault:%s h=%s -->", bl[0], contentHash(bl[1]))
		cur = parseNote(n.Rel, cur.Raw[:b.openStart]+marker+"\n\n"+bl[1]+"\n\n"+cur.Raw[b.contentEnd:])
	}
	g.Files[n.Rel] = cur.Raw
}

// Generate computes the generated files. It depends only on the notes, never
// on git state. force overwrites hand-edited blocks.
func Generate(v *Vault, force bool) GenResult {
	g := GenResult{Files: map[string]string{}}
	for _, m := range modules(v) {
		g.fill(m, force, [][2]string{{"algorithms", moduleBlock(v, m)}})
	}
	tks := tickets(v)
	kept, _, dropped := dedupe(v, collectTodos(v))
	if idx := v.byPath["index.md"]; idx != nil {
		g.fill(idx, force, [][2]string{
			{"status", statusBlock(v)},
			{"modules", modulesBlock(v)},
			{"tickets", ticketsBlock(tks, openP1(kept))},
		})
	}
	g.Files["todo.md"] = todoFile(tks, kept, dropped)
	return g
}
