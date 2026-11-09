package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func note(typ, extraFM, body string) string {
	return "---\ntype: " + typ + "\nstatus: active\ntags: [t]\n" + extraFM + "---\n" + body
}

func alg(module, status, testResult, complete, correct, summary, body string) string {
	return "---\ntype: algorithm\nstatus: " + status + "\ntags: [t]\nmodule: \"[[" + module + "]]\"\n" +
		"summary: \"" + summary + "\"\ntest_result: " + testResult + "\ncomplete: " + complete +
		"\ncorrect: " + correct + "\nevidence: \"run\"\n---\n" + body
}

func ticket(id, slug, title, status, ws string) (string, string) {
	return "tickets/" + id + "-" + slug + ".md", "---\nid: " + id + "\ntitle: " + title + "\ntype: ticket\nstatus: " + status +
		"\ntags: [t]\nowner: easyrobot\nworkstream: " + ws + "\n---\n# " + id + "\n"
}

const markersAlg = "<!-- vault:algorithms -->\n<!-- /vault:algorithms -->\n"

// baseVault is a small vault that passes check after gen.
func baseVault() map[string]string {
	files := map[string]string{
		"conventions.md": note("meta", "", "# Conventions\n\n## TODO lines\n\ngrammar\n\n## After a fix\n\ntext\n"),
		"index.md": note("moc", "", "# Home\n\nSee [[conventions#After a fix|fixing]] and [[tickets/index|tickets]].\n\n"+
			"<!-- vault:status -->\n<!-- /vault:status -->\n\n## Modules\n\n<!-- vault:modules -->\n<!-- /vault:modules -->\n\n"+
			"## Tickets\n\n<!-- vault:tickets -->\n<!-- /vault:tickets -->\n"),
		"tickets/index.md": note("ticket", "", "# Tickets\n\n[[index]]\n"),
		"modules/kin.md": note("module", "group: Control and planning\n", "# Kinematics\n\n"+markersAlg+
			"\n## TODOs\n\n- [ ] Module-level cleanup #p3\n"),
		"modules/sig.md": note("module", "group: Signal, codecs, audio\n", "# Signal\n\n"+markersAlg),
		"algorithms/fk.md": alg("kin", "tested", "pass", "yes", "yes", "FK exact | fast",
			"# FK\n\nSee [[ik#Completeness review]] and `m[[2,0,1]]` and ![[diagram.png]].\n\n"+
				"```go\n- [ ] fake inside fence #p1\n[[nonexistent-in-fence]]\n```\n\n"+
				"## TODOs\n\n- [ ] Sign error — `x/dh/dh.go:10-12` #p1 #ER-017\n- [ ] Overflow — util.go:5 #p1 #backlog\n- [ ] Add tests #p2\n"),
		"algorithms/ik.md": alg("kin", "partial", "fail", "no", "no", "",
			"# IK\n\n## Completeness review\n\ntext\n\n## TODOs\n\n- [ ] Diverges — ik.go:3 #p1 #ER-017\n- [ ] Stalls #p1 #ER-018\n- [ ] Docs #p3\n"),
		"algorithms/osc.md":      alg("sig", "implemented", "none", "yes", "unknown", "oscillator", "# Osc\n"),
		"workstreams/W1.md":      note("workstream", "", "# W1\n\nUses [[fk]].\n\n| Note |\n|---|\n| [[ik#Completeness review\\|IK]] |\n"),
		"templates/algorithm.md": "---\ntype: algorithm\nmodule: \"[[]]\"\n---\n# {{title}}\n[[nothing]]\n",
		"research/idea.md":       "# Idea without frontmatter\n",
		"planning/roadmap.md":    "---\nowner: product\n---\n# Roadmap\n\n- [ ] Ship #p1\n",
		"diagram.png":            "png",
	}
	for _, t := range [][4]string{
		{"ER-017", "fix-dh", "Fix DH", "ready"},
		{"ER-018", "ik-stall", "IK stall", "backlog"},
		{"ER-019", "old", "Old", "done"},
	} {
		p, c := ticket(t[0], t[1], t[2], t[3], "[W3, W6]")
		files[p] = c
	}
	// In tickets `owner` is the assignee: such tickets are still checked.
	p := "tickets/ER-018-ik-stall.md"
	files[p] = strings.Replace(files[p], "owner: easyrobot", "owner: unassigned-personal-profile", 1)
	return files
}

func writeVault(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for p, c := range files {
		full := filepath.Join(dir, filepath.FromSlash(p))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(c), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func load(t *testing.T, dir string) *Vault {
	t.Helper()
	v, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func runCmd(t *testing.T, dir, cmd string) (int, string) {
	t.Helper()
	var out, errb bytes.Buffer
	code := run([]string{"-docs", dir, "-repo", dir, cmd}, &out, &errb)
	return code, out.String() + errb.String()
}

func read(t *testing.T, dir, rel string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(rel)))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func genVault(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := writeVault(t, files)
	if code, out := runCmd(t, dir, "gen"); code != 0 {
		t.Fatalf("gen exit %d: %s", code, out)
	}
	return dir
}

func mustContain(t *testing.T, s string, subs ...string) {
	t.Helper()
	for _, sub := range subs {
		if !strings.Contains(s, sub) {
			t.Errorf("missing %q in:\n%s", sub, s)
		}
	}
}

func TestModuleTable(t *testing.T) {
	dir := genVault(t, baseVault())
	kin := read(t, dir, "modules/kin.md")
	mustContain(t, kin,
		"Counts: 1 tested · 0 implemented · 1 partial · 0 stub · 0 unverified (2 algorithms)",
		"| Algorithm | Status | Tests | Summary |\n|---|---|---|---|\n| [[fk]] | tested | pass | FK exact \\| fast |\n| [[ik]] | partial | fail |  |\n",
		"## TODOs\n\n- [ ] Module-level cleanup #p3\n")
}

func TestModuleWithoutAlgorithms(t *testing.T) {
	files := baseVault()
	files["modules/io.md"] = note("module", "group: Devices and apps\n", "# IO\n\n"+markersAlg)
	dir := genVault(t, files)
	io := read(t, dir, "modules/io.md")
	mustContain(t, io, "No algorithm notes belong to this module")
	if strings.Contains(io, "| Algorithm |") {
		t.Errorf("empty module got a table:\n%s", io)
	}
}

func TestIndexBlocks(t *testing.T) {
	dir := genVault(t, baseVault())
	idx := read(t, dir, "index.md")
	mustContain(t, idx,
		"State: 1 tested · 1 implemented · 1 partial · 0 stub · 0 unverified (3 algorithms)\n",
		"| **Control and planning** | | | | | | |\n| [[kin]] | 1 | 0 | 1 | 0 | 0 | 2 |\n| **Signal, codecs, audio** | | | | | | |\n| [[sig]] | 0 | 1 | 0 | 0 | 0 | 1 |\n| **All** | 1 | 1 | 1 | 0 | 0 | 3 |",
		"| Ticket | Title | Status | Workstream | Open p1 |\n|---|---|---|---|---|\n| [[ER-017-fix-dh\\|ER-017]] | Fix DH | ready | W3, W6 | 2 |\n| [[ER-018-ik-stall\\|ER-018]] | IK stall | backlog | W3, W6 | 1 |\n")
	if strings.Contains(idx, "ER-019") {
		t.Error("done ticket listed")
	}
}

func TestTodoRollup(t *testing.T) {
	dir := genVault(t, baseVault())
	todo := read(t, dir, "todo.md")
	mustContain(t, todo,
		"type: meta\nstatus: generated\n",
		"make docs",
		"**Totals:** 7 open after dedupe — p1 4, p2 1, p3 2. 0 duplicate lines dropped.",
		"0 duplicate lines dropped.\n\n## p1 by ticket",
		"### [[ER-017-fix-dh|ER-017 — Fix DH]]\n\n- [ ] Sign error — `x/dh/dh.go:10-12` #p1 #ER-017 ([[fk]])\n- [ ] Diverges — ik.go:3 #p1 #ER-017 ([[ik]])\n",
		"### [[ER-018-ik-stall|ER-018 — IK stall]]\n\n- [ ] Stalls #p1 #ER-018 ([[ik]])\n",
		"## p1 backlog by module\n\n### [[kin]]\n\n- [ ] Overflow — util.go:5 #p1 #backlog ([[fk]])\n",
		"## Untagged p1\n\n", "is not empty.\n\nNone.\n\n## p2 and p3",
		"| [[kin]] | 1 | 2 |")
	if strings.Contains(todo, "fake inside fence") || strings.Contains(todo, "Ship") {
		t.Error("fenced or foreign TODO collected")
	}
	if strings.Index(todo, "ER-017 — Fix DH") > strings.Index(todo, "ER-018 — IK stall") {
		t.Error("tickets out of id order")
	}
}

func TestIdempotentAndCheckClean(t *testing.T) {
	dir := genVault(t, baseVault())
	before := map[string]string{}
	for _, p := range []string{"index.md", "todo.md", "modules/kin.md", "modules/sig.md"} {
		before[p] = read(t, dir, p)
	}
	if code, out := runCmd(t, dir, "gen"); code != 0 {
		t.Fatal(out)
	}
	for p, b := range before {
		if a := read(t, dir, p); a != b {
			t.Errorf("%s changed on second gen:\n%s\n---\n%s", p, b, a)
		}
	}
	code, out := runCmd(t, dir, "check")
	if code != 0 {
		t.Fatalf("check on clean vault exit %d:\n%s", code, out)
	}
	mustContain(t, out, "0 errors", "foreign notes: planning/roadmap.md (link problems are warnings)")
}

func TestCheckStaleGenerated(t *testing.T) {
	dir := writeVault(t, baseVault())
	code, out := runCmd(t, dir, "check")
	if code != 1 {
		t.Fatalf("want exit 1, got %d", code)
	}
	mustContain(t, out, "[generated]")
}

func TestDedupe(t *testing.T) {
	files := baseVault()
	// Same anchor (basename+line, same module) in the module note and a workstream note.
	files["modules/kin.md"] = strings.Replace(files["modules/kin.md"], "## TODOs\n\n",
		"## TODOs\n\n- [ ] Sign wrong — dh/dh.go:10 #p1 #ER-017\n", 1)
	files["workstreams/W1.md"] += "\n## TODOs\n\n- [ ] Sign wrong again — `dh.go:10` #p1 #ER-017\n- [ ] Other — other.go:10 #p2\n"
	// Same basename+line as fk's util.go:5 but in another module: not a duplicate.
	files["algorithms/osc.md"] += "\n## TODOs\n\n- [ ] Unrelated — x/sig/util.go:5 #p2\n"
	v := load(t, writeVault(t, files))
	kept, groups, dropped := dedupe(v, collectTodos(v))
	if dropped != 2 {
		t.Errorf("dropped = %d, want 2", dropped)
	}
	if len(groups) != 1 || groups[0][0].Note.Name != "fk" {
		t.Fatalf("groups = %v", groups)
	}
	count := func(sub string) int {
		n := 0
		for _, td := range kept {
			if strings.Contains(td.Text, sub) {
				n++
			}
		}
		return n
	}
	if n := count("dh.go:10"); n != 1 {
		t.Errorf("kept %d dh.go:10 lines, want 1 (fk's)", n)
	}
	if n := count("util.go:5"); n != 2 {
		t.Errorf("kept %d util.go:5 lines, want 2", n)
	}
	gen := Generate(v, false).Files
	mustContain(t, gen["todo.md"], "2 duplicate lines dropped", "| [[W1]] | 1 | 0 |")
	mustContain(t, gen["index.md"], "| [[ER-017-fix-dh\\|ER-017]] | Fix DH | ready | W3, W6 | 2 |")
	rep := Check(v, StaleResult{})
	if !hasIssue(rep.Errors, "duplicate", "algorithms/fk.md") {
		t.Errorf("no duplicate error: %v", rep.Errors)
	}
}

func hasIssue(is []Issue, class, path string) bool {
	for _, i := range is {
		if i.Class == class && (path == "" || i.Path == path) {
			return true
		}
	}
	return false
}

func TestCheckErrors(t *testing.T) {
	cases := []struct {
		name, class, path string
		edit              func(f map[string]string)
	}{
		{"unresolved wikilink", "link", "workstreams/W1.md", func(f map[string]string) { f["workstreams/W1.md"] += "[[missing-note]]\n" }},
		{"bad heading", "link", "workstreams/W1.md", func(f map[string]string) { f["workstreams/W1.md"] += "[[ik#No such heading]]\n" }},
		{"missing embed", "link", "workstreams/W1.md", func(f map[string]string) { f["workstreams/W1.md"] += "![[nope.png]]\n" }},
		{"bad path link", "link", "workstreams/W1.md", func(f map[string]string) { f["workstreams/W1.md"] += "[[algorithms/nope|x]]\n" }},
		{"broken md link", "mdlink", "workstreams/W1.md", func(f map[string]string) { f["workstreams/W1.md"] += "[x](../algorithms/nope.md)\n" }},
		{"missing frontmatter", "frontmatter", "workstreams/W2.md", func(f map[string]string) { f["workstreams/W2.md"] = "# W2\n" }},
		{"missing tags", "frontmatter", "workstreams/W2.md", func(f map[string]string) {
			f["workstreams/W2.md"] = "---\ntype: workstream\nstatus: active\n---\n# W2\n"
		}},
		{"bad status", "algorithm", "algorithms/x.md", func(f map[string]string) {
			f["algorithms/x.md"] = alg("kin", "works", "pass", "yes", "yes", "", "# X\n")
		}},
		{"tested without pass", "algorithm", "algorithms/x.md", func(f map[string]string) {
			f["algorithms/x.md"] = alg("kin", "tested", "fail", "yes", "yes", "", "# X\n")
		}},
		{"implemented but wrong", "algorithm", "algorithms/x.md", func(f map[string]string) {
			f["algorithms/x.md"] = alg("kin", "implemented", "none", "yes", "no", "", "# X\n")
		}},
		{"implemented incomplete", "algorithm", "algorithms/x.md", func(f map[string]string) {
			f["algorithms/x.md"] = alg("kin", "implemented", "none", "no", "unknown", "", "# X\n")
		}},
		{"missing evidence", "algorithm", "algorithms/x.md", func(f map[string]string) {
			f["algorithms/x.md"] = "---\ntype: algorithm\nstatus: stub\ntags: [t]\nmodule: \"[[kin]]\"\ntest_result: none\ncomplete: no\ncorrect: unknown\n---\n# X\n"
		}},
		{"missing module note", "module", "algorithms/x.md", func(f map[string]string) {
			f["algorithms/x.md"] = alg("nomod", "stub", "none", "no", "unknown", "", "# X\n")
		}},
		{"module without markers", "markers", "modules/bare.md", func(f map[string]string) { f["modules/bare.md"] = note("module", "", "# Bare\n") }},
		{"untagged p1", "p1-tag", "algorithms/ik.md", func(f map[string]string) { f["algorithms/ik.md"] += "- [ ] Crash #p1\n" }},
		{"p1 outside TODO section", "p1-tag", "workstreams/W1.md", func(f map[string]string) { f["workstreams/W1.md"] += "\n## Notes\n\n- [ ] Crash in W1 #p1\n" }},
		{"unknown ticket", "ticket", "algorithms/ik.md", func(f map[string]string) { f["algorithms/ik.md"] += "- [ ] Crash #p1 #ER-099\n" }},
		{"duplicate anchor", "duplicate", "", func(f map[string]string) {
			f["algorithms/ik.md"] += "- [ ] Same defect — ik.go:3 #p2\n"
			f["modules/kin.md"] += "- [ ] Same defect — x/ik/ik.go:3 #p2\n"
		}},
		{"stale block", "generated", "modules/kin.md", func(f map[string]string) {
			f["algorithms/osc.md"] = alg("kin", "stub", "none", "no", "unknown", "", "# Osc\n")
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := genVault(t, baseVault())
			files := map[string]string{}
			for _, n := range load(t, dir).Notes {
				files[n.Rel] = n.Raw
			}
			before := map[string]string{}
			for p, c := range files {
				before[p] = c
			}
			tc.edit(files)
			for p, c := range files {
				if before[p] == c {
					continue
				}
				full := filepath.Join(dir, filepath.FromSlash(p))
				_ = os.MkdirAll(filepath.Dir(full), 0o755)
				if err := os.WriteFile(full, []byte(c), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			rep := Check(load(t, dir), StaleResult{})
			if !hasIssue(rep.Errors, tc.class, tc.path) {
				t.Errorf("want %s error at %q, got errors %v", tc.class, tc.path, rep.Errors)
			}
		})
	}
}

func TestCheckClassesSilentOnClean(t *testing.T) {
	dir := genVault(t, baseVault())
	rep := Check(load(t, dir), StaleResult{})
	if len(rep.Errors) != 0 {
		t.Fatalf("errors on clean vault: %v", rep.Errors)
	}
	if len(rep.Warnings) != 0 {
		t.Fatalf("warnings on clean vault: %v", rep.Warnings)
	}
}

func TestWarnings(t *testing.T) {
	files := baseVault()
	files["workstreams/W1.md"] += "[fk](../algorithms/fk.md)\n"
	files["evidence/fk.md"] = note("evidence", "", "# dup name\n")
	dir := genVault(t, files)
	rep := Check(load(t, dir), StaleResult{})
	if !hasIssue(rep.Warnings, "mdlink-style", "workstreams/W1.md") {
		t.Errorf("no style warning: %v", rep.Warnings)
	}
	if !hasIssue(rep.Warnings, "dup-name", "") {
		t.Errorf("no duplicate name warning: %v", rep.Warnings)
	}
	for _, w := range rep.Warnings {
		if w.Class == "dup-name" && strings.Contains(w.Msg, "index") {
			t.Errorf("index notes reported as duplicates: %v", w)
		}
	}
}

func TestMasking(t *testing.T) {
	n := parseNote("a.md", "# A\n\nx `[[a]]` y ``code ` [[b]]`` [[c]]\n```\n[[d]]\n```\n~~~~\n```\n[[e]]\n~~~~\n[[f]]\n")
	got := strings.Join(n.Masked, "\n")
	for _, bad := range []string{"[[a]]", "[[b]]", "[[d]]", "[[e]]"} {
		if strings.Contains(got, bad) {
			t.Errorf("%s not masked:\n%s", bad, got)
		}
	}
	for _, good := range []string{"[[c]]", "[[f]]"} {
		if !strings.Contains(got, good) {
			t.Errorf("%s masked:\n%s", good, got)
		}
	}
	if len(got) != len(n.Raw[n.BodyStart:]) {
		t.Error("masking changed length")
	}
}

func TestUnquotedModuleLink(t *testing.T) {
	n := parseNote("algorithms/a.md", "---\ntype: algorithm\nmodule: [[kin]]\n---\n")
	if got := n.Str("module"); got != "[[kin]]" {
		t.Errorf("module = %q", got)
	}
}

func git(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", dir, "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false"}, args...)...)
	cmd.Env = append(os.Environ(), "GIT_CONFIG_GLOBAL="+os.DevNull, "GIT_CONFIG_SYSTEM="+os.DevNull)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

func TestStaleDetection(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	files := baseVault()
	repoFiles := map[string]string{
		"x/a/a.go":     "package a\n",
		"x/c/c.go":     "package c\n",
		"x/pkg/one.go": "package pkg\n",
	}
	repo := writeVault(t, repoFiles)
	docs := filepath.Join(repo, "docs")
	for p, c := range files {
		full := filepath.Join(docs, filepath.FromSlash(p))
		_ = os.MkdirAll(filepath.Dir(full), 0o755)
		if err := os.WriteFile(full, []byte(c), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	git(t, repo, "init", "-q")
	git(t, repo, "add", "-A")
	git(t, repo, "commit", "-q", "-m", "init")
	rev := git(t, repo, "rev-parse", "HEAD")

	set := func(rel, content string) {
		if err := os.WriteFile(filepath.Join(docs, rel), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	set("conventions.md", note("meta", "vault_revision: "+rev+"\n", "# C\n\n## After a fix\n"))
	set("algorithms/fk.md", strings.Replace(files["algorithms/fk.md"], "---\n#", "source: [x/a/a.go]\n---\n#", 1))
	set("algorithms/ik.md", strings.Replace(files["algorithms/ik.md"], "---\n#", "source: [x/c/c.go]\nsource_revision: "+rev+"\n---\n#", 1))
	set("algorithms/osc.md", strings.Replace(files["algorithms/osc.md"], "---\n#", "source: [x/pkg]\n---\n#", 1))
	// fk: tracked file modified; osc: untracked file added to its source dir; ik unchanged.
	if err := os.WriteFile(filepath.Join(repo, "x/a/a.go"), []byte("package a\n\nvar X = 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "x/pkg/two.go"), []byte("package pkg\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	v := load(t, docs)
	st := findStale(v, repo)
	if st.Err != nil {
		t.Fatal(st.Err)
	}
	var names []string
	for _, s := range st.Notes {
		names = append(names, s.Note.Name+":"+strings.Join(s.Files, ","))
	}
	if got, want := strings.Join(names, " "), "fk:x/a/a.go osc:x/pkg/two.go"; got != want {
		t.Errorf("stale = %q, want %q", got, want)
	}
	for p, s := range Generate(v, false).Files {
		if strings.Contains(s, "x/a/a.go") || strings.Contains(s, "changed code") {
			t.Errorf("%s mentions stale state", p)
		}
	}
	rep := Check(v, st)
	if !hasIssue(rep.Warnings, "stale", "algorithms/fk.md") {
		t.Errorf("no stale warning: %v", rep.Warnings)
	}
}

func TestAnchorOf(t *testing.T) {
	for in, want := range map[string]string{
		"Use `mat.DLS` (pseudo_inverse.go:48) — joints/dh/denavithartenberg.go:216 #p1": "denavithartenberg.go:216",
		"Fix — `x/a/b.go:10-12`, c.go:3 #p1":                                            "b.go:10",
		"See x/mat/mat.go:5 for details #p2":                                            "mat.go:5",
		"No anchor — probe K #p2":                                                       "",
		"Port on example.com:8080 #p3":                                                  "",
	} {
		if got := anchorOf(in); got != want {
			t.Errorf("anchorOf(%q) = %q, want %q", in, got, want)
		}
	}
}
