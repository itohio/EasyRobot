package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func runArgs(t *testing.T, args ...string) (int, string, string) {
	t.Helper()
	var out, errb bytes.Buffer
	code := run(args, &out, &errb)
	return code, out.String(), errb.String()
}

// gitVault writes baseVault (plus extra) under repo/docs and the given repo
// files, commits everything and returns repo, docs and the commit id.
func gitVault(t *testing.T, extra map[string]string, repoFiles map[string]string) (string, string, string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
	repo := writeVault(t, repoFiles)
	docs := filepath.Join(repo, "docs")
	files := baseVault()
	for p, c := range extra {
		files[p] = c
	}
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
	return repo, docs, git(t, repo, "rev-parse", "HEAD")
}

func writeFile(t *testing.T, p, content string) {
	t.Helper()
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func withSource(algNote, fmLines string) string {
	return strings.Replace(algNote, "---\n#", fmLines+"---\n#", 1)
}

// D1: generated files do not depend on working-tree git state.
func TestGenIgnoresGitState(t *testing.T) {
	b := baseVault()
	repo, docs, rev := gitVault(t, map[string]string{
		"algorithms/fk.md": withSource(b["algorithms/fk.md"], "source: [x/a/a.go]\n"),
	}, map[string]string{"x/a/a.go": "package a\n"})
	writeFile(t, filepath.Join(docs, "conventions.md"), note("meta", "vault_revision: "+rev+"\n", "# C\n\n## TODO lines\n\n## After a fix\n"))
	if code, out, errs := runArgs(t, "-docs", docs, "-repo", repo, "gen"); code != 0 {
		t.Fatalf("gen: %d %s %s", code, out, errs)
	}
	snap := map[string]string{}
	for _, p := range []string{"index.md", "todo.md", "modules/kin.md"} {
		snap[p] = read(t, docs, p)
	}
	writeFile(t, filepath.Join(repo, "x/a/a.go"), "package a\n\n// dirty\n")
	if code, out, errs := runArgs(t, "-docs", docs, "-repo", repo, "gen"); code != 0 {
		t.Fatalf("gen: %d %s %s", code, out, errs)
	}
	for p, s := range snap {
		if got := read(t, docs, p); got != s {
			t.Errorf("%s changed after a cited file got dirty:\n%s\n---\n%s", p, s, got)
		}
	}
	code, out, errs := runArgs(t, "-docs", docs, "-repo", repo, "check")
	if code != 0 {
		t.Errorf("check exit %d with a dirty cited file:\n%s%s", code, out, errs)
	}
	mustContain(t, out, "[stale]", "x/a/a.go")
	code, out, errs = runArgs(t, "-docs", docs, "-repo", repo, "stale")
	if code != 0 {
		t.Errorf("stale exit %d: %s%s", code, out, errs)
	}
	mustContain(t, out, "algorithms/fk.md", "x/a/a.go")
}

// D2: one bad revision is an error on its note; other notes are still checked.
func TestBadRevisionIsolated(t *testing.T) {
	b := baseVault()
	repo, docs, rev := gitVault(t, map[string]string{
		"algorithms/fk.md":  withSource(b["algorithms/fk.md"], "source: [x/a/a.go]\nsource_revision: deadbeefdeadbeef\n"),
		"algorithms/osc.md": withSource(b["algorithms/osc.md"], "source: [x/c/c.go]\n"),
	}, map[string]string{"x/a/a.go": "package a\n", "x/c/c.go": "package c\n"})
	writeFile(t, filepath.Join(docs, "conventions.md"), note("meta", "vault_revision: "+rev+"\n", "# C\n\n## TODO lines\n\n## After a fix\n"))
	writeFile(t, filepath.Join(repo, "x/c/c.go"), "package c\n\n// dirty\n")
	v := load(t, docs)
	st := findStale(v, repo)
	if st.Err != nil {
		t.Fatalf("whole stale check failed: %v", st.Err)
	}
	if len(st.Notes) != 1 || st.Notes[0].Note.Name != "osc" {
		t.Errorf("stale = %+v, want osc only", st.Notes)
	}
	rep := Check(v, st)
	if !hasIssue(rep.Errors, "revision", "algorithms/fk.md") {
		t.Errorf("no revision error on fk: %v", rep.Errors)
	}
	if !hasIssue(rep.Warnings, "stale", "algorithms/osc.md") {
		t.Errorf("osc not reported stale: %v", rep.Warnings)
	}
}

// D3: text typed inside a generated block is not silently overwritten.
func TestHandEditedBlock(t *testing.T) {
	dir := genVault(t, baseVault())
	kin := read(t, dir, "modules/kin.md")
	if !strings.Contains(kin, "<!-- vault:algorithms h=") {
		t.Fatalf("opening marker carries no hash:\n%s", kin)
	}
	edited := strings.Replace(kin, "| [[fk]] |", "My hand note.\n\n| [[fk]] |", 1)
	writeFile(t, filepath.Join(dir, "modules/kin.md"), edited)
	rep := Check(load(t, dir), StaleResult{})
	if !hasIssue(rep.Errors, "hand-edited", "modules/kin.md") {
		t.Errorf("no hand-edited error: %v", rep.Errors)
	}
	code, _, errs := runArgs(t, "-docs", dir, "-repo", dir, "gen")
	if code != 1 || !strings.Contains(errs, "modules/kin.md") || !strings.Contains(errs, "-force") {
		t.Errorf("gen on hand-edited block: exit %d, stderr %q", code, errs)
	}
	if got := read(t, dir, "modules/kin.md"); got != edited {
		t.Error("gen overwrote a hand-edited block without -force")
	}
	if code, _, errs := runArgs(t, "-docs", dir, "-repo", dir, "-force", "gen"); code != 0 {
		t.Fatalf("gen -force: %d %s", code, errs)
	}
	if got := read(t, dir, "modules/kin.md"); got != kin {
		t.Errorf("gen -force did not restore the block:\n%s", got)
	}
}

// D4: a file changed on disk after load is skipped; writes go through rename.
func TestWriteSkipsFileChangedSinceLoad(t *testing.T) {
	dir := writeVault(t, baseVault())
	v := load(t, dir)
	res := Generate(v, false)
	changed := read(t, dir, "modules/kin.md") + "\nEdited meanwhile.\n"
	writeFile(t, filepath.Join(dir, "modules/kin.md"), changed)
	var out, errb bytes.Buffer
	written, ok := writeGenerated(v, res, &out, &errb)
	if ok {
		t.Error("writeGenerated reported success although a file changed since load")
	}
	if got := read(t, dir, "modules/kin.md"); got != changed {
		t.Error("file changed since load was overwritten")
	}
	mustContain(t, errb.String(), "modules/kin.md", "changed since")
	if written == 0 || !strings.Contains(read(t, dir, "modules/sig.md"), "[[osc]]") {
		t.Error("other files not written")
	}
	entries, _ := os.ReadDir(filepath.Join(dir, "modules"))
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".md") {
			t.Errorf("temp file left behind: %s", e.Name())
		}
	}
}

// D5: link problems in foreign notes are warnings.
func TestForeignLinksAreWarnings(t *testing.T) {
	files := baseVault()
	files["planning/roadmap.md"] += "[[gone]]\n"
	dir := genVault(t, files)
	rep := Check(load(t, dir), StaleResult{})
	if hasIssue(rep.Errors, "link", "planning/roadmap.md") {
		t.Errorf("foreign link reported as error: %v", rep.Errors)
	}
	if !hasIssue(rep.Warnings, "link", "planning/roadmap.md") {
		t.Errorf("foreign link not warned: %v", rep.Warnings)
	}
	mustContain(t, strings.Join(rep.Foreign, " "), "planning/roadmap.md")
}

// D6, D8: missing or unclosed markers make gen skip the file loudly.
func TestGenSkipsNotesWithoutMarkers(t *testing.T) {
	files := baseVault()
	files["modules/bare.md"] = note("module", "", "# Bare\n")
	files["modules/open.md"] = note("module", "", "# Open\n\n<!-- vault:algorithms -->\nold text\n")
	dir := writeVault(t, files)
	code, _, errs := runArgs(t, "-docs", dir, "-repo", dir, "gen")
	if code != 1 {
		t.Errorf("gen exit %d, want 1", code)
	}
	mustContain(t, errs, "skipped modules/bare.md: no markers", "skipped modules/open.md: no markers")
	if read(t, dir, "modules/open.md") != files["modules/open.md"] || read(t, dir, "modules/bare.md") != files["modules/bare.md"] {
		t.Error("note without complete markers was modified")
	}
	if !strings.Contains(read(t, dir, "modules/kin.md"), "[[fk]]") {
		t.Error("other notes not generated")
	}
	rep := Check(load(t, dir), StaleResult{})
	if !hasIssue(rep.Errors, "markers", "modules/open.md") {
		t.Errorf("unclosed marker not reported: %v", rep.Errors)
	}
}

// D7: an ambiguous base-name link is an error, except [[index]].
func TestAmbiguousLink(t *testing.T) {
	files := baseVault()
	files["evidence/fk.md"] = note("evidence", "", "# Other fk\n")
	dir := genVault(t, files)
	rep := Check(load(t, dir), StaleResult{})
	found := false
	for _, e := range rep.Errors {
		if e.Class == "link" && e.Path == "workstreams/W1.md" && strings.Contains(e.Msg, "ambiguous") {
			found = true
		}
		if strings.Contains(e.Msg, "[[index]]") {
			t.Errorf("[[index]] reported: %v", e)
		}
	}
	if !found {
		t.Errorf("no ambiguous-link error: %v", rep.Errors)
	}
	v := load(t, dir)
	for range 20 {
		if n := v.Resolve("fk"); n == nil || n.Rel != "algorithms/fk.md" {
			t.Fatalf("Resolve(fk) = %v, want algorithms/fk.md (deterministic)", n)
		}
		if n := v.Resolve("FK"); n == nil || n.Rel != "algorithms/fk.md" {
			t.Fatalf("Resolve(FK) = %v, want algorithms/fk.md (deterministic)", n)
		}
	}
}

// D8: text outside the markers survives gen byte for byte.
func TestTextOutsideMarkersSurvives(t *testing.T) {
	files := baseVault()
	before := "# Kinematics\n\nIntro with `<!-- vault:algorithms -->` in code and trailing spaces   \n\n```\n<!-- /vault:algorithms -->\n```\n\n"
	after := "\n\n## Notes\n\nTail text, no final newline"
	files["modules/kin.md"] = note("module", "group: Control and planning\n", before+"<!-- vault:algorithms -->\nstale\n<!-- /vault:algorithms -->"+after)
	dir := genVault(t, files)
	got := read(t, dir, "modules/kin.md")
	head := note("module", "group: Control and planning\n", before)
	if !strings.HasPrefix(got, head) {
		t.Errorf("text before the block changed:\n%s", got)
	}
	if !strings.HasSuffix(got, "<!-- /vault:algorithms -->"+after) {
		t.Errorf("text after the block changed:\n%s", got)
	}
	if !strings.Contains(got, "[[fk]]") || strings.Contains(got, "\nstale\n") {
		t.Errorf("block not regenerated:\n%s", got)
	}
}
