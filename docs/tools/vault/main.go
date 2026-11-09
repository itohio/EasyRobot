// Command vault maintains the EasyRobot Obsidian vault (docs/).
//
//	go run ./docs/tools/vault [-docs docs] [-repo .] [-force] gen|check|stale
//
// gen rewrites the generated blocks (<!-- vault:NAME h=HASH --> … <!-- /vault:NAME -->)
// in module notes and index.md and the whole of todo.md from the notes; it
// never reads git, skips notes without markers and blocks edited by hand
// (-force overwrites those) and files changed while it ran, and exits 1 when
// it skipped anything. check reports broken links, frontmatter and vocabulary
// errors, untagged p1 TODOs, repeated TODO anchors, bad revisions, hand-edited
// blocks and out-of-date generated files (exit 1), and warns about stale notes,
// relative Markdown links, duplicate note names and link problems in notes
// owned by other writers. stale lists notes whose source files changed since
// the revision they describe. Rules: docs/conventions.md.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("vault", flag.ContinueOnError)
	fs.SetOutput(stderr)
	docs := fs.String("docs", "docs", "vault directory")
	repo := fs.String("repo", ".", "git repository root the notes' source paths are relative to")
	force := fs.Bool("force", false, "gen: overwrite generated blocks that were edited by hand")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: vault [-docs docs] [-repo .] [-force] gen|check|stale")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}
	cmd := fs.Arg(0)
	if fs.NArg() != 1 || (cmd != "gen" && cmd != "check" && cmd != "stale") {
		fs.Usage()
		return 2
	}
	v, err := Load(*docs)
	if err != nil {
		fmt.Fprintln(stderr, "vault:", err)
		return 2
	}
	switch cmd {
	case "gen":
		return gen(v, *force, stdout, stderr)
	case "stale":
		return stale(v, findStale(v, *repo), stdout, stderr)
	}
	return check(v, findStale(v, *repo), stdout)
}

func gen(v *Vault, force bool, stdout, stderr io.Writer) int {
	res := Generate(v, force)
	for _, s := range res.Skipped {
		fmt.Fprintf(stderr, "skipped %s: %s\n", s.Path, s.Msg)
	}
	written, ok := writeGenerated(v, res, stdout, stderr)
	kept, _, dropped := dedupe(v, collectTodos(v))
	fmt.Fprintf(stdout, "gen: %d of %d generated files changed, %d skipped; %d open TODOs after dedupe (%d dropped)\n",
		written, len(res.Files), len(res.Skipped), len(kept), dropped)
	if !ok || len(res.Skipped) > 0 {
		return 1
	}
	return 0
}

// writeGenerated writes each changed file through a temp file and rename. A
// file that changed on disk since it was loaded is skipped (ok false).
func writeGenerated(v *Vault, res GenResult, stdout, stderr io.Writer) (int, bool) {
	written, ok := 0, true
	for _, p := range sortedIDs(res.Files) {
		n := v.byPath[p]
		if n != nil && n.Raw == res.Files[p] {
			continue
		}
		full := filepath.Join(v.Dir, filepath.FromSlash(p))
		cur, err := os.ReadFile(full)
		changed := (n == nil && err == nil) || (n != nil && (err != nil || string(cur) != n.Raw))
		if changed {
			fmt.Fprintf(stderr, "skipped %s: changed since gen loaded it; rerun make docs\n", p)
			ok = false
			continue
		}
		if err := writeAtomic(full, res.Files[p]); err != nil {
			fmt.Fprintln(stderr, "vault:", err)
			ok = false
			continue
		}
		fmt.Fprintln(stdout, "wrote", p)
		written++
	}
	return written, ok
}

func writeAtomic(path, content string) error {
	mode := os.FileMode(0o644)
	if fi, err := os.Stat(path); err == nil {
		mode = fi.Mode().Perm()
	}
	f, err := os.CreateTemp(filepath.Dir(path), ".vault-*.tmp")
	if err != nil {
		return err
	}
	tmp := f.Name()
	_, err = f.WriteString(content)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err == nil {
		err = os.Chmod(tmp, mode)
	}
	if err == nil {
		err = os.Rename(tmp, path)
	}
	if err != nil {
		os.Remove(tmp)
	}
	return err
}

func stale(v *Vault, st StaleResult, stdout, stderr io.Writer) int {
	if st.Err != nil {
		fmt.Fprintln(stderr, "vault: stale check unavailable:", st.Err)
		return 1
	}
	for _, s := range st.Notes {
		fmt.Fprintf(stdout, "%s — %s\n", s.Note.Rel, strings.Join(s.Files, ", "))
	}
	for _, b := range st.Bad {
		fmt.Fprintf(stderr, "%s: revision %q: %s\n", b.Note.Rel, b.Rev, b.Err)
	}
	fmt.Fprintf(stdout, "stale: %d notes describe changed code; %d bad revisions\n", len(st.Notes), len(st.Bad))
	if len(st.Bad) > 0 {
		return 1
	}
	return 0
}

func check(v *Vault, st StaleResult, stdout io.Writer) int {
	r := Check(v, st)
	for _, i := range r.Errors {
		fmt.Fprintln(stdout, "error:", i)
	}
	for _, i := range r.Warnings {
		fmt.Fprintln(stdout, "warning:", i)
	}
	fmt.Fprintf(stdout, "check: %d errors%s, %d warnings%s; %d notes, %d stale\n",
		len(r.Errors), classCounts(r.Errors), len(r.Warnings), classCounts(r.Warnings), len(v.Notes), r.Stale)
	if len(r.Foreign) > 0 {
		fmt.Fprintf(stdout, "foreign notes: %s (link problems are warnings)\n", strings.Join(r.Foreign, ", "))
	}
	if len(r.Errors) > 0 {
		return 1
	}
	return 0
}

func classCounts(is []Issue) string {
	if len(is) == 0 {
		return ""
	}
	c := map[string]int{}
	for _, i := range is {
		c[i.Class]++
	}
	var parts []string
	for k, n := range c {
		parts = append(parts, fmt.Sprintf("%s %d", k, n))
	}
	sort.Strings(parts)
	return " (" + strings.Join(parts, ", ") + ")"
}
