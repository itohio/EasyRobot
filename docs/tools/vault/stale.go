package main

import (
	"fmt"
	"os/exec"
	"path"
	"sort"
	"strings"
)

// StaleNote is a note whose source files changed since the revision it describes.
type StaleNote struct {
	Note  *Note
	Files []string
}

// BadRevision is a note whose revision does not resolve to a commit.
type BadRevision struct {
	Note *Note
	Rev  string
	Err  string
}

// StaleResult is the outcome of the stale check. Bad lists notes whose
// revision does not resolve (the others are still checked); Err set means git
// could not run at all.
type StaleResult struct {
	Notes []StaleNote
	Bad   []BadRevision
	Err   error
}

// findStale compares each note's `source` paths between its source_revision
// (default: vault_revision in conventions.md) and the working tree of repo:
// each revision is resolved on its own (`git rev-parse --verify REV^{commit}`),
// then one `git diff --name-only REV --` per good revision plus one
// `git status` for untracked and modified files.
func findStale(v *Vault, repo string) StaleResult {
	type entry struct {
		note *Note
		rev  string
		srcs []string
	}
	var res StaleResult
	var entries []entry
	byRev := map[string][]string{}
	var revs []string
	for _, n := range v.Notes {
		if n.template() {
			continue
		}
		var srcs []string
		for _, s := range n.List("source") {
			s = strings.TrimSuffix(path.Clean(strings.TrimSpace(s)), "/")
			if s == "." || s == "" || strings.HasPrefix(s, "../") || strings.HasPrefix(s, "/") {
				continue
			}
			srcs = append(srcs, s)
		}
		if len(srcs) == 0 {
			continue
		}
		rev := n.Str("source_revision")
		if rev == "" {
			rev = v.VaultRevision
		}
		if rev == "" {
			res.Bad = append(res.Bad, BadRevision{n, "", "no source_revision and no vault_revision in conventions.md"})
			continue
		}
		if _, ok := byRev[rev]; !ok {
			revs = append(revs, rev)
		}
		byRev[rev] = append(byRev[rev], srcs...)
		entries = append(entries, entry{n, rev, srcs})
	}
	if len(entries) == 0 {
		return res
	}
	if _, err := gitOut(repo, "rev-parse", "--git-dir"); err != nil {
		return StaleResult{Err: err}
	}
	changedAt := map[string][]string{}
	badRev := map[string]string{}
	for _, rev := range revs {
		if _, err := gitOut(repo, "rev-parse", "--verify", "--quiet", rev+"^{commit}"); err != nil {
			badRev[rev] = "revision does not resolve to a commit"
			continue
		}
		out, err := gitOut(repo, append([]string{"diff", "--name-only", rev, "--"}, uniq(byRev[rev])...)...)
		if err != nil {
			badRev[rev] = err.Error()
			continue
		}
		changedAt[rev] = splitNonEmpty(out, "\n")
	}
	out, err := gitOut(repo, "status", "--porcelain", "-z", "--untracked-files=all")
	if err != nil {
		return StaleResult{Err: err}
	}
	var dirty []string
	recs := strings.Split(out, "\x00") // not trimmed: " M path" starts with a space
	for i := 0; i < len(recs); i++ {
		r := recs[i]
		if len(r) < 4 {
			continue
		}
		dirty = append(dirty, r[3:])
		if r[0] == 'R' || r[0] == 'C' {
			i++ // the original path follows a rename/copy record
		}
	}
	conv := v.byPath["conventions.md"]
	reportedDefault := false
	for _, e := range entries {
		if msg, bad := badRev[e.rev]; bad {
			// A bad vault_revision is reported once, on conventions.md.
			if e.note.Str("source_revision") == "" && conv != nil {
				if !reportedDefault {
					res.Bad = append(res.Bad, BadRevision{conv, e.rev, "vault_revision: " + msg})
					reportedDefault = true
				}
				continue
			}
			res.Bad = append(res.Bad, BadRevision{e.note, e.rev, msg})
			continue
		}
		hit := map[string]bool{}
		for _, f := range append(append([]string{}, changedAt[e.rev]...), dirty...) {
			for _, s := range e.srcs {
				if f == s || strings.HasPrefix(f, s+"/") {
					hit[f] = true
				}
			}
		}
		if len(hit) == 0 {
			continue
		}
		files := make([]string, 0, len(hit))
		for f := range hit {
			files = append(files, f)
		}
		sort.Strings(files)
		res.Notes = append(res.Notes, StaleNote{Note: e.note, Files: files})
	}
	return res
}

func gitOut(repo string, args ...string) (string, error) {
	cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return "", fmt.Errorf("git %s: %v: %s", args[0], err, strings.TrimSpace(string(ee.Stderr)))
		}
		return "", fmt.Errorf("git %s: %v", args[0], err)
	}
	return string(out), nil
}

func splitNonEmpty(s, sep string) []string {
	var out []string
	for _, p := range strings.Split(s, sep) {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func uniq(ss []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, s := range ss {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}
