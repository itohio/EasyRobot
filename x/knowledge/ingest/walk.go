package ingest

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/itohio/EasyRobot/x/knowledge"
)

// entry is one accepted file found by the walk.
type entry struct {
	rel     string // slash path relative to the source root
	abs     string // absolute OS path
	id      knowledge.DocID
	scope   string
	size    int64
	mtime   time.Time
	chunker knowledge.Chunker
	kind    string
}

// skipDirs are never descended into (in addition to any dot-directory).
var skipDirs = map[string]bool{
	"node_modules": true,
	"vendor":       true,
	"_quarantine":  true,
	"_tmp":         true,
}

// secretName reports whether a file name looks like key material or
// credentials; such files are never read.
func secretName(name string) bool {
	n := strings.ToLower(name)
	switch {
	case n == ".env", strings.HasPrefix(n, ".env."):
		return true
	case strings.HasPrefix(n, "id_rsa"), strings.HasPrefix(n, "id_ed25519"):
		return true
	}
	switch path.Ext(n) {
	case ".pem", ".key", ".creds", ".nk", ".p12", ".keystore":
		return true
	}
	return false
}

// ignoreRule is one pattern from a .gitignore file.
type ignoreRule struct {
	base     string // directory holding the .gitignore, slash path relative to root ("" = root)
	pattern  string // path.Match pattern
	anchored bool   // match against the path relative to base, not the base name
	anyDepth bool   // leading "**/": match the path relative to base at any depth
	dirOnly  bool   // trailing "/" in the source pattern
}

// parseGitignore reads the simple subset of .gitignore syntax: names,
// path.Match globs, trailing "/" for directories, leading "/" (or any inner
// "/") anchoring to the .gitignore's directory, and a leading "**/" (match
// at any depth below that directory). Comments and blank lines are skipped;
// negation ("!") and "**" elsewhere are unsupported.
func parseGitignore(data []byte, base string) []ignoreRule {
	var rules []ignoreRule
	for line := range strings.SplitSeq(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || line[0] == '#' || line[0] == '!' {
			continue
		}
		r := ignoreRule{base: base}
		if strings.HasSuffix(line, "/") {
			r.dirOnly = true
			line = strings.TrimRight(line, "/")
		}
		switch {
		case strings.HasPrefix(line, "**/"):
			r.anyDepth = true
			line = strings.TrimLeft(strings.TrimPrefix(line, "**/"), "/")
		case strings.HasPrefix(line, "/"):
			r.anchored = true
			line = strings.TrimLeft(line, "/")
		case strings.Contains(line, "/"):
			r.anchored = true
		}
		if line == "" {
			continue
		}
		if _, err := path.Match(line, ""); err != nil {
			continue // malformed glob
		}
		r.pattern = line
		rules = append(rules, r)
	}
	return rules
}

func (r ignoreRule) match(rel string, isDir bool) bool {
	if r.dirOnly && !isDir {
		return false
	}
	sub := rel
	if r.base != "" {
		if !strings.HasPrefix(rel, r.base+"/") {
			return false
		}
		sub = rel[len(r.base)+1:]
	}
	switch {
	case r.anyDepth:
		for {
			if ok, _ := path.Match(r.pattern, sub); ok {
				return true
			}
			i := strings.IndexByte(sub, '/')
			if i < 0 {
				return false
			}
			sub = sub[i+1:]
		}
	case !r.anchored:
		sub = path.Base(rel)
	}
	ok, _ := path.Match(r.pattern, sub)
	return ok
}

func ignored(rules []ignoreRule, rel string, isDir bool) bool {
	for _, r := range rules {
		if r.match(rel, isDir) {
			return true
		}
	}
	return false
}

// walker collects accepted files under a root.
type walker struct {
	ing      *Ingestor
	root     string // absolute
	realRoot string // root with symlinks resolved
	entries  []entry
	failed   []string // rel paths that could not be read or stat'ed
	secrets  int
	skipped  int
	errs     []error
}

// walk visits the tree under w.root; only context errors and an unreadable
// root are fatal, other problems are collected in w.errs (and their paths
// in w.failed).
func (w *walker) walk(ctx context.Context) error {
	if _, err := os.ReadDir(w.root); err != nil {
		return fmt.Errorf("ingest: read root: %w", err)
	}
	rr, err := filepath.EvalSymlinks(w.root)
	if err != nil {
		return fmt.Errorf("ingest: resolve root: %w", err)
	}
	w.realRoot = rr
	return w.dir(ctx, w.root, "", nil)
}

// covers reports whether rel lies at or under a path the walk failed to
// read; such catalog documents are kept rather than deleted.
func (w *walker) covers(rel string) bool {
	for _, f := range w.failed {
		if f == "" || rel == f || strings.HasPrefix(rel, f+"/") {
			return true
		}
	}
	return false
}

// linkTarget resolves a file symlink; ok is false for dangling links, links
// to directories and links whose target lies outside the root.
func (w *walker) linkTarget(p string) (target string, info fs.FileInfo, ok bool) {
	target, err := filepath.EvalSymlinks(p)
	if err != nil {
		return "", nil, false
	}
	if rel, err := filepath.Rel(w.realRoot, target); err != nil || !filepath.IsLocal(rel) {
		return "", nil, false
	}
	fi, err := os.Stat(target)
	if err != nil || fi.IsDir() {
		return "", nil, false
	}
	return target, fi, true
}

func (w *walker) dir(ctx context.Context, abs, rel string, rules []ignoreRule) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	des, err := os.ReadDir(abs)
	if err != nil {
		w.failed = append(w.failed, rel)
		w.errs = append(w.errs, fmt.Errorf("ingest: read dir %q: %w", rel, err))
		return nil
	}
	if data, err := os.ReadFile(filepath.Join(abs, ".gitignore")); err == nil {
		rules = append(slices.Clip(rules), parseGitignore(data, rel)...)
	}
	for _, de := range des {
		if err := ctx.Err(); err != nil {
			return err
		}
		name := de.Name()
		r := path.Join(rel, name)
		p := filepath.Join(abs, name)
		var (
			info   fs.FileInfo
			target string
		)
		if de.Type()&fs.ModeSymlink != 0 {
			var ok bool
			if target, info, ok = w.linkTarget(p); !ok {
				continue
			}
		}
		if de.IsDir() {
			if name[0] == '.' || skipDirs[name] || w.excluded(r, name) || ignored(rules, r, true) {
				continue
			}
			if err := w.dir(ctx, p, r, rules); err != nil {
				return err
			}
			continue
		}
		if w.excluded(r, name) || ignored(rules, r, false) {
			continue
		}
		if secretName(name) || (target != "" && secretName(filepath.Base(target))) {
			w.secrets++
			continue
		}
		c := w.ing.chunkerFor(r)
		if c == nil {
			continue
		}
		if info == nil {
			if info, err = de.Info(); err != nil {
				w.failed = append(w.failed, r)
				w.errs = append(w.errs, fmt.Errorf("ingest: stat %q: %w", r, err))
				continue
			}
		}
		if !info.Mode().IsRegular() {
			continue
		}
		if info.Size() > w.ing.maxSize {
			w.skipped++
			continue
		}
		w.entries = append(w.entries, entry{
			rel:     r,
			abs:     p,
			id:      w.ing.docID(r),
			scope:   w.ing.docScope(r),
			size:    info.Size(),
			mtime:   info.ModTime(),
			chunker: c,
			kind:    kindOf(c, r),
		})
	}
	return nil
}

// excluded applies Source.Exclude: entries without "/" match base names
// (path.Match globs allowed), entries with "/" are path prefixes relative
// to the root.
func (w *walker) excluded(rel, name string) bool {
	for _, e := range w.ing.src.Exclude {
		e = strings.Trim(filepath.ToSlash(e), "/")
		if e == "" {
			continue
		}
		if strings.Contains(e, "/") {
			if rel == e || strings.HasPrefix(rel, e+"/") {
				return true
			}
			continue
		}
		if ok, _ := path.Match(e, name); ok {
			return true
		}
	}
	return false
}
