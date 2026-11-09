// Package bench measures retrievers against questions with known answers:
// question loading, a scoring harness (recall@1, recall@k, MRR, latency
// percentiles per question type) and Grep, a ripgrep-style baseline that
// models how peers search the workspace today.
//
// Tables rendered by this package carry scores and question ids only, never
// question text, paths or snippets, because live corpora hold private notes.
package bench

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"strings"
)

// Type classifies a benchmark question.
type Type string

// Question types.
const (
	TypeIdentifier Type = "identifier" // an id, symbol or title phrase
	TypeParaphrase Type = "paraphrase" // same meaning, different words
	TypeCrossLink  Type = "cross-link" // answer reachable through links
	TypeRecency    Type = "recency"    // the newer of two similar notes
	TypeCode       Type = "code"       // answer is (or includes) source code
)

// Types returns all question types in report order.
func Types() []Type {
	return []Type{TypeIdentifier, TypeParaphrase, TypeCrossLink, TypeRecency, TypeCode}
}

func (t Type) valid() bool {
	for _, k := range Types() {
		if t == k {
			return true
		}
	}
	return false
}

// Question is one benchmark question. ExpectAny lists corpus-relative slash
// paths (knowledge.DocID with an empty scope); finding any of them counts.
type Question struct {
	ID        string   `json:"id"`
	Q         string   `json:"q"`
	Type      Type     `json:"type"`
	ExpectAny []string `json:"expect_any"`
}

// LoadQuestions reads and validates a JSON array of questions.
func LoadQuestions(file string) ([]Question, error) {
	data, err := os.ReadFile(file)
	if err != nil {
		return nil, fmt.Errorf("bench: read questions: %w", err)
	}
	qs, err := parseQuestions(data)
	if err != nil {
		return nil, fmt.Errorf("bench: %s: %w", file, err)
	}
	return qs, nil
}

func parseQuestions(data []byte) ([]Question, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var qs []Question
	if err := dec.Decode(&qs); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return nil, errors.New("decode: trailing data after the question array")
	}
	if len(qs) == 0 {
		return nil, errors.New("no questions")
	}
	seen := make(map[string]bool, len(qs))
	for i, q := range qs {
		if err := q.validate(); err != nil {
			return nil, fmt.Errorf("question %d (%q): %w", i, q.ID, err)
		}
		if seen[q.ID] {
			return nil, fmt.Errorf("question %d: duplicate id %q", i, q.ID)
		}
		seen[q.ID] = true
	}
	return qs, nil
}

func (q Question) validate() error {
	switch {
	case strings.TrimSpace(q.ID) == "":
		return errors.New("empty id")
	case strings.TrimSpace(q.Q) == "":
		return errors.New("empty question")
	case !q.Type.valid():
		return fmt.Errorf("unknown type %q", q.Type)
	case len(q.ExpectAny) == 0:
		return errors.New("empty expect_any")
	}
	for _, p := range q.ExpectAny {
		if err := validPath(p); err != nil {
			return fmt.Errorf("expect_any %q: %w", p, err)
		}
	}
	return nil
}

// validPath accepts clean, relative slash paths only.
func validPath(p string) error {
	switch {
	case p == "":
		return errors.New("empty path")
	case strings.Contains(p, `\`):
		return errors.New("backslash in path; use slashes")
	case path.IsAbs(p):
		return errors.New("absolute path; use a corpus-relative path")
	case path.Clean(p) != p:
		return errors.New("path is not clean")
	case p == ".." || strings.HasPrefix(p, "../"):
		return errors.New("path escapes the corpus")
	}
	return nil
}
