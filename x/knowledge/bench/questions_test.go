package bench

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestLoadQuestions(t *testing.T) {
	path := filepath.Join(t.TempDir(), "qs.json")
	writeFile(t, path, `[
  {"id": "q1", "q": "RV-007", "type": "identifier", "expect_any": ["rover/tickets/RV-007.md"]},
  {"id": "q2", "q": "how is the loop tuned", "type": "paraphrase", "expect_any": ["a.md", "b/c.go"]},
  {"id": "q3", "q": "x", "type": "cross-link", "expect_any": ["x.md"]},
  {"id": "q4", "q": "y", "type": "recency", "expect_any": ["y.md"]},
  {"id": "q5", "q": "z", "type": "code", "expect_any": ["z.go"]}
]`)
	qs, err := LoadQuestions(path)
	if err != nil {
		t.Fatalf("LoadQuestions: %v", err)
	}
	want := []Question{
		{ID: "q1", Q: "RV-007", Type: TypeIdentifier, ExpectAny: []string{"rover/tickets/RV-007.md"}},
		{ID: "q2", Q: "how is the loop tuned", Type: TypeParaphrase, ExpectAny: []string{"a.md", "b/c.go"}},
		{ID: "q3", Q: "x", Type: TypeCrossLink, ExpectAny: []string{"x.md"}},
		{ID: "q4", Q: "y", Type: TypeRecency, ExpectAny: []string{"y.md"}},
		{ID: "q5", Q: "z", Type: TypeCode, ExpectAny: []string{"z.go"}},
	}
	if !reflect.DeepEqual(qs, want) {
		t.Fatalf("got %+v\nwant %+v", qs, want)
	}
}

func TestLoadQuestionsInvalid(t *testing.T) {
	cases := map[string]string{
		"unknown type":   `[{"id":"q1","q":"x","type":"fuzzy","expect_any":["a.md"]}]`,
		"empty expect":   `[{"id":"q1","q":"x","type":"code","expect_any":[]}]`,
		"missing expect": `[{"id":"q1","q":"x","type":"code"}]`,
		"empty path":     `[{"id":"q1","q":"x","type":"code","expect_any":[""]}]`,
		"absolute path":  `[{"id":"q1","q":"x","type":"code","expect_any":["/etc/a.md"]}]`,
		"dotdot path":    `[{"id":"q1","q":"x","type":"code","expect_any":["../a.md"]}]`,
		"backslash path": `[{"id":"q1","q":"x","type":"code","expect_any":["a\\b.md"]}]`,
		"empty id":       `[{"id":"","q":"x","type":"code","expect_any":["a.md"]}]`,
		"empty question": `[{"id":"q1","q":"  ","type":"code","expect_any":["a.md"]}]`,
		"duplicate id":   `[{"id":"q1","q":"x","type":"code","expect_any":["a.md"]},{"id":"q1","q":"y","type":"code","expect_any":["b.md"]}]`,
		"unknown field":  `[{"id":"q1","q":"x","type":"code","expect":["a.md"]}]`,
		"not an array":   `{"id":"q1"}`,
		"trailing data":  `[] []`,
		"empty list":     `[]`,
	}
	dir := t.TempDir()
	for name, content := range cases {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(dir, strings.ReplaceAll(name, " ", "_")+".json")
			writeFile(t, path, content)
			if qs, err := LoadQuestions(path); err == nil {
				t.Fatalf("expected error, got %+v", qs)
			}
		})
	}
	if _, err := LoadQuestions(filepath.Join(dir, "missing.json")); err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestLoadCommittedQuestionFiles(t *testing.T) {
	for _, tc := range []struct {
		path     string
		min, max int
	}{
		{"testdata/fixture-questions.json", 24, 30},
		{"testdata/live-questions.json", 28, 30},
	} {
		qs, err := LoadQuestions(tc.path)
		if err != nil {
			t.Fatalf("%s: %v", tc.path, err)
		}
		if len(qs) < tc.min || len(qs) > tc.max {
			t.Errorf("%s: %d questions, want %d..%d", tc.path, len(qs), tc.min, tc.max)
		}
		seen := map[Type]int{}
		for _, q := range qs {
			seen[q.Type]++
		}
		for _, typ := range Types() {
			if seen[typ] == 0 {
				t.Errorf("%s: no %s question", tc.path, typ)
			}
		}
	}
}
