package sqlite

import (
	"context"
	"database/sql"
	"fmt"
)

// schemaVersion is stored in PRAGMA user_version.
const schemaVersion = 1

// schema creates version 1. FTS rows are contentless (the text lives in
// chunks) and hold pre-tokenized text: package token's terms joined by
// spaces. tokenchars keeps the connectors of identifiers (ABU-061,
// abg.svc.find, snake_case) inside one FTS token, so FTS5 terms equal
// token terms. chunks_fts.rowid = chunks.rid.
var schema = []string{
	`CREATE TABLE IF NOT EXISTS meta (
		k TEXT PRIMARY KEY,
		v INTEGER NOT NULL
	) WITHOUT ROWID`,
	`INSERT OR IGNORE INTO meta(k, v) VALUES ('generation', 0)`,
	`CREATE TABLE IF NOT EXISTS docs (
		id       TEXT PRIMARY KEY,
		scope    TEXT NOT NULL,
		path     TEXT NOT NULL,
		kind     TEXT NOT NULL,
		title    TEXT NOT NULL,
		size     INTEGER NOT NULL,
		mtime_s  INTEGER NOT NULL,
		mtime_ns INTEGER NOT NULL,
		hash     BLOB NOT NULL,
		props    TEXT NOT NULL,
		weight   REAL NOT NULL
	) WITHOUT ROWID`,
	`CREATE TABLE IF NOT EXISTS chunks (
		rid     INTEGER PRIMARY KEY,
		id      BLOB NOT NULL UNIQUE,
		doc     TEXT NOT NULL REFERENCES docs(id) ON DELETE CASCADE ON UPDATE CASCADE,
		ord     INTEGER NOT NULL,
		heading TEXT NOT NULL,
		symbol  TEXT NOT NULL,
		l0      INTEGER NOT NULL,
		l1      INTEGER NOT NULL,
		text    TEXT NOT NULL,
		hash    BLOB NOT NULL,
		tokens  INTEGER NOT NULL
	)`,
	`CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc, ord)`,
	`CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
		heading, symbol, path, body,
		content='', contentless_delete=1,
		tokenize="unicode61 remove_diacritics 0 tokenchars '-_.'"
	)`,
	`CREATE TABLE IF NOT EXISTS edges (
		src    TEXT NOT NULL,
		dst    TEXT NOT NULL,
		kind   INTEGER NOT NULL,
		weight REAL NOT NULL,
		source TEXT NOT NULL,
		PRIMARY KEY (src, dst, kind, source)
	) WITHOUT ROWID`,
	`CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst)`,
	`CREATE INDEX IF NOT EXISTS edges_source ON edges(source)`,
}

// migrate creates the schema in an empty database and rejects databases
// written by an unknown schema version.
func migrate(ctx context.Context, db *sql.DB) error {
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	var v int
	if err := tx.QueryRowContext(ctx, `PRAGMA user_version`).Scan(&v); err != nil {
		return fmt.Errorf("read user_version: %w", err)
	}
	switch v {
	case schemaVersion:
		return nil
	case 0:
	default:
		return fmt.Errorf("unsupported schema version %d (want %d)", v, schemaVersion)
	}
	for _, stmt := range schema {
		if _, err := tx.ExecContext(ctx, stmt); err != nil {
			return fmt.Errorf("create schema: %w", err)
		}
	}
	if _, err := tx.ExecContext(ctx, fmt.Sprintf(`PRAGMA user_version = %d`, schemaVersion)); err != nil {
		return fmt.Errorf("set user_version: %w", err)
	}
	return tx.Commit()
}
