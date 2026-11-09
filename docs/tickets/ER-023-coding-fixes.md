---
id: ER-023
title: "Coding: GF(256) RS, CRC catalogue, line codes, LFSR, Gray, PEG"
type: ticket
status: backlog
kind: implementation
owner: easyrobot
workstream: W7
milestone: M-e
tags: [easyrobot, ticket, w7, coding]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-023 — Coding: GF(256) RS, CRC catalogue, line codes, LFSR, Gray, PEG

[[tickets/index|Ticket index]] · [[roadmap]] · ER-R2 (evidence: `~/src/docs/evidence/ER-R2.md`, outside the repo) · [[todo]]

## Goal and evidence

Correct the coding layer M-e (and M-b's SPI CRC) rely on ([[line-coding-and-fec]], [[peg-protocol-parser]]). Accepted ER-R2 decision 5: in-repo GF(256) Reed–Solomon with erasures.

- **RS** is not RS: implement generator, syndromes, Berlekamp–Massey, Chien, Forney, erasures — `x/math/dsp/coding/rs/rs_library.go:21-123`; final partial block — `rs_encoder.go:164-176` ([[reed-solomon-code]]).
- **CRC** polynomial/reflection mismatch — `crc/crc_encoder.go:23-44`, `:86-153` ([[crc-frame-check]]).
- **4B5B** nibble order — `four5b/fourb5b_decoder.go:129`; **8B10B** 0xFF sentinel — `eight10b/eightb10b_tables.go:82-94`.
- **LFSR** 7/23-bit preset taps (`0x44`, `0x400010`) — `lfsr/scrambler_encoder.go:22-35`.
- **Gray** widths 1–3, 5–7 — `gray/gray_encoder.go:106-109`, `gray_decoder.go:110-113`.
- **PEG** self-modification during eval — `x/math/protocol/peg/ast.go:62-96`; fields not reset per `Decide` — `program.go:24-30`; base-type detection — `parser.go:716-797`.

## Dependencies and scope

Dependencies: After roster task ER-AUDIO-1 (O4; same files and tests). Feeds [[ER-022-audio-modem-acquisition|ER-022]] and [[ER-031-spi-slave-feature-framing|ER-031]] (CRC-32).

Not in scope: the wireline placeholders (AMI, B8ZS, HDB3, RLL, PPM) were deleted in ER-cleanup (evidence: `~/src/docs/evidence/ER-cleanup.md`, outside the repo) (owner decision #5b: delete).

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] CRC: catalogue check values for "123456789" (CRC-8, CRC-16/CCITT, CRC-32/IEEE) fail first, then pass.
- [ ] RS(255,223): random error patterns up to t = 16 and erasure patterns up to 2t corrected; t+1 errors detected — failing first.
- [ ] 4B5B, 8B10B (0xFF), Gray (all widths): encode→decode round trip over all byte values fails first, then passes.
- [ ] LFSR: preset periods 127 and 8388607 measured.
- [ ] PEG: two consecutive `Decide` calls on different packets give independent fields; AST unchanged after eval.
- [ ] `go test -count=1 ./x/math/dsp/coding/... ./x/math/protocol/peg/...` passes.

## Effort uncertainty and revision trigger

Product's guess: part of M-e stage 2 (**L** with ER-022); this ticket ≈ **M**. Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.
