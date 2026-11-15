---
id: ER-022
title: "Audio modem acquisition, sync, drift and BER tests"
type: ticket
status: blocked
kind: implementation
owner: easyrobot
workstream: W7
milestone: M-e
tags: [easyrobot, ticket, w7, modem]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-022 — Audio modem acquisition, sync, drift and BER tests

[[tickets/index|Ticket index]] · [[roadmap]] · [[todo]]

## Goal and evidence

M-e stage 2 ([[roadmap#M-e — Robust voice modem (W7)|roadmap M-e]], [[W7-filters-codecs-voice]]): the ≈ 25–30 dB gap to theory comes from acquisition/sync and test design, not symbol decisions.

- **BER-vs-Eb/N0 harness** on a channel model (Gaussian noise, fractional delay, level, ppm, band-pass, echo); every modem change ships a BER curve.
- **Shared synchroniser**: CFAR energy detect, chirp/preamble correlation, fractional timing, frame end ([[modem-energy-detector-and-preamble-acquisition]]); ideation recommends an up+down chirp pair for CFO ([[2026-11-09-voice-modem-w7]] §3).
- **Oscillator drift**: phase-continuous NCO replacing float32-time oscillators — `x/math/dsp/generators_trig.go:240,258`.
- Framer and voice-band profile (500–3000 Hz; 8-FSK baseline per ideation §2) folded in here (roadmap §4 "or fold into ER-022").
- The `chain.Adapter` nil buffer and `cmd/audio` chunk packing (`x/chain/adaper.go:111,138`; `cmd/audio/common.go:304-318,502,542`) belong to the audio composition fixes; this ticket picks them up only if those did not.

## Dependencies and scope

Dependencies: After the audio composition fixes (O4); PLL/AGC part of [[ER-019-filter-correctness|ER-019]] first (O7); CRC/FEC from [[ER-023-coding-fixes|ER-023]]. **Blocked:** acquisition tuning needs the owner's channel choice (recommended speaker→mic). The channel-model harness itself is channel-independent and may start once the audio composition fixes land.

Not in scope: No real-channel run (stage 3, needs an approved audio device setup), no OFDM rate tier.

Source/test edits are approved; scheduling is decided separately. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] Harness: a test reproduces the current gap (BER at Eb/N0 = 10 dB) before any modem change.
- [ ] Failing test first: acquisition finds the frame start within ±1 sample under fractional delay and ±100 ppm.
- [ ] Failing test first: NCO phase stays continuous over 10⁶ samples (no drift > 1e-3 rad from a float64 reference).
- [ ] W7 target on the channel model: BER < 1e-4 at Eb/N0 ≈ 10 dB with 4/8-FSK + chirp preamble + CRC-16 + interleaved FEC.

## Effort uncertainty and revision trigger

Size guess: M-e stage 2 **L** (with ER-023). Sizes are rough guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.
