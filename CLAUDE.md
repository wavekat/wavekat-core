# WaveKat Core — Project Instructions

Shared types for the WaveKat audio processing ecosystem. This crate is a leaf dependency used by `wavekat-vad`, `wavekat-turn`, `wavekat-voice`, and any future WaveKat crates.

## Purpose

Provide the common audio primitives so that all WaveKat crates speak the same language. Nothing more.

## What Belongs Here

- Types shared across **2 or more** WaveKat crates
- Audio frame representation (`AudioFrame`, `IntoSamples`)
- Common constants (sample rates, format standards)
- Shared error types (only when genuinely shared)
- Format normalisation operations on `AudioFrame` (i16→f32, sample-rate conversion) — gated behind optional features

## What Does NOT Belong Here

- Backend implementations (VAD models, turn detectors, ASR engines)
- Domain-specific processing logic (filtering, feature extraction, VAD)
- Anything specific to a single crate — if only one crate uses it, it stays there

## Design Principles

1. **Zero required dependencies** — the default build has no external deps. Optional features (e.g. `wav`, `resample`) may pull deps but must be opt-in.
2. **Tiny surface area** — only add types when there is a concrete need from 2+ crates.
3. **Stable API** — downstream crates depend on this, so changes here ripple everywhere. Be conservative.
4. **Feature flags for heavy deps** — capabilities like WAV I/O (`hound`) and resampling (`rubato`) are gated behind feature flags so the default crate stays lightweight.

## Audio Format Standard

The WaveKat ecosystem standardizes on:
- **16 kHz** sample rate (the universal standard for speech ML models)
- **Mono** (single channel)
- **f32 normalized `[-1.0, 1.0]`** as the internal representation

`AudioFrame` accepts both `&[i16]` and `&[f32]` input transparently. The conversion happens once at construction. Downstream crates receive `&[f32]` from `frame.samples()`.

## Error Handling

`CoreError` is the unified error type for all fallible operations in `wavekat-core`.

| Variant | Wraps | When |
|---|---|---|
| `Io(std::io::Error)` | I/O errors | File/stream failures |
| `Audio(String)` | Format/codec errors | WAV encoding/decoding issues |

- Manually implements `Display`, `Error`, and `From<std::io::Error>` — no `thiserror` dependency.
- `From<hound::Error>` is gated behind `#[cfg(feature = "wav")]`: I/O errors map to `Io`, everything else maps to `Audio(msg)`.
- Resampling errors (from `rubato`) are mapped to `Audio(msg)` behind `#[cfg(feature = "resample")]`.
- Downstream crates can add `From<CoreError> for TheirError` to make `?` work naturally.

## Repository Structure

```
wavekat-core/
├── Cargo.toml                  # workspace root
├── crates/
│   └── wavekat-core/           # library crate
│       ├── src/
│       │   ├── lib.rs          # public API, re-exports
│       │   ├── audio.rs        # AudioFrame, IntoSamples, resample
│       │   └── error.rs        # CoreError
│       └── Cargo.toml
├── docs/                       # design documents
│   ├── 01-wav-io.md
│   ├── 02-test-coverage.md
│   └── 03-resample.md
├── LICENSE                     # Apache 2.0
└── CLAUDE.md                   # this file
```

## Code Quality

- `cargo fmt --all --check` — no formatting issues
- `cargo build` — clean build
- `cargo test` — all tests pass
- **All new features and bug fixes must include tests** — no PR should add or change behavior without corresponding test coverage
- `cargo clippy --workspace -- -D warnings` — no warnings
- No `unwrap()` in library code
- Error types use manual `Display`/`Error` impls (no `thiserror` — keeps zero required deps)
- `///` doc comments on all public items

## Conventions

- Keep the main branch stable and buildable
- **PR titles must use conventional commit format, max 50 characters** — e.g. `feat: add AudioFrame type`, `fix: normalize edge case`
- **Always use squash merge** when merging feature branches into main
