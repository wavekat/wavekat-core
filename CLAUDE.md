# WaveKat Core — Project Instructions

Shared types for the WaveKat audio processing ecosystem. This crate is a leaf dependency used by `wavekat-vad`, `wavekat-turn`, `wavekat-voice`, and any future WaveKat crates.

## Purpose

Provide the common audio primitives so that all WaveKat crates speak the same language. Nothing more.

## What Belongs Here

- Types shared across **2 or more** WaveKat crates
- Audio frame representation (`AudioFrame`, `IntoSamples`)
- Common constants (sample rates, format standards)
- Shared error types (only when genuinely shared)

## What Does NOT Belong Here

- Backend implementations (VAD models, turn detectors, ASR engines)
- Processing logic (resampling, filtering, feature extraction)
- Feature flags that pull heavy dependencies (`ort`, `ndarray`, `rubato`)
- Anything specific to a single crate — if only one crate uses it, it stays there

## Design Principles

1. **Zero dependencies** — this crate must have no external dependencies. It is a leaf.
2. **Tiny surface area** — only add types when there is a concrete need from 2+ crates.
3. **Stable API** — downstream crates depend on this, so changes here ripple everywhere. Be conservative.
4. **No feature flags** — keep it simple. If something needs a flag, it probably belongs in a downstream crate.

## Audio Format Standard

The WaveKat ecosystem standardizes on:
- **16 kHz** sample rate (the universal standard for speech ML models)
- **Mono** (single channel)
- **f32 normalized `[-1.0, 1.0]`** as the internal representation

`AudioFrame` accepts both `&[i16]` and `&[f32]` input transparently. The conversion happens once at construction. Downstream crates receive `&[f32]` from `frame.samples()`.

## Repository Structure

```
wavekat-core/
├── Cargo.toml                  # workspace root
├── crates/
│   └── wavekat-core/           # library crate
│       ├── src/
│       │   ├── lib.rs          # public API, re-exports
│       │   └── audio.rs        # AudioFrame, IntoSamples
│       └── Cargo.toml
├── LICENSE                     # Apache 2.0
└── CLAUDE.md                   # this file
```

## Code Quality

- `cargo fmt --all --check` — no formatting issues
- `cargo build` — clean build
- `cargo test` — all tests pass
- `cargo clippy --workspace -- -D warnings` — no warnings
- No `unwrap()` in library code
- Use `thiserror` for error types (only if error types are added)
- `///` doc comments on all public items

## Conventions

- Keep the main branch stable and buildable
- **PR titles must use conventional commit format, max 50 characters** — e.g. `feat: add AudioFrame type`, `fix: normalize edge case`
- **Always use squash merge** when merging feature branches into main
