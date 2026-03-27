<p align="center">
  <img src="https://github.com/wavekat/wavekat-brand/raw/main/assets/banners/wavekat-core-narrow.svg" alt="WaveKat Core">
</p>

[![Crates.io](https://img.shields.io/crates/v/wavekat-core.svg)](https://crates.io/crates/wavekat-core)
[![docs.rs](https://docs.rs/wavekat-core/badge.svg)](https://docs.rs/wavekat-core)

Shared types for the WaveKat audio processing ecosystem.

> [!WARNING]
> Early development. API may change.

## What's Inside

| Type | Description |
|------|-------------|
| `AudioFrame` | Audio samples with sample rate, accepts both `i16` and `f32` |
| `IntoSamples` | Trait for transparent sample format conversion |

## Quick Start

```sh
cargo add wavekat-core
```

```rust
use wavekat_core::AudioFrame;

// From f32 — zero-copy
let frame = AudioFrame::new(&f32_samples, 16000);

// From i16 — normalizes to f32 automatically
let frame = AudioFrame::new(&i16_samples, 16000);

// Same API regardless of input format
let samples: &[f32] = frame.samples();
let rate: u32 = frame.sample_rate();
```

## Audio Format Standard

The WaveKat ecosystem standardizes on **16 kHz, mono, f32 `[-1.0, 1.0]`**.
`AudioFrame` handles the conversion so downstream crates don't have to.

```
Your audio (any format)
        |
        v
   AudioFrame::new(samples, sample_rate)
        |
        +---> wavekat-vad
        +---> wavekat-turn
        +---> wavekat-asr (future)
```

## License

Licensed under [Apache 2.0](LICENSE).

Copyright 2026 WaveKat.
