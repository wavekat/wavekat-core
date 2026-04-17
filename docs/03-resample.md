# 03 — Resampling (`resample` feature)

## Overview

The optional `resample` feature extends `AudioFrame` with a `resample` method,
allowing any frame to be converted to a target sample rate. Backed by
[`rubato`](https://crates.io/crates/rubato) (high-quality sinc interpolation).

## Motivation

`AudioFrame` already performs format normalisation at construction: i16 → f32.
Sample-rate conversion is the same category of operation — it transforms raw
input into the format a consumer expects. Different WaveKat backends require
different rates (VAD: 16 kHz, TTS: 24 kHz), but audio sources arrive at
arbitrary rates (44.1 kHz, 48 kHz, etc.). Without a shared resampling
primitive, every consumer solves this independently.

## Enabling

```toml
wavekat-core = { version = "0.0.7", features = ["resample"] }
```

## API

### `AudioFrame::resample`

```rust
pub fn resample(&self, target_rate: u32) -> Result<AudioFrame<'static>, CoreError>
```

Returns a new owned `AudioFrame` at the target sample rate. If the frame is
already at the target rate, returns a clone without touching rubato.

Returns `CoreError::Audio` if rubato reports an error (e.g. zero sample rate).

## Example

```rust
use wavekat_core::AudioFrame;

// Load a 44.1 kHz WAV and resample to 24 kHz for TTS
let frame = AudioFrame::from_wav("reference.wav")?;
let frame = frame.resample(24000)?;
assert_eq!(frame.sample_rate(), 24000);
```

## Design notes

- Feature is named `resample` (capability) rather than `rubato`
  (implementation), consistent with the `wav`/`hound` convention.
- Returns `Result` because rubato can fail on invalid parameters (e.g. zero
  rates). The no-op path (same rate) is infallible but wrapped in `Ok` for a
  uniform API.
- Always returns an owned `AudioFrame<'static>` — resampling necessarily
  produces new data.
- Uses `SincFixedIn` with `SincInterpolationParameters` defaults
  (quality 256, Linear interpolation, Blackman-Harris2 window) — good
  balance of quality and speed for speech audio.
