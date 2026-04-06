# 01 — WAV I/O (`wav` feature)

## Overview

The optional `wav` feature extends `AudioFrame` with `write_wav` and `from_wav`,
providing a single canonical implementation for WAV I/O across the WaveKat
ecosystem. Backed by [`hound`](https://crates.io/crates/hound).

## Enabling

```toml
wavekat-core = { version = "0.0.5", features = ["wav"] }
```

## API

### `AudioFrame::write_wav`

```rust
pub fn write_wav(&self, path: impl AsRef<Path>) -> Result<(), hound::Error>
```

Writes the frame to a WAV file. Always mono, f32 PCM, at the frame's native
sample rate.

### `AudioFrame::from_wav`

```rust
pub fn from_wav(path: impl AsRef<Path>) -> Result<AudioFrame<'static>, hound::Error>
```

Reads a mono WAV file and returns an owned `AudioFrame`. Accepts both f32 and
i16 files; i16 samples are normalised to `[-1.0, 1.0]` (divided by 32768).

## Example

```rust
use wavekat_core::AudioFrame;

let frame = AudioFrame::from_vec(vec![0.0f32; 16000], 16000);
frame.write_wav("output.wav")?;

let loaded = AudioFrame::from_wav("output.wav")?;
assert_eq!(loaded.sample_rate(), 16000);
assert_eq!(loaded.len(), 16000);
```

## Design notes

- Feature is named `wav` (capability) rather than `hound` (implementation), so
  the underlying library can change without a breaking API surface change.
- Multi-channel WAV files are not rejected at read time — `hound` interleaves
  channels. Callers that need strict mono validation should check
  `reader.spec().channels` themselves; wavekat-core does not add that constraint
  here since the ecosystem already standardises on mono at the producer level.
