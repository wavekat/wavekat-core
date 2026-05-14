# 04 — Resample drops short inputs when upsampling

> Status: bug · Date: 2026-05-14

## Summary

`AudioFrame::resample` returns `CoreError::Audio("Insufficient output
buffer size N, expected M frames")` for any input whose length is
**less than rubato's internal chunk size (1024 frames)** when the
target rate is much higher than the source rate. Every call from
`wavekat-voice`'s RTP receive path (160-sample G.711 frames →
44.1 kHz speaker) hits this and the frame is dropped — the user hears
silence from the remote even though signaling and RTP are working.

The bug is in the output-buffer sizing, not in the resampler itself.

## Reproduction

```rust
use wavekat_core::AudioFrame;

let g711_frame: Vec<f32> = vec![0.0; 160];           // 20 ms at 8 kHz
let frame = AudioFrame::from_vec(g711_frame, 8000);
let resampled = frame.resample(44_100);              // Err
```

Observed log line from `wavekat-voice` during a live call:

```
WARN wavekat_voice::audio_io: resample failed — dropping frame
  err=Audio("Insufficient output buffer size 1906, expected 5633 frames")
```

## Root cause

`crates/wavekat-core/src/audio.rs:152` builds the resampler with a
fixed input chunk size and an explicit `FixedAsync::Input` mode:

```rust
let mut resampler = Async::<f32>::new_sinc(
    ratio, 1.0, &params, /* chunk_size: */ 1024, 1, FixedAsync::Input,
)?;
```

Rubato in `FixedAsync::Input` mode processes input in fixed
`chunk_size`-sized chunks, **regardless of how much input the caller
actually supplied.** Each chunk produces `chunk_size * ratio` output
frames.

Line 156 sizes the output buffer for the *actual* input length:

```rust
let out_len = (nbr_input_frames as f64 * ratio) as usize + 1024;
```

For a 160-sample input upsampled 8 kHz → 44.1 kHz:

| Quantity                                | Value |
|-----------------------------------------|-------|
| `nbr_input_frames`                      | 160   |
| `ratio = 44_100 / 8_000`                | 5.5125 |
| `out_len` (sized by caller)             | 160 × 5.5125 + 1024 = **1906** |
| Output rubato actually requires         | 1024 × 5.5125 ≈ **5633** |

Rubato sees the output buffer is too small and returns
`InsufficientOutputBufferSize`. The wavekat-voice receive loop logs
the error and drops the frame.

Why this hasn't bitten before:

- **Downsample paths** (e.g. 48 kHz mic → 8 kHz for G.711 encode): the
  *output* per chunk is smaller than 1024, the heuristic `out_len`
  almost always exceeds it. No failures.
- **Upsample with large inputs** (e.g. a 1 s WAV at 8 kHz → 16 kHz for
  VAD): `nbr_input_frames` ≥ chunk_size, so `nbr_input_frames * ratio`
  already covers `chunk_size * ratio`.
- **Upsample with short inputs and a small ratio** (e.g. 8 kHz →
  16 kHz, 160 input frames → 320 output): `out_len = 1184`, rubato
  needs `1024 × 2 = 2048`. Still fails, but the symptom is the same;
  has just not been exercised.

The bug is specifically when `nbr_input_frames < chunk_size` and
`target_rate > source_rate`.

## Recommended fix

Two changes in `resample()`:

1. **Size the output buffer with rubato's own API.** `Resampler` exposes
   `output_frames_max()` (and `output_frames_next()` for streaming
   callers), which returns the largest output any single chunk could
   produce. Use it instead of the input-times-ratio heuristic:

   ```rust
   let out_len = resampler.output_frames_max() + 1024; // small headroom
   let mut outdata = vec![0.0f32; out_len];
   ```

   This is what rubato's own examples recommend; it removes the
   "guess and pad" pattern.

2. **(Optional) Match the chunk size to short inputs.** For one-shot
   `AudioFrame::resample` callers, processing one 160-sample input
   in 1024-sample chunks is wasteful — rubato pads the chunk
   internally and the extra output is truncated by `out_produced`.
   Pick a chunk size matched to the input when the input is shorter
   than the default:

   ```rust
   let chunk_size = nbr_input_frames.min(1024);
   ```

   This saves both CPU and memory on the hot path the RTP receive
   loop drives.

Either fix alone resolves the bug; doing both is the right shape.

## Test additions (must land with the fix)

Per the project's "tests ship with features" rule, add unit tests in
`audio::tests` that fail before the fix and pass after:

1. **`resample_short_input_upsample_large_ratio`** — the exact case
   that bit us: 160 samples at 8 kHz → 44.1 kHz. Must return `Ok`
   with an output length close to `160 * 5.5125 = 882` (allow a small
   tolerance, rubato's edge behavior).
2. **`resample_short_input_upsample_small_ratio`** — 160 samples at
   8 kHz → 16 kHz. Today this also fails (output buffer 1344, rubato
   needs 2048); the fix must cover it.
3. **`resample_single_g711_frame_to_48k`** — 160 → 48 kHz (the
   other common device rate); same shape.
4. **`resample_does_not_regress_on_long_inputs`** — keep the existing
   `resample_upsample` / `resample_downsample` tests passing
   (regression guard).

## Downstream impact

After the fix lands:

- **`wavekat-core` 0.0.10** (or whatever the next release-please bump
  is).
- **`wavekat-voice`** picks up the new version via workspace dep bump;
  the existing per-frame `AudioFrame::resample` call in
  `LocalDeviceSink` starts working. No other changes needed on the
  consumer side.

A streaming-resampler refactor in `LocalDeviceSink` (build one
resampler at stream-open, feed it sample-by-sample, keep state across
frames) is a separate, larger improvement and not required for this
fix. The motivations for that refactor are CPU efficiency (no per-frame
allocator churn) and quality (avoid edge artifacts at chunk
boundaries), not correctness — once the buffer-sizing bug is fixed,
per-frame stateless resampling is *correct*, just not optimal.
