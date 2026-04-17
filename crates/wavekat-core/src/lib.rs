//! Shared types for the WaveKat audio processing ecosystem.
//!
//! This crate provides the common audio primitives used across all WaveKat
//! crates (`wavekat-vad`, `wavekat-turn`, `wavekat-voice`, etc.).
//!
//! # Audio Format Standard
//!
//! The WaveKat ecosystem standardizes on **16 kHz, mono, f32 `[-1.0, 1.0]`**
//! as the internal audio format. [`AudioFrame`] accepts both `i16` and `f32`
//! input transparently via [`IntoSamples`].
//!
//! ```
//! use wavekat_core::AudioFrame;
//!
//! // From f32 — zero-copy
//! let f32_data = vec![0.0f32; 160];
//! let frame = AudioFrame::new(&f32_data, 16000);
//!
//! // From i16 — converts automatically
//! let i16_data = vec![0i16; 160];
//! let frame = AudioFrame::new(&i16_data, 16000);
//!
//! assert_eq!(frame.sample_rate(), 16000);
//! assert_eq!(frame.len(), 160);
//! ```

mod audio;
mod error;

pub use audio::{AudioFrame, IntoSamples};
pub use error::CoreError;
