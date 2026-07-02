//! Telephony / streaming audio codecs.
//!
//! Each codec lives in a submodule so the public surface stays
//! deliberately granular: a consumer that only needs G.711 imports
//! `wavekat_core::codec::g711`. Future additions (iLBC, …) live beside
//! them and stay independently selectable.
//!
//! Opus is gated behind the `opus` cargo feature because it pulls a C
//! build dependency (vendored `libopus` via `audiopus_sys`); the crate
//! stays pure-Rust for consumers that don't opt in.

pub mod g711;
#[cfg(feature = "opus")]
pub mod opus;
