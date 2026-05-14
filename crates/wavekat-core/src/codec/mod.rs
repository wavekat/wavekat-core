//! Telephony / streaming audio codecs.
//!
//! Each codec lives in a submodule so the public surface stays
//! deliberately granular: a consumer that only needs G.711 imports
//! `wavekat_core::codec::g711`. Future additions (Opus, iLBC, …) live
//! beside it and stay independently selectable.

pub mod g711;
