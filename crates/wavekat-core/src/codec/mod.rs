//! Telephony / streaming audio codecs.
//!
//! Each codec lives in a submodule so the public surface stays
//! deliberately granular: a consumer that only needs G.711 imports
//! `wavekat_core::codec::g711`, one that wants wideband on a plain SIP
//! line imports `wavekat_core::codec::g722`. Future additions (iLBC, …)
//! live beside them and stay independently selectable.
//!
//! Opus is gated behind the `opus` cargo feature because it pulls a C
//! build dependency (vendored `libopus` via `audiopus_sys`); the crate
//! stays pure-Rust for consumers that don't opt in. The feature also
//! unlocks [`Encoder`] / [`Decoder`] — the codec-agnostic seam for
//! consumers that negotiate their codec per call.

pub mod g711;
pub mod g722;
#[cfg(feature = "opus")]
pub mod opus;

#[cfg(feature = "opus")]
pub use seam::{Decoder, Encoder};

/// The per-call codec seam: one enum per direction, dispatching to the
/// stateless G.711 tables or a stateful G.722 or Opus coder. Split from the
/// submodules so `g711`/`opus` stay usable standalone.
#[cfg(feature = "opus")]
mod seam {
    use super::g711::{G711Codec, G711_FRAME_SAMPLES, G711_SAMPLE_RATE};
    use super::g722::{G722Decoder, G722Encoder, G722_FRAME_SAMPLES, G722_PCM_SAMPLE_RATE};
    use super::opus::{OpusDecoder, OpusEncoder, OPUS_FRAME_SAMPLES, OPUS_PCM_SAMPLE_RATE};
    use crate::CoreError;

    /// The encode half of a negotiated call codec.
    ///
    /// G.711 is a stateless table lookup (the enum just carries the
    /// μ-law/A-law choice); G.722 keeps its predictor state inline;
    /// Opus owns a heap-backed encoder with inter-frame state. In every
    /// case: feed one 20 ms frame at
    /// [`pcm_sample_rate`](Self::pcm_sample_rate) per call, in order.
    // G.722's predictor state is a few hundred bytes held inline; one
    // coder lives for a whole call, so boxing it would buy nothing.
    #[allow(clippy::large_enum_variant)]
    pub enum Encoder {
        /// PCMU or PCMA.
        G711(G711Codec),
        /// G.722, wideband at 64 kbit/s.
        G722(G722Encoder),
        /// Opus, wideband voice profile.
        Opus(OpusEncoder),
    }

    impl Encoder {
        /// The PCM rate this encoder consumes: 8 kHz for G.711, 16 kHz
        /// for G.722 and Opus. Resample the microphone to this, not to a global
        /// constant.
        pub fn pcm_sample_rate(&self) -> u32 {
            match self {
                Encoder::G711(_) => G711_SAMPLE_RATE,
                Encoder::G722(_) => G722_PCM_SAMPLE_RATE,
                Encoder::Opus(_) => OPUS_PCM_SAMPLE_RATE,
            }
        }

        /// Samples in one 20 ms frame at [`pcm_sample_rate`](Self::pcm_sample_rate):
        /// 160 for G.711, 320 for G.722 and Opus.
        pub fn frame_samples(&self) -> usize {
            match self {
                Encoder::G711(_) => G711_FRAME_SAMPLES,
                Encoder::G722(_) => G722_FRAME_SAMPLES,
                Encoder::Opus(_) => OPUS_FRAME_SAMPLES,
            }
        }

        /// Encode one 20 ms frame, appending the payload bytes to `out`
        /// and returning the payload length. G.711 cannot fail; G.722
        /// refuses an odd sample count; Opus errors surface (wrong
        /// frame length, libopus failure).
        pub fn encode(&mut self, pcm: &[i16], out: &mut Vec<u8>) -> Result<usize, CoreError> {
            match self {
                Encoder::G711(codec) => {
                    codec.encode(pcm, out);
                    Ok(pcm.len())
                }
                Encoder::G722(enc) => enc.encode(pcm, out),
                Encoder::Opus(enc) => enc.encode(pcm, out),
            }
        }
    }

    /// The decode half of a negotiated call codec.
    ///
    /// The loss-recovery calls are uniform so a receive loop can drive
    /// them without codec branches: for G.711 and G.722 they are no-ops
    /// (append nothing, return 0) because neither carries redundancy and
    /// the historical loss behavior — silence by absence — is the
    /// correct one to keep.
    // G.722's predictor state is a few hundred bytes held inline; one
    // coder lives for a whole call, so boxing it would buy nothing.
    #[allow(clippy::large_enum_variant)]
    pub enum Decoder {
        /// PCMU or PCMA.
        G711(G711Codec),
        /// G.722, wideband at 64 kbit/s.
        G722(G722Decoder),
        /// Opus, with in-band FEC recovery and PLC.
        Opus(OpusDecoder),
    }

    impl Decoder {
        /// The PCM rate decoded frames come out at: 8 kHz for G.711,
        /// 16 kHz for G.722 and Opus.
        pub fn pcm_sample_rate(&self) -> u32 {
            match self {
                Decoder::G711(_) => G711_SAMPLE_RATE,
                Decoder::G722(_) => G722_PCM_SAMPLE_RATE,
                Decoder::Opus(_) => OPUS_PCM_SAMPLE_RATE,
            }
        }

        /// Decode one received payload, appending PCM to `out` and
        /// returning the sample count. Use the returned length — Opus
        /// peers choose their own frame size.
        pub fn decode(&mut self, payload: &[u8], out: &mut Vec<i16>) -> Result<usize, CoreError> {
            match self {
                Decoder::G711(codec) => {
                    codec.decode(payload, out);
                    Ok(payload.len())
                }
                Decoder::G722(dec) => Ok(dec.decode(payload, out)),
                Decoder::Opus(dec) => dec.decode(payload, out),
            }
        }

        /// Recover a single lost frame from the packet that followed it
        /// (drive this on a one-packet RTP sequence gap). Opus decodes
        /// the next packet's in-band FEC; G.711 has nothing to recover
        /// and appends nothing.
        pub fn recover_lost(
            &mut self,
            next_payload: &[u8],
            out: &mut Vec<i16>,
        ) -> Result<usize, CoreError> {
            match self {
                Decoder::G711(_) | Decoder::G722(_) => Ok(0),
                Decoder::Opus(dec) => dec.decode_fec(next_payload, out),
            }
        }

        /// Produce one concealment frame for a packet that is not
        /// coming back (multi-packet gap). Opus extrapolates via PLC;
        /// G.711 appends nothing — absence already sounds like silence.
        pub fn conceal(&mut self, out: &mut Vec<i16>) -> Result<usize, CoreError> {
            match self {
                Decoder::G711(_) | Decoder::G722(_) => Ok(0),
                Decoder::Opus(dec) => dec.conceal(out),
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn tone(n: usize, rate: u32) -> Vec<i16> {
            (0..n)
                .map(|i| {
                    let t = i as f32 / rate as f32;
                    ((t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 8000.0) as i16
                })
                .collect()
        }

        #[test]
        fn g711_seam_matches_direct_codec() {
            // The enum must dispatch to the same tables as calling
            // G711Codec directly — not a re-implementation.
            let pcm = tone(G711_FRAME_SAMPLES, G711_SAMPLE_RATE);
            let mut via_seam = Vec::new();
            Encoder::G711(G711Codec::Pcmu)
                .encode(&pcm, &mut via_seam)
                .unwrap();
            let mut direct = Vec::new();
            G711Codec::Pcmu.encode(&pcm, &mut direct);
            assert_eq!(via_seam, direct);

            let mut decoded = Vec::new();
            let n = Decoder::G711(G711Codec::Pcmu)
                .decode(&via_seam, &mut decoded)
                .unwrap();
            assert_eq!(n, G711_FRAME_SAMPLES);
        }

        #[test]
        fn seam_reports_per_codec_rates_and_frames() {
            let g711 = Encoder::G711(G711Codec::Pcma);
            assert_eq!(g711.pcm_sample_rate(), 8_000);
            assert_eq!(g711.frame_samples(), 160);
            let opus = Encoder::Opus(OpusEncoder::new().unwrap());
            assert_eq!(opus.pcm_sample_rate(), 16_000);
            assert_eq!(opus.frame_samples(), 320);
            assert_eq!(Decoder::G711(G711Codec::Pcmu).pcm_sample_rate(), 8_000);
            assert_eq!(
                Decoder::Opus(OpusDecoder::new().unwrap()).pcm_sample_rate(),
                16_000
            );
        }

        #[test]
        fn g722_seam_matches_direct_codec() {
            let pcm = tone(G722_FRAME_SAMPLES, G722_PCM_SAMPLE_RATE);
            let mut enc = Encoder::G722(G722Encoder::new());
            assert_eq!(enc.pcm_sample_rate(), 16_000);
            assert_eq!(enc.frame_samples(), 320);
            let mut via_seam = Vec::new();
            assert_eq!(enc.encode(&pcm, &mut via_seam).unwrap(), 160);
            let mut direct = Vec::new();
            G722Encoder::new().encode(&pcm, &mut direct).unwrap();
            assert_eq!(via_seam, direct);
            assert!(enc.encode(&pcm[..319], &mut via_seam).is_err());

            let mut dec = Decoder::G722(G722Decoder::new());
            assert_eq!(dec.pcm_sample_rate(), 16_000);
            let mut out = Vec::new();
            assert_eq!(dec.decode(&direct, &mut out).unwrap(), G722_FRAME_SAMPLES);
            // No redundancy to recover from: loss stays silence.
            assert_eq!(dec.recover_lost(&direct, &mut out).unwrap(), 0);
            assert_eq!(dec.conceal(&mut out).unwrap(), 0);
            assert_eq!(out.len(), G722_FRAME_SAMPLES);
        }

        #[test]
        fn opus_seam_round_trips() {
            let mut enc = Encoder::Opus(OpusEncoder::new().unwrap());
            let mut dec = Decoder::Opus(OpusDecoder::new().unwrap());
            let pcm = tone(OPUS_FRAME_SAMPLES, OPUS_PCM_SAMPLE_RATE);
            let mut packet = Vec::new();
            enc.encode(&pcm, &mut packet).unwrap();
            let mut out = Vec::new();
            let n = dec.decode(&packet, &mut out).unwrap();
            assert_eq!(n, OPUS_FRAME_SAMPLES);
        }

        #[test]
        fn g711_recovery_is_a_noop_by_design() {
            // The receive loop drives recover_lost/conceal without
            // codec branches; for G.711 both must keep today's
            // behavior — nothing inserted, loss stays silence.
            let mut dec = Decoder::G711(G711Codec::Pcmu);
            let mut out = vec![1i16, 2, 3];
            assert_eq!(dec.recover_lost(&[0xFF; 20], &mut out).unwrap(), 0);
            assert_eq!(dec.conceal(&mut out).unwrap(), 0);
            assert_eq!(out, vec![1, 2, 3], "G.711 recovery must not touch out");
        }

        #[test]
        fn opus_seam_recovers_and_conceals() {
            let mut enc = Encoder::Opus(OpusEncoder::new().unwrap());
            let mut dec = Decoder::Opus(OpusDecoder::new().unwrap());
            let mut packets = Vec::new();
            for i in 0..4 {
                let pcm: Vec<i16> = (0..OPUS_FRAME_SAMPLES)
                    .map(|j| {
                        let t = (i * OPUS_FRAME_SAMPLES + j) as f32 / OPUS_PCM_SAMPLE_RATE as f32;
                        ((t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 8000.0) as i16
                    })
                    .collect();
                let mut packet = Vec::new();
                enc.encode(&pcm, &mut packet).unwrap();
                packets.push(packet);
            }
            let mut out = Vec::new();
            dec.decode(&packets[0], &mut out).unwrap();
            out.clear();
            dec.decode(&packets[1], &mut out).unwrap();
            // Packet 2 "lost": recover from packet 3's FEC, then conceal
            // one more as if a second gap followed.
            out.clear();
            assert_eq!(
                dec.recover_lost(&packets[3], &mut out).unwrap(),
                OPUS_FRAME_SAMPLES
            );
            assert_eq!(dec.conceal(&mut out).unwrap(), OPUS_FRAME_SAMPLES);
            assert_eq!(out.len(), 2 * OPUS_FRAME_SAMPLES);
        }
    }
}
