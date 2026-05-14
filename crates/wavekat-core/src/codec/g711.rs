//! G.711 μ-law (PCMU) and A-law (PCMA) codecs.
//!
//! All four functions are byte-for-byte conversions: one 16-bit PCM
//! sample ↔ one 8-bit codeword. A 20 ms RTP frame at 8 kHz is therefore
//! 160 samples / 160 bytes — no length surprises.
//!
//! The tables follow ITU-T G.711; see
//! <https://www.itu.int/rec/T-REC-G.711> for the recommendation and
//! <https://en.wikipedia.org/wiki/G.711> for a readable summary.
//! Implementations are cross-checked against the reference vectors in
//! Sun Microsystems' `g711.c` and SpanDSP's reference.
//!
//! G.711 lives in `wavekat-core` (not `wavekat-sip`) because codecs are
//! a consumer-layer choice — `wavekat-sip` deliberately stays
//! codec-agnostic; SDP advertises both PCMU and PCMA and the consumer
//! picks one after answering.

/// SDP / RTP static payload type for μ-law (G.711U).
pub const PCMU_PAYLOAD_TYPE: u8 = 0;
/// SDP / RTP static payload type for A-law (G.711A).
pub const PCMA_PAYLOAD_TYPE: u8 = 8;

/// Sample rate of every static G.711 stream. The wire format does not
/// carry the rate; both endpoints just know.
pub const G711_SAMPLE_RATE: u32 = 8000;
/// Samples in a 20 ms G.711 frame (the standard RTP packetization
/// interval).
pub const G711_FRAME_SAMPLES: usize = 160;

const CLIP: i32 = 32635;
const BIAS: i32 = 0x84;
const SIGN_BIT: u8 = 0x80;
const QUANT_MASK: u8 = 0x0F;
const SEG_SHIFT: u8 = 4;
const SEG_MASK: u8 = 0x70;

// G.711 segment index for `pcm` in `[0, 0x7FFF]`. The segment is the
// position of the highest set bit above bit 7, clamped to 0 for
// `pcm < 0x100`. Callers in this file bound their inputs to `≤ 0x7FFF`
// (μ-law clips to CLIP+BIAS=0x7FFF; A-law masks with 0x7FFF), so we
// pick the bit-math form that needs no out-of-range fallback.
#[inline]
fn seg_for(pcm: i32) -> u32 {
    if pcm < 0x100 {
        0
    } else {
        31 - (pcm as u32).leading_zeros() - 7
    }
}

/// Encode one 16-bit PCM sample to a μ-law byte (G.711U).
pub fn linear_to_ulaw(pcm: i16) -> u8 {
    let mut pcm = pcm as i32;
    let sign = if pcm < 0 {
        pcm = -pcm;
        0x7F
    } else {
        0xFF
    };
    if pcm > CLIP {
        pcm = CLIP;
    }
    pcm += BIAS;

    let seg = seg_for(pcm);
    let mantissa = ((pcm >> (seg + 3)) & 0x0F) as u8;
    let coded = ((seg as u8) << 4) | mantissa;
    coded ^ sign
}

/// Decode one μ-law byte to a 16-bit PCM sample.
pub fn ulaw_to_linear(ulaw: u8) -> i16 {
    let ulaw = !ulaw;
    let sign = (ulaw & SIGN_BIT) != 0;
    let exponent = (ulaw & SEG_MASK) >> SEG_SHIFT;
    let mantissa = ulaw & QUANT_MASK;
    let mut sample = (((mantissa as i32) << 3) + BIAS) << exponent;
    sample -= BIAS;
    if sign {
        -sample as i16
    } else {
        sample as i16
    }
}

/// Encode one 16-bit PCM sample to an A-law byte (G.711A).
pub fn linear_to_alaw(pcm: i16) -> u8 {
    let (pcm, mask) = if pcm >= 0 {
        (pcm as i32, 0xD5u8)
    } else {
        (((!pcm) as i32) & 0x7FFF, 0x55u8)
    };

    let seg = seg_for(pcm);
    let mantissa = if seg < 1 {
        ((pcm >> 4) & 0x0F) as u8
    } else {
        ((pcm >> (seg + 3)) & 0x0F) as u8
    };
    let coded = ((seg as u8) << 4) | mantissa;
    coded ^ mask
}

/// Decode one A-law byte to a 16-bit PCM sample.
///
/// A-law's sign-bit convention is opposite to μ-law's: after XOR with
/// `0x55`, sign bit set means *positive* (see ITU-T G.711 §2.3, or
/// SpanDSP's reference implementation).
pub fn alaw_to_linear(alaw: u8) -> i16 {
    let alaw = alaw ^ 0x55;
    let sign_set = (alaw & SIGN_BIT) != 0;
    let exponent = (alaw & SEG_MASK) >> SEG_SHIFT;
    let mantissa = alaw & QUANT_MASK;
    let mut sample = ((mantissa as i32) << 4) + 8;
    if exponent != 0 {
        sample = (sample + 0x100) << (exponent - 1);
    }
    if sign_set {
        sample as i16
    } else {
        -sample as i16
    }
}

/// Codec selection for a session. The wire payload-type number
/// (`0`/`8`) is the canonical identifier; this enum is the typed
/// version we pass around in code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum G711Codec {
    /// μ-law (G.711U) — North America / Japan default, RTP payload type `0`.
    Pcmu,
    /// A-law (G.711A) — Europe / rest-of-world default, RTP payload type `8`.
    Pcma,
}

impl G711Codec {
    /// The static RTP payload-type number for this codec — `0` for PCMU,
    /// `8` for PCMA (RFC 3551 §6).
    pub fn payload_type(self) -> u8 {
        match self {
            G711Codec::Pcmu => PCMU_PAYLOAD_TYPE,
            G711Codec::Pcma => PCMA_PAYLOAD_TYPE,
        }
    }

    /// Resolve from a SIP/RTP payload type number. Returns `None` for
    /// any non-G.711 payload type — the caller decides whether to fall
    /// through (e.g. accept anyway, ask for re-INVITE, reject).
    pub fn from_payload_type(pt: u8) -> Option<Self> {
        match pt {
            PCMU_PAYLOAD_TYPE => Some(G711Codec::Pcmu),
            PCMA_PAYLOAD_TYPE => Some(G711Codec::Pcma),
            _ => None,
        }
    }

    /// Encode a slice of 16-bit PCM samples into G.711 bytes, one byte
    /// per sample. Appends to `out`.
    pub fn encode(self, pcm: &[i16], out: &mut Vec<u8>) {
        out.reserve(pcm.len());
        match self {
            G711Codec::Pcmu => out.extend(pcm.iter().map(|&s| linear_to_ulaw(s))),
            G711Codec::Pcma => out.extend(pcm.iter().map(|&s| linear_to_alaw(s))),
        }
    }

    /// Decode G.711 bytes into 16-bit PCM samples, one sample per byte.
    /// Appends to `out`.
    pub fn decode(self, encoded: &[u8], out: &mut Vec<i16>) {
        out.reserve(encoded.len());
        match self {
            G711Codec::Pcmu => out.extend(encoded.iter().map(|&b| ulaw_to_linear(b))),
            G711Codec::Pcma => out.extend(encoded.iter().map(|&b| alaw_to_linear(b))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulaw_round_trip_silence() {
        assert_eq!(linear_to_ulaw(0), 0xFF);
        // μ-law silence (0xFF) decodes to a small non-zero residue — the
        // codec is not loss-free near zero. The residue should round back
        // to 0xFF on re-encode, which is the property that matters for
        // end-to-end stability.
        let s = ulaw_to_linear(0xFF);
        assert_eq!(linear_to_ulaw(s), 0xFF);
    }

    #[test]
    fn alaw_round_trip_silence() {
        let encoded = linear_to_alaw(0);
        let s = alaw_to_linear(encoded);
        assert_eq!(linear_to_alaw(s), encoded);
    }

    #[test]
    fn ulaw_handles_full_scale() {
        assert_eq!(linear_to_ulaw(i16::MAX), 0x80);
        assert_eq!(linear_to_ulaw(i16::MIN), 0x00);
    }

    #[test]
    fn alaw_handles_full_scale() {
        assert_eq!(linear_to_alaw(i16::MAX), 0xD5 ^ 0x7F);
        assert_eq!(linear_to_alaw(i16::MIN), 0x55 ^ 0x7F);
    }

    #[test]
    fn codec_encode_decode_length_matches_samples() {
        let pcm: Vec<i16> = (0..160).map(|i| (i * 200) as i16).collect();
        let mut encoded = Vec::new();
        G711Codec::Pcmu.encode(&pcm, &mut encoded);
        assert_eq!(encoded.len(), pcm.len());
        let mut decoded = Vec::new();
        G711Codec::Pcmu.decode(&encoded, &mut decoded);
        assert_eq!(decoded.len(), encoded.len());
    }

    #[test]
    fn payload_type_round_trips() {
        assert_eq!(G711Codec::from_payload_type(0), Some(G711Codec::Pcmu));
        assert_eq!(G711Codec::from_payload_type(8), Some(G711Codec::Pcma));
        assert_eq!(G711Codec::from_payload_type(127), None);
        assert_eq!(G711Codec::Pcmu.payload_type(), 0);
        assert_eq!(G711Codec::Pcma.payload_type(), 8);
    }

    #[test]
    fn ulaw_round_trip_preserves_loud_samples_within_codec_step() {
        let inputs: &[i16] = &[1000, -1000, 8000, -8000, 16000, -16000];
        for &s in inputs {
            let encoded = linear_to_ulaw(s);
            let decoded = ulaw_to_linear(encoded);
            let diff = (s as i32 - decoded as i32).abs();
            assert!(
                diff < 400,
                "μ-law round-trip drift too large: {s} → {decoded} (Δ={diff})"
            );
        }
    }

    #[test]
    fn alaw_round_trip_preserves_loud_samples_within_codec_step() {
        let inputs: &[i16] = &[1000, -1000, 8000, -8000, 16000, -16000];
        for &s in inputs {
            let encoded = linear_to_alaw(s);
            let decoded = alaw_to_linear(encoded);
            let diff = (s as i32 - decoded as i32).abs();
            assert!(
                diff < 400,
                "A-law round-trip drift too large: {s} → {decoded} (Δ={diff})"
            );
        }
    }

    #[test]
    fn ulaw_decode_is_a_fixed_point_for_every_codeword() {
        // The right invariant: a decoded sample is the canonical form
        // of its codeword, so decode(encode(decode(b))) must equal
        // decode(b) for every codeword. The weaker variant
        // (encode→decode→encode == encode for every i16) fails near
        // zero because μ-law has separate +0/-0 codewords that
        // collapse to the same decoded sample; that's a property of
        // the codec, not a bug.
        for b in 0u8..=255 {
            let mid = ulaw_to_linear(b);
            let again = ulaw_to_linear(linear_to_ulaw(mid));
            assert_eq!(again, mid, "μ-law decode not fixed-point at {b:#x}");
        }
    }

    #[test]
    fn alaw_decode_is_a_fixed_point_for_every_codeword() {
        for b in 0u8..=255 {
            let mid = alaw_to_linear(b);
            let again = alaw_to_linear(linear_to_alaw(mid));
            assert_eq!(again, mid, "A-law decode not fixed-point at {b:#x}");
        }
    }

    #[test]
    fn ulaw_decode_covers_full_codeword_space_without_panic() {
        // 256 possible codewords. Decoding all of them must not panic
        // and must stay in i16 range.
        for b in 0u8..=255 {
            let _ = ulaw_to_linear(b);
        }
    }

    #[test]
    fn alaw_decode_covers_full_codeword_space_without_panic() {
        for b in 0u8..=255 {
            let _ = alaw_to_linear(b);
        }
    }

    #[test]
    fn ulaw_zero_is_distinct_from_full_scale() {
        // Sanity: a non-trivial codec maps 0 and i16::MAX to different
        // bytes. Guards against a stub impl that returns a constant.
        assert_ne!(linear_to_ulaw(0), linear_to_ulaw(i16::MAX));
        assert_ne!(linear_to_ulaw(0), linear_to_ulaw(i16::MIN));
    }

    #[test]
    fn alaw_zero_is_distinct_from_full_scale() {
        assert_ne!(linear_to_alaw(0), linear_to_alaw(i16::MAX));
        assert_ne!(linear_to_alaw(0), linear_to_alaw(i16::MIN));
    }

    #[test]
    fn pcmu_and_pcma_produce_different_bytes_for_the_same_input() {
        // Guards against accidentally aliasing the two paths (e.g. a
        // typo wiring Pcma to linear_to_ulaw). PCMU and PCMA share the
        // shape but pick different quantisation tables and silence
        // codewords; they should not match on a non-trivial sample.
        let s = 12345i16;
        assert_ne!(linear_to_ulaw(s), linear_to_alaw(s));
    }

    #[test]
    fn codec_enum_dispatches_to_the_right_path() {
        // Crossing the enum boundary must end up at the matching
        // function — not swapped, not aliased.
        let pcm = vec![1000i16, -2000, 3000];

        let mut a = Vec::new();
        G711Codec::Pcmu.encode(&pcm, &mut a);
        let mut b = Vec::new();
        for &s in &pcm {
            b.push(linear_to_ulaw(s));
        }
        assert_eq!(a, b);

        let mut c = Vec::new();
        G711Codec::Pcma.encode(&pcm, &mut c);
        let mut d = Vec::new();
        for &s in &pcm {
            d.push(linear_to_alaw(s));
        }
        assert_eq!(c, d);
    }

    #[test]
    fn slice_encode_then_decode_recovers_signal_within_codec_drift() {
        // Twenty-millisecond G.711 frame of a small sine — encode,
        // decode, and compare against the input. The codec is lossy
        // (log-PCM quantisation), so we allow a per-sample drift, but
        // the average error must be small for an "audible" path.
        let samples: Vec<i16> = (0..G711_FRAME_SAMPLES)
            .map(|i| {
                let t = i as f32 / G711_SAMPLE_RATE as f32;
                ((t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 8000.0) as i16
            })
            .collect();

        for codec in [G711Codec::Pcmu, G711Codec::Pcma] {
            let mut encoded = Vec::new();
            codec.encode(&samples, &mut encoded);
            assert_eq!(encoded.len(), G711_FRAME_SAMPLES);

            let mut decoded = Vec::new();
            codec.decode(&encoded, &mut decoded);
            assert_eq!(decoded.len(), G711_FRAME_SAMPLES);

            let mean_abs_error: f64 = samples
                .iter()
                .zip(decoded.iter())
                .map(|(a, b)| (*a as i32 - *b as i32).abs() as f64)
                .sum::<f64>()
                / samples.len() as f64;
            // 200 i16 units ≈ 0.6% of full scale — comfortably below
            // perceptible degradation for telephony.
            assert!(
                mean_abs_error < 200.0,
                "{codec:?}: mean abs error {mean_abs_error} too high"
            );
        }
    }

    #[test]
    fn encode_appends_rather_than_replacing() {
        // The slice-level encode/decode take `&mut Vec<…>` and append.
        // Verifying that explicitly so callers can reuse buffers
        // across RTP packets without per-packet alloc.
        let mut buf = vec![0xFFu8, 0xFEu8];
        let pcm = vec![0i16; 3];
        G711Codec::Pcmu.encode(&pcm, &mut buf);
        assert_eq!(buf.len(), 5);
        assert_eq!(&buf[..2], &[0xFF, 0xFE]);
    }

    #[test]
    fn payload_type_constants_match_rfc3551() {
        // RFC 3551 §6 pins PCMU=0 and PCMA=8. Hard-coding these
        // numbers in tests protects against a casual rename that would
        // silently break SDP negotiation against any real PBX.
        assert_eq!(PCMU_PAYLOAD_TYPE, 0);
        assert_eq!(PCMA_PAYLOAD_TYPE, 8);
        assert_eq!(G711_SAMPLE_RATE, 8000);
        assert_eq!(G711_FRAME_SAMPLES, 160);
    }
}
