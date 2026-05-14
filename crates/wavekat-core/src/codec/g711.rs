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

#[inline]
fn seg_for(pcm: i32, seg_end: &[i32; 8]) -> usize {
    for (i, &end) in seg_end.iter().enumerate() {
        if pcm <= end {
            return i;
        }
    }
    seg_end.len()
}

/// Encode one 16-bit PCM sample to a μ-law byte (G.711U).
pub fn linear_to_ulaw(pcm: i16) -> u8 {
    const SEG_END: [i32; 8] = [0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF];

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

    let seg = seg_for(pcm, &SEG_END);
    if seg >= 8 {
        0x7F ^ sign
    } else {
        let mantissa = ((pcm >> (seg + 3)) & 0x0F) as u8;
        let coded = ((seg as u8) << 4) | mantissa;
        coded ^ sign
    }
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
    const SEG_END: [i32; 8] = [0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF];

    let (pcm, mask) = if pcm >= 0 {
        (pcm as i32, 0xD5u8)
    } else {
        (((!pcm) as i32) & 0x7FFF, 0x55u8)
    };

    let seg = seg_for(pcm, &SEG_END);
    if seg >= 8 {
        0x7F ^ mask
    } else {
        let mantissa = if seg < 1 {
            ((pcm >> 4) & 0x0F) as u8
        } else {
            ((pcm >> (seg + 3)) & 0x0F) as u8
        };
        let coded = ((seg as u8) << 4) | mantissa;
        coded ^ mask
    }
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
    Pcmu,
    Pcma,
}

impl G711Codec {
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
}
