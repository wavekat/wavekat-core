//! G.722 wideband codec (64 kbit/s, mode 1).
//!
//! G.722 carries 16 kHz speech (50 Hz – 7 kHz) in the same 64 kbit/s as
//! G.711: the signal is split into a low and a high band by a 24-tap
//! quadrature mirror filter, and each band is coded with ADPCM (6 bits
//! for the low band, 2 for the high band). Two PCM samples in, one byte
//! out, so a 20 ms RTP frame is 320 samples / 160 bytes.
//!
//! It is the wideband codec most SIP providers and desk phones speak,
//! which makes it the "HD voice" option on lines that refuse Opus.
//!
//! One historical trap: RFC 3551 gives G.722 an RTP clock rate of
//! **8000** although the audio is sampled at 16 kHz, so the RTP
//! timestamp advances by 160 per 20 ms frame, not 320. See
//! [`G722_RTP_CLOCK_RATE`] and [`G722_RTP_SAMPLES_PER_FRAME`].
//!
//! The tables and block structure follow ITU-T G.722
//! (<https://www.itu.int/rec/T-REC-G.722>); the arithmetic follows the
//! widely used public-domain reference lineage (CMU / SpanDSP). Only
//! the 64 kbit/s mode is implemented: it is the only one RTP uses.
//!
//! Unlike G.711 the coder is stateful (adaptive predictors and filter
//! history), so keep one [`G722Encoder`] / [`G722Decoder`] per stream
//! and feed frames in order.

use crate::CoreError;

/// SDP / RTP static payload type for G.722.
pub const G722_PAYLOAD_TYPE: u8 = 9;
/// The PCM rate the encoder consumes and the decoder produces.
pub const G722_PCM_SAMPLE_RATE: u32 = 16_000;
/// The RTP clock rate for G.722. It is 8000, not 16000: an erratum in
/// the original RTP profile that RFC 3551 §4.5.2 keeps for
/// compatibility. SDP says `G722/8000`.
pub const G722_RTP_CLOCK_RATE: u32 = 8_000;
/// RTP timestamp units in one 20 ms frame (at the 8 kHz RTP clock).
pub const G722_RTP_SAMPLES_PER_FRAME: u32 = 160;
/// PCM samples in a 20 ms frame at 16 kHz.
pub const G722_FRAME_SAMPLES: usize = 320;
/// Payload bytes in a 20 ms frame.
pub const G722_FRAME_BYTES: usize = 160;

/// Transmit / receive QMF coefficients (half of the symmetric 24 taps).
const QMF: [i32; 12] = [3, -11, 12, 32, -210, 951, 3876, -805, 362, -156, 53, -11];

/// Low-band quantizer decision levels.
const Q6: [i32; 32] = [
    0, 35, 72, 110, 150, 190, 233, 276, 323, 370, 422, 473, 530, 587, 650, 714, 786, 858, 940,
    1023, 1121, 1219, 1339, 1458, 1612, 1765, 1980, 2195, 2557, 2919, 0, 0,
];
/// Low-band codewords for a negative / positive difference.
const ILN: [i32; 32] = [
    0, 63, 62, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
    10, 9, 8, 7, 6, 5, 4, 0,
];
const ILP: [i32; 32] = [
    0, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39,
    38, 37, 36, 35, 34, 33, 32, 0,
];
/// Low-band log scale factor adaptation.
const WL: [i32; 8] = [-60, -30, 58, 172, 334, 538, 1198, 3042];
const RL42: [usize; 16] = [0, 7, 6, 5, 4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 1, 0];
/// Log-to-linear scale factor table.
const ILB: [i32; 32] = [
    2048, 2093, 2139, 2186, 2233, 2282, 2332, 2383, 2435, 2489, 2543, 2599, 2656, 2714, 2774, 2834,
    2896, 2960, 3025, 3091, 3158, 3228, 3298, 3371, 3444, 3520, 3597, 3676, 3756, 3838, 3922, 4008,
];
/// Low-band inverse quantizers: 4 bits (predictor) and 6 bits (output).
const QM4: [i32; 16] = [
    0, -20456, -12896, -8968, -6288, -4240, -2584, -1200, 20456, 12896, 8968, 6288, 4240, 2584,
    1200, 0,
];
const QM6: [i32; 64] = [
    -136, -136, -136, -136, -24808, -21904, -19008, -16704, -14984, -13512, -12280, -11192, -10232,
    -9360, -8576, -7856, -7192, -6576, -6000, -5456, -4944, -4464, -4008, -3576, -3168, -2776,
    -2400, -2032, -1688, -1360, -1040, -728, 24808, 21904, 19008, 16704, 14984, 13512, 12280,
    11192, 10232, 9360, 8576, 7856, 7192, 6576, 6000, 5456, 4944, 4464, 4008, 3576, 3168, 2776,
    2400, 2032, 1688, 1360, 1040, 728, 432, 136, -432, -136,
];
/// High-band tables.
const QM2: [i32; 4] = [-7408, -1616, 7408, 1616];
const IHN: [i32; 3] = [0, 1, 0];
const IHP: [i32; 3] = [0, 3, 2];
const WH: [i32; 3] = [0, -214, 798];
const RH2: [usize; 4] = [2, 1, 2, 1];

#[inline]
fn sat(x: i32) -> i32 {
    x.clamp(i16::MIN as i32, i16::MAX as i32)
}

/// The adaptive predictor state of one sub-band (shared by the encoder
/// and the decoder, which must track each other exactly).
#[derive(Clone, Debug)]
struct Band {
    s: i32,
    sp: i32,
    sz: i32,
    r: [i32; 3],
    a: [i32; 3],
    ap: [i32; 3],
    p: [i32; 3],
    d: [i32; 7],
    b: [i32; 7],
    bp: [i32; 7],
    nb: i32,
    det: i32,
}

impl Band {
    fn new(det: i32) -> Self {
        Self {
            s: 0,
            sp: 0,
            sz: 0,
            r: [0; 3],
            a: [0; 3],
            ap: [0; 3],
            p: [0; 3],
            d: [0; 7],
            b: [0; 7],
            bp: [0; 7],
            nb: 0,
            det,
        }
    }

    /// Update the log scale factor from an adaptation step and derive
    /// the linear one. `limit` and `shift` differ per band (LOGSCL +
    /// SCALEL, LOGSCH + SCALEH).
    fn scale(&mut self, step: i32, limit: i32, shift: i32) {
        self.nb = (((self.nb * 127) >> 7) + step).clamp(0, limit);
        let wd1 = ((self.nb >> 6) & 31) as usize;
        let wd2 = shift - (self.nb >> 11);
        let wd3 = if wd2 < 0 {
            ILB[wd1] << -wd2
        } else {
            ILB[wd1] >> wd2
        };
        self.det = wd3 << 2;
    }

    /// Block 4 of the recommendation: reconstruct, adapt the pole and
    /// zero predictors, and form the next signal estimate.
    fn block4(&mut self, d: i32) {
        // RECONS, PARREC
        self.d[0] = d;
        self.r[0] = sat(self.s + d);
        self.p[0] = sat(self.sz + d);

        // UPPOL2
        let sg = [self.p[0] >> 15, self.p[1] >> 15, self.p[2] >> 15];
        let wd1 = sat(self.a[1] << 2);
        let wd2 = if sg[0] == sg[1] { -wd1 } else { wd1 }.min(32767);
        let mut wd3 = (wd2 >> 7) + if sg[0] == sg[2] { 128 } else { -128 };
        wd3 += (self.a[2] * 32512) >> 15;
        self.ap[2] = wd3.clamp(-12288, 12288);

        // UPPOL1
        let wd1 = if sg[0] == sg[1] { 192 } else { -192 };
        let wd2 = (self.a[1] * 32640) >> 15;
        let limit = sat(15360 - self.ap[2]);
        self.ap[1] = sat(wd1 + wd2).clamp(-limit, limit);

        // UPZERO
        let wd1 = if d == 0 { 0 } else { 128 };
        let sg0 = d >> 15;
        for i in 1..7 {
            let wd2 = if (self.d[i] >> 15) == sg0 { wd1 } else { -wd1 };
            let wd3 = (self.b[i] * 32640) >> 15;
            self.bp[i] = sat(wd2 + wd3);
        }

        // DELAYA
        for i in (1..7).rev() {
            self.d[i] = self.d[i - 1];
            self.b[i] = self.bp[i];
        }
        for i in (1..3).rev() {
            self.r[i] = self.r[i - 1];
            self.p[i] = self.p[i - 1];
            self.a[i] = self.ap[i];
        }

        // FILTEP
        let wd1 = (self.a[1] * sat(self.r[1] + self.r[1])) >> 15;
        let wd2 = (self.a[2] * sat(self.r[2] + self.r[2])) >> 15;
        self.sp = sat(wd1 + wd2);

        // FILTEZ
        let mut sz = 0;
        for i in (1..7).rev() {
            sz += (self.b[i] * sat(self.d[i] + self.d[i])) >> 15;
        }
        self.sz = sat(sz);

        // PREDIC
        self.s = sat(self.sp + self.sz);
    }

    fn low() -> Self {
        Self::new(32)
    }

    fn high() -> Self {
        Self::new(8)
    }

    fn scale_low(&mut self, code4: usize) {
        self.scale(WL[RL42[code4]], 18432, 8);
    }

    fn scale_high(&mut self, code2: usize) {
        self.scale(WH[RH2[code2]], 22528, 10);
    }
}

/// Both halves of the QMF over the 24-sample history: the sum over the
/// even taps and the sum over the odd taps.
#[inline]
fn qmf_sums(x: &[i32; 24]) -> (i32, i32) {
    let mut even = 0;
    let mut odd = 0;
    for i in 0..12 {
        even += x[2 * i] * QMF[i];
        odd += x[2 * i + 1] * QMF[11 - i];
    }
    (even, odd)
}

/// G.722 encoder for one stream: 16 kHz PCM in, 64 kbit/s out.
#[derive(Clone, Debug)]
pub struct G722Encoder {
    low: Band,
    high: Band,
    x: [i32; 24],
}

impl Default for G722Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl G722Encoder {
    /// A fresh encoder, in the reset state the recommendation defines.
    pub fn new() -> Self {
        Self {
            low: Band::low(),
            high: Band::high(),
            x: [0; 24],
        }
    }

    /// Encode 16 kHz PCM, appending one byte per two samples to `out`
    /// and returning the number of bytes written. `pcm` must hold an
    /// even number of samples (a 20 ms frame is
    /// [`G722_FRAME_SAMPLES`]); an odd length is an error and encodes
    /// nothing.
    pub fn encode(&mut self, pcm: &[i16], out: &mut Vec<u8>) -> Result<usize, CoreError> {
        if pcm.len() & 1 != 0 {
            return Err(CoreError::Audio(format!(
                "G.722 codes samples in pairs, got an odd count ({})",
                pcm.len()
            )));
        }
        out.reserve(pcm.len() / 2);
        let mut samples = pcm.iter();
        while let (Some(&first), Some(&second)) = (samples.next(), samples.next()) {
            out.push(self.encode_pair(first, second));
        }
        Ok(pcm.len() / 2)
    }

    fn encode_pair(&mut self, first: i16, second: i16) -> u8 {
        // Transmit QMF: split into the two half-rate bands.
        self.x.copy_within(2.., 0);
        self.x[22] = first as i32;
        self.x[23] = second as i32;
        let (sum_odd, sum_even) = qmf_sums(&self.x);
        let xlow = (sum_even + sum_odd) >> 14;
        let xhigh = (sum_even - sum_odd) >> 14;

        // Low band: 6-bit quantizer on the prediction error.
        let el = sat(xlow - self.low.s);
        let wd = if el >= 0 { el } else { -(el + 1) };
        let mut i = 1;
        while i < 30 {
            if wd < (Q6[i] * self.low.det) >> 12 {
                break;
            }
            i += 1;
        }
        let ilow = if el < 0 { ILN[i] } else { ILP[i] };
        let code4 = (ilow >> 2) as usize;
        let dlow = (self.low.det * QM4[code4]) >> 15;
        self.low.scale_low(code4);
        self.low.block4(dlow);

        // High band: 2-bit quantizer.
        let eh = sat(xhigh - self.high.s);
        let wd = if eh >= 0 { eh } else { -(eh + 1) };
        let mih = if wd >= (564 * self.high.det) >> 12 {
            2
        } else {
            1
        };
        let ihigh = if eh < 0 { IHN[mih] } else { IHP[mih] };
        let dhigh = (self.high.det * QM2[ihigh as usize]) >> 15;
        self.high.scale_high(ihigh as usize);
        self.high.block4(dhigh);

        ((ihigh << 6) | ilow) as u8
    }
}

/// G.722 decoder for one stream: 64 kbit/s in, 16 kHz PCM out.
#[derive(Clone, Debug)]
pub struct G722Decoder {
    low: Band,
    high: Band,
    x: [i32; 24],
}

impl Default for G722Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl G722Decoder {
    /// A fresh decoder, in the reset state the recommendation defines.
    pub fn new() -> Self {
        Self {
            low: Band::low(),
            high: Band::high(),
            x: [0; 24],
        }
    }

    /// Decode a payload, appending two 16 kHz samples per byte to `out`
    /// and returning the number of samples written.
    pub fn decode(&mut self, payload: &[u8], out: &mut Vec<i16>) -> usize {
        out.reserve(payload.len() * 2);
        for &code in payload {
            let (first, second) = self.decode_byte(code);
            out.push(first);
            out.push(second);
        }
        payload.len() * 2
    }

    fn decode_byte(&mut self, code: u8) -> (i16, i16) {
        let ilow = (code & 0x3F) as usize;
        let ihigh = (code >> 6) as usize;

        // Low band: the 6-bit value reconstructs the output, the top
        // 4 bits drive the predictor (as in the encoder).
        let rlow = (self.low.s + ((self.low.det * QM6[ilow]) >> 15)).clamp(-16384, 16383);
        let code4 = ilow >> 2;
        let dlow = (self.low.det * QM4[code4]) >> 15;
        self.low.scale_low(code4);
        self.low.block4(dlow);

        // High band.
        let dhigh = (self.high.det * QM2[ihigh]) >> 15;
        let rhigh = (dhigh + self.high.s).clamp(-16384, 16383);
        self.high.scale_high(ihigh);
        self.high.block4(dhigh);

        // Receive QMF: recombine the bands at 16 kHz.
        self.x.copy_within(2.., 0);
        self.x[22] = rlow + rhigh;
        self.x[23] = rlow - rhigh;
        let (xout2, xout1) = qmf_sums(&self.x);
        (sat(xout1 >> 11) as i16, sat(xout2 >> 11) as i16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f32, n: usize, amp: f32) -> Vec<i16> {
        (0..n)
            .map(|i| {
                let t = i as f32 / G722_PCM_SAMPLE_RATE as f32;
                ((t * hz * 2.0 * std::f32::consts::PI).sin() * amp) as i16
            })
            .collect()
    }

    /// Signal-to-noise ratio in dB of `got` against `want`, with `got`
    /// delayed by `delay` samples (the two QMFs add 22).
    fn snr_db(want: &[i16], got: &[i16], delay: usize) -> f64 {
        let mut sig = 0f64;
        let mut noise = 0f64;
        // Skip the start, where the predictors are still adapting.
        for i in 800..want.len() - delay {
            let w = want[i] as f64;
            let g = got[i + delay] as f64;
            sig += w * w;
            noise += (w - g) * (w - g);
        }
        10.0 * (sig / noise.max(1.0)).log10()
    }

    fn round_trip(pcm: &[i16]) -> Vec<i16> {
        let mut bytes = Vec::new();
        G722Encoder::new().encode(pcm, &mut bytes).unwrap();
        let mut out = Vec::new();
        G722Decoder::new().decode(&bytes, &mut out);
        out
    }

    #[test]
    fn output_matches_the_reference_implementation_bit_for_bit() {
        // An integer-only signal (so the vector is the same on every
        // platform) coded by an independent G.722 implementation of
        // the same reference lineage. Any change to a table or to the
        // arithmetic shows up here.
        let mut seed: u32 = 1;
        let pcm: Vec<i16> = (0..64)
            .map(|i| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                let noise = ((seed >> 16) & 0x3FF) as i32 - 512;
                let tri = (i % 16 - 8_i32).abs() * 3000 - 12000;
                (tri + noise) as i16
            })
            .collect();
        let mut bytes = Vec::new();
        G722Encoder::new().encode(&pcm, &mut bytes).unwrap();
        assert_eq!(
            bytes,
            [
                144, 32, 132, 32, 132, 32, 160, 57, 19, 138, 142, 29, 176, 39, 235, 186, 213, 139,
                145, 254, 243, 40, 110, 188, 22, 141, 146, 63, 242, 42, 111, 191
            ]
        );
        let mut out = Vec::new();
        G722Decoder::new().decode(&bytes, &mut out);
        assert_eq!(
            out[32..],
            [
                -6026, -2806, 189, 2978, 6393, 9892, 11350, 9528, 6187, 2945, 30, -2484, -6474,
                -10556, -12006, -9327, -5951, -2509, -146, 2352, 6055, 10102, 11687, 9100, 5663,
                2826, -117, -2939, -5933, -9506, -11491, -8895
            ]
        );
    }

    #[test]
    fn a_frame_is_half_as_many_bytes_as_samples() {
        let pcm = tone(440.0, G722_FRAME_SAMPLES, 8000.0);
        let mut bytes = Vec::new();
        let n = G722Encoder::new().encode(&pcm, &mut bytes).unwrap();
        assert_eq!(n, G722_FRAME_BYTES);
        assert_eq!(bytes.len(), G722_FRAME_BYTES);
        let mut out = Vec::new();
        assert_eq!(
            G722Decoder::new().decode(&bytes, &mut out),
            G722_FRAME_SAMPLES
        );
        assert_eq!(out.len(), G722_FRAME_SAMPLES);
    }

    #[test]
    fn output_is_appended_not_overwritten() {
        let pcm = tone(440.0, 320, 8000.0);
        let mut bytes = vec![0xAA];
        G722Encoder::new().encode(&pcm, &mut bytes).unwrap();
        assert_eq!(bytes.len(), 161);
        assert_eq!(bytes[0], 0xAA);
        let mut out = vec![7i16];
        G722Decoder::new().decode(&bytes[1..], &mut out);
        assert_eq!(out.len(), 321);
        assert_eq!(out[0], 7);
    }

    #[test]
    fn an_odd_sample_count_is_refused_and_encodes_nothing() {
        let mut bytes = Vec::new();
        let err = G722Encoder::new()
            .encode(&[0i16; 321], &mut bytes)
            .unwrap_err();
        assert!(err.to_string().contains("321"), "{err}");
        assert!(bytes.is_empty());
    }

    #[test]
    fn silence_encodes_to_the_idle_pattern_and_decodes_to_silence() {
        let out = round_trip(&[0i16; 640]);
        assert!(out.iter().all(|s| s.abs() <= 8), "{:?}", &out[..16]);
    }

    #[test]
    fn a_low_band_tone_survives_the_round_trip() {
        let pcm = tone(1000.0, 16_000, 8000.0);
        let out = round_trip(&pcm);
        let snr = snr_db(&pcm, &out, 22);
        assert!(snr > 25.0, "1 kHz SNR was {snr:.1} dB");
    }

    #[test]
    fn a_high_band_tone_survives_the_round_trip() {
        // 5.5 kHz is above anything G.711 can carry: it only comes back
        // if the high band works. Two bits per sample there, so the
        // bar is lower than for the low band.
        let pcm = tone(5500.0, 16_000, 8000.0);
        let out = round_trip(&pcm);
        let snr = snr_db(&pcm, &out, 22);
        assert!(snr > 8.0, "5.5 kHz SNR was {snr:.1} dB");
        let energy: f64 = out[800..].iter().map(|&s| (s as f64).powi(2)).sum();
        let want: f64 = pcm[800..].iter().map(|&s| (s as f64).powi(2)).sum();
        assert!(energy > want * 0.5, "the high band came back too quiet");
    }

    #[test]
    fn full_scale_input_does_not_overflow() {
        // Worst case for the saturating arithmetic: a full-scale square
        // wave. It must neither panic (debug overflow checks) nor wrap.
        let pcm: Vec<i16> = (0..3200)
            .map(|i| {
                if (i / 20) % 2 == 0 {
                    i16::MAX
                } else {
                    i16::MIN
                }
            })
            .collect();
        let out = round_trip(&pcm);
        assert_eq!(out.len(), pcm.len());
    }

    #[test]
    fn frames_fed_one_at_a_time_match_one_long_call() {
        // The coder is stateful: splitting a stream into RTP frames
        // must give exactly the bytes of coding it in one go.
        let pcm = tone(700.0, G722_FRAME_SAMPLES * 5, 6000.0);
        let mut whole = Vec::new();
        G722Encoder::new().encode(&pcm, &mut whole).unwrap();
        let mut enc = G722Encoder::new();
        let mut framed = Vec::new();
        for frame in pcm.chunks(G722_FRAME_SAMPLES) {
            enc.encode(frame, &mut framed).unwrap();
        }
        assert_eq!(whole, framed);

        let mut whole_pcm = Vec::new();
        G722Decoder::new().decode(&whole, &mut whole_pcm);
        let mut dec = G722Decoder::new();
        let mut framed_pcm = Vec::new();
        for frame in whole.chunks(G722_FRAME_BYTES) {
            dec.decode(frame, &mut framed_pcm);
        }
        assert_eq!(whole_pcm, framed_pcm);
    }

    #[test]
    fn the_rtp_clock_is_half_the_sample_rate() {
        // RFC 3551 §4.5.2: G722/8000 on the wire, 16 kHz audio.
        assert_eq!(G722_RTP_CLOCK_RATE * 2, G722_PCM_SAMPLE_RATE);
        assert_eq!(G722_RTP_SAMPLES_PER_FRAME as usize * 2, G722_FRAME_SAMPLES);
        assert_eq!(G722_RTP_SAMPLES_PER_FRAME as usize, G722_FRAME_BYTES);
    }
}
