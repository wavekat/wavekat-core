//! Opus codec (RFC 6716) — the wideband voice profile for telephony.
//!
//! Wraps the reference C `libopus` via the [`opus`](https://docs.rs/opus)
//! crate. The profile is fixed for the WaveKat call path: 16 kHz mono
//! ("wideband"), `Application::Voip`, in-band FEC on, DTX off. A 20 ms
//! frame is therefore 320 PCM samples in, one variable-size packet out.
//!
//! # Wire constants vs. audio reality
//!
//! Three numbers here look contradictory and are not:
//!
//! - The SDP rtpmap is always `opus/48000/2` and the RTP timestamp
//!   advances by [`OPUS_RTP_SAMPLES_PER_FRAME`] (960) per 20 ms packet.
//!   RFC 7587 §4.1 pins the RTP clock to 48 kHz and the channel count
//!   to 2 **regardless of what the codec actually does** — they are
//!   wire-format constants, not a description of the stream.
//! - The PCM on either side of the codec is [`OPUS_PCM_SAMPLE_RATE`]
//!   (16 kHz) mono. That's the audio reality: wideband speech.
//!
//! Confusing the 48 kHz wire clock with the 16 kHz PCM rate is the
//! classic Opus-over-RTP interop bug; the constants below exist so
//! consumers never re-derive these numbers.
//!
//! # Loss recovery is driven by the consumer
//!
//! Unlike G.711, Opus can *recover* lost packets — but only if the
//! receive path notices the loss (an RTP sequence-number gap) and asks:
//!
//! - [`OpusDecoder::decode_fec`] — recover the lost frame from the
//!   *next* packet's embedded redundancy (in-band FEC / LBRR). The
//!   encoder only embeds that redundancy because we set a nonzero
//!   expected packet loss; `set_inband_fec(true)` alone puts nothing
//!   on the wire.
//! - [`OpusDecoder::conceal`] — packet-loss concealment when nothing
//!   usable arrived at all: the decoder extrapolates a plausible frame
//!   instead of emitting a click or silence.
//!
//! Opus lives in `wavekat-core` (not `wavekat-sip`) for the same reason
//! G.711 does: codecs are a consumer-layer choice — `wavekat-sip`
//! deliberately stays codec-agnostic.

use crate::CoreError;
use ::opus::{Application, Bitrate, Channels};

/// De-facto default dynamic RTP payload type for Opus in SDP offers.
///
/// Unlike PCMU (0) / PCMA (8) this is **not** a static assignment —
/// Opus rides the dynamic range (96–127) and the peer may pick any
/// number. 111 is only what *we* offer; the negotiated value comes from
/// SDP and must be carried per-call, never assumed.
pub const OPUS_DEFAULT_PAYLOAD_TYPE: u8 = 111;

/// The RTP timestamp clock rate for Opus — always 48 kHz per RFC 7587
/// §4.1, independent of the actual encode rate. Wire constant.
pub const OPUS_RTP_CLOCK_RATE: u32 = 48_000;

/// RTP timestamp advance per 20 ms Opus packet: 20 ms at the mandatory
/// 48 kHz wire clock. Feed this to the RTP sender's `samples_per_frame`
/// (the G.711 equivalent is 160).
pub const OPUS_RTP_SAMPLES_PER_FRAME: u32 = 960;

/// PCM sample rate on both sides of the codec: 16 kHz wideband, the
/// locked profile from doc 45. This — not 48 kHz — is the rate the
/// microphone is resampled to before encode and the rate decoded frames
/// come out at.
pub const OPUS_PCM_SAMPLE_RATE: u32 = 16_000;

/// PCM samples in one 20 ms frame at [`OPUS_PCM_SAMPLE_RATE`]: what
/// [`OpusEncoder::encode`] consumes per call and what a normal 20 ms
/// packet decodes to.
pub const OPUS_FRAME_SAMPLES: usize = 320;

/// PCM samples in the largest legal Opus frame (120 ms) at
/// [`OPUS_PCM_SAMPLE_RATE`]. The far end chooses its own frame size, so
/// decode buffers must size for this, not for 20 ms.
pub const OPUS_MAX_FRAME_SAMPLES: usize = 1920;

/// Target bitrate: the middle of the 24–32 kbps wideband-voice sweet
/// spot from doc 45 — a large quality jump over G.711's 64 kbps
/// narrowband at under half the bandwidth.
pub const OPUS_DEFAULT_BITRATE: i32 = 28_000;

/// Expected packet-loss percentage told to the encoder. Nonzero is what
/// makes libopus actually spend bits on in-band FEC redundancy.
pub const OPUS_DEFAULT_PACKET_LOSS_PERC: i32 = 10;

/// Scratch headroom for one encoded packet. A 20 ms mono voice frame at
/// our bitrate is a few hundred bytes at most; the theoretical Opus
/// maximum for one frame is 1275.
const MAX_PACKET_BYTES: usize = 1500;

fn opus_err(op: &str, e: ::opus::Error) -> CoreError {
    CoreError::Audio(format!("opus {op}: {e}"))
}

/// Stateful Opus encoder with the fixed WaveKat voice profile.
///
/// Unlike [`G711Codec`](super::g711::G711Codec) this is not `Copy` —
/// libopus carries inter-frame prediction state, so one encoder serves
/// exactly one outbound stream and must be fed every frame in order.
pub struct OpusEncoder {
    inner: ::opus::Encoder,
}

impl OpusEncoder {
    /// Create an encoder with the locked profile: 16 kHz mono VoIP,
    /// ~28 kbps, in-band FEC on (with expected loss set so it actually
    /// engages), DTX off.
    pub fn new() -> Result<Self, CoreError> {
        let mut inner =
            ::opus::Encoder::new(OPUS_PCM_SAMPLE_RATE, Channels::Mono, Application::Voip)
                .map_err(|e| opus_err("encoder create", e))?;
        inner
            .set_bitrate(Bitrate::Bits(OPUS_DEFAULT_BITRATE))
            .map_err(|e| opus_err("set_bitrate", e))?;
        inner
            .set_inband_fec(true)
            .map_err(|e| opus_err("set_inband_fec", e))?;
        inner
            .set_packet_loss_perc(OPUS_DEFAULT_PACKET_LOSS_PERC)
            .map_err(|e| opus_err("set_packet_loss_perc", e))?;
        inner.set_dtx(false).map_err(|e| opus_err("set_dtx", e))?;
        Ok(Self { inner })
    }

    /// Encode exactly one 20 ms frame ([`OPUS_FRAME_SAMPLES`] samples at
    /// [`OPUS_PCM_SAMPLE_RATE`]) into one Opus packet. Appends the packet
    /// bytes to `out` and returns the packet length.
    ///
    /// The length is enforced because libopus only accepts legal frame
    /// durations and the WaveKat send path packetizes at 20 ms; a wrong
    /// slice length here is always a caller bug, not a stream condition.
    pub fn encode(&mut self, pcm: &[i16], out: &mut Vec<u8>) -> Result<usize, CoreError> {
        if pcm.len() != OPUS_FRAME_SAMPLES {
            return Err(CoreError::Audio(format!(
                "opus encode: expected one 20 ms frame ({OPUS_FRAME_SAMPLES} samples), got {}",
                pcm.len()
            )));
        }
        let start = out.len();
        out.resize(start + MAX_PACKET_BYTES, 0);
        let n = self
            .inner
            .encode(pcm, &mut out[start..])
            .map_err(|e| opus_err("encode", e))?;
        out.truncate(start + n);
        Ok(n)
    }
}

/// Stateful Opus decoder for one inbound stream.
///
/// Also not `Copy` — the decoder state is what makes
/// [`conceal`](Self::conceal) and [`decode_fec`](Self::decode_fec)
/// possible, so it must see every packet of its stream in order.
pub struct OpusDecoder {
    inner: ::opus::Decoder,
}

impl OpusDecoder {
    /// Create a decoder producing 16 kHz mono PCM.
    pub fn new() -> Result<Self, CoreError> {
        let inner = ::opus::Decoder::new(OPUS_PCM_SAMPLE_RATE, Channels::Mono)
            .map_err(|e| opus_err("decoder create", e))?;
        Ok(Self { inner })
    }

    /// Decode one received Opus packet. Appends the PCM samples to `out`
    /// and returns how many were produced.
    ///
    /// The count is normally [`OPUS_FRAME_SAMPLES`] but the far end
    /// chooses its own frame size — up to [`OPUS_MAX_FRAME_SAMPLES`] —
    /// so consumers must use the returned length, not assume 20 ms.
    pub fn decode(&mut self, payload: &[u8], out: &mut Vec<i16>) -> Result<usize, CoreError> {
        self.decode_inner(payload, out, false)
    }

    /// Recover a **lost** frame from the packet that followed it.
    ///
    /// Call this when a sequence gap shows exactly one packet was lost:
    /// pass the packet *after* the gap, and the decoder reconstructs the
    /// missing frame from that packet's embedded redundancy (falling
    /// back to concealment when the sender embedded none). Then decode
    /// the passed packet normally with [`decode`](Self::decode) — this
    /// call consumes only its redundancy, not its own frame.
    pub fn decode_fec(
        &mut self,
        next_payload: &[u8],
        out: &mut Vec<i16>,
    ) -> Result<usize, CoreError> {
        self.decode_inner(next_payload, out, true)
    }

    /// Produce one concealment frame for a packet that never arrived and
    /// can't be recovered (no follow-up packet yet, or a multi-packet
    /// gap). The decoder extrapolates from what it last heard, which
    /// sounds far better than a gap of silence.
    pub fn conceal(&mut self, out: &mut Vec<i16>) -> Result<usize, CoreError> {
        // An empty input slice is the libopus idiom for "the packet was
        // lost" — the decoder synthesizes `out.len()` samples of PLC. We
        // ask for one nominal 20 ms frame per lost packet.
        let start = out.len();
        out.resize(start + OPUS_FRAME_SAMPLES, 0);
        let n = self
            .inner
            .decode(&[], &mut out[start..], false)
            .map_err(|e| opus_err("plc decode", e))?;
        out.truncate(start + n);
        Ok(n)
    }

    fn decode_inner(
        &mut self,
        payload: &[u8],
        out: &mut Vec<i16>,
        fec: bool,
    ) -> Result<usize, CoreError> {
        let start = out.len();
        if fec {
            // FEC reconstruction is duration-driven: the output slice
            // length tells libopus how much audio to recover. The
            // redundancy in a packet mirrors that packet's own frame
            // duration, so size the request from the packet itself.
            let lost = ::opus::packet::get_nb_samples(payload, OPUS_PCM_SAMPLE_RATE)
                .map_err(|e| opus_err("fec frame sizing", e))?;
            out.resize(start + lost, 0);
        } else {
            out.resize(start + OPUS_MAX_FRAME_SAMPLES, 0);
        }
        let n = self
            .inner
            .decode(payload, &mut out[start..], fec)
            .map_err(|e| opus_err(if fec { "fec decode" } else { "decode" }, e))?;
        out.truncate(start + n);
        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_frame(freq: f32, amplitude: f32, offset: usize) -> Vec<i16> {
        (0..OPUS_FRAME_SAMPLES)
            .map(|i| {
                let t = (offset + i) as f32 / OPUS_PCM_SAMPLE_RATE as f32;
                ((t * freq * 2.0 * std::f32::consts::PI).sin() * amplitude) as i16
            })
            .collect()
    }

    fn rms(pcm: &[i16]) -> f64 {
        if pcm.is_empty() {
            return 0.0;
        }
        let sum: f64 = pcm.iter().map(|&s| (s as f64) * (s as f64)).sum();
        (sum / pcm.len() as f64).sqrt()
    }

    #[test]
    fn constants_match_rfc7587_and_the_locked_profile() {
        // RFC 7587 §4.1 pins the wire clock; doc 45 pins the profile.
        // Hard-coding them in a test protects SDP/RTP interop against a
        // casual "cleanup" the same way the G.711 constants test does.
        assert_eq!(OPUS_DEFAULT_PAYLOAD_TYPE, 111);
        assert_eq!(OPUS_RTP_CLOCK_RATE, 48_000);
        assert_eq!(OPUS_RTP_SAMPLES_PER_FRAME, 960);
        assert_eq!(OPUS_PCM_SAMPLE_RATE, 16_000);
        assert_eq!(OPUS_FRAME_SAMPLES, 320);
        assert_eq!(OPUS_MAX_FRAME_SAMPLES, 1920);
        // 20 ms at each clock, cross-checked.
        assert_eq!(OPUS_RTP_SAMPLES_PER_FRAME, OPUS_RTP_CLOCK_RATE / 50);
        assert_eq!(OPUS_FRAME_SAMPLES as u32, OPUS_PCM_SAMPLE_RATE / 50);
    }

    #[test]
    fn encode_decode_round_trip_preserves_signal_energy() {
        let mut enc = OpusEncoder::new().unwrap();
        let mut dec = OpusDecoder::new().unwrap();

        let mut input_rms = 0.0;
        let mut decoded_tail = Vec::new();
        for i in 0..20 {
            let frame = sine_frame(440.0, 8000.0, i * OPUS_FRAME_SAMPLES);
            input_rms = rms(&frame);

            let mut packet = Vec::new();
            let n = enc.encode(&frame, &mut packet).unwrap();
            assert!(n > 0 && n <= 1275, "packet size {n} out of range");
            assert_eq!(packet.len(), n);

            let mut pcm = Vec::new();
            let produced = dec.decode(&packet, &mut pcm).unwrap();
            assert_eq!(produced, OPUS_FRAME_SAMPLES);
            assert_eq!(pcm.len(), OPUS_FRAME_SAMPLES);
            // Skip the first frames: codec lookahead delay means the
            // start of the stream decodes quiet. Steady state is what
            // matters.
            if i >= 5 {
                decoded_tail.extend_from_slice(&pcm);
            }
        }

        let out_rms = rms(&decoded_tail);
        assert!(
            (out_rms - input_rms).abs() / input_rms < 0.5,
            "round-trip energy drifted too far: in {input_rms:.0}, out {out_rms:.0}"
        );
    }

    #[test]
    fn encoder_profile_is_the_locked_one() {
        // FEC on, expected loss nonzero (without it FEC is silent), DTX
        // off — the doc-45 profile, read back through libopus getters so
        // a dropped setter call fails loudly.
        let mut enc = OpusEncoder::new().unwrap();
        assert!(enc.inner.get_inband_fec().unwrap());
        assert_eq!(
            enc.inner.get_packet_loss_perc().unwrap(),
            OPUS_DEFAULT_PACKET_LOSS_PERC
        );
        assert!(!enc.inner.get_dtx().unwrap());
        assert_eq!(
            enc.inner.get_bitrate().unwrap(),
            Bitrate::Bits(OPUS_DEFAULT_BITRATE)
        );
    }

    #[test]
    fn encode_rejects_non_20ms_input() {
        let mut enc = OpusEncoder::new().unwrap();
        let mut out = Vec::new();
        assert!(enc.encode(&[0i16; 160], &mut out).is_err());
        assert!(enc.encode(&[0i16; 321], &mut out).is_err());
        assert!(out.is_empty(), "failed encode must not write output");
    }

    #[test]
    fn encode_appends_rather_than_replacing() {
        // Same buffer-reuse contract as G711Codec::encode.
        let mut enc = OpusEncoder::new().unwrap();
        let mut out = vec![0xAAu8, 0xBB];
        let frame = sine_frame(440.0, 8000.0, 0);
        let n = enc.encode(&frame, &mut out).unwrap();
        assert_eq!(out.len(), 2 + n);
        assert_eq!(&out[..2], &[0xAA, 0xBB]);
    }

    // Encode a stream that is near-silent except for one loud frame,
    // "lose" the loud frame, and try to recover it from the next
    // packet. With FEC engaged the recovery carries the loud frame's
    // energy; without FEC the decoder can only extrapolate from the
    // silent context and produces near-silence. Amplitude is the
    // discriminator because it survives codec coloration.
    fn lost_loud_frame_recovery_rms(fec: bool) -> f64 {
        let mut enc = if fec {
            OpusEncoder::new().unwrap()
        } else {
            let mut inner =
                ::opus::Encoder::new(OPUS_PCM_SAMPLE_RATE, Channels::Mono, Application::Voip)
                    .unwrap();
            inner
                .set_bitrate(Bitrate::Bits(OPUS_DEFAULT_BITRATE))
                .unwrap();
            inner.set_inband_fec(false).unwrap();
            OpusEncoder { inner }
        };
        let mut dec = OpusDecoder::new().unwrap();

        const LOUD: usize = 6;
        let mut packets = Vec::new();
        for i in 0..=LOUD + 1 {
            let frame = if i == LOUD {
                sine_frame(440.0, 12000.0, i * OPUS_FRAME_SAMPLES)
            } else {
                sine_frame(440.0, 30.0, i * OPUS_FRAME_SAMPLES)
            };
            let mut packet = Vec::new();
            enc.encode(&frame, &mut packet).unwrap();
            packets.push(packet);
        }

        let mut pcm = Vec::new();
        for packet in &packets[..LOUD] {
            pcm.clear();
            dec.decode(packet, &mut pcm).unwrap();
        }
        // Packet LOUD never arrives; recover it from packet LOUD+1.
        let mut recovered = Vec::new();
        let n = dec.decode_fec(&packets[LOUD + 1], &mut recovered).unwrap();
        assert_eq!(n, OPUS_FRAME_SAMPLES);
        rms(&recovered)
    }

    #[test]
    fn fec_actually_recovers_a_lost_frame() {
        let with_fec = lost_loud_frame_recovery_rms(true);
        let without_fec = lost_loud_frame_recovery_rms(false);
        // The FEC recovery must carry real signal, and beat the
        // conceal-from-silence baseline by a wide margin — this is the
        // "FEC engages on the wire" proof, not just an API smoke test.
        assert!(
            with_fec > 1000.0,
            "FEC recovery is near-silent (rms {with_fec:.1}) — redundancy not embedded?"
        );
        assert!(
            with_fec > without_fec * 4.0,
            "FEC recovery ({with_fec:.1}) not clearly better than PLC baseline ({without_fec:.1})"
        );
    }

    #[test]
    fn conceal_produces_a_plausible_frame_after_loss() {
        let mut enc = OpusEncoder::new().unwrap();
        let mut dec = OpusDecoder::new().unwrap();

        let mut pcm = Vec::new();
        for i in 0..10 {
            let frame = sine_frame(440.0, 8000.0, i * OPUS_FRAME_SAMPLES);
            let mut packet = Vec::new();
            enc.encode(&frame, &mut packet).unwrap();
            pcm.clear();
            dec.decode(&packet, &mut pcm).unwrap();
        }

        let mut concealed = Vec::new();
        let n = dec.conceal(&mut concealed).unwrap();
        assert_eq!(n, OPUS_FRAME_SAMPLES);
        // PLC extrapolates the ongoing tone — it must be audible, not a
        // silence gap.
        assert!(
            rms(&concealed) > 500.0,
            "concealment is near-silent (rms {:.1})",
            rms(&concealed)
        );
    }

    #[test]
    fn decode_handles_frames_larger_than_20ms() {
        // The far end picks its own frame size. Build a 40 ms packet
        // with a raw libopus encoder and make sure our decoder sizes for
        // it instead of assuming 20 ms.
        let mut enc =
            ::opus::Encoder::new(OPUS_PCM_SAMPLE_RATE, Channels::Mono, Application::Voip).unwrap();
        let frame: Vec<i16> = (0..OPUS_FRAME_SAMPLES * 2)
            .map(|i| {
                let t = i as f32 / OPUS_PCM_SAMPLE_RATE as f32;
                ((t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 8000.0) as i16
            })
            .collect();
        let packet = enc.encode_vec(&frame, MAX_PACKET_BYTES).unwrap();

        let mut dec = OpusDecoder::new().unwrap();
        let mut pcm = Vec::new();
        let n = dec.decode(&packet, &mut pcm).unwrap();
        assert_eq!(n, OPUS_FRAME_SAMPLES * 2, "40 ms frame must decode fully");
        assert_eq!(pcm.len(), n);
    }

    #[test]
    fn decode_rejects_garbage() {
        let mut dec = OpusDecoder::new().unwrap();
        let mut pcm = Vec::new();
        // An invalid TOC byte sequence must surface an error, not panic
        // or emit junk audio.
        let garbage = vec![0xFFu8; 3];
        assert!(dec.decode(&garbage, &mut pcm).is_err());
    }
}
