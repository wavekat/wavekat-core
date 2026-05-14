use std::borrow::Cow;

/// A frame of audio samples with associated sample rate.
///
/// `AudioFrame` is the standard audio input type across the WaveKat ecosystem.
/// It stores samples as f32 normalized to `[-1.0, 1.0]`, regardless of the
/// original input format.
///
/// Construct via [`AudioFrame::new`], which accepts both `&[f32]` (zero-copy)
/// and `&[i16]` (converts once) through the [`IntoSamples`] trait.
///
/// # Examples
///
/// ```
/// use wavekat_core::AudioFrame;
///
/// // f32 input — zero-copy via Cow::Borrowed
/// let samples = [0.1f32, -0.2, 0.3];
/// let frame = AudioFrame::new(&samples, 16000);
/// assert_eq!(frame.samples(), &[0.1, -0.2, 0.3]);
///
/// // i16 input — normalized to f32 [-1.0, 1.0]
/// let samples = [i16::MAX, 0, i16::MIN];
/// let frame = AudioFrame::new(&samples, 16000);
/// assert!((frame.samples()[0] - 1.0).abs() < 0.001);
/// ```
#[derive(Debug, Clone)]
pub struct AudioFrame<'a> {
    samples: Cow<'a, [f32]>,
    sample_rate: u32,
}

impl<'a> AudioFrame<'a> {
    /// Create a new audio frame from any supported sample type.
    ///
    /// Accepts `&[f32]` (zero-copy) or `&[i16]` (converts to normalized f32).
    pub fn new(samples: impl IntoSamples<'a>, sample_rate: u32) -> Self {
        Self {
            samples: samples.into_samples(),
            sample_rate,
        }
    }

    /// The audio samples as f32 normalized to `[-1.0, 1.0]`.
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Sample rate in Hz (e.g. 16000).
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Number of samples in the frame.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns `true` if the frame contains no samples.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Duration of this frame in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }

    /// Consume the frame and return the owned samples.
    pub fn into_owned(self) -> AudioFrame<'static> {
        AudioFrame {
            samples: Cow::Owned(self.samples.into_owned()),
            sample_rate: self.sample_rate,
        }
    }
}

impl AudioFrame<'static> {
    /// Construct an owned frame directly from a `Vec<f32>`.
    ///
    /// Zero-copy — wraps the vec as `Cow::Owned` without cloning.
    /// Intended for audio producers (TTS, ASR) that generate owned data.
    ///
    /// # Example
    ///
    /// ```
    /// use wavekat_core::AudioFrame;
    ///
    /// let samples = vec![0.5f32, -0.5, 0.3];
    /// let frame = AudioFrame::from_vec(samples, 24000);
    /// assert_eq!(frame.sample_rate(), 24000);
    /// assert_eq!(frame.len(), 3);
    /// ```
    pub fn from_vec(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples: Cow::Owned(samples),
            sample_rate,
        }
    }
}

#[cfg(feature = "resample")]
impl AudioFrame<'_> {
    /// Resample this frame to a different sample rate.
    ///
    /// Returns a new owned `AudioFrame` at `target_rate`. If the frame is
    /// already at the target rate, returns a clone without touching the
    /// resampler.
    ///
    /// Uses high-quality sinc interpolation via [`rubato`].
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::Audio`] if the resampler cannot be constructed
    /// (e.g. zero sample rate) or if processing fails.
    ///
    /// # Example
    ///
    /// ```
    /// use wavekat_core::AudioFrame;
    ///
    /// let frame = AudioFrame::from_vec(vec![0.0f32; 4410], 44100);
    /// let resampled = frame.resample(16000).unwrap();
    /// assert_eq!(resampled.sample_rate(), 16000);
    /// ```
    pub fn resample(&self, target_rate: u32) -> Result<AudioFrame<'static>, crate::CoreError> {
        use rubato::audioadapter_buffers::direct::InterleavedSlice;
        use rubato::Resampler;

        if self.sample_rate == target_rate {
            return Ok(self.clone().into_owned());
        }

        if self.is_empty() {
            return Ok(AudioFrame::from_vec(Vec::new(), target_rate));
        }

        let nbr_input_frames = self.samples.len();
        // Match chunk size to input when shorter than the default — avoids
        // wasting work padding a 160-sample G.711 frame up to 1024 samples.
        let chunk_size = nbr_input_frames.min(1024);
        let mut resampler = build_sinc_resampler(self.sample_rate, target_rate, chunk_size)?;

        // Ask rubato exactly how much output space `process_all_into_buffer`
        // needs — it accounts for the per-chunk pad-up, the resampler's
        // internal delay, and the input-length-times-ratio expected output.
        let out_len = resampler.process_all_needed_output_len(nbr_input_frames);
        let mut outdata = vec![0.0f32; out_len];

        let input_adapter = InterleavedSlice::new(self.samples.as_ref(), 1, nbr_input_frames)
            .map_err(|e| crate::CoreError::Audio(e.to_string()))?;
        let mut output_adapter = InterleavedSlice::new_mut(&mut outdata, 1, out_len)
            .map_err(|e| crate::CoreError::Audio(e.to_string()))?;

        let (_in_consumed, out_produced) = resampler
            .process_all_into_buffer(&input_adapter, &mut output_adapter, nbr_input_frames, None)
            .map_err(|e| crate::CoreError::Audio(e.to_string()))?;

        outdata.truncate(out_produced);
        Ok(AudioFrame::from_vec(outdata, target_rate))
    }
}

/// Shared rubato builder used by both [`AudioFrame::resample`] and
/// [`StreamingResampler`]. Keeps the sinc parameters (and the version
/// bumps that come with rubato API churn) in one place.
#[cfg(feature = "resample")]
fn build_sinc_resampler(
    source_rate: u32,
    target_rate: u32,
    chunk_size: usize,
) -> Result<rubato::Async<f32>, crate::CoreError> {
    use rubato::{
        Async, FixedAsync, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    if source_rate == 0 || target_rate == 0 {
        return Err(crate::CoreError::Audio(
            "sample rate must be non-zero".into(),
        ));
    }
    if chunk_size == 0 {
        return Err(crate::CoreError::Audio(
            "chunk_size must be non-zero".into(),
        ));
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Cubic,
        oversampling_factor: 128,
        window: WindowFunction::BlackmanHarris2,
    };
    let ratio = target_rate as f64 / source_rate as f64;
    Async::<f32>::new_sinc(ratio, 1.0, &params, chunk_size, 1, FixedAsync::Input)
        .map_err(|e| crate::CoreError::Audio(e.to_string()))
}

/// Stateful streaming resampler.
///
/// [`AudioFrame::resample`] is convenient but constructs a fresh rubato
/// resampler per call. For real-time pipelines that hand the resampler
/// short frames (e.g. 20 ms G.711 packets off an RTP socket) the per-call
/// resampler has no state to carry across frame boundaries, and sinc
/// reconstruction produces audible edge artifacts at the frame rate —
/// 50 Hz for 20 ms packets, perceived as continuous noise/buzz over the
/// voice. `StreamingResampler` builds rubato once at stream open and
/// reuses its internal filter state for every call, so output samples
/// stitch together cleanly.
///
/// Build it with [`StreamingResampler::new`], then call
/// [`process`](Self::process) for each arriving block of audio. Samples
/// accumulate inside the resampler until a full `chunk_size` is ready,
/// then a chunk's worth of output is appended to the caller's buffer.
///
/// If `source_rate == target_rate`, `process` becomes a pure copy and
/// `chunk_size` is ignored.
///
/// # Example
///
/// ```
/// use wavekat_core::StreamingResampler;
///
/// // 8 kHz → 44.1 kHz, 160-sample input chunks (matches 20 ms G.711).
/// let mut resampler = StreamingResampler::new(8000, 44100, 160).unwrap();
///
/// let mut out = Vec::new();
/// for _packet in 0..5 {
///     let input = vec![0.0f32; 160]; // 20 ms of silence per packet
///     resampler.process(&input, &mut out).unwrap();
/// }
/// // Five 160-sample inputs at 8 kHz expand to roughly 5 × 882 samples
/// // at 44.1 kHz (the exact count depends on rubato's edge handling).
/// assert!(out.len() > 4000);
/// ```
#[cfg(feature = "resample")]
pub struct StreamingResampler {
    // `None` when source_rate == target_rate (pass-through fast path).
    inner: Option<rubato::Async<f32>>,
    source_rate: u32,
    target_rate: u32,
    chunk_size: usize,
    // Accumulates partial input across calls until we have `chunk_size`
    // samples for the next rubato step.
    input_buf: Vec<f32>,
    // Reusable scratch sized to `output_frames_max()` so we don't
    // re-allocate on every chunk.
    output_buf: Vec<f32>,
}

#[cfg(feature = "resample")]
impl StreamingResampler {
    /// Build a streaming resampler.
    ///
    /// `chunk_size` is how many input samples are processed per internal
    /// rubato step. Match it to the natural arrival size of your input
    /// — e.g. 160 for 20 ms G.711 frames at 8 kHz. Smaller chunks mean
    /// lower latency; larger chunks are marginally more efficient.
    ///
    /// Returns [`CoreError::Audio`] if the resampler cannot be built
    /// (zero rate, zero chunk size, or rubato rejects the ratio).
    pub fn new(
        source_rate: u32,
        target_rate: u32,
        chunk_size: usize,
    ) -> Result<Self, crate::CoreError> {
        if source_rate == target_rate {
            // Pass-through still validates the rates so calling code
            // can't smuggle a zero rate past us.
            if source_rate == 0 {
                return Err(crate::CoreError::Audio(
                    "sample rate must be non-zero".into(),
                ));
            }
            return Ok(Self {
                inner: None,
                source_rate,
                target_rate,
                chunk_size,
                input_buf: Vec::new(),
                output_buf: Vec::new(),
            });
        }

        let inner = build_sinc_resampler(source_rate, target_rate, chunk_size)?;
        let out_max = {
            use rubato::Resampler;
            inner.output_frames_max()
        };
        Ok(Self {
            inner: Some(inner),
            source_rate,
            target_rate,
            chunk_size,
            input_buf: Vec::with_capacity(chunk_size),
            output_buf: vec![0.0; out_max],
        })
    }

    /// Source sample rate this resampler was built for.
    pub fn source_rate(&self) -> u32 {
        self.source_rate
    }

    /// Target sample rate this resampler emits.
    pub fn target_rate(&self) -> u32 {
        self.target_rate
    }

    /// Input chunk size — how many samples per internal step.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Resample `input` and append the output samples to `out`.
    ///
    /// Input is buffered internally until a full `chunk_size` has been
    /// received; partial chunks remain buffered until the next call.
    /// State is carried across calls so there are no boundary artifacts
    /// — feeding two adjacent 160-sample chunks is equivalent to
    /// feeding one 320-sample chunk (modulo the resampler's group
    /// delay, paid once at the start of the stream).
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<(), crate::CoreError> {
        let Some(inner) = self.inner.as_mut() else {
            out.extend_from_slice(input);
            return Ok(());
        };
        use rubato::audioadapter_buffers::direct::InterleavedSlice;
        use rubato::Resampler;

        let mut remaining = input;
        while !remaining.is_empty() {
            let need = self.chunk_size - self.input_buf.len();
            let take = need.min(remaining.len());
            self.input_buf.extend_from_slice(&remaining[..take]);
            remaining = &remaining[take..];

            if self.input_buf.len() < self.chunk_size {
                break;
            }

            let in_adapter = InterleavedSlice::new(&self.input_buf[..], 1, self.chunk_size)
                .map_err(|e| crate::CoreError::Audio(e.to_string()))?;
            let out_buf_len = self.output_buf.len();
            let mut out_adapter =
                InterleavedSlice::new_mut(&mut self.output_buf[..], 1, out_buf_len)
                    .map_err(|e| crate::CoreError::Audio(e.to_string()))?;
            let (_in_used, out_produced) = inner
                .process_into_buffer(&in_adapter, &mut out_adapter, None)
                .map_err(|e| crate::CoreError::Audio(e.to_string()))?;
            out.extend_from_slice(&self.output_buf[..out_produced]);
            self.input_buf.clear();
        }
        Ok(())
    }
}

#[cfg(feature = "wav")]
impl AudioFrame<'_> {
    /// Write this frame to a WAV file at `path`.
    ///
    /// Always writes mono f32 PCM at the frame's native sample rate.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use wavekat_core::AudioFrame;
    ///
    /// let frame = AudioFrame::from_vec(vec![0.0f32; 16000], 16000);
    /// frame.write_wav("output.wav").unwrap();
    /// ```
    pub fn write_wav(&self, path: impl AsRef<std::path::Path>) -> Result<(), crate::CoreError> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(path, spec)?;
        for &sample in self.samples() {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
        Ok(())
    }
}

#[cfg(feature = "wav")]
impl AudioFrame<'static> {
    /// Read a mono WAV file and return an owned `AudioFrame`.
    ///
    /// Accepts both f32 and i16 WAV files. i16 samples are normalised to
    /// `[-1.0, 1.0]` (divided by 32768).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use wavekat_core::AudioFrame;
    ///
    /// let frame = AudioFrame::from_wav("input.wav").unwrap();
    /// println!("{} Hz, {} samples", frame.sample_rate(), frame.len());
    /// ```
    pub fn from_wav(path: impl AsRef<std::path::Path>) -> Result<Self, crate::CoreError> {
        let mut reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| s.map(|v| v as f32 / 32768.0))
                .collect::<Result<_, _>>()?,
        };
        Ok(AudioFrame::from_vec(samples, sample_rate))
    }
}

/// Trait for types that can be converted into audio samples.
///
/// Implemented for `&[f32]` (zero-copy) and `&[i16]` (normalized conversion).
pub trait IntoSamples<'a> {
    /// Convert into f32 samples normalized to `[-1.0, 1.0]`.
    fn into_samples(self) -> Cow<'a, [f32]>;
}

impl<'a> IntoSamples<'a> for &'a [f32] {
    #[inline]
    fn into_samples(self) -> Cow<'a, [f32]> {
        Cow::Borrowed(self)
    }
}

impl<'a> IntoSamples<'a> for &'a Vec<f32> {
    #[inline]
    fn into_samples(self) -> Cow<'a, [f32]> {
        Cow::Borrowed(self.as_slice())
    }
}

impl<'a, const N: usize> IntoSamples<'a> for &'a [f32; N] {
    #[inline]
    fn into_samples(self) -> Cow<'a, [f32]> {
        Cow::Borrowed(self.as_slice())
    }
}

impl<'a> IntoSamples<'a> for &'a [i16] {
    #[inline]
    fn into_samples(self) -> Cow<'a, [f32]> {
        Cow::Owned(self.iter().map(|&s| s as f32 / 32768.0).collect())
    }
}

impl<'a> IntoSamples<'a> for &'a Vec<i16> {
    #[inline]
    fn into_samples(self) -> Cow<'a, [f32]> {
        Cow::Owned(self.iter().map(|&s| s as f32 / 32768.0).collect())
    }
}

impl<'a, const N: usize> IntoSamples<'a> for &'a [i16; N] {
    #[inline]
    fn into_samples(self) -> Cow<'a, [f32]> {
        Cow::Owned(self.iter().map(|&s| s as f32 / 32768.0).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_is_zero_copy() {
        let samples = vec![0.1f32, -0.2, 0.3];
        let frame = AudioFrame::new(samples.as_slice(), 16000);
        // Cow::Borrowed — the pointer should be the same
        assert!(matches!(frame.samples, Cow::Borrowed(_)));
        assert_eq!(frame.samples(), &[0.1, -0.2, 0.3]);
    }

    #[test]
    fn i16_normalizes_to_f32() {
        let samples: Vec<i16> = vec![0, 16384, -16384, i16::MAX, i16::MIN];
        let frame = AudioFrame::new(samples.as_slice(), 16000);
        assert!(matches!(frame.samples, Cow::Owned(_)));

        let s = frame.samples();
        assert!((s[0] - 0.0).abs() < f32::EPSILON);
        assert!((s[1] - 0.5).abs() < 0.001);
        assert!((s[2] - -0.5).abs() < 0.001);
        assert!((s[3] - (i16::MAX as f32 / 32768.0)).abs() < f32::EPSILON);
        assert!((s[4] - -1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn metadata() {
        let samples = vec![0.0f32; 160];
        let frame = AudioFrame::new(samples.as_slice(), 16000);
        assert_eq!(frame.sample_rate(), 16000);
        assert_eq!(frame.len(), 160);
        assert!(!frame.is_empty());
        assert!((frame.duration_secs() - 0.01).abs() < 1e-9);
    }

    #[test]
    fn empty_frame() {
        let samples: &[f32] = &[];
        let frame = AudioFrame::new(samples, 16000);
        assert!(frame.is_empty());
        assert_eq!(frame.len(), 0);
    }

    #[test]
    fn into_owned() {
        let samples = vec![0.5f32, -0.5];
        let frame = AudioFrame::new(samples.as_slice(), 16000);
        let owned: AudioFrame<'static> = frame.into_owned();
        assert_eq!(owned.samples(), &[0.5, -0.5]);
        assert_eq!(owned.sample_rate(), 16000);
    }

    #[cfg(feature = "wav")]
    #[test]
    fn wav_read_i16() {
        // Write an i16 WAV directly via hound, then read it with from_wav.
        let path = std::env::temp_dir().join("wavekat_test_i16.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let i16_samples: &[i16] = &[0, i16::MAX, i16::MIN, 16384];
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        for &s in i16_samples {
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();

        let frame = AudioFrame::from_wav(&path).unwrap();
        assert_eq!(frame.sample_rate(), 16000);
        assert_eq!(frame.len(), 4);
        let s = frame.samples();
        assert!((s[0] - 0.0).abs() < 1e-6);
        assert!((s[1] - (i16::MAX as f32 / 32768.0)).abs() < 1e-6);
        assert!((s[2] - -1.0).abs() < 1e-6);
        assert!((s[3] - 0.5).abs() < 1e-4);
    }

    #[cfg(feature = "wav")]
    #[test]
    fn wav_round_trip() {
        let original = AudioFrame::from_vec(vec![0.5f32, -0.5, 0.0, 1.0], 16000);
        let path = std::env::temp_dir().join("wavekat_test.wav");
        original.write_wav(&path).unwrap();
        let loaded = AudioFrame::from_wav(&path).unwrap();
        assert_eq!(loaded.sample_rate(), 16000);
        for (a, b) in original.samples().iter().zip(loaded.samples()) {
            assert!((a - b).abs() < 1e-6, "sample mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn from_vec_is_zero_copy() {
        let samples = vec![0.5f32, -0.5];
        let ptr = samples.as_ptr();
        let frame = AudioFrame::from_vec(samples, 24000);
        assert_eq!(frame.samples().as_ptr(), ptr);
        assert_eq!(frame.sample_rate(), 24000);
    }

    #[test]
    fn into_samples_vec_f32() {
        let samples = vec![0.1f32, -0.2, 0.3];
        let frame = AudioFrame::new(&samples, 16000);
        assert!(matches!(frame.samples, Cow::Borrowed(_)));
        assert_eq!(frame.samples(), &[0.1, -0.2, 0.3]);
    }

    #[test]
    fn into_samples_array_f32() {
        let samples = [0.1f32, -0.2, 0.3];
        let frame = AudioFrame::new(&samples, 16000);
        assert!(matches!(frame.samples, Cow::Borrowed(_)));
        assert_eq!(frame.samples(), &[0.1, -0.2, 0.3]);
    }

    #[test]
    fn into_samples_vec_i16() {
        let samples: Vec<i16> = vec![0, 16384, i16::MIN];
        let frame = AudioFrame::new(&samples, 16000);
        assert!(matches!(frame.samples, Cow::Owned(_)));
        let s = frame.samples();
        assert!((s[0] - 0.0).abs() < f32::EPSILON);
        assert!((s[1] - 0.5).abs() < 0.001);
        assert!((s[2] - -1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn into_samples_array_i16() {
        let samples: [i16; 3] = [0, 16384, i16::MIN];
        let frame = AudioFrame::new(&samples, 16000);
        assert!(matches!(frame.samples, Cow::Owned(_)));
        let s = frame.samples();
        assert!((s[0] - 0.0).abs() < f32::EPSILON);
        assert!((s[1] - 0.5).abs() < 0.001);
        assert!((s[2] - -1.0).abs() < f32::EPSILON);
    }

    #[cfg(feature = "resample")]
    #[test]
    fn resample_noop_same_rate() {
        let samples = vec![0.1f32, -0.2, 0.3, 0.4, 0.5];
        let frame = AudioFrame::from_vec(samples.clone(), 16000);
        let resampled = frame.resample(16000).unwrap();
        assert_eq!(resampled.sample_rate(), 16000);
        assert_eq!(resampled.samples(), &samples[..]);
    }

    #[cfg(feature = "resample")]
    #[test]
    fn resample_empty_frame() {
        let frame = AudioFrame::from_vec(Vec::new(), 44100);
        let resampled = frame.resample(16000).unwrap();
        assert_eq!(resampled.sample_rate(), 16000);
        assert!(resampled.is_empty());
    }

    #[cfg(feature = "resample")]
    #[test]
    fn resample_downsample() {
        // 1 second of silence at 48 kHz → 16 kHz
        let frame = AudioFrame::from_vec(vec![0.0f32; 48000], 48000);
        let resampled = frame.resample(16000).unwrap();
        assert_eq!(resampled.sample_rate(), 16000);
        // Should produce ~16000 samples (allow small tolerance from resampler)
        let expected = 16000;
        let tolerance = 50;
        assert!(
            (resampled.len() as i64 - expected as i64).unsigned_abs() < tolerance,
            "expected ~{expected} samples, got {}",
            resampled.len()
        );
    }

    #[cfg(feature = "resample")]
    #[test]
    fn resample_upsample() {
        // 1 second at 16 kHz → 24 kHz
        let frame = AudioFrame::from_vec(vec![0.0f32; 16000], 16000);
        let resampled = frame.resample(24000).unwrap();
        assert_eq!(resampled.sample_rate(), 24000);
        let expected = 24000;
        let tolerance = 50;
        assert!(
            (resampled.len() as i64 - expected as i64).unsigned_abs() < tolerance,
            "expected ~{expected} samples, got {}",
            resampled.len()
        );
    }

    #[cfg(feature = "resample")]
    #[test]
    fn resample_short_input_upsample_large_ratio() {
        // The exact case from the wavekat-voice RTP path: a 20 ms G.711 frame
        // (160 samples @ 8 kHz) upsampled to 44.1 kHz. Before the fix this
        // returned `InsufficientOutputBufferSize`.
        let frame = AudioFrame::from_vec(vec![0.0f32; 160], 8000);
        let resampled = frame.resample(44_100).unwrap();
        assert_eq!(resampled.sample_rate(), 44_100);
        let expected = (160.0 * 44_100.0 / 8_000.0) as i64; // 882
        let actual = resampled.len() as i64;
        assert!(
            (actual - expected).unsigned_abs() < 50,
            "expected ~{expected} samples, got {actual}"
        );
    }

    #[cfg(feature = "resample")]
    #[test]
    fn resample_short_input_upsample_small_ratio() {
        // 160 samples @ 8 kHz → 16 kHz. Also failed before the fix even
        // though the ratio is modest, because nbr_input_frames < chunk_size.
        let frame = AudioFrame::from_vec(vec![0.0f32; 160], 8000);
        let resampled = frame.resample(16_000).unwrap();
        assert_eq!(resampled.sample_rate(), 16_000);
        let expected: i64 = 320;
        let actual = resampled.len() as i64;
        assert!(
            (actual - expected).unsigned_abs() < 50,
            "expected ~{expected} samples, got {actual}"
        );
    }

    #[cfg(feature = "resample")]
    #[test]
    fn resample_single_g711_frame_to_48k() {
        // The other common device rate: 160 @ 8 kHz → 48 kHz.
        let frame = AudioFrame::from_vec(vec![0.0f32; 160], 8000);
        let resampled = frame.resample(48_000).unwrap();
        assert_eq!(resampled.sample_rate(), 48_000);
        let expected: i64 = 960;
        let actual = resampled.len() as i64;
        assert!(
            (actual - expected).unsigned_abs() < 50,
            "expected ~{expected} samples, got {actual}"
        );
    }

    #[cfg(feature = "resample")]
    #[test]
    fn resample_preserves_sine_frequency() {
        // Generate a 440 Hz sine at 44100 Hz, resample to 16000 Hz,
        // then verify the dominant frequency is still ~440 Hz by
        // checking zero-crossing rate.
        let sr_in: u32 = 44100;
        let sr_out: u32 = 16000;
        let duration_secs = 1.0;
        let freq = 440.0;
        let n = (sr_in as f64 * duration_secs) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sr_in as f64).sin() as f32)
            .collect();

        let frame = AudioFrame::from_vec(samples, sr_in);
        let resampled = frame.resample(sr_out).unwrap();

        // Count zero crossings (sign changes)
        let s = resampled.samples();
        let crossings: usize = s
            .windows(2)
            .filter(|w| w[0].signum() != w[1].signum())
            .count();
        // A pure sine at f Hz has 2*f zero crossings per second
        let measured_freq = crossings as f64 / (2.0 * duration_secs);
        assert!(
            (measured_freq - freq).abs() < 5.0,
            "expected ~{freq} Hz, measured {measured_freq} Hz"
        );
    }

    #[cfg(feature = "resample")]
    #[test]
    fn streaming_resampler_same_rate_is_passthrough() {
        // No-op short-circuit: no resampler is built, no work is done,
        // samples pass through verbatim. Guards against accidentally
        // putting a same-rate stream through rubato (which adds group
        // delay we don't want).
        use crate::StreamingResampler;
        let mut r = StreamingResampler::new(16000, 16000, 160).unwrap();
        let input = vec![0.1, -0.2, 0.3, -0.4];
        let mut out = Vec::new();
        r.process(&input, &mut out).unwrap();
        assert_eq!(out, input);
    }

    #[cfg(feature = "resample")]
    #[test]
    fn streaming_resampler_short_input_chunked_calls() {
        // The exact shape `wavekat-voice`'s RTP receive path drives:
        // repeated 160-sample inputs at 8 kHz → 44.1 kHz. Each call
        // produces ~882 output samples; total over N calls is ~N × 882
        // (the first chunk may emit slightly less while rubato fills
        // its filter delay).
        use crate::StreamingResampler;
        let mut r = StreamingResampler::new(8000, 44100, 160).unwrap();
        let mut out = Vec::new();
        for _ in 0..10 {
            let input = vec![0.0f32; 160];
            r.process(&input, &mut out).unwrap();
        }
        // 10 × 160 input @ 8k = 200 ms; @ 44.1k that's ~8820 samples.
        // Allow generous tolerance for rubato's initial transient.
        let expected = (10 * 160 * 44100 / 8000) as i64;
        let actual = out.len() as i64;
        assert!(
            (actual - expected).unsigned_abs() < 2000,
            "expected ~{expected} samples, got {actual}"
        );
    }

    #[cfg(feature = "resample")]
    #[test]
    fn streaming_resampler_buffers_across_partial_calls() {
        // Splitting an input across two `process` calls must produce
        // the same output as one big call. Catches a regression where
        // partial input is dropped on the floor instead of buffered.
        use crate::StreamingResampler;
        let input: Vec<f32> = (0..320).map(|i| (i as f32) * 0.01).collect();

        let mut split_out = Vec::new();
        let mut r1 = StreamingResampler::new(8000, 16000, 160).unwrap();
        r1.process(&input[..50], &mut split_out).unwrap();
        // No full chunk yet — buffered.
        assert!(split_out.is_empty(), "no output before a full chunk");
        r1.process(&input[50..], &mut split_out).unwrap();

        let mut whole_out = Vec::new();
        let mut r2 = StreamingResampler::new(8000, 16000, 160).unwrap();
        r2.process(&input, &mut whole_out).unwrap();

        assert_eq!(
            split_out.len(),
            whole_out.len(),
            "split call must produce same number of samples as one-shot"
        );
        // The samples themselves must match too — same rubato state
        // either way.
        for (i, (a, b)) in split_out.iter().zip(whole_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "split vs whole differ at {i}: {a} vs {b}"
            );
        }
    }

    #[cfg(feature = "resample")]
    #[test]
    fn streaming_resampler_avoids_per_frame_edge_artifacts() {
        // The motivating regression: a stateless per-call resampler
        // (i.e. `AudioFrame::resample` invoked on each 160-sample
        // chunk) produces edge artifacts at every chunk boundary,
        // because rubato assumes silence before t=0 and after t=N for
        // each isolated chunk — sinc reconstruction near the edges
        // sees an abrupt step.
        //
        // We don't compare against a reference signal (group-delay
        // offsets across approaches make sample-index alignment
        // unreliable). Instead we check the output's own *smoothness*:
        // a band-limited signal at the input rate, upsampled, produces
        // a band-limited output, so consecutive-sample deltas are
        // bounded by `2π × freq / sr_out`. Edge artifacts show up as
        // spikes in that consecutive delta — much larger than the
        // smooth bound.
        use crate::StreamingResampler;
        let sr_in: u32 = 8000;
        let sr_out: u32 = 44100;
        let chunks = 30;
        let chunk_size = 160;

        // Mid-band sine that exercises the sinc filter without
        // touching the anti-aliasing edge.
        let freq = 600.0_f32;
        let signal: Vec<f32> = (0..chunks * chunk_size)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr_in as f32).sin())
            .collect();

        // Streaming: state carried across calls.
        let mut streaming = StreamingResampler::new(sr_in, sr_out, chunk_size).unwrap();
        let mut streaming_out: Vec<f32> = Vec::new();
        for c in 0..chunks {
            streaming
                .process(
                    &signal[c * chunk_size..(c + 1) * chunk_size],
                    &mut streaming_out,
                )
                .unwrap();
        }

        // Stateless per-chunk: fresh resampler every call (the bug).
        let mut stateless_out: Vec<f32> = Vec::new();
        for c in 0..chunks {
            let chunk =
                AudioFrame::from_vec(signal[c * chunk_size..(c + 1) * chunk_size].to_vec(), sr_in);
            let resampled = chunk.resample(sr_out).unwrap();
            stateless_out.extend_from_slice(resampled.samples());
        }

        // Skip the initial group-delay transient and the trailing
        // tail; we want steady-state behavior.
        let skip = 1500;
        let tail = 500;

        // Smooth bound: for a 600 Hz sine sampled at 44.1 kHz, the
        // maximum delta between adjacent samples is ~ 2π × 600 / 44100
        // ≈ 0.085. Allow generous headroom (4×) before we call a delta
        // "spiky."
        let expected_max_delta = 2.0 * std::f32::consts::PI * freq / sr_out as f32;
        let spike_threshold = expected_max_delta * 4.0;

        let count_spikes = |samples: &[f32], skip: usize, tail: usize| -> usize {
            if samples.len() < skip + tail {
                return 0;
            }
            samples[skip..samples.len() - tail]
                .windows(2)
                .filter(|w| (w[1] - w[0]).abs() > spike_threshold)
                .count()
        };

        let streaming_spikes = count_spikes(&streaming_out, skip, tail);
        let stateless_spikes = count_spikes(&stateless_out, skip, tail);

        // Streaming output should be smooth: essentially zero spikes
        // in steady state.
        assert!(
            streaming_spikes < 10,
            "streaming output should be smooth, found {streaming_spikes} sample-delta spikes (threshold {spike_threshold})"
        );
        // Stateless per-chunk output should have many spikes — one
        // per chunk boundary, at minimum. We have ~25 chunks in the
        // compared range, so expect at least 25 spikes.
        assert!(
            stateless_spikes > streaming_spikes * 5,
            "stateless per-chunk should have far more spikes than streaming; got stateless={stateless_spikes}, streaming={streaming_spikes}"
        );
    }

    #[cfg(feature = "resample")]
    #[test]
    fn streaming_resampler_rejects_zero_rate() {
        use crate::StreamingResampler;
        assert!(StreamingResampler::new(0, 16000, 160).is_err());
        assert!(StreamingResampler::new(16000, 0, 160).is_err());
        assert!(StreamingResampler::new(0, 0, 160).is_err());
    }

    #[cfg(feature = "resample")]
    #[test]
    fn streaming_resampler_rejects_zero_chunk_size() {
        use crate::StreamingResampler;
        assert!(StreamingResampler::new(8000, 16000, 0).is_err());
    }
}
