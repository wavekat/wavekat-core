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
        use rubato::{
            Async, FixedAsync, Resampler, SincInterpolationParameters, SincInterpolationType,
            WindowFunction,
        };

        if self.sample_rate == target_rate {
            return Ok(self.clone().into_owned());
        }

        if self.is_empty() {
            return Ok(AudioFrame::from_vec(Vec::new(), target_rate));
        }

        let ratio = target_rate as f64 / self.sample_rate as f64;
        let nbr_input_frames = self.samples.len();

        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Cubic,
            oversampling_factor: 128,
            window: WindowFunction::BlackmanHarris2,
        };

        let mut resampler = Async::<f32>::new_sinc(ratio, 1.0, &params, 1024, 1, FixedAsync::Input)
            .map_err(|e| crate::CoreError::Audio(e.to_string()))?;

        // Allocate output buffer with headroom
        let out_len = (nbr_input_frames as f64 * ratio) as usize + 1024;
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
}
