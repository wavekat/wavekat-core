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

    #[test]
    fn from_vec_is_zero_copy() {
        let samples = vec![0.5f32, -0.5];
        let ptr = samples.as_ptr();
        let frame = AudioFrame::from_vec(samples, 24000);
        assert_eq!(frame.samples().as_ptr(), ptr);
        assert_eq!(frame.sample_rate(), 24000);
    }
}
