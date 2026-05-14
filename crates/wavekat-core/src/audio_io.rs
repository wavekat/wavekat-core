//! Audio source/sink traits.
//!
//! These are the seam the WaveKat audio pipeline is drawn against:
//! whatever produces audio (a microphone, a TTS engine, a WAV file)
//! implements [`AudioSource`]; whatever consumes it (a speaker, an
//! RTP encoder, an ASR worker) implements [`AudioSink`]. Concrete
//! impls live in the consuming crates — cpal-backed mic/speaker in
//! `wavekat-voice`, a future agent-driven impl in `wavekat-agent`,
//! and so on — so that adding a new producer or consumer is "implement
//! the trait" rather than "rewrite the RTP path."
//!
//! The traits speak in [`AudioFrame<'static>`]: sample-rate-tagged
//! frames so consumers can resample to whatever rate the codec wants
//! without either side of the trait having to know the codec exists.

use core::future::Future;

use crate::AudioFrame;

/// Produces owned [`AudioFrame`]s. `next_frame().await` returns the
/// next frame when one is available, or `None` once the source has
/// run out (file ended, device closed, dialogue terminated). Each
/// frame's [`AudioFrame::sample_rate`] is set by the implementation —
/// consumers resample as needed.
pub trait AudioSource: Send {
    fn next_frame(&mut self) -> impl Future<Output = Option<AudioFrame<'static>>> + Send;
}

/// Consumes audio frames. Implementations may drop frames on
/// backpressure rather than block the caller; the alternative —
/// stalling — is worse on the RTP receive path, where it stalls the
/// whole pipeline.
pub trait AudioSink: Send {
    fn write_frame(&mut self, frame: AudioFrame<'_>) -> impl Future<Output = ()> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smallest possible impl-pair: confirms the traits can be
    /// implemented (in particular, that the `impl Future` returns are
    /// `Send`) and that frames flow through them end-to-end.
    #[derive(Default)]
    struct VecSink {
        frames: Vec<AudioFrame<'static>>,
    }

    impl AudioSink for VecSink {
        async fn write_frame(&mut self, frame: AudioFrame<'_>) {
            self.frames.push(frame.into_owned());
        }
    }

    struct OnceSource {
        frame: Option<AudioFrame<'static>>,
    }

    impl AudioSource for OnceSource {
        async fn next_frame(&mut self) -> Option<AudioFrame<'static>> {
            self.frame.take()
        }
    }

    #[tokio::test]
    async fn traits_compose_end_to_end() {
        let mut source = OnceSource {
            frame: Some(AudioFrame::from_vec(vec![0.5, -0.5], 8000)),
        };
        let mut sink = VecSink::default();

        let frame = source.next_frame().await.expect("frame");
        sink.write_frame(frame).await;
        assert!(source.next_frame().await.is_none());

        assert_eq!(sink.frames.len(), 1);
        assert_eq!(sink.frames[0].samples(), &[0.5, -0.5]);
        assert_eq!(sink.frames[0].sample_rate(), 8000);
    }
}
