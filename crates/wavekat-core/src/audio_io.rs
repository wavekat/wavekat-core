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

    /// In-memory sink: collects frames in a Vec.
    #[derive(Default)]
    struct VecSink {
        frames: Vec<AudioFrame<'static>>,
    }

    impl AudioSink for VecSink {
        async fn write_frame(&mut self, frame: AudioFrame<'_>) {
            self.frames.push(frame.into_owned());
        }
    }

    /// Source that yields a single frame then ends.
    struct OnceSource {
        frame: Option<AudioFrame<'static>>,
    }

    impl AudioSource for OnceSource {
        async fn next_frame(&mut self) -> Option<AudioFrame<'static>> {
            self.frame.take()
        }
    }

    /// Source that drains a queue of pre-loaded frames, then signals
    /// exhaustion with `None`. Mirrors what a file-backed or
    /// agent-backed source looks like in practice.
    struct QueueSource {
        frames: std::collections::VecDeque<AudioFrame<'static>>,
    }

    impl AudioSource for QueueSource {
        async fn next_frame(&mut self) -> Option<AudioFrame<'static>> {
            self.frames.pop_front()
        }
    }

    /// Sink that drops every frame past `cap`. Models the "drop on
    /// backpressure" pattern the trait docs call out.
    struct CapSink {
        cap: usize,
        frames: Vec<AudioFrame<'static>>,
        dropped: usize,
    }

    impl AudioSink for CapSink {
        async fn write_frame(&mut self, frame: AudioFrame<'_>) {
            if self.frames.len() >= self.cap {
                self.dropped += 1;
                return;
            }
            self.frames.push(frame.into_owned());
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

    #[tokio::test]
    async fn source_drains_in_order_then_returns_none() {
        let mut source = QueueSource {
            frames: vec![
                AudioFrame::from_vec(vec![0.1], 8000),
                AudioFrame::from_vec(vec![0.2], 8000),
                AudioFrame::from_vec(vec![0.3], 8000),
            ]
            .into(),
        };
        let mut sink = VecSink::default();
        while let Some(f) = source.next_frame().await {
            sink.write_frame(f).await;
        }
        assert_eq!(sink.frames.len(), 3);
        assert_eq!(sink.frames[0].samples(), &[0.1]);
        assert_eq!(sink.frames[1].samples(), &[0.2]);
        assert_eq!(sink.frames[2].samples(), &[0.3]);

        // Past exhaustion the source must keep returning None — callers
        // rely on this to unblock their drain loop.
        assert!(source.next_frame().await.is_none());
        assert!(source.next_frame().await.is_none());
    }

    #[tokio::test]
    async fn sink_with_capacity_drops_overflow() {
        let mut sink = CapSink {
            cap: 2,
            frames: Vec::new(),
            dropped: 0,
        };
        for i in 0..5 {
            sink.write_frame(AudioFrame::from_vec(vec![i as f32], 8000))
                .await;
        }
        assert_eq!(sink.frames.len(), 2);
        assert_eq!(sink.dropped, 3);
        // Cap policy is FIFO-keep / tail-drop here; the first two went
        // through, the rest got dropped.
        assert_eq!(sink.frames[0].samples(), &[0.0]);
        assert_eq!(sink.frames[1].samples(), &[1.0]);
    }

    #[tokio::test]
    async fn frame_sample_rate_round_trips_through_sink() {
        // Different impls emit at their native rates — the sink must
        // preserve `sample_rate` so the consumer can resample
        // downstream without losing the original rate.
        let mut sink = VecSink::default();
        sink.write_frame(AudioFrame::from_vec(vec![0.0], 8000))
            .await;
        sink.write_frame(AudioFrame::from_vec(vec![0.0], 16_000))
            .await;
        sink.write_frame(AudioFrame::from_vec(vec![0.0], 48_000))
            .await;
        let rates: Vec<u32> = sink.frames.iter().map(|f| f.sample_rate()).collect();
        assert_eq!(rates, vec![8000, 16_000, 48_000]);
    }

    /// Compile-time check that the trait bounds (the trait is `Send`,
    /// and so are the returned futures) actually compose. If anything
    /// drops the `Send` bound, this test stops compiling — which is
    /// what we want, because the RTP path holds these impls across
    /// `.await` points in a multi-threaded runtime.
    #[test]
    fn impls_are_send() {
        fn assert_send<T: Send>(_: &T) {}
        let source = OnceSource { frame: None };
        let sink = VecSink::default();
        assert_send(&source);
        assert_send(&sink);
    }
}
