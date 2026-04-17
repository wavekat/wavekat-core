/// Unified error type for `wavekat-core`.
///
/// Covers all fallible operations in this crate (currently WAV I/O).
/// Downstream crates can convert via `From<CoreError>` for ergonomic `?` usage.
///
/// # Variants
///
/// - [`Io`](CoreError::Io) — file or stream I/O failure
/// - [`Audio`](CoreError::Audio) — format or codec error (e.g. invalid WAV)
#[derive(Debug)]
pub enum CoreError {
    /// File or stream I/O failure.
    Io(std::io::Error),
    /// Audio format or codec error (e.g. unsupported WAV format).
    Audio(String),
}

impl std::fmt::Display for CoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoreError::Io(e) => write!(f, "{e}"),
            CoreError::Audio(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for CoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CoreError::Io(e) => Some(e),
            CoreError::Audio(_) => None,
        }
    }
}

impl From<std::io::Error> for CoreError {
    fn from(e: std::io::Error) -> Self {
        CoreError::Io(e)
    }
}

#[cfg(feature = "wav")]
impl From<hound::Error> for CoreError {
    fn from(e: hound::Error) -> Self {
        match e {
            hound::Error::IoError(io) => CoreError::Io(io),
            other => CoreError::Audio(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn display_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err = CoreError::Io(io_err);
        assert_eq!(err.to_string(), "file missing");
    }

    #[test]
    fn display_audio() {
        let err = CoreError::Audio("bad format".into());
        assert_eq!(err.to_string(), "bad format");
    }

    #[test]
    fn source_io_returns_inner() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let err = CoreError::Io(io_err);
        assert!(err.source().is_some());
    }

    #[test]
    fn source_audio_returns_none() {
        let err = CoreError::Audio("oops".into());
        assert!(err.source().is_none());
    }

    #[test]
    fn from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let err = CoreError::from(io_err);
        assert!(matches!(err, CoreError::Io(_)));
        assert_eq!(err.to_string(), "denied");
    }

    #[cfg(feature = "wav")]
    #[test]
    fn from_hound_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "not found");
        let hound_err = hound::Error::IoError(io_err);
        let err = CoreError::from(hound_err);
        assert!(matches!(err, CoreError::Io(_)));
        assert_eq!(err.to_string(), "not found");
    }

    #[cfg(feature = "wav")]
    #[test]
    fn from_hound_format_error() {
        let hound_err = hound::Error::Unsupported;
        let err = CoreError::from(hound_err);
        assert!(matches!(err, CoreError::Audio(_)));
    }
}
