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
