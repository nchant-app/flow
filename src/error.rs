//! Error types for the mai-timing crate.

/// Errors that can occur during timing model operations.
#[derive(Debug, thiserror::Error)]
pub enum TimingError {
    /// Failed to read or write a file.
    #[error("IO error at '{path}': {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// Failed to parse a YAML file.
    #[error("YAML parse error at '{path}': {source}")]
    Yaml {
        path: String,
        #[source]
        source: serde_yaml::Error,
    },

    /// Expected a directory but got something else.
    #[error("'{path}' is not a directory")]
    NotADirectory { path: String },

    /// No TextGrid files found in the given directory.
    #[error("no TextGrid files found in '{dir}'")]
    NoTextGridFiles { dir: String },

    /// A TextGrid file could not be parsed.
    #[error("TextGrid error in '{path}': {reason}")]
    TextGrid { path: String, reason: String },

    /// Empty input was provided where phonemes were expected.
    #[error("empty input: no phonemes provided")]
    EmptyInput,

    /// A catch-all for context-specific errors (e.g. CLI messages).
    #[error("{0}")]
    Other(String),
}

impl TimingError {
    /// Create an IO error with path context.
    pub fn io(path: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    /// Create a YAML parse error with path context.
    pub fn yaml(path: impl Into<String>, source: serde_yaml::Error) -> Self {
        Self::Yaml {
            path: path.into(),
            source,
        }
    }
}
