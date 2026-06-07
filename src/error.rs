// error.rs — unified error type for all providers

use std::fmt;

/// All errors that can arise when calling an LLM API.
#[derive(Debug, Clone)]
pub enum SamvadSetuError {
    /// The server returned a non-success HTTP status.
    Http { status: u16, body: String },

    /// The provider returned a structured error in the response body.
    Provider {
        /// Provider-specific error type string (e.g. "invalid_request_error").
        error_type: String,
        /// Human-readable error message.
        message: String,
        /// The parameter that caused the error, if reported.
        param: Option<String>,
        /// Provider-specific error code, if reported.
        code: Option<String>,
    },

    /// The API rate limit was exceeded.
    RateLimit {
        /// Seconds to wait before retrying, if the server provided it.
        retry_after_secs: Option<u64>,
        message: String,
    },

    /// Failed to serialize the request or deserialize the response.
    Parse {
        message: String,
        /// The raw response text that failed to parse, when available.
        raw_response: Option<String>,
    },

    /// A required field was missing or invalid in the library configuration.
    Config(String),

    /// Authentication failed (missing or invalid API key).
    Auth(String),

    /// The request timed out before a response was received.
    Timeout,

    /// A lower-level network error occurred.
    Network(String),

    /// A batch job was queried but has not finished yet.
    BatchNotComplete { batch_id: String, status: String },

    /// The requested feature is not supported by this provider.
    UnsupportedFeature { provider: String, feature: String },
}

impl fmt::Display for SamvadSetuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http { status, body } => write!(f, "HTTP {status}: {body}"),
            Self::Provider { error_type, message, code, param } => {
                write!(f, "Provider error [{error_type}]")?;
                if let Some(c) = code {
                    write!(f, " (code: {c})")?;
                }
                if let Some(p) = param {
                    write!(f, " (param: {p})")?;
                }
                write!(f, ": {message}")
            }
            Self::RateLimit { retry_after_secs, message } => match retry_after_secs {
                Some(s) => write!(f, "Rate limited (retry after {s}s): {message}"),
                None => write!(f, "Rate limited: {message}"),
            },
            Self::Parse { message, .. } => write!(f, "Parse error: {message}"),
            Self::Config(msg) => write!(f, "Config error: {msg}"),
            Self::Auth(msg) => write!(f, "Auth error: {msg}"),
            Self::Timeout => write!(f, "Request timed out"),
            Self::Network(msg) => write!(f, "Network error: {msg}"),
            Self::BatchNotComplete { batch_id, status } => {
                write!(f, "Batch {batch_id} not complete (status: {status})")
            }
            Self::UnsupportedFeature { provider, feature } => {
                write!(f, "{provider} does not support {feature}")
            }
        }
    }
}

impl std::error::Error for SamvadSetuError {}

/// Parse a reqwest error into a `SamvadSetuError`.
pub(crate) fn from_reqwest_error(e: reqwest::Error) -> SamvadSetuError {
    if e.is_timeout() {
        SamvadSetuError::Timeout
    } else if e.is_connect() {
        SamvadSetuError::Network(format!("Connection failed: {e}"))
    } else {
        SamvadSetuError::Network(e.to_string())
    }
}

/// Try to extract a structured `Provider` error from a JSON error body returned by
/// OpenAI-compatible APIs ({"error": {"type", "message", "param", "code"}}).
pub(crate) fn parse_openai_error_body(
    status: u16,
    body: &str,
) -> SamvadSetuError {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(body)
        && let Some(err) = val.get("error")
    {
        return SamvadSetuError::Provider {
            error_type: err
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            message: err
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or(body)
                .to_string(),
            param: err.get("param").and_then(|v| v.as_str()).map(str::to_string),
            code: err.get("code").and_then(|v| v.as_str()).map(str::to_string),
        };
    }
    SamvadSetuError::Http { status, body: body.to_string() }
}

/// Try to extract a structured error from an Anthropic error body
/// ({"type":"error","error":{"type","message"}}).
pub(crate) fn parse_anthropic_error_body(status: u16, body: &str) -> SamvadSetuError {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(body)
        && let Some(err) = val.get("error")
    {
        return SamvadSetuError::Provider {
            error_type: err
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("api_error")
                .to_string(),
            message: err
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or(body)
                .to_string(),
            param: None,
            code: None,
        };
    }
    SamvadSetuError::Http { status, body: body.to_string() }
}
