// types.rs — shared data types used across all providers

use serde::{Deserialize, Serialize};

// ── Message types ────────────────────────────────────────────────────────────

/// The role of a message participant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

/// A single message in a conversation, provider-agnostic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: MessageContent,
    /// For `Role::Tool` messages: the `id` of the tool call this is a result for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// For `Role::Tool` messages: the function name (required by some providers).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: MessageContent::Text(content.into()),
            tool_call_id: None,
            name: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: MessageContent::Text(content.into()),
            tool_call_id: None,
            name: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::Text(content.into()),
            tool_call_id: None,
            name: None,
        }
    }

    /// Build an assistant message that contains one or more tool calls.
    pub fn assistant_with_tool_calls(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: MessageContent::ToolCalls(tool_calls),
            tool_call_id: None,
            name: None,
        }
    }

    /// Build a tool-result message (returned to the model after executing a tool).
    pub fn tool_result(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        result: impl Into<String>,
    ) -> Self {
        Self {
            role: Role::Tool,
            content: MessageContent::Text(result.into()),
            tool_call_id: Some(tool_call_id.into()),
            name: Some(tool_name.into()),
        }
    }

    /// Return the text of this message, if it is a plain-text message.
    pub fn text(&self) -> Option<&str> {
        match &self.content {
            MessageContent::Text(t) => Some(t),
            _ => None,
        }
    }
}

/// The payload of a [`ChatMessage`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text (the common case).
    Text(String),
    /// An assistant message that requested one or more tool calls.
    ToolCalls(Vec<ToolCall>),
    /// Anthropic-style multi-block content (text + tool_use + tool_result).
    Blocks(Vec<ContentBlock>),
}

/// A typed content block used by the Anthropic Messages API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
}

// ── Tool types ────────────────────────────────────────────────────────────────

/// A tool/function definition provided to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Unique name for the tool (a-z, A-Z, 0-9, underscores; max 64 chars).
    pub name: String,
    /// Human-readable description used by the model to decide when to call it.
    pub description: String,
    /// JSON Schema object describing the function parameters.
    pub parameters: serde_json::Value,
}

impl ToolDefinition {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
}

/// A tool call requested by the model.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolCall {
    /// Provider-assigned unique ID for this call instance.
    pub id: String,
    /// Name of the tool/function to invoke.
    pub name: String,
    /// Arguments to pass, as parsed JSON (not a string).
    pub arguments: serde_json::Value,
}

// ── Logprob types ─────────────────────────────────────────────────────────────

/// Log-probability data for a single generated token.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenLogprob {
    /// The token text.
    pub token: String,
    /// Natural-log probability. Convert to linear with `logprob.exp()`.
    pub logprob: f64,
    /// Raw UTF-8 bytes of the token, when provided.
    pub bytes: Vec<u8>,
    /// The top-N alternative tokens at this position.
    pub top_alternatives: Vec<TopTokenAlternative>,
}

impl TokenLogprob {
    /// Linear probability ∈ (0, 1].
    pub fn probability(&self) -> f64 {
        self.logprob.exp()
    }
}

/// One alternative token candidate at a given position.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TopTokenAlternative {
    pub token: String,
    pub logprob: f64,
}

// ── Response types ────────────────────────────────────────────────────────────

/// Why the model stopped generating.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum StopReason {
    /// Normal completion.
    #[default]
    Stop,
    /// Hit the `max_tokens` limit.
    MaxTokens,
    /// Model wants to call a tool.
    ToolUse,
    /// Content was filtered by safety policy.
    ContentFilter,
    /// Some other reason (contains the raw value from the API).
    Other(String),
}

impl StopReason {
    pub(crate) fn from_str(s: &str) -> Self {
        match s {
            "stop" | "end_turn" => StopReason::Stop,
            "length" | "max_tokens" => StopReason::MaxTokens,
            "tool_calls" | "tool_use" => StopReason::ToolUse,
            "content_filter" => StopReason::ContentFilter,
            other => StopReason::Other(other.to_string()),
        }
    }
}

/// The desired output format for structured generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseFormat {
    /// Plain text (default).
    Text,
    /// JSON object — the model is instructed to return valid JSON.
    /// Note: you must also ask for JSON in your prompt.
    JsonObject,
    /// Constrained JSON matching a JSON Schema.
    /// Supported by OpenAI (structured outputs), llama.cpp, and Ollama.
    JsonSchema {
        schema: serde_json::Value,
        /// Optional schema name shown to the model.
        name: Option<String>,
    },
}

/// The result of a successful LLM completion call.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LlmApiResult {
    /// The primary text generated by the model.
    pub generated_text: String,
    /// Number of tokens in the prompt/input.
    pub input_tokens_count: u64,
    /// Number of tokens generated.
    pub output_tokens_count: u64,
    /// Why generation stopped.
    pub stop_reason: StopReason,
    /// The exact model version that generated the response.
    pub model_used: String,
    /// Per-token log-probabilities. Empty if the provider or request didn't
    /// enable logprobs.
    pub logprobs: Vec<TokenLogprob>,
    /// Tool/function calls the model wants to make. Non-empty when
    /// `stop_reason == StopReason::ToolUse`.
    pub tool_calls: Vec<ToolCall>,
    /// Chain-of-thought / reasoning text from models that expose it
    /// (DeepSeek R1/V3, Claude with thinking, Ollama `think` param).
    pub reasoning_content: Option<String>,
}

impl LlmApiResult {
    /// Mean log-probability across all generated tokens.
    /// Returns `None` if logprobs were not requested.
    pub fn mean_logprob(&self) -> Option<f64> {
        if self.logprobs.is_empty() {
            return None;
        }
        let sum: f64 = self.logprobs.iter().map(|t| t.logprob).sum();
        Some(sum / self.logprobs.len() as f64)
    }

    /// Geometric mean probability — a rough proxy for generation confidence.
    /// Higher values (closer to 1.0) suggest lower hallucination risk.
    /// Returns `None` if logprobs were not requested.
    pub fn mean_probability(&self) -> Option<f64> {
        self.mean_logprob().map(f64::exp)
    }

    /// Minimum per-token probability — highlights the single weakest token,
    /// which can indicate a hallucination hotspot.
    pub fn min_token_probability(&self) -> Option<f64> {
        self.logprobs
            .iter()
            .map(|t| t.logprob)
            .reduce(f64::min)
            .map(f64::exp)
    }
}

// ── Batch types ────────────────────────────────────────────────────────────────

/// A single request in a batch submission.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    /// Caller-assigned unique ID (used to correlate results).
    pub custom_id: String,
    /// The messages for this request.
    pub messages: Vec<ChatMessage>,
    /// Optional tools available for this request.
    pub tools: Option<Vec<ToolDefinition>>,
    /// Optional override of the generator's system prompt.
    pub system_prompt: Option<String>,
}

impl BatchRequest {
    pub fn new(custom_id: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            custom_id: custom_id.into(),
            messages,
            tools: None,
            system_prompt: None,
        }
    }
}

/// A handle returned after successfully submitting a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchHandle {
    /// The provider-assigned batch ID.
    pub batch_id: String,
    /// The provider name (e.g. "chatgpt", "claude").
    pub provider: String,
    /// The current status of the batch.
    pub status: BatchStatus,
    /// URL or file ID to retrieve results from (once completed).
    pub output_file_id: Option<String>,
}

/// Processing status of a batch job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    Validating,
    InProgress,
    Finalizing,
    Completed,
    Failed,
    Expired,
    Cancelling,
    Cancelled,
}

impl BatchStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            BatchStatus::Completed | BatchStatus::Failed | BatchStatus::Expired | BatchStatus::Cancelled
        )
    }

    pub(crate) fn from_str(s: &str) -> Self {
        match s {
            "validating" => BatchStatus::Validating,
            "in_progress" => BatchStatus::InProgress,
            "finalizing" => BatchStatus::Finalizing,
            "completed" => BatchStatus::Completed,
            "failed" => BatchStatus::Failed,
            "expired" => BatchStatus::Expired,
            "cancelling" => BatchStatus::Cancelling,
            "cancelled" => BatchStatus::Cancelled,
            _ => BatchStatus::InProgress,
        }
    }
}

/// The result of a single request within a completed batch.
#[derive(Debug, Clone)]
pub struct BatchRequestResult {
    /// The `custom_id` from the original `BatchRequest`.
    pub custom_id: String,
    pub result: Result<LlmApiResult, crate::error::SamvadSetuError>,
}
