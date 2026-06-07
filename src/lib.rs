//! # SamvadSetu — Multi-provider LLM API client
//!
//! A Rust-native library for sending chat-completion requests to multiple LLM
//! providers with a single, unified API.  The name is Sanskrit for "dialogue
//! bridge": saṃvāda (संवाद) = dialogue, setu (सेतु) = bridge.
//!
//! ## Supported providers
//!
//! | Provider          | `llm_service` string | Auth env var          |
//! |-------------------|---------------------|-----------------------|
//! | OpenAI / ChatGPT  | `"chatgpt"` / `"openai"` | `OPENAI_API_KEY`  |
//! | DeepSeek          | `"deepseek"`        | `DEEPSEEK_API_KEY`    |
//! | Alibaba Qwen      | `"qwen"`            | `DASHSCOPE_API_KEY`   |
//! | llama.cpp server  | `"llamacpp"`        | `LLAMACPP_API_KEY`    |
//! | Anthropic Claude  | `"claude"` / `"anthropic"` | `ANTHROPIC_API_KEY` |
//! | Google Gemini     | `"gemini"`          | `GOOGLE_API_KEY`      |
//! | Google GenAI      | `"google_genai"`    | `GOOGLE_API_KEY`      |
//! | Ollama            | `"ollama"`          | _(none)_              |
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use samvadsetu::llm::LLMTextGenBuilder;
//! use samvadsetu::types::ChatMessage;
//!
//! let llm_gen = LLMTextGenBuilder::build("chatgpt", "gpt-4o-mini", 60, None, None)
//!     .expect("build failed");
//!
//! let msgs = vec![
//!     ChatMessage::system("You are a helpful assistant."),
//!     ChatMessage::user("How is a rainbow formed?"),
//! ];
//!
//! let result = llm_gen.generate_text(&msgs, None, None).expect("API call failed");
//! println!("{}", result.generated_text);
//!
//! // Token-level confidence (hallucination proxy)
//! if let Some(p) = result.mean_probability() {
//!     println!("Mean token confidence: {:.1}%", p * 100.0);
//! }
//! ```

pub mod error;
pub mod llm;
pub mod types;

pub mod providers {
    pub mod anthropic;
    pub mod google;
    pub mod ollama;
    pub mod openai;
}

pub mod batch;

// Re-export the most commonly used items at the crate root for convenience.
pub use error::SamvadSetuError;
pub use llm::{LLMTextGenBuilder, LLMTextGenerator};
pub use types::{
    BatchHandle, BatchRequest, BatchRequestResult, BatchStatus, ChatMessage, ContentBlock,
    LlmApiResult, MessageContent, ResponseFormat, Role, StopReason, TokenLogprob, ToolCall,
    ToolDefinition, TopTokenAlternative,
};
