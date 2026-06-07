// llm.rs — core generator struct, builder, and HTTP client factory

use crate::error::SamvadSetuError;
use crate::providers::{anthropic, google, ollama, openai};
use crate::types::{
    BatchHandle, BatchRequest, BatchRequestResult, ChatMessage, LlmApiResult, ResponseFormat,
    ToolDefinition,
};
use config::Config;
use log::{error, info};
use reqwest::header::{HeaderMap, HeaderValue};
use std::cmp::max;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::thread;

// ── Default base URLs ─────────────────────────────────────────────────────────

pub const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";
pub const DEEPSEEK_API_URL: &str = "https://api.deepseek.com/chat/completions";
/// Default is Singapore region; override via config `api_url` or `svc_base_url`.
pub const QWEN_API_URL: &str =
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions";
pub const LLAMACPP_API_URL: &str = "http://127.0.0.1:8080/v1/chat/completions";
pub const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
pub const GEMINI_API_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models";
pub const OLLAMA_CHAT_URL: &str = "http://127.0.0.1:11434/api/chat";

/// Minimum enforced gap between successive requests to the same API key
/// when `shared_lock` rate-limiting is enabled.
pub const MIN_GAP_BTWN_RQST_SECS: u64 = 6;

// ── Generator struct ──────────────────────────────────────────────────────────

/// A configured LLM API client for a specific provider and model.
///
/// Build one with [`LLMTextGenBuilder::build`] or
/// [`LLMTextGenBuilder::build_from_config`], then call [`generate_text`] as
/// many times as needed.  The struct is intentionally stateless with respect
/// to conversation history — callers own and pass the full `messages` slice on
/// every call.
///
/// [`generate_text`]: LLMTextGenerator::generate_text
#[derive(Debug)]
pub struct LLMTextGenerator {
    /// Provider identifier: "chatgpt", "deepseek", "qwen", "llamacpp",
    /// "claude", "gemini", "google_genai", "ollama".
    pub llm_service: String,

    /// Pre-configured blocking HTTP client (with auth headers attached).
    pub api_client: reqwest::blocking::Client,

    /// API key stored as a query-string parameter (legacy Gemini only).
    /// For all other providers the key is embedded in the client headers.
    pub api_key: String,

    /// Request timeout in seconds (0 = use the client's built-in timeout).
    pub fetch_timeout: u64,

    /// Sampling temperature (0.0 = deterministic, 1.0+ = creative).
    pub model_temperature: f64,

    /// Maximum tokens to generate per call.
    pub max_tok_gen: usize,

    /// Model identifier string sent to the API.
    pub model_name: String,

    /// Context-window length (mainly used by Ollama).
    pub num_context: usize,

    /// Base URL of the API service.
    pub svc_base_url: String,

    /// Optional default system prompt.  If `None`, callers should include a
    /// `ChatMessage::system(…)` in the messages slice instead.
    pub system_prompt: Option<String>,

    /// Shared mutex for rate-limiting across threads.  The stored value is a
    /// UNIX timestamp (seconds) of the last successful request.
    pub shared_lock: Option<Arc<Mutex<isize>>>,

    /// Minimum gap (seconds) between requests when `shared_lock` is active.
    pub min_gap_btwn_rqsts_secs: u64,
}

impl LLMTextGenerator {
    /// Generate a completion for the given conversation.
    ///
    /// * `messages` – Full conversation history (include a
    ///   [`ChatMessage::system`] entry if you want a system prompt).
    /// * `tools` – Optional tool definitions the model may call.
    /// * `response_format` – Optional structured-output specification.
    ///
    /// When `shared_lock` is set, calls are serialised and rate-limited.
    pub fn generate_text(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[ToolDefinition]>,
        response_format: Option<&ResponseFormat>,
    ) -> Result<LlmApiResult, SamvadSetuError> {
        if let Some(lock) = self.shared_lock.clone() {
            match lock.lock() {
                Ok(mut last_ts) => {
                    let result = self.generate_text_inner(messages, tools, response_format);
                    // Enforce minimum gap between requests.
                    let now_secs = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let elapsed = now_secs.saturating_sub(max(*last_ts, 0) as u64);
                    if elapsed < self.min_gap_btwn_rqsts_secs {
                        let wait = self.min_gap_btwn_rqsts_secs - elapsed;
                        thread::sleep(Duration::from_secs(wait));
                    }
                    *last_ts = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs() as isize;
                    result
                }
                Err(_) => Err(SamvadSetuError::Network(
                    "Rate-limit mutex is poisoned".to_string(),
                )),
            }
        } else {
            self.generate_text_inner(messages, tools, response_format)
        }
    }

    /// Internal dispatch — no rate-limit mutex.
    pub fn generate_text_inner(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[ToolDefinition]>,
        response_format: Option<&ResponseFormat>,
    ) -> Result<LlmApiResult, SamvadSetuError> {
        match self.llm_service.as_str() {
            "chatgpt" | "openai" | "deepseek" | "qwen" | "llamacpp" => {
                openai::http_post_openai_compat(
                    self,
                    &self.api_client,
                    messages,
                    tools,
                    response_format,
                )
            }
            "claude" | "anthropic" => anthropic::http_post_anthropic(
                self,
                &self.api_client,
                messages,
                tools,
            ),
            "gemini" => google::http_post_gemini(self, &self.api_client, messages, tools),
            "google_genai" => {
                google::http_post_google_genai(self, &self.api_client, messages, tools, response_format)
            }
            "ollama" => {
                ollama::http_post_chat_ollama(self, &self.api_client, messages, tools, response_format)
            }
            unknown => Err(SamvadSetuError::Config(format!(
                "Unknown LLM service: '{unknown}'. Valid values: chatgpt, openai, deepseek, \
                 qwen, llamacpp, claude, anthropic, gemini, google_genai, ollama"
            ))),
        }
    }

    // ── Batch helpers ─────────────────────────────────────────────────────────

    /// Submit a batch of requests (OpenAI or Anthropic only).
    ///
    /// Returns a [`BatchHandle`] that can be polled with [`check_batch_status`]
    /// and retrieved with [`retrieve_batch_results`].
    ///
    /// [`check_batch_status`]: LLMTextGenerator::check_batch_status
    /// [`retrieve_batch_results`]: LLMTextGenerator::retrieve_batch_results
    pub fn submit_batch(
        &self,
        requests: Vec<BatchRequest>,
    ) -> Result<BatchHandle, SamvadSetuError> {
        match self.llm_service.as_str() {
            "chatgpt" | "openai" | "deepseek" | "qwen" => {
                crate::batch::openai::submit_batch(self, &self.api_client, requests)
            }
            "claude" | "anthropic" => {
                crate::batch::anthropic::submit_batch(self, &self.api_client, requests)
            }
            svc => Err(SamvadSetuError::UnsupportedFeature {
                provider: svc.to_string(),
                feature: "batch API".to_string(),
            }),
        }
    }

    /// Poll the status of a previously submitted batch.
    pub fn check_batch_status(
        &self,
        handle: &BatchHandle,
    ) -> Result<BatchHandle, SamvadSetuError> {
        match self.llm_service.as_str() {
            "chatgpt" | "openai" | "deepseek" | "qwen" => {
                crate::batch::openai::get_batch_status(self, &self.api_client, handle)
            }
            "claude" | "anthropic" => {
                crate::batch::anthropic::get_batch_status(self, &self.api_client, handle)
            }
            svc => Err(SamvadSetuError::UnsupportedFeature {
                provider: svc.to_string(),
                feature: "batch API".to_string(),
            }),
        }
    }

    /// Download results for a completed batch.
    pub fn retrieve_batch_results(
        &self,
        handle: &BatchHandle,
    ) -> Result<Vec<BatchRequestResult>, SamvadSetuError> {
        match self.llm_service.as_str() {
            "chatgpt" | "openai" | "deepseek" | "qwen" => {
                crate::batch::openai::retrieve_batch_results(self, &self.api_client, handle)
            }
            "claude" | "anthropic" => {
                crate::batch::anthropic::retrieve_batch_results(self, &self.api_client, handle)
            }
            svc => Err(SamvadSetuError::UnsupportedFeature {
                provider: svc.to_string(),
                feature: "batch API".to_string(),
            }),
        }
    }
}

// ── Builder ───────────────────────────────────────────────────────────────────

pub struct LLMTextGenBuilder;

impl LLMTextGenBuilder {
    /// Build a generator with sensible defaults for the given provider.
    ///
    /// # Arguments
    ///
    /// * `llm_api_name`         – Provider name (see [`LLMTextGenerator::llm_service`]).
    /// * `model_name`           – Model identifier to send in requests.
    /// * `network_timeout_secs` – Seconds before a request times out.
    /// * `proxy_server`         – Optional HTTPS proxy URL.
    /// * `api_access_mutex`     – Optional shared mutex for cross-thread rate-limiting.
    pub fn build(
        llm_api_name: &str,
        model_name: &str,
        network_timeout_secs: u64,
        proxy_server: Option<String>,
        api_access_mutex: Option<Arc<Mutex<isize>>>,
    ) -> Option<LLMTextGenerator> {
        let (svc_url, api_key, custom_headers, min_gap) = match llm_api_name {
            "chatgpt" | "openai" => {
                let key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
                let headers = openai::prepare_bearer_headers(&key);
                (OPENAI_API_URL.to_string(), String::new(), Some(headers), MIN_GAP_BTWN_RQST_SECS)
            }
            "deepseek" => {
                let key = std::env::var("DEEPSEEK_API_KEY").unwrap_or_default();
                let headers = openai::prepare_bearer_headers(&key);
                (DEEPSEEK_API_URL.to_string(), String::new(), Some(headers), MIN_GAP_BTWN_RQST_SECS)
            }
            "qwen" => {
                let key = std::env::var("DASHSCOPE_API_KEY").unwrap_or_default();
                let headers = openai::prepare_bearer_headers(&key);
                (QWEN_API_URL.to_string(), String::new(), Some(headers), MIN_GAP_BTWN_RQST_SECS)
            }
            "llamacpp" => {
                // llama.cpp server usually needs no auth; allow override via env.
                let key = std::env::var("LLAMACPP_API_KEY").unwrap_or_default();
                let headers = if key.is_empty() {
                    None
                } else {
                    Some(openai::prepare_bearer_headers(&key))
                };
                (LLAMACPP_API_URL.to_string(), String::new(), headers, 0)
            }
            "claude" | "anthropic" => {
                let key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
                let headers = anthropic::prepare_anthropic_headers(&key);
                (ANTHROPIC_API_URL.to_string(), String::new(), Some(headers), MIN_GAP_BTWN_RQST_SECS)
            }
            "gemini" => {
                let key = std::env::var("GOOGLE_API_KEY").unwrap_or_default();
                // For legacy Gemini the key goes in the URL as ?key=…
                (GEMINI_API_URL.to_string(), key, None, MIN_GAP_BTWN_RQST_SECS)
            }
            "google_genai" => {
                let key = std::env::var("GOOGLE_API_KEY").unwrap_or_default();
                let headers = google::prepare_googlegenai_headers(&key);
                (GEMINI_API_URL.to_string(), String::new(), Some(headers), MIN_GAP_BTWN_RQST_SECS)
            }
            "ollama" => {
                (OLLAMA_CHAT_URL.to_string(), String::new(), None, 0)
            }
            _ => return None,
        };

        let client = build_llm_api_client(
            network_timeout_secs,
            network_timeout_secs,
            proxy_server,
            custom_headers,
        );

        Some(LLMTextGenerator {
            llm_service: llm_api_name.to_string(),
            api_client: client,
            api_key,
            fetch_timeout: network_timeout_secs,
            model_temperature: 0.0,
            max_tok_gen: 8192,
            model_name: model_name.to_string(),
            num_context: 8192,
            svc_base_url: svc_url,
            system_prompt: None,
            shared_lock: api_access_mutex,
            min_gap_btwn_rqsts_secs: min_gap,
        })
    }

    /// Build a generator from a `config::Config` object.
    ///
    /// Reads from the `[llm_apis."<name>"]` table.  Supported keys:
    /// `model_name`, `api_url`, `temperature`, `max_gen_tokens`,
    /// `max_context_len`, `min_gap_btwn_rqsts_secs`, `model_api_timeout`,
    /// `system_prompt`.
    pub fn build_from_config(app_config: &Config, llm_svc_name: &str) -> Option<LLMTextGenerator> {
        let mutex = Arc::new(Mutex::new(0));
        let mut llm_gen = LLMTextGenBuilder::build(llm_svc_name, "", 60, None, Some(mutex))?;

        let config_table = match app_config.get_table("llm_apis") {
            Ok(t) => t,
            Err(e) => {
                error!("Config: missing [llm_apis] table: {e}");
                return None;
            }
        };

        let entry = match config_table.get(llm_svc_name) {
            Some(v) => v.clone(),
            None => {
                error!("Config: no entry for '{llm_svc_name}' in [llm_apis]");
                return None;
            }
        };

        let table = match entry.into_table() {
            Ok(t) => t,
            Err(e) => {
                error!("Config: [llm_apis.\"{llm_svc_name}\"] is not a table: {e}");
                return None;
            }
        };

        info!("Loading LLM config for '{llm_svc_name}'");

        macro_rules! get_int {
            ($key:literal, $field:ident) => {
                if let Some(v) = table.get($key) {
                    llm_gen.$field = max(0, v.clone().into_int().unwrap_or_default()) as _;
                }
            };
        }
        macro_rules! get_float {
            ($key:literal, $field:ident) => {
                if let Some(v) = table.get($key) {
                    llm_gen.$field = v.clone().into_float().unwrap_or_default();
                }
            };
        }
        macro_rules! get_string {
            ($key:literal, $field:ident) => {
                if let Some(v) = table.get($key) {
                    llm_gen.$field = v.clone().into_string().unwrap_or_default();
                }
            };
        }

        get_string!("model_name", model_name);
        get_string!("api_url", svc_base_url);
        get_float!("temperature", model_temperature);
        get_int!("max_gen_tokens", max_tok_gen);
        get_int!("max_context_len", num_context);
        get_int!("min_gap_btwn_rqsts_secs", min_gap_btwn_rqsts_secs);
        get_int!("model_api_timeout", fetch_timeout);

        if let Some(v) = table.get("system_prompt")
            && let Ok(s) = v.clone().into_string()
            && !s.is_empty()
        {
            llm_gen.system_prompt = Some(s);
        }

        Some(llm_gen)
    }
}

// ── HTTP client factory ───────────────────────────────────────────────────────

/// Build a blocking `reqwest::Client` with the given settings.
pub fn build_llm_api_client(
    connect_timeout: u64,
    fetch_timeout: u64,
    proxy_url: Option<String>,
    custom_headers: Option<HeaderMap>,
) -> reqwest::blocking::Client {
    let pool_idle_timeout = (connect_timeout + fetch_timeout) * 5;

    let mut headers = custom_headers.unwrap_or_default();
    headers.insert(
        reqwest::header::CONNECTION,
        HeaderValue::from_static("keep-alive"),
    );
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );

    let builder = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(fetch_timeout))
        .connect_timeout(Duration::from_secs(connect_timeout))
        .default_headers(headers)
        .gzip(true)
        .pool_idle_timeout(Duration::from_secs(pool_idle_timeout))
        .pool_max_idle_per_host(1);

    let builder = if let Some(proxy_str) = proxy_url {
        match reqwest::Proxy::https(&proxy_str) {
            Ok(proxy) => builder.proxy(proxy),
            Err(e) => {
                error!("Cannot configure proxy '{proxy_str}': {e}");
                builder
            }
        }
    } else {
        builder
    };

    builder
        .build()
        .expect("Failed to build HTTP client — check system TLS configuration")
}
