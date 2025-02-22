// file: llm.rs

#![allow(unused_imports)]
#![warn(clippy::unwrap_used)]
#![warn(clippy::get_unwrap)]
#![warn(clippy::manual_unwrap_or)]
#![warn(clippy::manual_unwrap_or_default)]
#![warn(clippy::map_unwrap_or)]
#![warn(clippy::option_env_unwrap)]
#![warn(clippy::or_then_unwrap)]
#![warn(clippy::panicking_unwrap)]
#![warn(clippy::unnecessary_literal_unwrap)]
#![warn(clippy::unnecessary_unwrap)]
#![warn(clippy::unwrap_in_result)]
#![warn(clippy::unwrap_used)]

use crate::providers::google::{
    http_post_json_gemini, http_post_json_google_genai, prepare_gemini_api_payload,
    prepare_google_genai_api_payload, prepare_googlegenai_headers,
};
use crate::providers::ollama::{http_post_json_ollama, prepare_ollama_payload};
use crate::providers::openai::{
    http_post_json_chatgpt, prepare_chatgpt_headers, prepare_chatgpt_payload,
};
use config;
use config::Config;
use log::{debug, error, info};
use reqwest;
use reqwest::header::{HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};
use std::cmp::max;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use std::{fmt, thread};

pub const MIN_GAP_BTWN_RQST_SECS: u64 = 6;

#[derive(Debug)]
pub struct LLMTextGenerator {
    pub llm_service: String,
    pub api_client: reqwest::blocking::Client,
    pub api_key: String,
    pub fetch_timeout: u64,
    pub overwrite_existing_value: bool,
    pub save_intermediate: bool,
    pub avg_tokens_per_word: f64,
    pub model_temperature: f64,
    pub user_prompt: String,
    pub max_tok_gen: usize,
    pub model_name: String,
    pub num_context: usize,
    pub svc_base_url: String,
    pub system_context: String,
    pub input_tokens_count: u64,
    pub output_tokens_count: u64,
    pub shared_lock: Option<Arc<Mutex<isize>>>,
    pub min_gap_btwn_rqsts_secs: u64,
}

impl LLMTextGenerator {
    pub fn generate_text(
        &self,
        context_prefix: &str,
        context_suffix: &str,
    ) -> Result<LlmApiResult, String> {
        if let Some(api_access_mutex) = self.shared_lock.clone() {
            // attempt lock, retrieve value of mutux
            if let Ok(mut shared_val) = api_access_mutex.lock() {
                // then execute llm service call,
                let result = self.generate_text_no_mutex(context_prefix, context_suffix);
                let duration_previous = Duration::from_secs(max(*shared_val, 0) as u64);
                // get current timestamp:
                let mut duration_now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default();
                let seconds_elapsed = (duration_now - duration_previous).as_secs();
                // check if current tiemstamp is more than given duration from mutex value,
                if seconds_elapsed < self.min_gap_btwn_rqsts_secs {
                    // add delay in seconds to make up for the remaining time:
                    println!("API accessed {} seconds ago, hence waiting for {} seconds to limit API requests/sec", seconds_elapsed, self.min_gap_btwn_rqsts_secs-seconds_elapsed);
                    thread::sleep(Duration::from_secs(
                        self.min_gap_btwn_rqsts_secs - seconds_elapsed,
                    ));
                }
                // set current timestamp and then return
                duration_now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default();
                *shared_val = duration_now.as_secs() as isize;
                result
            }else{
                Err("Unable to access the shared lock for the API Service.".to_string())
            }
        } else {
            self.generate_text_no_mutex(context_prefix, context_suffix)
        }
    }

    /// Generates text using the llm service.
    /// This version of the functiondoes not use a shared mutex lock to coordinate access to the
    /// LLM API
    ///
    /// # Arguments
    ///
    /// * `context_prefix`: The part of the context that needs to be prefixed to the user prompt.
    ///   Pass an empty string in case a prefix is not required.
    /// * `context_suffix`: The context that follows the user prompt
    ///
    /// returns: Result<LlmApiResult, String>
    pub fn generate_text_no_mutex(
        &self,
        context_prefix: &str,
        context_suffix: &str,
    ) -> Result<LlmApiResult, String> {

        let prompt = format!("{}\n{}\n{}", context_prefix, self.user_prompt, context_suffix);

        self.http_api_request(&(self.api_client), prompt)
    }

    pub fn http_api_request(
        &self,
        client: &reqwest::blocking::Client,
        complete_prompt_input: String,
    ) -> Result<LlmApiResult, String> {
        // create payload based on api service type:
        match self.llm_service.as_str() {
            "chatgpt" => {
                let json_payload = prepare_chatgpt_payload(complete_prompt_input, self);
                http_post_json_chatgpt(self, client, json_payload)
            }
            "gemini" => {
                let json_payload = prepare_gemini_api_payload(complete_prompt_input, self);
                http_post_json_gemini(self, client, json_payload)
            }
            "google_genai" => {
                let json_payload = prepare_google_genai_api_payload(complete_prompt_input, self);
                http_post_json_google_genai(self, client, json_payload)
            }
            "ollama" => {
                let json_payload = prepare_ollama_payload(complete_prompt_input, self);
                http_post_json_ollama(self, client, json_payload)
            }
            _ => Err(format!("Unknown LLM service: {}", self.llm_service)),
        }
    }
}

pub struct LLMTextGenBuilder {
    pub llm_gen: LLMTextGenerator,
}

impl LLMTextGenBuilder {
    /// Builds an instance of LLMTextGenerator if successful, or None if an error occurs.
    ///
    /// # Arguments
    ///
    /// * `llm_api_name`:
    /// * `model_name`:
    /// * `network_timeout_secs`:
    ///
    /// returns: Option<LLMTextGenerator>
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    pub fn build(
        llm_api_name: &str,
        model_name: &str,
        network_timeout_secs: u64,
        proxy_server: Option<String>,
        api_access_mutex: Option<Arc<Mutex<isize>>>,
    ) -> Option<LLMTextGenerator> {

        match llm_api_name {

            "chatgpt" => {
                let api_key = std::env::var("OPENAI_API_KEY").unwrap_or(String::from(""));
                let chatgpt_headers = prepare_chatgpt_headers(api_key);
                let client: reqwest::blocking::Client = build_llm_api_client(
                    network_timeout_secs,
                    network_timeout_secs,
                    proxy_server,
                    Some(chatgpt_headers),
                );
                Some(LLMTextGenerator {
                    llm_service: llm_api_name.to_string(),
                    api_client: client,
                    api_key: "".to_string(),
                    fetch_timeout: 0,
                    overwrite_existing_value: false,
                    save_intermediate: false,
                    avg_tokens_per_word: 1.3,
                    model_temperature: 0.0,
                    user_prompt: "".to_string(),
                    max_tok_gen: 8192,
                    model_name: model_name.to_string(),
                    num_context: 8192,
                    svc_base_url: "https://api.openai.com/v1/chat/completions".to_string(),
                    system_context: "You are an expert".to_string(),
                    input_tokens_count: 0,
                    output_tokens_count: 0,
                    shared_lock: api_access_mutex,
                    min_gap_btwn_rqsts_secs: 6,
                })
            },

            "google_genai" => {
                let api_key = std::env::var("GOOGLE_API_KEY").unwrap_or(String::from(""));
                let googlegenai_headers = prepare_googlegenai_headers(api_key);
                let client: reqwest::blocking::Client = build_llm_api_client(
                    network_timeout_secs,
                    network_timeout_secs,
                    proxy_server,
                    Some(googlegenai_headers),
                );
                Some(LLMTextGenerator {
                    llm_service: llm_api_name.to_string(),
                    api_client: client,
                    api_key: "".to_string(),
                    fetch_timeout: 0,
                    overwrite_existing_value: false,
                    save_intermediate: false,
                    avg_tokens_per_word: 1.3,
                    model_temperature: 0.0,
                    user_prompt: "".to_string(),
                    max_tok_gen: 8192,
                    model_name: model_name.to_string(),
                    num_context: 8192,
                    svc_base_url: "https://generativelanguage.googleapis.com/v1beta/models"
                        .to_string(),
                    system_context: "You are an expert".to_string(),
                    input_tokens_count: 0,
                    output_tokens_count: 0,
                    shared_lock: api_access_mutex,
                    min_gap_btwn_rqsts_secs: 6,
                })
            },

            "gemini" => {
                let api_key = std::env::var("GOOGLE_API_KEY").unwrap_or(String::from(""));
                let client: reqwest::blocking::Client = build_llm_api_client(
                    network_timeout_secs,
                    network_timeout_secs,
                    proxy_server,
                    None,
                );
                Some(LLMTextGenerator {
                    llm_service: llm_api_name.to_string(),
                    api_client: client,
                    api_key,
                    fetch_timeout: 0,
                    overwrite_existing_value: false,
                    save_intermediate: false,
                    avg_tokens_per_word: 1.3,
                    model_temperature: 0.0,
                    user_prompt: "".to_string(),
                    max_tok_gen: 8192,
                    model_name: model_name.to_string(),
                    num_context: 8192,
                    svc_base_url: "https://generativelanguage.googleapis.com/v1beta/models"
                        .to_string(),
                    system_context: "You are an expert".to_string(),
                    input_tokens_count: 0,
                    output_tokens_count: 0,
                    shared_lock: api_access_mutex,
                    min_gap_btwn_rqsts_secs: 6,
                })
            },

            "ollama" => {
                let client: reqwest::blocking::Client = build_llm_api_client(
                    network_timeout_secs,
                    network_timeout_secs,
                    proxy_server,
                    None,
                );
                Some(LLMTextGenerator {
                    llm_service: llm_api_name.to_string(),
                    api_client: client,
                    api_key: "".to_string(),
                    fetch_timeout: 0,
                    overwrite_existing_value: false,
                    save_intermediate: false,
                    avg_tokens_per_word: 1.3,
                    model_temperature: 0.0,
                    user_prompt: "".to_string(),
                    max_tok_gen: 8192,
                    model_name: model_name.to_string(),
                    num_context: 8192,
                    svc_base_url: "http://127.0.0.1:11434/api/generate".to_string(),
                    system_context: "You are an expert".to_string(),
                    input_tokens_count: 0,
                    output_tokens_count: 0,
                    shared_lock: api_access_mutex,
                    min_gap_btwn_rqsts_secs: 0,
                })
            }
            _ => {
                None
            }
        }
    }

    pub fn build_from_config(app_confg: &Config, llm_svc_name: &str) -> Option<LLMTextGenerator> {

        let api_access_mutex = Arc::new(Mutex::new(0));

        // start with a generator with default values:
        if let Some(mut llm_gen) =
            LLMTextGenBuilder::build(llm_svc_name, "", 60, None, Some(api_access_mutex))
        {
            if let Ok(config_table) = app_confg.get_table("llm_apis") {
                if let Some((llm_name, llm_val)) = config_table.get_key_value(llm_svc_name) {
                    info!("Loading LLM text generation configuration from entry: {}", llm_name);
                    match llm_val.clone().into_table() {
                        Ok(entry_table) => {
                            match entry_table.get("max_gen_tokens") {
                                None => {}
                                Some(max_gen_tokens_val) => {
                                    llm_gen.max_tok_gen = max(
                                        0,
                                        max_gen_tokens_val.clone().into_int().unwrap_or_default(),
                                    ) as usize;
                                }
                            }
                            match entry_table.get("max_context_len") {
                                None => {}
                                Some(max_context_len_val) => {
                                    llm_gen.num_context = max(
                                        0,
                                        max_context_len_val.clone().into_int().unwrap_or_default(),
                                    ) as usize;
                                }
                            }
                            match entry_table.get("min_gap_btwn_rqsts_secs") {
                                None => {}
                                Some(min_gap_btwn_rqsts_secs_val) => {
                                    llm_gen.min_gap_btwn_rqsts_secs = max(
                                        0,
                                        min_gap_btwn_rqsts_secs_val
                                            .clone()
                                            .into_int()
                                            .unwrap_or_default(),
                                    )
                                        as u64;
                                }
                            }
                            match entry_table.get("temperature") {
                                None => {}
                                Some(temperature_val) => {
                                    llm_gen.model_temperature =
                                        temperature_val.clone().into_float().unwrap_or_default();
                                }
                            }
                            match entry_table.get("api_url") {
                                None => {}
                                Some(api_url_val) => {
                                    llm_gen.svc_base_url =
                                        api_url_val.clone().into_string().unwrap_or_default();
                                }
                            }
                            match entry_table.get("model_name") {
                                None => {}
                                Some(model_name_val) => {
                                    llm_gen.model_name =
                                        model_name_val.clone().into_string().unwrap_or_default();
                                }
                            }
                            match entry_table.get("model_api_timeout") {
                                None => {}
                                Some(model_api_timeout_val) => {
                                    llm_gen.fetch_timeout = max(
                                        0,
                                        model_api_timeout_val.clone().into_int().unwrap_or_default(),
                                    ) as u64;
                                }
                            }
                            return Some(llm_gen);
                        }
                        Err(er) => {
                            error!("{}", er.to_string());
                        }
                    }
                }
            }else{
                return None;
            }
        }
        None
    }
}

pub fn build_llm_api_client(
    connect_timeout: u64,
    fetch_timeout: u64,
    proxy_url: Option<String>,
    custom_headers: Option<HeaderMap>,
) -> reqwest::blocking::Client {
    let pool_idle_timeout: u64 = (connect_timeout + fetch_timeout) * 5;
    let pool_max_idle_connections: usize = 1;

    let mut headers = HeaderMap::new();
    if let Some(custom_header_map) = custom_headers {
        headers = custom_header_map;
    }
    // prepare headers:
    headers.insert(
        reqwest::header::CONNECTION,
        HeaderValue::from_static("keep-alive"),
    );
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );

    // build client:
    let client_builder = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(fetch_timeout))
        .connect_timeout(Duration::from_secs(connect_timeout))
        .default_headers(headers)
        .gzip(true)
        .pool_idle_timeout(Duration::from_secs(pool_idle_timeout))
        .pool_max_idle_per_host(pool_max_idle_connections);
    if proxy_url.is_some() {
        if let Some(proxy_url_str) = proxy_url {
            // if proxy is configured, then add proxy with https rule:
            match reqwest::Proxy::https(proxy_url_str.as_str()) {
                Ok(proxy_obj) => {
                    let client: reqwest::blocking::Client = client_builder
                        .proxy(proxy_obj)
                        .build()
                        .expect("Require valid parameters for building HTTP client");
                    return client;
                }
                Err(e) => {
                    error!(
                        "Unable to use proxy, Error when setting the proxy server: {}",
                        e
                    );
                }
            }
        }
    }
    let client_no_proxy: reqwest::blocking::Client = client_builder
        .build()
        .expect("Require valid parameters for building REST API client");
    client_no_proxy
}

#[derive(Serialize, Deserialize, PartialEq, Default)]
pub struct LlmApiResult {
    // The text generated by the model
    pub generated_text: String,
    // The count of input tokens used up for the ptompt / text generation input
    pub input_tokens_count: u64,
    // The count of generated tokens
    pub output_tokens_count: u64,
    // The reason for stopping text generation
    pub stop_reason: String,
    // The model used for geneeration
    pub model_used: String,
    // The log probabilities of the tokens generated.
    // Convert to linear probabilities = round(exp(logprob) * 100, 2)
    pub logprobs: Vec<f64>,
}

impl LlmApiResult {
    pub fn from(
        generated_text: String,
        input_tokens_count: u64,
        output_tokens_count: u64,
        stop_reason: String,
        model_used: String,
    ) -> LlmApiResult {
        LlmApiResult {
            generated_text,
            input_tokens_count,
            output_tokens_count,
            stop_reason,
            model_used,
            logprobs: vec![],
        }
    }
    pub fn error(error_message: String) -> Result<LlmApiResult, String> {
        Err(error_message)
    }
}

impl fmt::Debug for LlmApiResult {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("LLM API Result\n")
            .field(
                "\nLLM generated text",
                &format_args!("{}", self.generated_text),
            )
            .field(
                "\nInput tokens",
                &format_args!("{}", self.input_tokens_count),
            )
            .field(
                "\nOutput tokens",
                &format_args!("{}", self.output_tokens_count),
            ) //&format_args!("{}", self.addr))
            .field(
                "\nReason to stop generating",
                &format_args!("{}", self.stop_reason),
            )
            .field(
                "\nModel that generated this text",
                &format_args!("{}", self.model_used),
            )
            .field(
                "\nLog probabilities of generated tokens",
                &format_args!("{:?}\n", self.logprobs),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {}
