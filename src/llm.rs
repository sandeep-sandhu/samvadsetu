// file: llm.rs

use std::cmp::{max, min};
use std::collections::HashMap;
use std::error::Error;
use std::ops::Deref;
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::{fmt, thread};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use config::Config;
use log::{debug, error, info};
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use crate::providers::google::{http_post_json_gemini, http_post_json_google_genai, prepare_gemini_api_payload, prepare_google_genai_api_payload, prepare_googlegenai_headers};
use crate::providers::ollama::{http_post_json_ollama, prepare_ollama_payload};
use crate::providers::openai::{http_post_json_chatgpt, prepare_chatgpt_headers, prepare_chatgpt_payload};

pub const MIN_GAP_BTWN_RQST_SECS: u64 = 6;

#[derive(Debug)]
pub struct LLMTextGenerator {
    pub llm_service: String,
    pub api_client: reqwest::blocking::Client,
    pub api_key: String,
    pub fetch_timeout: u64,
    pub overwrite_existing_value: bool,
    pub save_intermediate: bool,
    pub avg_tokens_per_word: f32,
    pub model_temperature: f32,
    pub prompt: String,
    pub max_tok_gen: usize,
    pub model_name: String,
    pub num_context: usize,
    pub svc_base_url: String,
    pub system_context: String,
    pub input_tokens_count: u64,
    pub output_tokens_count: u64,
    pub shared_lock: Option<Arc<Mutex<isize>>>,
    pub min_gap_btwn_rqsts_secs: u64
}

impl LLMTextGenerator {

    pub fn generate_text(&self, context_prefix: &str, context_suffix: &str) -> Result<LlmApiResult, String> {
        if let Some(api_access_mutex) = self.shared_lock.clone() {
            // attempt lock, retrieve value of mutux
            let mut shared_val = api_access_mutex.lock().unwrap();
            // then execute llm service call,
            let result = self.generate_text_no_mutex(context_prefix, context_suffix);
            let duration_previous = Duration::from_secs(max(*shared_val,0) as u64);
            // get current timestamp:
            let mut duration_now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
            let seconds_elapsed = (duration_now-duration_previous).as_secs();
            // check if current tiemstamp is more than given duration from mutex value,
            if seconds_elapsed < self.min_gap_btwn_rqsts_secs {
                // add delay in seconds to make up for the remaining time:
                println!("API accessed {} seconds ago, hence waiting for {} seconds to limit API requests/sec", seconds_elapsed, self.min_gap_btwn_rqsts_secs-seconds_elapsed);
                thread::sleep(Duration::from_secs(self.min_gap_btwn_rqsts_secs -seconds_elapsed));
            }
            // set current timestamp and then return
            duration_now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
            *shared_val = duration_now.as_secs() as isize;
            return result;
        } else {
            return self.generate_text_no_mutex(context_prefix, context_suffix);
        }
    }

    pub fn generate_text_no_mutex(&self, context_prefix: &str, context_suffix: &str) -> Result<LlmApiResult, String> {

        let prompt = format!("{}\n{}\n{}", context_prefix, self.prompt, context_suffix);

        let llm_output= self.http_api_request(&(self.api_client), prompt);

        return llm_output;
    }

    fn http_api_request(&self, client: &Client, complete_prompt_input: String) -> Result<LlmApiResult, String> {
        // create payload based on api service type:
        match self.llm_service.as_str() {
            "chatgpt" => {
                let json_payload = prepare_chatgpt_payload(complete_prompt_input, self);
                return http_post_json_chatgpt(self, client, json_payload)
            },
            "gemini" => {
                let json_payload = prepare_gemini_api_payload(complete_prompt_input, self);
                return http_post_json_gemini(self, client, json_payload)
            },
            "google_genai" => {
                let json_payload = prepare_google_genai_api_payload(complete_prompt_input, self);
                return http_post_json_google_genai(self, client, json_payload)
            },
            "ollama" => {
                let json_payload = prepare_ollama_payload(complete_prompt_input, self);
                return http_post_json_ollama(self, client, json_payload)
            },
            _ => {
                return Err(format!("Unknown LLM service: {}", self.llm_service))
            }
        }
    }
}

pub struct LLMTextGenBuilder{
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
    pub fn build(llm_api_name: &str, model_name: &str, network_timeout_secs: u64, proxy_server: Option<String>, api_access_mutex: Option<Arc<Mutex<isize>>>) -> Option<LLMTextGenerator> {
        match llm_api_name {
            "chatgpt" => {
                let api_key = std::env::var("OPENAI_API_KEY").unwrap_or(String::from(""));
                let chatgpt_headers = prepare_chatgpt_headers(api_key);
                let client: reqwest::blocking::Client = build_llm_api_client(network_timeout_secs, network_timeout_secs, proxy_server, Some(chatgpt_headers));
                return Some(LLMTextGenerator {
                    llm_service: llm_api_name.to_string(),
                    api_client: client,
                    api_key: "".to_string(),
                    fetch_timeout: 0,
                    overwrite_existing_value: false,
                    save_intermediate: false,
                    avg_tokens_per_word: 1.3,
                    model_temperature: 0.0,
                    prompt: "".to_string(),
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
                let client: reqwest::blocking::Client = build_llm_api_client(network_timeout_secs, network_timeout_secs, proxy_server, Some(googlegenai_headers));
                return Some(LLMTextGenerator{
                    llm_service: llm_api_name.to_string(),
                    api_client: client,
                    api_key: "".to_string(),
                    fetch_timeout: 0,
                    overwrite_existing_value: false,
                    save_intermediate: false,
                    avg_tokens_per_word: 1.3,
                    model_temperature: 0.0,
                    prompt: "".to_string(),
                    max_tok_gen: 8192,
                    model_name: model_name.to_string(),
                    num_context: 8192,
                    svc_base_url: "https://generativelanguage.googleapis.com/v1beta/models".to_string(),
                    system_context: "You are an expert".to_string(),
                    input_tokens_count: 0,
                    output_tokens_count: 0,
                    shared_lock: api_access_mutex,
                    min_gap_btwn_rqsts_secs: 6,
                });
            },
            "gemini" => {
                let api_key = std::env::var("GOOGLE_API_KEY").unwrap_or(String::from(""));
                let client: reqwest::blocking::Client = build_llm_api_client(network_timeout_secs, network_timeout_secs, proxy_server, None);
                return Some(LLMTextGenerator{
                    llm_service: llm_api_name.to_string(),
                    api_client: client,
                    api_key: api_key,
                    fetch_timeout: 0,
                    overwrite_existing_value: false,
                    save_intermediate: false,
                    avg_tokens_per_word: 1.3,
                    model_temperature: 0.0,
                    prompt: "".to_string(),
                    max_tok_gen: 8192,
                    model_name: model_name.to_string(),
                    num_context: 8192,
                    svc_base_url: "https://generativelanguage.googleapis.com/v1beta/models".to_string(),
                    system_context: "You are an expert".to_string(),
                    input_tokens_count: 0,
                    output_tokens_count: 0,
                    shared_lock: api_access_mutex,
                    min_gap_btwn_rqsts_secs: 6,
                });
            }            ,
            "ollama" => {
                let client: reqwest::blocking::Client = build_llm_api_client(network_timeout_secs, network_timeout_secs, proxy_server, None);
                return Some(LLMTextGenerator{
                    llm_service: llm_api_name.to_string(),
                    api_client: client,
                    api_key: "".to_string(),
                    fetch_timeout: 0,
                    overwrite_existing_value: false,
                    save_intermediate: false,
                    avg_tokens_per_word: 1.3,
                    model_temperature: 0.0,
                    prompt: "".to_string(),
                    max_tok_gen: 8192,
                    model_name: model_name.to_string(),
                    num_context: 8192,
                    svc_base_url: "http://127.0.0.1:11434/api/generate".to_string(),
                    system_context: "You are an expert".to_string(),
                    input_tokens_count: 0,
                    output_tokens_count: 0,
                    shared_lock: api_access_mutex,
                    min_gap_btwn_rqsts_secs: 0,
                });
            },
            _ => {
                return None;
            }
        }
    }
}




pub fn build_llm_api_client(connect_timeout: u64, fetch_timeout: u64, proxy_url: Option<String>, custom_headers: Option<HeaderMap>) -> reqwest::blocking::Client {

    let pool_idle_timeout: u64 = (connect_timeout + fetch_timeout) * 5;
    let pool_max_idle_connections: usize = 1;

    let mut headers = HeaderMap::new();
    if let Some(custom_header_map) = custom_headers {
        headers = custom_header_map;
    }
    // prepare headers:
    headers.insert(reqwest::header::CONNECTION, HeaderValue::from_static("keep-alive"));
    headers.insert(reqwest::header::CONTENT_TYPE, HeaderValue::from_static("application/json"));

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
                    error!("Unable to use proxy, Error when setting the proxy server: {}", e);
                }
            }
        }
    }
    let client_no_proxy: reqwest::blocking::Client = client_builder
        .build()
        .expect("Require valid parameters for building REST API client");
    return client_no_proxy;
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
    // The log probabilities of the tokens generated. Linear probabilities = round(exp(logprob) * 100, 2)
    pub logprobs: Vec<f64>
}

impl LlmApiResult {
    pub fn from(generated_text: String, input_tokens_count: u64, output_tokens_count: u64, stop_reason: String, model_used: String) -> LlmApiResult {
        return LlmApiResult{
            generated_text: generated_text,
            input_tokens_count: input_tokens_count,
            output_tokens_count: output_tokens_count,
            stop_reason: stop_reason,
            model_used: model_used,
            logprobs: vec![]
        }
    }
    pub fn error(error_message: String) -> Result<LlmApiResult,String> {
        return Err(error_message);
    }
}

impl fmt::Debug for LlmApiResult {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("LLM API Result\n")
            .field("\nLLM generated text", &format_args!("{}", self.generated_text))
            .field("\nInput tokens", &format_args!("{}", self.input_tokens_count))
            .field("\nOutput tokens", &format_args!("{}", self.output_tokens_count))//&format_args!("{}", self.addr))
            .field("\nReason to stop generating", &format_args!("{}", self.stop_reason))
            .field("\nModel that generated this text", &format_args!("{}", self.model_used))
            .field("\nLog probabilities of generated tokens", &format_args!("{:?}\n", self.logprobs))
            .finish()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use log::{debug, info, error};

}
