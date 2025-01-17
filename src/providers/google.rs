use std::collections::HashMap;
use std::error::Error;
use log::{debug, error, info};
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use crate::llm::{LLMTextGenBuilder, LLMTextGenerator, LlmApiResult};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct GeminiRequestPayload {
    pub contents: Vec<HashMap<String, Vec<HashMap<String, String>>>>,
    #[serde(rename = "safetySettings")]
    pub safety_settings: Vec<HashMap<String, String>>,
    #[serde(rename = "generationConfig")]
    pub generation_config: HashMap<String, usize>,
}


#[derive(Serialize, Deserialize, Debug)]
pub struct GenerationConfig {
    pub temperature: usize,
    #[serde(rename = "maxOutputTokens")]
    pub max_output_tokens: usize,
    pub response_modalities: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Parts {
    pub text: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Contents {
    pub role: String,
    pub parts: Parts,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GoogleGenAIRequestPayload {
    pub contents: Contents,
    #[serde(rename = "safety_settings")]
    pub safety_settings: Vec<HashMap<String, String>>,
    pub generation_config: GenerationConfig,
}



/// Add headers for google gen ai api:
/// "x-goog-api-key: PUT-YOUR-API-KEY-HERE"
/// "Content-Type: application/json"
///
/// # Arguments
///
/// * `app_config`: The application configuration
///
/// returns: HeaderMap<HeaderValue>
pub fn prepare_googlegenai_headers(api_key: String) -> HeaderMap {
    let mut custom_headers = HeaderMap::new();
    const GOOG_API_HEADER: reqwest::header::HeaderName = reqwest::header::HeaderName::from_static("x-goog-api-key");

    if let Ok(header_apikey_val) = HeaderValue::from_str(api_key.as_str()) {
        custom_headers.insert(GOOG_API_HEADER, header_apikey_val);
    }
    custom_headers.insert(reqwest::header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    custom_headers
}



pub fn prepare_google_genai_api_payload(prompt: String, llm_params: &LLMTextGenerator) -> GoogleGenAIRequestPayload {
    // put the parameters into the structure
    let json_payload = GoogleGenAIRequestPayload {
        contents: Contents{
            role: "USER".to_string(),
            parts: Parts { text: prompt },
        },
        safety_settings: vec![
            HashMap::from([
                ("category".to_string(), "HARM_CATEGORY_DANGEROUS_CONTENT".to_string()),
                ("threshold".to_string(), "BLOCK_ONLY_HIGH".to_string()),
            ])
        ],
        generation_config: GenerationConfig{
            temperature: llm_params.model_temperature as usize,
            max_output_tokens: llm_params.max_tok_gen,
            response_modalities: "TEXT".to_string(),
        },
    };
    json_payload
}


/// Prepare the JSON payload for sending to the Gemini LLM API service.
///
/// # Arguments
///
/// * `prompt`: The prompt to the model.
/// * `llm_params`: the LLMParameters struct with various params, e.g. temperature, num_ctx, max_gen
///
/// returns: RequestPayload
pub fn prepare_gemini_api_payload(prompt: String, llm_params: &LLMTextGenerator) -> GeminiRequestPayload {
    // put the parameters into the structure
    let json_payload = GeminiRequestPayload {
        contents: vec![
            HashMap::from([
                ("parts".to_string(),
                 vec![HashMap::from([
                     ("text".to_string(), prompt)
                 ])]
                )
            ])],
        safety_settings: vec![HashMap::from([
            ("category".to_string(), "HARM_CATEGORY_DANGEROUS_CONTENT".to_string()),
            ("threshold".to_string(), "BLOCK_ONLY_HIGH".to_string())
        ])],
        generation_config: HashMap::from([
            ("temperature".to_string(), llm_params.model_temperature as usize),
            ("maxOutputTokens".to_string(), llm_params.max_tok_gen),
        ]),
    };
    return json_payload;
}

pub fn http_post_json_google_genai<'post>(llm_params: &LLMTextGenerator, client: &reqwest::blocking::Client, json_payload: GoogleGenAIRequestPayload) -> Result<LlmApiResult, String> {

    let api_url = format!("{}/{}:generateContent", llm_params.svc_base_url, llm_params.model_name);
    // Generate URL of the format: https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent

    // add json payload to body
    match client.post(api_url)
        .json(&json_payload)
        .send() {
        Result::Ok(resp) => {
            match resp.status() {
                StatusCode::OK => {
                    match resp.json::<serde_json::value::Value>(){
                        Result::Ok( json ) => {
                            debug!("Google GenAI API response:\n{:?}", json);
                            let mut llm_response = LlmApiResult::default();
                            if let Some(resp_error) = json.get("error"){
                                if let Some(error_message) = resp_error.get("message"){
                                    if let Some(err_message) = error_message.as_str(){
                                        error!("API Error message: {}", err_message);
                                        return Err(format!("Google GenAI error: {}", err_message));
                                    }
                                }
                            }
                            if let Some(resp_candidates) = json.get("candidates"){

                                if let Some(first_candidate) = resp_candidates.get(0) {

                                    if let Some(resp_content) = first_candidate.get("content") {
                                        if let Some(parts) = resp_content.get("parts") {

                                            if let Some(first_part) = parts.get(0) {

                                                if let Some(text_part) = first_part.get("text") {
                                                    llm_response.generated_text = text_part.to_string();
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            if let Some(resp_usage_metadata) = json.get("usageMetadata"){
                                if let Some(prompt_token_count_str) = resp_usage_metadata.get("promptTokenCount"){
                                    llm_response.input_tokens_count = prompt_token_count_str.as_u64().unwrap_or_default();
                                }
                                if let Some(candidates_token_count_str) = resp_usage_metadata.get("candidatesTokenCount"){
                                    llm_response.output_tokens_count = candidates_token_count_str.as_u64().unwrap_or_default();
                                }
                            }
                            // "modelVersion": "gemini-1.5-flash-001"
                            if let Some(model_version_val) = json.get("modelVersion") {
                                llm_response.model_used = model_version_val.to_string();
                            }
                            return Ok(llm_response);
                        },
                        Err(e) => {
                            error!("When retrieving json from Google GenAI API response: {}", e);
                            if let Some(err_source) = e.source(){
                                error!("Caused by: {}", err_source);
                                return Err(format!("Google GenAI error: {}, caused by {}", e.to_string(), err_source));
                            }
                        },
                    }
                },
                StatusCode::UNAUTHORIZED => {
                    error!("Google GenAI API: Unauthorised!");
                    return Err("Google GenAI API: Unauthorised".to_string());
                },
                StatusCode::NOT_FOUND => {
                    error!("Google GenAI API: Service not found!");
                    return Err("Google GenAI API: Service not found".to_string());
                },
                StatusCode::PAYLOAD_TOO_LARGE => {
                    error!("Google GenAI API: Request payload is too large!");
                    return Err("Google GenAI API: Request payload is too large".to_string());
                },
                StatusCode::TOO_MANY_REQUESTS => {
                    error!("Google GenAI API: Too many requests. Exceeded the Provisioned Throughput.");
                    return Err("Google GenAI API: Too many requests. Exceeded the Provisioned Throughput.".to_string());
                },
                s => {
                    error!("Google GenAI API response status: {s:?}");
                    return Err(format!("Google GenAI error, HTTP response status code: {:?}", s));
                },
            };
        }
        Err(e) => {
            error!("When posting json payload to Google GenAI API service: {}", e);
            if let Some(err_source) = e.source(){
                error!("Caused by: {}", err_source);
                return Err(format!("Google GenAI error: {}, caused by {}", e.to_string(), err_source));
            }
        }
    }
    return LlmApiResult::error("Google GenAI API: did not generate any text".to_string());
}

/// Posts the json payload with the prompt to generate text using the Gemini LLM API REST service and retrieves back the result.
/// Converts the url, model and api key to full url for the api service for non-stream
/// content generation:
/// e.g. https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=$GOOGLE_API_KEY
/// First, the payload is prepared in json format.
/// Then, it is HTTP POST(ed) to the URL and the response payload is retrieved and converted
/// from json to struct to extract and return the model generated output text.
///
/// # Arguments
///
/// * `json_payload`: The prompt + context input to the model service
/// * `llm_params`: The API parameters to be used, e.g. temperature, max token count, model, etc.
///
/// returns: Result
pub fn http_post_json_gemini<'post>(llm_params: &LLMTextGenerator, client: &Client, json_payload: GeminiRequestPayload) -> Result<LlmApiResult, String> {

    let api_url = format!("{}/{}:generateContent?key={}", llm_params.svc_base_url, llm_params.model_name, llm_params.api_key);

    // add json payload to body
    match client.post(api_url)
        .json(&json_payload)
        .send() {
        Result::Ok(resp) => {
            match resp.status() {
                StatusCode::OK => {
                    match resp.json::<serde_json::value::Value>() {
                        Result::Ok(json) => {
                            println!("Google Gemini API response:\n{:?}", json);
                            let mut llm_response = LlmApiResult::default();
                            if let Some(resp_candidates) = json.get("candidates") {
                                if let Some(first_candidate) = resp_candidates.get(0) {
                                    if let Some(resp_content) = first_candidate.get("content") {
                                        if let Some(parts) = resp_content.get("parts") {
                                            if let Some(first_part) = parts.get(0) {
                                                if let Some(text_part) = first_part.get("text") {
                                                    if let Some(response_str) = text_part.as_str() {
                                                        llm_response.generated_text = response_str.to_string();
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            if let Some(resp_error) = json.get("error") {
                                if let Some(resp_error_msg_val) = resp_error.get("message") {
                                    let error_message = resp_error_msg_val.as_str().unwrap_or_default();
                                    return LlmApiResult::error(format!("Gemini API error: {}", error_message));
                                }
                            }
                            // TODO: get stop_reason:
                            if let Some(resp_usage_metadata) = json.get("usageMetadata"){
                                if let Some(prompt_token_count_str) = resp_usage_metadata.get("promptTokenCount"){
                                    llm_response.input_tokens_count = prompt_token_count_str.as_u64().unwrap_or_default();
                                }
                                if let Some(candidates_token_count_str) = resp_usage_metadata.get("candidatesTokenCount"){
                                    llm_response.output_tokens_count = candidates_token_count_str.as_u64().unwrap_or_default();
                                }
                            }
                            // "modelVersion": "gemini-1.5-flash-001"
                            if let Some(model_version_val) = json.get("modelVersion") {
                                llm_response.model_used = model_version_val.to_string();
                            }
                            return Ok(llm_response);
                        },
                        Err(e) => {
                            error!("Gemini API: When retrieving json from response: {}", e);
                            if let Some(err_source) = e.source() {
                                error!("Caused by: {}", err_source);
                                return LlmApiResult::error(format!("Gemini API error When retrieving json: {}, caused by: {}", e, err_source));
                            }
                        },
                    }
                },
                StatusCode::NOT_FOUND => {
                    error!("Gemini API: Service not found!");
                    return LlmApiResult::error("Gemini API: Service not found".to_string());
                },
                StatusCode::PAYLOAD_TOO_LARGE => {
                    error!("Gemini API: Request payload is too large!");
                    return LlmApiResult::error("Gemini API: Request payload is too large".to_string());
                },
                StatusCode::TOO_MANY_REQUESTS => {
                    error!("Gemini API: Too many requests. Exceeded the Provisioned Throughput.");
                    return LlmApiResult::error("Gemini API: Too many requests. Exceeded the Provisioned Throughput.".to_string());
                }
                s => {
                    error!("Gemini API: Received response status: {s:?}");
                    return LlmApiResult::error(
                        format!(
                            "Gemini API: Received response status: {s:?}"
                        )
                    );
                }
            }
        }
        Err(e) => {
            error!("When posting json payload to service: {}", e);
            if let Some(err_source) = e.source(){
                error!("Caused by: {}", err_source);
                return LlmApiResult::error(
                    format!(
                        "Gemini API: When posting json payload to service error {}, caused by: {}",
                        e,
                        err_source)
                );
            }
        }
    }
    return LlmApiResult::error("Gemini API: did not generate any text".to_string());
}


#[cfg(test)]
mod tests {
    use super::*;
    use log::{debug, info, error};

    #[test]
    fn test_generate_using_gemini_llm(){
        let mut llmgen = LLMTextGenBuilder::build("gemini", "gemini-1.5-flash-latest", 60, None, None).unwrap();
        llmgen.max_tok_gen = 16000;
        let answer1 = llmgen.generate_text("", "How is a rainbow created in the sky? Respond very concisely.");
        if let Ok(llm_response) = answer1 {
            println!("---Answer 1---\n{:?}", llm_response);
            assert_eq!(true, true);
        }
        assert_eq!(true, true);
    }

    #[test]
    fn test_generate_using_google_genai_llm(){
        let mut llmgen = LLMTextGenBuilder::build("google_genai", "gemini-2.0-flash-exp", 60, None, None).unwrap();
        llmgen.max_tok_gen = 16000;
        let answer11 = llmgen.generate_text("", "How is a rainbow created in the sky? Respond very concisely.");
        if let Ok(llm_response) = answer11 {
            println!("---Answer 11---\n{:?}", llm_response);
            assert_eq!(true, true);
        }
        assert_eq!(true, true);
    }

}
