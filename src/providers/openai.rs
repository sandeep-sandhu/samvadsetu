use std::cmp::max;
use std::collections::HashMap;
use std::error::Error;
use log::{debug, error, info};
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use crate::llm::{LLMTextGenBuilder, LLMTextGenerator, LlmApiResult};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct ChatGPTRequestPayload {
    pub model: String,
    pub messages: Vec<HashMap<String, String>>,
    pub temperature: f64,
    max_completion_tokens: usize,
    logprobs: bool,
}


/// Prepare custom headers for OpenAI's ChatGPT API
///
/// # Arguments
///
/// * `api_key`: The API key for the service
///
/// returns: HeaderMap<HeaderValue>
pub fn prepare_chatgpt_headers(api_key: String) -> HeaderMap {

    let mut custom_headers = HeaderMap::new();

    // set header "Authorization: Bearer $OPENAI_API_KEY"
    let api_key = format!("Bearer {}", api_key);
    if let Ok(header_val) = HeaderValue::from_str(api_key.as_str()) {
        custom_headers.insert(reqwest::header::AUTHORIZATION, header_val);
    }

    // set header: "OpenAI-Project: $PROJECT_ID"
    // let project_id = std::env::var("PROJECT_ID").unwrap_or(String::from(""));
    // if let Ok(header_val) = HeaderValue::from_str(project_id.as_str()){
    //     let proj_id = HeaderName::from_lowercase(b"OpenAI-Project").unwrap();
    //     custom_headers.insert(proj_id, header_val);
    // }

    return custom_headers;
}

/// Generate payload of the format:
///     {
//           "model": "gpt-4o-mini",
//           "messages": [{"role": "user", "content": "Say this is a test!"}],
//           "temperature": 0.7
//         }
/// # Arguments
///
/// * `prompt`: The prompt to the model.
/// * `llm_params`: The LLMParameters object with relevant parameters to be used.
///
/// returns: ChatGPTRequestPayload
pub fn prepare_chatgpt_payload(prompt: String, llm_params: &LLMTextGenerator) -> ChatGPTRequestPayload {

    // put the parameters into the structure
    let json_payload = ChatGPTRequestPayload {
        model: llm_params.model_name.clone(),
        messages: vec![
            HashMap::from([
                ("role".to_string(), "system".to_string()),
                ("content".to_string(), llm_params.system_context.clone())
            ]),
            HashMap::from([
                ("role".to_string(), "user".to_string()),
                ("content".to_string(), prompt)
            ]),
        ],
        temperature: llm_params.model_temperature as f64,
        max_completion_tokens: llm_params.max_tok_gen,
        logprobs: true,
    };
    return json_payload;
}


/// Posts the json payload to REST service and retrieves back the result.
///
/// # Arguments
///
/// * `service_url`:
/// * `client`:
/// * `json_payload`:
///
/// returns: String
pub fn http_post_json_chatgpt(llm_params: &LLMTextGenerator, client: &Client, json_payload: ChatGPTRequestPayload) -> Result<LlmApiResult, String>{

    // add json payload to body
    match client.post(llm_params.svc_base_url.clone())
        .json(&json_payload)
        .send() {
        Ok(resp) => {
            match resp.status() {
                StatusCode::OK => {
                    match resp.json::<serde_json::value::Value>() {
                        Ok(json) => {
                            debug!("ChatGPT API: model response:\n{:?}", json);
                            let mut llm_response = LlmApiResult::default();
                            if let Some(choices) = json.get("choices") {
                                if let Some(first_choice) = choices.get(0) {
                                    if let Some(message) = first_choice.get("message") {
                                        if let Some(content) = message.get("content") {
                                            llm_response.generated_text = content.to_string();
                                        }
                                    }
                                    if let Some(logprobs) = first_choice.get("logprobs") {
                                        // get object: "logprobs" , get attrib: "content" array of Object -> {"bytes", "logprob"}
                                        if let Some(logprobs_content) = logprobs.get("content") {
                                            match logprobs_content.as_array() {
                                                None => {}
                                                Some(logprobs_vec) => {
                                                    for log_prob_obj in logprobs_vec{
                                                        if let Some(logprob_val) = log_prob_obj.get("logprob") {
                                                            llm_response.logprobs.push(logprob_val.as_f64().unwrap_or_default());
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    // "finish_reason": String("stop")
                                    if let Some(finish_reason_val) = first_choice.get("finish_reason") {
                                        llm_response.stop_reason = finish_reason_val.to_string();
                                    }
                                }
                            }
                            // "model": String("gpt-4o-mini-2024-07-18")
                            if let Some(model_used_val) = json.get("model") {
                                llm_response.model_used = model_used_val.to_string();
                            }
                            // "usage" -> get integer attributes: "prompt_tokens", "completion_tokens"
                            if let Some(usage_val) = json.get("usage") {
                                if let Some(prompt_tokens_val) = usage_val.get("prompt_tokens") {
                                    llm_response.input_tokens_count = prompt_tokens_val.to_string().parse::<u64>().unwrap_or_default();
                                }
                                if let Some(completion_tokens_val) = usage_val.get("completion_tokens") {
                                    llm_response.output_tokens_count = completion_tokens_val.to_string().parse::<u64>().unwrap_or_default();
                                }
                            }
                            return Ok(llm_response);
                        },
                        Err(e) => {
                            error!("ChatGPT API: When retrieving json from response: {}", e);
                            if let Some(err_source) = e.source() {
                                error!("Caused by: {}", err_source);
                                return LlmApiResult::error(format!("ChatGPT error When retrieving json: {}, caused by: {}", e, err_source));
                            }
                            info!("ChatGPT Payload that resulted in error: {:?}", json_payload);
                        },
                    }
                },
                StatusCode::NOT_FOUND => {
                    error!("ChatGPT API: Service not found!");
                    return LlmApiResult::error("ChatGPT API: Service not found".to_string());
                },
                StatusCode::PAYLOAD_TOO_LARGE => {
                    error!("ChatGPT API: Request payload is too large!");
                    return LlmApiResult::error("ChatGPT API: Request payload is too large".to_string());
                },
                StatusCode::TOO_MANY_REQUESTS => {
                    error!("ChatGPT API: Too many requests. Exceeded the Provisioned Throughput.");
                    return LlmApiResult::error("ChatGPT API: Too many requests. Exceeded the Provisioned Throughput.".to_string());
                }
                s => {
                    error!("ChatGPT API: Received response status: {s:?}");
                    return LlmApiResult::error(
                        format!(
                            "ChatGPT API: Received response status: {s:?}"
                        )
                    );
                }
            }
        }
        Err(e) => {
            error!("ChatGPT API: When posting json payload to service: {}", e);
            if let Some(err_source) = e.source(){
                error!("Caused by: {}", err_source);
                return LlmApiResult::error(
                    format!(
                        "ChatGPT API: When posting json payload to service error {}, caused by: {}",
                        e,
                        err_source)
                );
            }
            info!("ChatGPT Payload: {:?}", json_payload);
        }
    }
    return LlmApiResult::error("ChatGPT API: did not generate any text".to_string());
}


#[cfg(test)]
mod tests {
    use super::*;
    use log::{debug, info, error};

    #[test]
    fn test_generate_using_chatgpt_llm(){
        let mut llmgen = LLMTextGenBuilder::build("chatgpt", "gpt-4o-mini",60, None, None).unwrap();
        llmgen.max_tok_gen = 16000;
        let answer1 = llmgen.generate_text("", "How is a rainbow created in the sky? Respond very concisely.");
        if let Ok(llm_response) = answer1 {
            println!("---Answer 1---\n{:?}", llm_response);
        }
        assert_eq!(true, true);
    }

}
