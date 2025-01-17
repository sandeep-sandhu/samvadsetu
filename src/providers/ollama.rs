use std::collections::HashMap;
use std::error::Error;
use std::sync::{Arc, Mutex};
use log::{debug, error};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use crate::llm::{LlmApiResult, LLMTextGenerator, LLMTextGenBuilder};



#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct OllamaPayload {
    pub model: String,
    #[serde(rename = "taskID")]
    pub task_id: usize,
    pub keep_alive: String,
    /// For options such as: "temperature": 0, "num_predict": 8192, "num_ctx": 8192,
    pub options: HashMap<String, usize>,
    pub prompt: String,
    pub stream: bool,
}

pub fn prepare_ollama_payload(prompt: String, llm_params: &LLMTextGenerator) -> OllamaPayload {
    let temperature: usize = llm_params.model_temperature as usize;
    // put the parameters into the structure
    let json_payload = OllamaPayload {
        model: llm_params.model_name.to_string(),
        task_id: 42, // what else!
        keep_alive: String::from("10m"),
        options: HashMap::from([
            ("temperature".to_string(), temperature),
            ("num_predict".to_string(), llm_params.max_tok_gen),
            ("num_ctx".to_string(), llm_params.num_context)
        ]),
        prompt: prompt.to_string(),
        stream: false
    };
    return json_payload;
}



pub fn build_llm_prompt(model_name: &str, system_context: &str, user_context: &str, input_text: &str) -> String {
    if model_name.contains("llama") {
        return prepare_llama_prompt(system_context, user_context, input_text);
    } else if model_name.contains("gemma") {
        return prepare_gemma_prompt(system_context, user_context, input_text);
    }
    else {
        return format!("{}\n{}\n{}", system_context, user_context, input_text).to_string();
    }
}

pub fn prepare_gemma_prompt(system_context: &str, user_context: &str, input_text: &str) -> String{
    return format!("<start_of_turn>user\
        {}\
        \
        {}<end_of_turn><start_of_turn>model", user_context, input_text).to_string();
}

pub fn prepare_llama_prompt(system_context: &str, user_context: &str, input_text: &str) -> String {
    return format!("<|begin_of_text|><|start_header_id|>system<|end_header_id|>{}\
        <|eot_id|><|start_header_id|>user<|end_header_id|>{}\
        \n\n{}<|eot_id|> <|start_header_id|>assistant<|end_header_id|>", system_context, user_context, input_text).to_string();
}

/// Posts the json payload to Ollama REST service and retrieves back the result.
///
/// # Arguments
///
/// * `service_url`:
/// * `client`:
/// * `json_payload`:
///
/// returns: String
pub fn http_post_json_ollama(llm_params: &LLMTextGenerator, client: &reqwest::blocking::Client, json_payload: OllamaPayload) -> Result<LlmApiResult, String> {
    // add json payload to body
    match client.post(llm_params.svc_base_url.clone())
        .json(&json_payload)
        .send() {
        Result::Ok(resp) => {
            match resp.status(){
                StatusCode::OK => {
                    match resp.json::<serde_json::value::Value>() {
                        Ok(json) => {
                            let mut llm_response = LlmApiResult::default();
                            debug!("ollama API: model response:\n{:?}", json);
                            if let Some(response) = json.get("response") {
                                llm_response.generated_text = response.to_string();
                            }
                            // get token counts - "eval_count" "prompt_eval_count"
                            if let Some(eval_count_val) = json.get("eval_count") {
                                llm_response.output_tokens_count = eval_count_val.to_string().parse::<u64>().unwrap_or_default();
                            }
                            if let Some(prompt_eval_count_val) = json.get("prompt_eval_count") {
                                llm_response.input_tokens_count = prompt_eval_count_val.to_string().parse::<u64>().unwrap_or_default();
                            }
                            if let Some(done_reason_val) = json.get("done_reason") {
                                llm_response.stop_reason = done_reason_val.to_string();
                            }
                            if let Some(model_used_val) = json.get("model") {
                                llm_response.model_used = model_used_val.to_string();
                            }
                            return Ok(llm_response);
                        }
                        Err(e) => {
                            error!("ollama API: json decode error:\n{:?}", e);
                            if let Some(err_source) = e.source() {
                                return Err(format!("Error {} caused by {}", e.to_string(), err_source.to_string()));
                            }
                        }
                    }
                },
                StatusCode::NOT_FOUND => {
                    error!("Ollama: Service not found!");
                },
                StatusCode::UNAUTHORIZED => {
                    error!("Ollama: Unauthorized access");
                },
                s => {
                    error!("Ollama: When retrieving response, status code: {}", s);
                    return Err(format!("Ollama API Error: When retrieving response, status code: {:?}", s));
                },
            }
        },
        Err(e) => {
            error!("Ollama: When posting json payload to service: {}", e);
            if let Some(err_source) = e.source(){
                error!("Caused by: {}", err_source);
                return LlmApiResult::error(format!("Ollama API error {} Caused by: {}", e.to_string(), err_source));
            }
        }
    }
    return Err(String::from("Ollama did not generate any content"));
}


#[cfg(test)]
mod tests {
    use super::*;
    use log::{debug, info, error};

    #[test]
    fn test_generate_using_ollama_llm(){
        let api_mutex = Arc::new(Mutex::new(0));
        let mut llmgen = LLMTextGenBuilder::build("ollama", "gemma2", 60, None, Some(api_mutex)).unwrap();
        llmgen.max_tok_gen = 8192;
        llmgen.min_gap_btwn_rqsts_secs = 20;
        llmgen.svc_base_url = "http://10.13.31.113:11434/api/generate".to_string();
        let answer1 = llmgen.generate_text("", "How is a rainbow created in the sky? Respond very concisely.");
        if let Ok(llm_response) = answer1 {
            println!("---Answer 1---\n{:?}", llm_response);
            assert_eq!(true, true);
        }

        // let answer3 = llmgen.generate_text("", "Prove maxwells equations analytically. Reply very concisely.");
        // println!("Response from ollama model = {:?}", answer3);
        // let answer2 = llmgen.generate_text("", "Why is the sky blue? Respond very concisely.");
        // println!("Response from ollama model = {:?}", answer2);
        assert_eq!(1,0);

    }

}
