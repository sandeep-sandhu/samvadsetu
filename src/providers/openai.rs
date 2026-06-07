// providers/openai.rs — OpenAI Chat Completions API + compatible providers
//
// Handles: chatgpt, openai, deepseek, qwen, llamacpp
// All use the same JSON schema; only the base URL and API key differ.

use crate::error::{from_reqwest_error, parse_openai_error_body, SamvadSetuError};
use crate::llm::LLMTextGenerator;
use crate::types::{
    ChatMessage, LlmApiResult, MessageContent, ResponseFormat, Role, StopReason,
    TokenLogprob, ToolCall, ToolDefinition, TopTokenAlternative,
};
use log::debug;
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::{json, Value};

// ── Header helpers ────────────────────────────────────────────────────────────

/// Build a `HeaderMap` containing `Authorization: Bearer <key>`.
pub fn prepare_bearer_headers(api_key: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    let value = format!("Bearer {api_key}");
    if let Ok(hv) = HeaderValue::from_str(&value) {
        headers.insert(reqwest::header::AUTHORIZATION, hv);
    }
    headers
}

// ── Payload construction ──────────────────────────────────────────────────────

/// Convert our `ChatMessage` slice into the OpenAI messages array.
fn build_messages_json(messages: &[ChatMessage], system_fallback: Option<&str>) -> Value {
    let mut out: Vec<Value> = Vec::new();

    // Prepend a system message from the generator if none is in the slice.
    let has_system = messages.iter().any(|m| m.role == Role::System);
    if !has_system && let Some(sp) = system_fallback.filter(|s| !s.is_empty()) {
        out.push(json!({"role": "system", "content": sp}));
    }

    for msg in messages {
        let role = msg.role.as_str();
        let mut obj = match &msg.content {
            MessageContent::Text(text) => {
                json!({"role": role, "content": text})
            }
            MessageContent::ToolCalls(calls) => {
                let tc_json: Vec<Value> = calls
                    .iter()
                    .map(|tc| {
                        json!({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments.to_string()
                            }
                        })
                    })
                    .collect();
                json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": tc_json
                })
            }
            // Blocks are flattened to text for OpenAI-compat
            MessageContent::Blocks(blocks) => {
                use crate::types::ContentBlock;
                let text = blocks
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::Text { text } = b {
                            Some(text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                json!({"role": role, "content": text})
            }
        };

        if let Some(id) = &msg.tool_call_id {
            obj["tool_call_id"] = json!(id);
        }
        if let Some(name) = &msg.name {
            obj["name"] = json!(name);
        }

        out.push(obj);
    }

    json!(out)
}

/// Build the full request JSON for an OpenAI-compatible completion call.
pub fn prepare_openai_compat_payload(
    messages: &[ChatMessage],
    tools: Option<&[ToolDefinition]>,
    response_format: Option<&ResponseFormat>,
    params: &LLMTextGenerator,
) -> Value {
    let messages_json = build_messages_json(
        messages,
        params.system_prompt.as_deref(),
    );

    let mut payload = json!({
        "model": params.model_name,
        "messages": messages_json,
        "temperature": params.model_temperature,
        "logprobs": true,
        "top_logprobs": 5,
        "stream": false,
    });

    // max_tokens field — OpenAI renamed it to max_completion_tokens but the
    // old name still works; compatible APIs (DeepSeek, Qwen, llama.cpp) use
    // max_tokens.  We send both to maximise compat.
    payload["max_tokens"] = json!(params.max_tok_gen);
    payload["max_completion_tokens"] = json!(params.max_tok_gen);

    if let Some(tool_defs) = tools {
        let tools_json: Vec<Value> = tool_defs
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters
                    }
                })
            })
            .collect();
        payload["tools"] = json!(tools_json);
        payload["tool_choice"] = json!("auto");
    }

    if let Some(fmt) = response_format {
        payload["response_format"] = match fmt {
            ResponseFormat::Text => json!({"type": "text"}),
            ResponseFormat::JsonObject => json!({"type": "json_object"}),
            ResponseFormat::JsonSchema { schema, name } => json!({
                "type": "json_schema",
                "json_schema": {
                    "name": name.as_deref().unwrap_or("response"),
                    "schema": schema,
                    "strict": true
                }
            }),
        };
    }

    payload
}

// ── Response parsing ──────────────────────────────────────────────────────────

pub(crate) fn parse_openai_compat_response(json: &Value) -> Result<LlmApiResult, SamvadSetuError> {
    // Some providers embed errors in a 200 response body.
    if let Some(err) = json.get("error") {
        return Err(SamvadSetuError::Provider {
            error_type: err
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("api_error")
                .to_string(),
            message: err
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown error")
                .to_string(),
            param: err.get("param").and_then(|v| v.as_str()).map(str::to_string),
            code: err.get("code").and_then(|v| v.as_str()).map(str::to_string),
        });
    }

    let mut result = LlmApiResult {
        model_used: json
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        ..Default::default()
    };

    if let Some(usage) = json.get("usage") {
        result.input_tokens_count = usage
            .get("prompt_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        result.output_tokens_count = usage
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
    }

    let choice = match json.get("choices").and_then(|c| c.get(0)) {
        Some(c) => c,
        None => {
            return Err(SamvadSetuError::Parse {
                message: "No choices in response".to_string(),
                raw_response: Some(json.to_string()),
            })
        }
    };

    if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
        result.stop_reason = StopReason::from_str(reason);
    }

    if let Some(message) = choice.get("message") {
        // Text content
        match message.get("content") {
            Some(v) if !v.is_null() => {
                result.generated_text = v
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| v.to_string());
            }
            _ => {}
        }

        // DeepSeek chain-of-thought
        if let Some(rc) = message.get("reasoning_content").and_then(|v| v.as_str()) {
            result.reasoning_content = (!rc.is_empty()).then(|| rc.to_string());
        }

        // Tool calls
        if let Some(calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in calls {
                let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                if let Some(func) = tc.get("function") {
                    let name = func
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let args_str = func
                        .get("arguments")
                        .and_then(|v| v.as_str())
                        .unwrap_or("{}");
                    let arguments: Value =
                        serde_json::from_str(args_str).unwrap_or_else(|_| json!({}));
                    result.tool_calls.push(ToolCall { id, name, arguments });
                }
            }
        }
    }

    // Logprobs
    if let Some(lp_obj) = choice.get("logprobs")
        && let Some(content) = lp_obj.get("content").and_then(|v| v.as_array())
    {
            for lp in content {
                let token =
                    lp.get("token").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let logprob = lp.get("logprob").and_then(|v| v.as_f64()).unwrap_or(f64::NEG_INFINITY);
                let bytes: Vec<u8> = lp
                    .get("bytes")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|b| b.as_u64().map(|n| n as u8))
                            .collect()
                    })
                    .unwrap_or_default();
                let top_alternatives: Vec<TopTokenAlternative> = lp
                    .get("top_logprobs")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .map(|alt| TopTokenAlternative {
                                token: alt
                                    .get("token")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                logprob: alt
                                    .get("logprob")
                                    .and_then(|v| v.as_f64())
                                    .unwrap_or(f64::NEG_INFINITY),
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                result.logprobs.push(TokenLogprob {
                    token,
                    logprob,
                    bytes,
                    top_alternatives,
                });
            }
    }

    Ok(result)
}

// ── HTTP call ─────────────────────────────────────────────────────────────────

/// Execute an OpenAI-compatible chat completions request.
pub fn http_post_openai_compat(
    params: &LLMTextGenerator,
    client: &Client,
    messages: &[ChatMessage],
    tools: Option<&[ToolDefinition]>,
    response_format: Option<&ResponseFormat>,
) -> Result<LlmApiResult, SamvadSetuError> {
    let payload = prepare_openai_compat_payload(messages, tools, response_format, params);
    debug!("{} request to {}", params.llm_service, params.svc_base_url);

    match client.post(&params.svc_base_url).json(&payload).send() {
        Ok(resp) => {
            let status = resp.status();
            let status_u16 = status.as_u16();

            if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                let retry_after = resp
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok());
                let body = resp.text().unwrap_or_default();
                return Err(SamvadSetuError::RateLimit {
                    retry_after_secs: retry_after,
                    message: body,
                });
            }

            if status == reqwest::StatusCode::UNAUTHORIZED {
                let body = resp.text().unwrap_or_default();
                return Err(SamvadSetuError::Auth(body));
            }

            let body = resp.text().map_err(|e| SamvadSetuError::Network(e.to_string()))?;

            if !status.is_success() {
                return Err(parse_openai_error_body(status_u16, &body));
            }

            let json: Value = serde_json::from_str(&body).map_err(|e| SamvadSetuError::Parse {
                message: e.to_string(),
                raw_response: Some(body.clone()),
            })?;

            debug!("{} response: {json:.200}", params.llm_service);
            parse_openai_compat_response(&json)
        }
        Err(e) => Err(from_reqwest_error(e)),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::LLMTextGenBuilder;
    use crate::types::{ChatMessage, ToolDefinition};
    use serde_json::json;

    fn chatgpt_gen() -> LLMTextGenerator {
        LLMTextGenBuilder::build("chatgpt", "gpt-4o-mini", 60, None, None).unwrap()
    }

    #[test]
    fn test_bearer_header_format() {
        let headers = prepare_bearer_headers("test-key-123");
        let auth = headers
            .get(reqwest::header::AUTHORIZATION)
            .unwrap()
            .to_str()
            .unwrap();
        assert_eq!(auth, "Bearer test-key-123");
    }

    #[test]
    fn test_payload_includes_logprobs() {
        let llm_gen = chatgpt_gen();
        let msgs = vec![ChatMessage::user("Hello")];
        let payload = prepare_openai_compat_payload(&msgs, None, None, &llm_gen);
        assert_eq!(payload["logprobs"], json!(true));
        assert_eq!(payload["top_logprobs"], json!(5));
    }

    #[test]
    fn test_payload_with_tools() {
        let llm_gen = chatgpt_gen();
        let msgs = vec![ChatMessage::user("What is the weather?")];
        let tools = vec![ToolDefinition::new(
            "get_weather",
            "Get current weather",
            json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
        )];
        let payload = prepare_openai_compat_payload(&msgs, Some(&tools), None, &llm_gen);
        assert!(payload["tools"].is_array());
        assert_eq!(payload["tools"][0]["function"]["name"], json!("get_weather"));
        assert_eq!(payload["tool_choice"], json!("auto"));
    }

    #[test]
    fn test_payload_json_mode() {
        let llm_gen = chatgpt_gen();
        let msgs = vec![ChatMessage::user("Return JSON")];
        let payload =
            prepare_openai_compat_payload(&msgs, None, Some(&ResponseFormat::JsonObject), &llm_gen);
        assert_eq!(payload["response_format"]["type"], json!("json_object"));
    }

    #[test]
    fn test_parse_response_extracts_text() {
        let json = json!({
            "model": "gpt-4o-mini",
            "choices": [{
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Hello world"},
                "logprobs": null
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });
        let result = parse_openai_compat_response(&json).unwrap();
        assert_eq!(result.generated_text, "Hello world");
        assert_eq!(result.input_tokens_count, 10);
        assert_eq!(result.output_tokens_count, 5);
        assert_eq!(result.stop_reason, StopReason::Stop);
    }

    #[test]
    fn test_parse_response_extracts_tool_calls() {
        let json = json!({
            "model": "gpt-4o-mini",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Paris\"}"
                        }
                    }]
                },
                "logprobs": null
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15}
        });
        let result = parse_openai_compat_response(&json).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert_eq!(result.tool_calls[0].id, "call_abc");
        assert_eq!(result.tool_calls[0].arguments["city"], "Paris");
        assert_eq!(result.stop_reason, StopReason::ToolUse);
    }

    #[test]
    fn test_parse_response_extracts_logprobs() {
        let json = json!({
            "model": "gpt-4o-mini",
            "choices": [{
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Hi"},
                "logprobs": {
                    "content": [
                        {
                            "token": "Hi",
                            "logprob": -0.5,
                            "bytes": [72, 105],
                            "top_logprobs": [
                                {"token": "Hi", "logprob": -0.5},
                                {"token": "Hello", "logprob": -1.2}
                            ]
                        }
                    ]
                }
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1}
        });
        let result = parse_openai_compat_response(&json).unwrap();
        assert_eq!(result.logprobs.len(), 1);
        assert_eq!(result.logprobs[0].token, "Hi");
        assert!((result.logprobs[0].logprob - (-0.5)).abs() < 1e-9);
        assert_eq!(result.logprobs[0].top_alternatives.len(), 2);
    }

    #[test]
    fn test_parse_provider_error_in_response() {
        let json = json!({
            "error": {
                "type": "invalid_request_error",
                "message": "The model does not exist",
                "code": "model_not_found"
            }
        });
        let err = parse_openai_compat_response(&json).unwrap_err();
        match err {
            SamvadSetuError::Provider { error_type, message, code, .. } => {
                assert_eq!(error_type, "invalid_request_error");
                assert!(message.contains("does not exist"));
                assert_eq!(code, Some("model_not_found".to_string()));
            }
            _ => panic!("Expected Provider error"),
        }
    }

    #[test]
    fn test_deepseek_builder() {
        let llm_gen = LLMTextGenBuilder::build("deepseek", "deepseek-v4-pro", 60, None, None).unwrap();
        assert_eq!(llm_gen.llm_service, "deepseek");
        assert!(llm_gen.svc_base_url.contains("deepseek.com"));
    }

    #[test]
    fn test_qwen_builder() {
        let llm_gen = LLMTextGenBuilder::build("qwen", "qwen-plus", 60, None, None).unwrap();
        assert_eq!(llm_gen.llm_service, "qwen");
        assert!(llm_gen.svc_base_url.contains("aliyuncs.com"));
    }

    #[test]
    fn test_llamacpp_builder() {
        let llm_gen =
            LLMTextGenBuilder::build("llamacpp", "llama-3.2-3b", 60, None, None).unwrap();
        assert_eq!(llm_gen.llm_service, "llamacpp");
        assert!(llm_gen.svc_base_url.contains("8080"));
    }

    #[test]
    fn test_system_fallback_added_when_missing() {
        let mut llm_gen = chatgpt_gen();
        llm_gen.system_prompt = Some("You are a test assistant.".to_string());
        let msgs = vec![ChatMessage::user("Hello")];
        let payload = prepare_openai_compat_payload(&msgs, None, None, &llm_gen);
        let messages = payload["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are a test assistant.");
    }

    #[test]
    fn test_system_message_not_duplicated() {
        let mut llm_gen = chatgpt_gen();
        llm_gen.system_prompt = Some("From struct".to_string());
        let msgs = vec![
            ChatMessage::system("From caller"),
            ChatMessage::user("Hello"),
        ];
        let payload = prepare_openai_compat_payload(&msgs, None, None, &llm_gen);
        let messages = payload["messages"].as_array().unwrap();
        // Should have exactly the caller's system message, not duplicated
        let system_msgs: Vec<_> = messages
            .iter()
            .filter(|m| m["role"] == "system")
            .collect();
        assert_eq!(system_msgs.len(), 1);
        assert_eq!(system_msgs[0]["content"], "From caller");
    }

    // Live API test — disabled in CI (no network or keys)
    #[test]
    #[ignore]
    fn test_live_chatgpt_call() {
        let llm_gen = chatgpt_gen();
        let msgs = vec![ChatMessage::user(
            "How is a rainbow formed? Reply in one sentence.",
        )];
        let result = llm_gen.generate_text(&msgs, None, None).unwrap();
        assert!(!result.generated_text.is_empty());
        assert!(result.input_tokens_count > 0);
        assert!(!result.logprobs.is_empty());
    }

    #[test]
    #[ignore]
    fn test_live_deepseek_call() {
        let llm_gen =
            LLMTextGenBuilder::build("deepseek", "deepseek-v4-pro", 60, None, None).unwrap();
        let msgs = vec![ChatMessage::user("What is 2+2? Reply with just the number.")];
        let result = llm_gen.generate_text(&msgs, None, None).unwrap();
        assert!(result.generated_text.contains('4'));
    }
}
