// providers/anthropic.rs — Anthropic Messages API (Claude)
//
// Reference: https://platform.claude.com/docs/en/api/messages
//
// Key differences from OpenAI format:
//  • Auth via `x-api-key` header (not `Authorization: Bearer`)
//  • `anthropic-version` header required
//  • System prompt is a top-level `system` field, not a message
//  • Tool calls are content blocks (`tool_use`), not a separate array
//  • Tool results are `user` messages containing `tool_result` blocks
//  • No logprobs support

use crate::error::{from_reqwest_error, parse_anthropic_error_body, SamvadSetuError};
use crate::llm::LLMTextGenerator;
use crate::types::{
    ChatMessage, ContentBlock, LlmApiResult, MessageContent, Role, StopReason, ToolCall,
    ToolDefinition,
};
use log::debug;
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::{json, Value};

pub const ANTHROPIC_VERSION: &str = "2023-06-01";

// ── Header helpers ────────────────────────────────────────────────────────────

pub fn prepare_anthropic_headers(api_key: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    if let Ok(hv) = HeaderValue::from_str(api_key) {
        headers.insert("x-api-key", hv);
    }
    headers.insert(
        "anthropic-version",
        HeaderValue::from_static(ANTHROPIC_VERSION),
    );
    headers
}

// ── Payload construction ──────────────────────────────────────────────────────

/// Convert our `ChatMessage` slice into the Anthropic messages array.
/// System messages are extracted separately and returned as a string.
fn build_anthropic_messages(messages: &[ChatMessage]) -> (Option<String>, Vec<Value>) {
    let system: Option<String> = {
        let parts: Vec<String> = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .filter_map(|m| m.text().map(str::to_string))
            .collect();
        if parts.is_empty() { None } else { Some(parts.join("\n")) }
    };

    let mut api_messages: Vec<Value> = Vec::new();

    for msg in messages.iter().filter(|m| m.role != Role::System) {
        let role = match msg.role {
            Role::User | Role::Tool => "user",
            Role::Assistant => "assistant",
            Role::System => continue,
        };

        let content_val: Value = match &msg.content {
            MessageContent::Text(text) => {
                if msg.role == Role::Tool {
                    // Tool results must be wrapped in a tool_result block
                    let tool_use_id = msg
                        .tool_call_id
                        .as_deref()
                        .unwrap_or("");
                    json!([{
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": text
                    }])
                } else {
                    json!(text)
                }
            }
            MessageContent::ToolCalls(calls) => {
                // Assistant requested tool calls → tool_use content blocks
                let blocks: Vec<Value> = calls
                    .iter()
                    .map(|tc| {
                        json!({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments
                        })
                    })
                    .collect();
                json!(blocks)
            }
            MessageContent::Blocks(blocks) => {
                let api_blocks: Vec<Value> = blocks
                    .iter()
                    .map(|b| match b {
                        ContentBlock::Text { text } => json!({"type": "text", "text": text}),
                        ContentBlock::ToolUse { id, name, input } => json!({
                            "type": "tool_use",
                            "id": id,
                            "name": name,
                            "input": input
                        }),
                        ContentBlock::ToolResult { tool_use_id, content, is_error } => json!({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": content,
                            "is_error": is_error
                        }),
                    })
                    .collect();
                json!(api_blocks)
            }
        };

        api_messages.push(json!({"role": role, "content": content_val}));
    }

    (system, api_messages)
}

pub fn prepare_anthropic_payload(
    messages: &[ChatMessage],
    tools: Option<&[ToolDefinition]>,
    params: &LLMTextGenerator,
) -> Value {
    let (system_from_messages, api_messages) = build_anthropic_messages(messages);

    // Resolve system prompt: message-level takes priority over struct-level.
    let system = system_from_messages
        .or_else(|| params.system_prompt.clone())
        .filter(|s| !s.is_empty());

    let mut payload = json!({
        "model": params.model_name,
        "max_tokens": params.max_tok_gen,
        "messages": api_messages,
    });

    if let Some(sp) = system {
        payload["system"] = json!(sp);
    }

    if params.model_temperature > 0.0 {
        payload["temperature"] = json!(params.model_temperature);
    }

    if let Some(tool_defs) = tools {
        let tools_json: Vec<Value> = tool_defs
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters
                })
            })
            .collect();
        payload["tools"] = json!(tools_json);
    }

    payload
}

// ── Response parsing ──────────────────────────────────────────────────────────

pub(crate) fn parse_anthropic_response(json: &Value) -> Result<LlmApiResult, SamvadSetuError> {
    // Top-level error object
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
            param: None,
            code: None,
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
        result.input_tokens_count = usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
        result.output_tokens_count =
            usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
    }

    if let Some(stop_reason) = json.get("stop_reason").and_then(|v| v.as_str()) {
        result.stop_reason = StopReason::from_str(stop_reason);
    }

    // Content blocks
    if let Some(content) = json.get("content").and_then(|v| v.as_array()) {
        for block in content {
            match block.get("type").and_then(|v| v.as_str()) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                        if !result.generated_text.is_empty() {
                            result.generated_text.push('\n');
                        }
                        result.generated_text.push_str(text);
                    }
                }
                Some("tool_use") => {
                    let id = block
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let name = block
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let arguments = block
                        .get("input")
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    result.tool_calls.push(ToolCall { id, name, arguments });
                }
                Some("thinking") => {
                    if let Some(thinking) = block.get("thinking").and_then(|v| v.as_str()) {
                        result.reasoning_content = Some(thinking.to_string());
                    }
                }
                _ => {}
            }
        }
    }

    Ok(result)
}

// ── HTTP call ─────────────────────────────────────────────────────────────────

pub fn http_post_anthropic(
    params: &LLMTextGenerator,
    client: &Client,
    messages: &[ChatMessage],
    tools: Option<&[ToolDefinition]>,
) -> Result<LlmApiResult, SamvadSetuError> {
    let payload = prepare_anthropic_payload(messages, tools, params);
    debug!("Anthropic request to {}", params.svc_base_url);

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
                return Err(parse_anthropic_error_body(status_u16, &body));
            }

            let json: Value = serde_json::from_str(&body).map_err(|e| SamvadSetuError::Parse {
                message: e.to_string(),
                raw_response: Some(body.clone()),
            })?;

            debug!("Anthropic response: {json:.200}");
            parse_anthropic_response(&json)
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

    fn claude_gen() -> LLMTextGenerator {
        LLMTextGenBuilder::build("claude", "claude-haiku-4-5", 60, None, None).unwrap()
    }

    #[test]
    fn test_anthropic_headers_contain_version() {
        let headers = prepare_anthropic_headers("key-123");
        assert!(headers.contains_key("x-api-key"));
        assert!(headers.contains_key("anthropic-version"));
    }

    #[test]
    fn test_system_extracted_from_messages() {
        let llm_gen = claude_gen();
        let msgs = vec![
            ChatMessage::system("Be helpful."),
            ChatMessage::user("Hello"),
        ];
        let payload = prepare_anthropic_payload(&msgs, None, &llm_gen);
        assert_eq!(payload["system"], json!("Be helpful."));
        let messages = payload["messages"].as_array().unwrap();
        // System message must NOT appear in the messages array
        assert!(!messages.iter().any(|m| m["role"] == "system"));
    }

    #[test]
    fn test_system_from_struct_when_no_system_message() {
        let mut llm_gen = claude_gen();
        llm_gen.system_prompt = Some("Struct system".to_string());
        let msgs = vec![ChatMessage::user("Hi")];
        let payload = prepare_anthropic_payload(&msgs, None, &llm_gen);
        assert_eq!(payload["system"], json!("Struct system"));
    }

    #[test]
    fn test_tool_definitions_use_input_schema_key() {
        let llm_gen = claude_gen();
        let msgs = vec![ChatMessage::user("What's the weather?")];
        let tools = vec![ToolDefinition::new(
            "get_weather",
            "Returns weather data",
            json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        )];
        let payload = prepare_anthropic_payload(&msgs, Some(&tools), &llm_gen);
        assert!(payload["tools"].is_array());
        // Anthropic uses "input_schema" not "parameters"
        assert!(!payload["tools"][0]["input_schema"].is_null());
        assert!(payload["tools"][0]["parameters"].is_null());
    }

    #[test]
    fn test_tool_result_message_becomes_user_with_block() {
        let llm_gen = claude_gen();
        let msgs = vec![
            ChatMessage::user("Use weather tool"),
            ChatMessage::tool_result("call_1", "get_weather", "Sunny, 22°C"),
        ];
        let payload = prepare_anthropic_payload(&msgs, None, &llm_gen);
        let messages = payload["messages"].as_array().unwrap();
        let tool_msg = messages.iter().find(|m| m["role"] == "user" && m["content"].is_array());
        assert!(tool_msg.is_some());
        let block = &tool_msg.unwrap()["content"][0];
        assert_eq!(block["type"], "tool_result");
        assert_eq!(block["tool_use_id"], "call_1");
    }

    #[test]
    fn test_assistant_tool_calls_become_blocks() {
        use crate::types::ToolCall;
        let llm_gen = claude_gen();
        let msgs = vec![ChatMessage::assistant_with_tool_calls(vec![ToolCall {
            id: "call_abc".to_string(),
            name: "get_weather".to_string(),
            arguments: json!({"city": "Paris"}),
        }])];
        let payload = prepare_anthropic_payload(&msgs, None, &llm_gen);
        let messages = payload["messages"].as_array().unwrap();
        let asst = messages.iter().find(|m| m["role"] == "assistant").unwrap();
        let block = &asst["content"][0];
        assert_eq!(block["type"], "tool_use");
        assert_eq!(block["name"], "get_weather");
    }

    #[test]
    fn test_parse_text_response() {
        let json = json!({
            "type": "message",
            "id": "msg_01",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Hello there!"}],
            "usage": {"input_tokens": 12, "output_tokens": 3}
        });
        let result = parse_anthropic_response(&json).unwrap();
        assert_eq!(result.generated_text, "Hello there!");
        assert_eq!(result.input_tokens_count, 12);
        assert_eq!(result.stop_reason, StopReason::Stop);
    }

    #[test]
    fn test_parse_tool_use_response() {
        let json = json!({
            "type": "message",
            "id": "msg_02",
            "role": "assistant",
            "model": "claude-opus-4-6",
            "stop_reason": "tool_use",
            "content": [{
                "type": "tool_use",
                "id": "toolu_01",
                "name": "get_weather",
                "input": {"city": "Tokyo"}
            }],
            "usage": {"input_tokens": 50, "output_tokens": 20}
        });
        let result = parse_anthropic_response(&json).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert_eq!(result.tool_calls[0].arguments["city"], "Tokyo");
        assert_eq!(result.stop_reason, StopReason::ToolUse);
    }

    #[test]
    fn test_parse_error_response() {
        let json = json!({
            "type": "error",
            "error": {
                "type": "authentication_error",
                "message": "Invalid API key"
            }
        });
        let err = parse_anthropic_response(&json).unwrap_err();
        match err {
            SamvadSetuError::Provider { error_type, message, .. } => {
                assert_eq!(error_type, "authentication_error");
                assert!(message.contains("Invalid"));
            }
            _ => panic!("Expected Provider error"),
        }
    }

    #[test]
    fn test_no_logprobs_does_not_error() {
        // Claude doesn't return logprobs; result should just have empty vec.
        let json = json!({
            "type": "message",
            "id": "msg_03",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Answer"}],
            "usage": {"input_tokens": 5, "output_tokens": 1}
        });
        let result = parse_anthropic_response(&json).unwrap();
        assert!(result.logprobs.is_empty());
        assert!(result.mean_logprob().is_none());
    }

    #[test]
    fn test_thinking_block_captured() {
        let json = json!({
            "type": "message",
            "id": "msg_04",
            "role": "assistant",
            "model": "claude-opus-4-8",
            "stop_reason": "end_turn",
            "content": [
                {"type": "thinking", "thinking": "Let me reason through this..."},
                {"type": "text", "text": "The answer is 42."}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20}
        });
        let result = parse_anthropic_response(&json).unwrap();
        assert_eq!(result.generated_text, "The answer is 42.");
        assert_eq!(
            result.reasoning_content,
            Some("Let me reason through this...".to_string())
        );
    }

    #[test]
    #[ignore]
    fn test_live_claude_call() {
        let llm_gen = claude_gen();
        let msgs = vec![ChatMessage::user(
            "What is 2 + 2? Reply with only the number.",
        )];
        let result = llm_gen.generate_text(&msgs, None, None).unwrap();
        assert!(result.generated_text.contains('4'));
    }
}
