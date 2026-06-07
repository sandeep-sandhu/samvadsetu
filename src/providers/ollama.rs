// providers/ollama.rs — Ollama /api/chat endpoint
//
// Reference: https://docs.ollama.com/api/chat
//
// Notable differences from the old /api/generate endpoint:
//  • Uses /api/chat with a "messages" array (OpenAI-style roles)
//  • Supports tool calling, logprobs, thinking traces, and JSON schema output
//  • Streaming is on by default; we always send stream=false

use crate::error::{from_reqwest_error, SamvadSetuError};
use crate::llm::LLMTextGenerator;
use crate::types::{
    ChatMessage, LlmApiResult, MessageContent, ResponseFormat, Role, StopReason, ToolCall,
    ToolDefinition, TokenLogprob, TopTokenAlternative,
};
use log::debug;
use reqwest::blocking::Client;
use serde_json::{json, Value};

// ── Helpers for model-specific prompt formatting ──────────────────────────────

/// Build a Llama-style prompt string (for use with /api/generate if needed).
pub fn prepare_llama_prompt(system: &str, context: &str, input: &str) -> String {
    format!(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system}\
        <|eot_id|><|start_header_id|>user<|end_header_id|>{context}\n\n{input}\
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )
}

/// Build a Gemma-style prompt string.
pub fn prepare_gemma_prompt(context: &str, input: &str) -> String {
    format!(
        "<start_of_turn>user{context}{input}<end_of_turn><start_of_turn>model"
    )
}

// ── Payload construction ──────────────────────────────────────────────────────

fn build_ollama_messages(messages: &[ChatMessage], system_fallback: Option<&str>) -> Value {
    let mut out: Vec<Value> = Vec::new();

    let has_system = messages.iter().any(|m| m.role == Role::System);
    if !has_system && let Some(sp) = system_fallback.filter(|s| !s.is_empty()) {
        out.push(json!({"role": "system", "content": sp}));
    }

    for msg in messages {
        let role = msg.role.as_str();
        let mut obj = match &msg.content {
            MessageContent::Text(text) => json!({"role": role, "content": text}),
            MessageContent::ToolCalls(calls) => {
                let tc_json: Vec<Value> = calls
                    .iter()
                    .map(|tc| {
                        json!({
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments
                            }
                        })
                    })
                    .collect();
                json!({"role": "assistant", "content": "", "tool_calls": tc_json})
            }
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
        out.push(obj);
    }

    json!(out)
}

pub fn prepare_ollama_chat_payload(
    messages: &[ChatMessage],
    tools: Option<&[ToolDefinition]>,
    response_format: Option<&ResponseFormat>,
    params: &LLMTextGenerator,
) -> Value {
    let messages_json = build_ollama_messages(messages, params.system_prompt.as_deref());

    let mut payload = json!({
        "model": params.model_name,
        "messages": messages_json,
        "stream": false,
        "options": {
            "temperature": params.model_temperature,
            "num_predict": params.max_tok_gen,
            "num_ctx": params.num_context
        },
        "logprobs": true,
        "top_logprobs": 5
    });

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
    }

    if let Some(fmt) = response_format {
        payload["format"] = match fmt {
            ResponseFormat::Text => Value::Null,
            ResponseFormat::JsonObject => json!("json"),
            ResponseFormat::JsonSchema { schema, .. } => schema.clone(),
        };
    }

    payload
}

// ── Response parsing ──────────────────────────────────────────────────────────

pub(crate) fn parse_ollama_chat_response(
    json: &Value,
) -> Result<LlmApiResult, SamvadSetuError> {
    let mut result = LlmApiResult {
        model_used: json
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        ..Default::default()
    };

    if let Some(reason) = json.get("done_reason").and_then(|v| v.as_str()) {
        result.stop_reason = match reason {
            "stop" => StopReason::Stop,
            "length" => StopReason::MaxTokens,
            other => StopReason::Other(other.to_string()),
        };
    }

    result.input_tokens_count = json
        .get("prompt_eval_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    result.output_tokens_count = json
        .get("eval_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    if let Some(message) = json.get("message") {
        if let Some(content) = message.get("content").and_then(|v| v.as_str()) {
            result.generated_text = content.to_string();
        }

        // Thinking trace (Ollama `think` param)
        if let Some(thinking) = message.get("thinking").and_then(|v| v.as_str()) {
            result.reasoning_content = (!thinking.is_empty()).then(|| thinking.to_string());
        }

        // Tool calls
        if let Some(calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in calls {
                if let Some(func) = tc.get("function") {
                    let name = func
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    // Ollama returns arguments as an already-parsed object
                    let arguments = func
                        .get("arguments")
                        .cloned()
                        .unwrap_or_else(|| json!({}));
                    result.tool_calls.push(ToolCall {
                        id: format!("call_{}", result.tool_calls.len()),
                        name,
                        arguments,
                    });
                }
            }
            if !result.tool_calls.is_empty() {
                result.stop_reason = StopReason::ToolUse;
            }
        }
    }

    // Logprobs (returned at the top level in Ollama's response)
    if let Some(logprobs_arr) = json.get("logprobs").and_then(|v| v.as_array()) {
        for lp in logprobs_arr {
            let token = lp
                .get("token")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let logprob = lp
                .get("logprob")
                .and_then(|v| v.as_f64())
                .unwrap_or(f64::NEG_INFINITY);
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
                bytes: vec![],
                top_alternatives,
            });
        }
    }

    Ok(result)
}

// ── HTTP call ─────────────────────────────────────────────────────────────────

pub fn http_post_chat_ollama(
    params: &LLMTextGenerator,
    client: &Client,
    messages: &[ChatMessage],
    tools: Option<&[ToolDefinition]>,
    response_format: Option<&ResponseFormat>,
) -> Result<LlmApiResult, SamvadSetuError> {
    let payload = prepare_ollama_chat_payload(messages, tools, response_format, params);
    debug!("Ollama request to {}", params.svc_base_url);

    match client.post(&params.svc_base_url).json(&payload).send() {
        Ok(resp) => {
            let status = resp.status();
            let status_u16 = status.as_u16();

            let body = resp.text().map_err(|e| SamvadSetuError::Network(e.to_string()))?;

            if !status.is_success() {
                return Err(SamvadSetuError::Http { status: status_u16, body });
            }

            let json: Value =
                serde_json::from_str(&body).map_err(|e| SamvadSetuError::Parse {
                    message: e.to_string(),
                    raw_response: Some(body.clone()),
                })?;

            debug!("Ollama response: {json:.200}");
            parse_ollama_chat_response(&json)
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

    fn ollama_gen() -> LLMTextGenerator {
        LLMTextGenBuilder::build("ollama", "gemma3", 60, None, None).unwrap()
    }

    #[test]
    fn test_ollama_uses_chat_endpoint() {
        let llm_gen = ollama_gen();
        assert!(llm_gen.svc_base_url.contains("/api/chat"));
    }

    #[test]
    fn test_payload_structure() {
        let llm_gen = ollama_gen();
        let msgs = vec![ChatMessage::user("Hello")];
        let payload = prepare_ollama_chat_payload(&msgs, None, None, &llm_gen);
        assert_eq!(payload["model"], "gemma3");
        assert_eq!(payload["stream"], json!(false));
        assert!(payload["messages"].is_array());
    }

    #[test]
    fn test_json_mode_sets_format() {
        let llm_gen = ollama_gen();
        let msgs = vec![ChatMessage::user("Return JSON")];
        let payload =
            prepare_ollama_chat_payload(&msgs, None, Some(&ResponseFormat::JsonObject), &llm_gen);
        assert_eq!(payload["format"], json!("json"));
    }

    #[test]
    fn test_json_schema_sets_format_object() {
        let llm_gen = ollama_gen();
        let msgs = vec![ChatMessage::user("Return structured data")];
        let schema = json!({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        });
        let payload = prepare_ollama_chat_payload(
            &msgs,
            None,
            Some(&ResponseFormat::JsonSchema { schema: schema.clone(), name: None }),
            &llm_gen,
        );
        assert_eq!(payload["format"], schema);
    }

    #[test]
    fn test_tools_in_payload() {
        let llm_gen = ollama_gen();
        let msgs = vec![ChatMessage::user("Search something")];
        let tools = vec![ToolDefinition::new(
            "search",
            "Search the web",
            json!({"type": "object", "properties": {"query": {"type": "string"}}}),
        )];
        let payload = prepare_ollama_chat_payload(&msgs, Some(&tools), None, &llm_gen);
        assert!(payload["tools"].is_array());
        assert_eq!(payload["tools"][0]["function"]["name"], "search");
    }

    #[test]
    fn test_parse_text_response() {
        let json = json!({
            "model": "gemma3",
            "done": true,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "Hello there!"},
            "prompt_eval_count": 10,
            "eval_count": 5
        });
        let result = parse_ollama_chat_response(&json).unwrap();
        assert_eq!(result.generated_text, "Hello there!");
        assert_eq!(result.input_tokens_count, 10);
        assert_eq!(result.output_tokens_count, 5);
        assert_eq!(result.stop_reason, StopReason::Stop);
    }

    #[test]
    fn test_parse_tool_call_response() {
        let json = json!({
            "model": "llama3.2",
            "done": true,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Berlin"}
                    }
                }]
            },
            "prompt_eval_count": 30,
            "eval_count": 20
        });
        let result = parse_ollama_chat_response(&json).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert_eq!(result.tool_calls[0].arguments["city"], "Berlin");
        assert_eq!(result.stop_reason, StopReason::ToolUse);
    }

    #[test]
    fn test_parse_logprobs() {
        let json = json!({
            "model": "gemma3",
            "done": true,
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "Hi"},
            "logprobs": [
                {
                    "token": "Hi",
                    "logprob": -0.4,
                    "top_logprobs": [
                        {"token": "Hi",    "logprob": -0.4},
                        {"token": "Hello", "logprob": -0.9}
                    ]
                }
            ],
            "prompt_eval_count": 5,
            "eval_count": 1
        });
        let result = parse_ollama_chat_response(&json).unwrap();
        assert_eq!(result.logprobs.len(), 1);
        assert_eq!(result.logprobs[0].token, "Hi");
        assert_eq!(result.logprobs[0].top_alternatives.len(), 2);
    }

    #[test]
    fn test_thinking_captured() {
        let json = json!({
            "model": "deepseek-r1",
            "done": true,
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "42",
                "thinking": "Let me calculate..."
            },
            "prompt_eval_count": 8,
            "eval_count": 2
        });
        let result = parse_ollama_chat_response(&json).unwrap();
        assert_eq!(result.generated_text, "42");
        assert_eq!(
            result.reasoning_content,
            Some("Let me calculate...".to_string())
        );
    }

    #[test]
    fn test_prompt_formatting_llama() {
        let prompt = prepare_llama_prompt("Be helpful", "Context here", "What is 2+2?");
        assert!(prompt.contains("<|begin_of_text|>"));
        assert!(prompt.contains("Be helpful"));
        assert!(prompt.contains("What is 2+2?"));
        assert!(prompt.contains("<|start_header_id|>assistant<|end_header_id|>"));
    }

    #[test]
    fn test_prompt_formatting_gemma() {
        let prompt = prepare_gemma_prompt("Context here", "Question?");
        assert!(prompt.contains("<start_of_turn>user"));
        assert!(prompt.contains("<start_of_turn>model"));
    }

    #[test]
    #[ignore]
    fn test_live_ollama_call() {
        let llm_gen = ollama_gen();
        let msgs = vec![ChatMessage::user(
            "What is 1 + 1? Answer with only the number.",
        )];
        let result = llm_gen.generate_text(&msgs, None, None).unwrap();
        assert!(result.generated_text.contains('2'));
    }
}
