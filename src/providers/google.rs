// providers/google.rs — Google Gemini API (legacy key-in-URL) + Google GenAI API
//
// Two endpoint variants:
//  • "gemini"      — legacy, API key as ?key= query param
//  • "google_genai" — newer, API key as x-goog-api-key header
// Both use the same response format and share response-parsing logic.

use crate::error::{from_reqwest_error, SamvadSetuError};
use crate::llm::LLMTextGenerator;
use crate::types::{
    ChatMessage, LlmApiResult, MessageContent, ResponseFormat, Role, StopReason, ToolCall,
    ToolDefinition, TokenLogprob, TopTokenAlternative,
};
use log::debug;
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::{json, Value};

// ── Header helpers ────────────────────────────────────────────────────────────

pub fn prepare_googlegenai_headers(api_key: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    if let Ok(hv) = HeaderValue::from_str(api_key)
        && let Ok(name) = reqwest::header::HeaderName::from_bytes(b"x-goog-api-key")
    {
        headers.insert(name, hv);
    }
    headers
}

// ── Payload helpers ───────────────────────────────────────────────────────────

/// Convert our `ChatMessage` slice into Gemini's `contents` array.
/// Returns (system_instruction_text, contents_array).
fn build_gemini_contents(messages: &[ChatMessage]) -> (Option<String>, Vec<Value>) {
    let system: Option<String> = {
        let parts: Vec<_> = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .filter_map(|m| m.text().map(str::to_string))
            .collect();
        if parts.is_empty() { None } else { Some(parts.join("\n")) }
    };

    let contents: Vec<Value> = messages
        .iter()
        .filter(|m| m.role != Role::System)
        .map(|msg| {
            let role = match msg.role {
                Role::User | Role::Tool => "user",
                Role::Assistant => "model",
                Role::System => "user",
            };

            let parts: Value = match &msg.content {
                MessageContent::Text(text) => {
                    if msg.role == Role::Tool {
                        // Tool result → functionResponse part
                        json!([{
                            "functionResponse": {
                                "name": msg.name.as_deref().unwrap_or("unknown"),
                                "id": msg.tool_call_id.as_deref().unwrap_or(""),
                                "response": {
                                    "content": text
                                }
                            }
                        }])
                    } else {
                        json!([{"text": text}])
                    }
                }
                MessageContent::ToolCalls(calls) => {
                    let parts: Vec<Value> = calls
                        .iter()
                        .map(|tc| {
                            json!({
                                "functionCall": {
                                    "id": tc.id,
                                    "name": tc.name,
                                    "args": tc.arguments
                                }
                            })
                        })
                        .collect();
                    json!(parts)
                }
                MessageContent::Blocks(blocks) => {
                    use crate::types::ContentBlock;
                    let parts: Vec<Value> = blocks
                        .iter()
                        .map(|b| match b {
                            ContentBlock::Text { text } => json!({"text": text}),
                            ContentBlock::ToolUse { id, name, input } => json!({
                                "functionCall": {"id": id, "name": name, "args": input}
                            }),
                            ContentBlock::ToolResult { tool_use_id, content, .. } => json!({
                                "functionResponse": {
                                    "id": tool_use_id,
                                    "name": "function",
                                    "response": {"content": content}
                                }
                            }),
                        })
                        .collect();
                    json!(parts)
                }
            };

            json!({"role": role, "parts": parts})
        })
        .collect();

    (system, contents)
}

/// Build `tools` array in Gemini format.
fn build_gemini_tools(tools: &[ToolDefinition]) -> Value {
    let declarations: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters
            })
        })
        .collect();
    json!([{"functionDeclarations": declarations}])
}

pub fn prepare_google_genai_payload(
    messages: &[ChatMessage],
    tools: Option<&[ToolDefinition]>,
    response_format: Option<&ResponseFormat>,
    params: &LLMTextGenerator,
) -> Value {
    let (system_text, contents) = build_gemini_contents(messages);
    let sys = system_text
        .or_else(|| params.system_prompt.clone())
        .filter(|s| !s.is_empty());

    let mut generation_config = json!({
        "temperature": params.model_temperature,
        "maxOutputTokens": params.max_tok_gen,
        "responseLogprobs": true,
        "logprobs": 5
    });

    match response_format {
        Some(ResponseFormat::JsonObject) => {
            generation_config["responseMimeType"] = json!("application/json");
        }
        Some(ResponseFormat::JsonSchema { schema, .. }) => {
            generation_config["responseMimeType"] = json!("application/json");
            generation_config["responseSchema"] = schema.clone();
        }
        _ => {}
    }

    let mut payload = json!({
        "contents": contents,
        "generationConfig": generation_config,
        "safetySettings": [
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}
        ]
    });

    if let Some(sp) = sys {
        payload["systemInstruction"] = json!({"parts": [{"text": sp}]});
    }

    if let Some(td) = tools {
        payload["tools"] = build_gemini_tools(td);
        payload["toolConfig"] = json!({
            "functionCallingConfig": {"mode": "AUTO"}
        });
    }

    payload
}

/// Build legacy Gemini payload (same shape, no system_instruction).
pub fn prepare_gemini_legacy_payload(
    messages: &[ChatMessage],
    tools: Option<&[ToolDefinition]>,
    params: &LLMTextGenerator,
) -> Value {
    prepare_google_genai_payload(messages, tools, None, params)
}

// ── Response parsing ──────────────────────────────────────────────────────────

pub(crate) fn parse_gemini_response(json: &Value) -> Result<LlmApiResult, SamvadSetuError> {
    // Error in body
    if let Some(err) = json.get("error") {
        return Err(SamvadSetuError::Provider {
            error_type: "gemini_api_error".to_string(),
            message: err
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown Gemini error")
                .to_string(),
            param: None,
            code: err
                .get("code")
                .and_then(|v| v.as_u64())
                .map(|c| c.to_string()),
        });
    }

    let mut result = LlmApiResult::default();

    if let Some(mv) = json.get("modelVersion").and_then(|v| v.as_str()) {
        result.model_used = mv.to_string();
    }

    if let Some(usage) = json.get("usageMetadata") {
        result.input_tokens_count = usage
            .get("promptTokenCount")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        result.output_tokens_count = usage
            .get("candidatesTokenCount")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
    }

    let candidate = match json.get("candidates").and_then(|c| c.get(0)) {
        Some(c) => c,
        None => {
            return Err(SamvadSetuError::Parse {
                message: "No candidates in Gemini response".to_string(),
                raw_response: Some(json.to_string()),
            })
        }
    };

    if let Some(reason) = candidate.get("finishReason").and_then(|v| v.as_str()) {
        result.stop_reason = match reason {
            "STOP" => StopReason::Stop,
            "MAX_TOKENS" => StopReason::MaxTokens,
            "SAFETY" => StopReason::ContentFilter,
            other => StopReason::Other(other.to_string()),
        };
    }

    if let Some(content) = candidate.get("content")
        && let Some(parts) = content.get("parts").and_then(|v| v.as_array())
    {
        for part in parts {
                if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                    if !result.generated_text.is_empty() {
                        result.generated_text.push('\n');
                    }
                    result.generated_text.push_str(text);
                }

                if let Some(fc) = part.get("functionCall") {
                    let id = fc
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let name = fc
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let arguments = fc.get("args").cloned().unwrap_or_else(|| json!({}));
                    result.tool_calls.push(ToolCall { id, name, arguments });
                }
            }
    }

    if !result.tool_calls.is_empty() {
        result.stop_reason = StopReason::ToolUse;
    }

    // Logprobs (available in newer Gemini models when responseLogprobs=true)
    if let Some(lp_result) = candidate.get("logprobsResult")
        && let Some(chosen) = lp_result.get("chosenCandidates").and_then(|v| v.as_array())
    {
            let top_candidates: Vec<&Value> = lp_result
                .get("topCandidates")
                .and_then(|v| v.as_array())
                .map(|v| v.iter().collect())
                .unwrap_or_default();

            for (i, entry) in chosen.iter().enumerate() {
                let token = entry
                    .get("token")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let logprob = entry.get("logProbability").and_then(|v| v.as_f64()).unwrap_or(f64::NEG_INFINITY);

                let top_alternatives: Vec<TopTokenAlternative> = top_candidates
                    .get(i)
                    .and_then(|tc| tc.get("candidates").and_then(|v| v.as_array()))
                    .map(|alts| {
                        alts.iter()
                            .map(|alt| TopTokenAlternative {
                                token: alt
                                    .get("token")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                logprob: alt
                                    .get("logProbability")
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

// ── HTTP calls ────────────────────────────────────────────────────────────────

fn post_to_gemini(
    url: &str,
    client: &Client,
    payload: &Value,
    provider_name: &str,
) -> Result<LlmApiResult, SamvadSetuError> {
    match client.post(url).json(payload).send() {
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
                return Err(SamvadSetuError::Http { status: status_u16, body });
            }

            let json: Value =
                serde_json::from_str(&body).map_err(|e| SamvadSetuError::Parse {
                    message: format!("{provider_name} JSON parse: {e}"),
                    raw_response: Some(body),
                })?;

            debug!("{provider_name} response: {json:.200}");
            parse_gemini_response(&json)
        }
        Err(e) => Err(from_reqwest_error(e)),
    }
}

/// POST to the legacy Gemini endpoint (`?key=…` auth).
pub fn http_post_gemini(
    params: &LLMTextGenerator,
    client: &Client,
    messages: &[ChatMessage],
    tools: Option<&[ToolDefinition]>,
) -> Result<LlmApiResult, SamvadSetuError> {
    let payload = prepare_gemini_legacy_payload(messages, tools, params);
    let url = format!(
        "{}/{}:generateContent?key={}",
        params.svc_base_url, params.model_name, params.api_key
    );
    debug!("Gemini request to {url}");
    post_to_gemini(&url, client, &payload, "Gemini")
}

/// POST to the Google GenAI endpoint (`x-goog-api-key` header auth).
pub fn http_post_google_genai(
    params: &LLMTextGenerator,
    client: &Client,
    messages: &[ChatMessage],
    tools: Option<&[ToolDefinition]>,
    response_format: Option<&ResponseFormat>,
) -> Result<LlmApiResult, SamvadSetuError> {
    let payload =
        prepare_google_genai_payload(messages, tools, response_format, params);
    let url = format!(
        "{}/{}:generateContent",
        params.svc_base_url, params.model_name
    );
    debug!("Google GenAI request to {url}");
    post_to_gemini(&url, client, &payload, "Google GenAI")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::LLMTextGenBuilder;
    use crate::types::{ChatMessage, ToolDefinition};
    use serde_json::json;

    fn gemini_gen() -> LLMTextGenerator {
        LLMTextGenBuilder::build("gemini", "gemini-2.0-flash", 60, None, None).unwrap()
    }
    fn genai_gen() -> LLMTextGenerator {
        LLMTextGenBuilder::build("google_genai", "gemini-2.0-flash-exp", 60, None, None).unwrap()
    }

    #[test]
    fn test_gemini_payload_has_contents() {
        let llm_gen = gemini_gen();
        let msgs = vec![ChatMessage::user("Hello")];
        let payload = prepare_gemini_legacy_payload(&msgs, None, &llm_gen);
        assert!(payload["contents"].is_array());
        assert_eq!(payload["contents"][0]["role"], "user");
    }

    #[test]
    fn test_system_becomes_system_instruction() {
        let llm_gen = genai_gen();
        let msgs = vec![
            ChatMessage::system("You are a pirate."),
            ChatMessage::user("Hello"),
        ];
        let payload = prepare_google_genai_payload(&msgs, None, None, &llm_gen);
        assert!(!payload["systemInstruction"].is_null());
        let contents = payload["contents"].as_array().unwrap();
        // System message must not appear in contents
        assert!(!contents.iter().any(|c| c["role"] == "system"));
    }

    #[test]
    fn test_tools_become_function_declarations() {
        let llm_gen = genai_gen();
        let msgs = vec![ChatMessage::user("Search for weather")];
        let tools = vec![ToolDefinition::new(
            "get_weather",
            "Get the weather",
            json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        )];
        let payload = prepare_google_genai_payload(&msgs, Some(&tools), None, &llm_gen);
        assert!(payload["tools"].is_array());
        let decls = &payload["tools"][0]["functionDeclarations"];
        assert_eq!(decls[0]["name"], "get_weather");
    }

    #[test]
    fn test_json_mode_sets_response_mime_type() {
        let llm_gen = genai_gen();
        let msgs = vec![ChatMessage::user("Return JSON")];
        let payload =
            prepare_google_genai_payload(&msgs, None, Some(&ResponseFormat::JsonObject), &llm_gen);
        assert_eq!(
            payload["generationConfig"]["responseMimeType"],
            json!("application/json")
        );
    }

    #[test]
    fn test_parse_text_candidate() {
        let json = json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "The answer is 42."}]
                },
                "finishReason": "STOP"
            }],
            "modelVersion": "gemini-2.0-flash",
            "usageMetadata": {
                "promptTokenCount": 8,
                "candidatesTokenCount": 5
            }
        });
        let result = parse_gemini_response(&json).unwrap();
        assert_eq!(result.generated_text, "The answer is 42.");
        assert_eq!(result.model_used, "gemini-2.0-flash");
        assert_eq!(result.stop_reason, StopReason::Stop);
    }

    #[test]
    fn test_parse_function_call_candidate() {
        let json = json!({
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "id": "fc_001",
                            "name": "get_weather",
                            "args": {"city": "Rome"}
                        }
                    }]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 10}
        });
        let result = parse_gemini_response(&json).unwrap();
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "get_weather");
        assert_eq!(result.tool_calls[0].arguments["city"], "Rome");
        assert_eq!(result.stop_reason, StopReason::ToolUse);
    }

    #[test]
    fn test_parse_logprobs_when_present() {
        let json = json!({
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "Hi"}]},
                "finishReason": "STOP",
                "logprobsResult": {
                    "chosenCandidates": [
                        {"token": "Hi", "logProbability": -0.3}
                    ],
                    "topCandidates": [
                        {"candidates": [
                            {"token": "Hi",    "logProbability": -0.3},
                            {"token": "Hello", "logProbability": -1.1}
                        ]}
                    ]
                }
            }],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1}
        });
        let result = parse_gemini_response(&json).unwrap();
        assert_eq!(result.logprobs.len(), 1);
        assert_eq!(result.logprobs[0].token, "Hi");
        assert_eq!(result.logprobs[0].top_alternatives.len(), 2);
    }

    #[test]
    fn test_parse_error_body() {
        let json = json!({
            "error": {
                "code": 400,
                "message": "API key not valid.",
                "status": "INVALID_ARGUMENT"
            }
        });
        let err = parse_gemini_response(&json).unwrap_err();
        match err {
            SamvadSetuError::Provider { message, .. } => {
                assert!(message.contains("API key"));
            }
            _ => panic!("Expected Provider error"),
        }
    }

    #[test]
    #[ignore]
    fn test_live_gemini_call() {
        let llm_gen = gemini_gen();
        let msgs = vec![ChatMessage::user("What is 1 + 1? Reply with just the number.")];
        let result = llm_gen.generate_text(&msgs, None, None).unwrap();
        assert!(result.generated_text.contains('2'));
    }
}
