# samvadsetu | संवाद-सेतु

[![Crates.io](https://img.shields.io/crates/v/samvadsetu.svg)](https://crates.io/crates/samvadsetu)
[![Docs.rs](https://docs.rs/samvadsetu/badge.svg)](https://docs.rs/samvadsetu)
[![CI](https://github.com/sandeep-sandhu/samvadsetu/workflows/CI/badge.svg)](https://github.com/sandeep-sandhu/samvadsetu/actions)

A Rust-native library for sending chat-completion requests to multiple LLM providers through a single, unified API.

**Sanskrit**: saṃvāda (संवाद) = dialogue · setu (सेतु) = bridge.

---

## Supported Providers

| Provider | `llm_service` string | Auth env var | Batch API |
|---|---|---|---|
| OpenAI / ChatGPT | `"chatgpt"` / `"openai"` | `OPENAI_API_KEY` | ✓ |
| DeepSeek | `"deepseek"` | `DEEPSEEK_API_KEY` | ✓ |
| Alibaba Qwen (DashScope) | `"qwen"` | `DASHSCOPE_API_KEY` | ✓ |
| llama.cpp server | `"llamacpp"` | `LLAMACPP_API_KEY` (opt.) | — |
| Anthropic Claude | `"claude"` / `"anthropic"` | `ANTHROPIC_API_KEY` | ✓ |
| Google Gemini (legacy) | `"gemini"` | `GOOGLE_API_KEY` | — |
| Google GenAI | `"google_genai"` | `GOOGLE_API_KEY` | — |
| Ollama (local) | `"ollama"` | _(none)_ | — |

---

## Installation

```toml
[dependencies]
samvadsetu = "0.2"
```

Optional async support (future):

```toml
samvadsetu = { version = "0.2", features = ["async"] }
```

---

## Quick Start

```rust
use samvadsetu::llm::LLMTextGenBuilder;
use samvadsetu::types::ChatMessage;

fn main() {
    let llm_gen = LLMTextGenBuilder::build("chatgpt", "gpt-4o-mini", 60, None, None)
        .expect("build failed — is OPENAI_API_KEY set?");

    let messages = vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("How is a rainbow formed? Reply in one sentence."),
    ];

    let result = llm_gen.generate_text(&messages, None, None)
        .expect("API call failed");

    println!("{}", result.generated_text);
}
```

Set the API key in your environment before running:

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Core Types

### `ChatMessage` — Building conversations

```rust
use samvadsetu::types::ChatMessage;

// Convenience constructors:
let msg = ChatMessage::system("You are a helpful assistant.");
let msg = ChatMessage::user("What is the capital of France?");
let msg = ChatMessage::assistant("Paris.");

// Returning a tool result to the model:
let msg = ChatMessage::tool_result("call_abc123", "get_weather", "Sunny, 22°C");

// Building an assistant message that made tool calls:
use samvadsetu::types::ToolCall;
use serde_json::json;
let msg = ChatMessage::assistant_with_tool_calls(vec![ToolCall {
    id: "call_abc123".into(),
    name: "get_weather".into(),
    arguments: json!({"city": "Paris"}),
}]);
```

### `LlmApiResult` — What you get back

```rust
use samvadsetu::types::LlmApiResult;

let result: LlmApiResult = /* ... */;

println!("Text:  {}", result.generated_text);
println!("Input tokens:  {}", result.input_tokens_count);
println!("Output tokens: {}", result.output_tokens_count);
println!("Stop reason: {:?}", result.stop_reason);
println!("Model: {}", result.model_used);

// Tool calls the model wants to make:
for tc in &result.tool_calls {
    println!("Call {} → {}({:?})", tc.id, tc.name, tc.arguments);
}

// Chain-of-thought (DeepSeek R1, Ollama `think`, Claude with thinking):
if let Some(cot) = &result.reasoning_content {
    println!("Reasoning: {cot}");
}
```

---

## Feature: Token Log-Probabilities (Hallucination Proxy)

Token log-probabilities are returned by OpenAI-family models (ChatGPT, DeepSeek,
Qwen, llama.cpp) and newer Gemini models.  Claude does **not** support logprobs.

```rust
use samvadsetu::llm::LLMTextGenBuilder;
use samvadsetu::types::ChatMessage;

let llm_gen = LLMTextGenBuilder::build("chatgpt", "gpt-4o-mini", 60, None, None).unwrap();
let messages = vec![ChatMessage::user("Who invented the telephone?")];

let result = llm_gen.generate_text(&messages, None, None).unwrap();

// Aggregate confidence metrics:
if let Some(mean_p) = result.mean_probability() {
    println!("Mean token confidence: {:.1}%", mean_p * 100.0);
}
if let Some(min_p) = result.min_token_probability() {
    println!("Weakest token: {:.1}% (hallucination hotspot)", min_p * 100.0);
}

// Per-token breakdown:
for lp in &result.logprobs {
    println!(
        "  token={:?}  logprob={:.3}  p={:.1}%",
        lp.token, lp.logprob,
        lp.probability() * 100.0
    );
    for alt in &lp.top_alternatives {
        println!("    alt={:?}  logprob={:.3}", alt.token, alt.logprob);
    }
}
```

> **Rule of thumb**: `mean_probability` > 0.9 generally indicates high-confidence
> generation; values < 0.7 warrant extra scrutiny.

---

## Feature: Tool Calling / Function Calling

```rust
use samvadsetu::llm::LLMTextGenBuilder;
use samvadsetu::types::{ChatMessage, ToolDefinition};
use serde_json::json;

let llm_gen = LLMTextGenBuilder::build("chatgpt", "gpt-4o-mini", 60, None, None).unwrap();

let tools = vec![
    ToolDefinition::new(
        "get_weather",
        "Returns current weather for a city.",
        json!({
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }),
    ),
];

let mut messages = vec![ChatMessage::user("What's the weather in Tokyo?")];

// First turn: model requests a tool call
let r1 = llm_gen.generate_text(&messages, Some(&tools), None).unwrap();

for tc in &r1.tool_calls {
    println!("Model wants: {}({:?})", tc.name, tc.arguments);

    // Execute the tool yourself, then add the result to the conversation:
    messages.push(ChatMessage::assistant_with_tool_calls(r1.tool_calls.clone()));
    messages.push(ChatMessage::tool_result(&tc.id, &tc.name, "Sunny, 25°C"));
}

// Second turn: model uses the tool result to answer
let r2 = llm_gen.generate_text(&messages, Some(&tools), None).unwrap();
println!("{}", r2.generated_text);
```

---

## Feature: Structured Output / JSON Mode

```rust
use samvadsetu::types::ResponseFormat;
use serde_json::json;

// JSON Object mode (model must output valid JSON; you define the shape in the prompt)
let result = llm_gen.generate_text(&messages, None, Some(&ResponseFormat::JsonObject)).unwrap();

// JSON Schema mode (model output is constrained to match your schema)
let schema = json!({
    "type": "object",
    "properties": {
        "name":  {"type": "string"},
        "score": {"type": "number"}
    },
    "required": ["name", "score"]
});
let fmt = ResponseFormat::JsonSchema {
    schema,
    name: Some("PersonScore".to_string()),
};
let result = llm_gen.generate_text(&messages, None, Some(&fmt)).unwrap();
let parsed: serde_json::Value = serde_json::from_str(&result.generated_text).unwrap();
```

> JSON Schema mode is fully supported by OpenAI (structured outputs), llama.cpp,
> and Ollama.  DeepSeek and Qwen support JSON object mode.  Gemini supports
> both via `response_mime_type` and `response_schema`.

---

## Feature: Batch Processing

Batch APIs let you submit hundreds or thousands of requests at lower cost and
process the results asynchronously (OpenAI: up to 50 % cost reduction; Anthropic: 50 %).

```rust
use samvadsetu::llm::LLMTextGenBuilder;
use samvadsetu::types::{BatchRequest, BatchStatus, ChatMessage};
use std::time::Duration;

let llm_gen = LLMTextGenBuilder::build("chatgpt", "gpt-4o-mini", 60, None, None).unwrap();

let requests = vec![
    BatchRequest::new("req-1", vec![ChatMessage::user("What is 1 + 1?")]),
    BatchRequest::new("req-2", vec![ChatMessage::user("What is the capital of Spain?")]),
];

// Submit
let mut handle = llm_gen.submit_batch(requests).unwrap();
println!("Batch submitted: {}", handle.batch_id);

// Poll until complete
loop {
    std::thread::sleep(Duration::from_secs(30));
    handle = llm_gen.check_batch_status(&handle).unwrap();
    println!("Status: {:?}", handle.status);
    if handle.status.is_terminal() {
        break;
    }
}

// Retrieve results
if handle.status == BatchStatus::Completed {
    for item in llm_gen.retrieve_batch_results(&handle).unwrap() {
        match item.result {
            Ok(r)  => println!("[{}] {}", item.custom_id, r.generated_text),
            Err(e) => println!("[{}] ERROR: {e}", item.custom_id),
        }
    }
}
```

---

## Configuration via TOML File

```toml
# app_config.toml

[llm_apis.chatgpt]
model_name         = "gpt-4o-mini"
api_url            = "https://api.openai.com/v1/chat/completions"
temperature        = 0.0
max_gen_tokens     = 8192
max_context_len    = 16384
model_api_timeout  = 120
system_prompt      = "You are a helpful assistant."

[llm_apis.claude]
model_name         = "claude-haiku-4-5"
api_url            = "https://api.anthropic.com/v1/messages"
temperature        = 0.0
max_gen_tokens     = 4096
model_api_timeout  = 120

[llm_apis.ollama]
model_name         = "llama3.2"
api_url            = "http://localhost:11434/api/chat"
temperature        = 0.0
max_gen_tokens     = 4096
max_context_len    = 8192
min_gap_btwn_rqsts_secs = 0
```

```rust
use config::{Config, FileFormat};
use samvadsetu::llm::LLMTextGenBuilder;
use samvadsetu::types::ChatMessage;

let app_config = Config::builder()
    .add_source(config::File::new("app_config.toml", FileFormat::Toml))
    .build()
    .expect("Failed to load config");

let llm_gen = LLMTextGenBuilder::build_from_config(&app_config, "chatgpt")
    .expect("Failed to build LLM generator from config");

let messages = vec![ChatMessage::user("Hello!")];
let result = llm_gen.generate_text(&messages, None, None).unwrap();
println!("{}", result.generated_text);
```

---

## Multi-Turn Conversation

The library is stateless — callers own the message history:

```rust
let mut history: Vec<ChatMessage> = vec![
    ChatMessage::system("You are a friendly tutor."),
];

loop {
    let user_input = /* read from stdin */;
    history.push(ChatMessage::user(&user_input));

    let result = llm_gen.generate_text(&history, None, None).unwrap();
    println!("Assistant: {}", result.generated_text);

    history.push(ChatMessage::assistant(&result.generated_text));
}
```

---

## Using Different Providers

```rust
use samvadsetu::llm::LLMTextGenBuilder;
use samvadsetu::types::ChatMessage;

// DeepSeek (OpenAI-compatible)
let llm_gen = LLMTextGenBuilder::build("deepseek", "deepseek-v4-pro", 60, None, None).unwrap();

// Qwen (DashScope, Singapore region by default; override api_url for other regions)
let llm_gen = LLMTextGenBuilder::build("qwen", "qwen-plus", 60, None, None).unwrap();

// Anthropic Claude (no logprobs available)
let llm_gen = LLMTextGenBuilder::build("claude", "claude-haiku-4-5", 60, None, None).unwrap();

// Local llama.cpp server
let mut llm_gen = LLMTextGenBuilder::build("llamacpp", "llama-3.2-3b", 60, None, None).unwrap();
llm_gen.svc_base_url = "http://192.168.1.100:8080/v1/chat/completions".to_string();

// Local Ollama
let mut llm_gen = LLMTextGenBuilder::build("ollama", "gemma3", 60, None, None).unwrap();
llm_gen.svc_base_url = "http://10.13.31.113:11434/api/chat".to_string();
```

---

## Rate Limiting (Multi-threaded)

```rust
use std::sync::{Arc, Mutex};
use samvadsetu::llm::LLMTextGenBuilder;

// Share a mutex across threads; calls will be spaced at least 6 seconds apart.
let rate_limit_mutex = Arc::new(Mutex::new(0isize));

let llm_gen = LLMTextGenBuilder::build(
    "chatgpt",
    "gpt-4o-mini",
    60,
    None,
    Some(Arc::clone(&rate_limit_mutex)),
).unwrap();
```

---

## Error Handling

All errors return `SamvadSetuError`:

```rust
use samvadsetu::error::SamvadSetuError;

match llm_gen.generate_text(&messages, None, None) {
    Ok(result) => println!("{}", result.generated_text),
    Err(SamvadSetuError::RateLimit { retry_after_secs, message }) => {
        if let Some(secs) = retry_after_secs {
            eprintln!("Rate limited. Retry after {secs}s: {message}");
        }
    }
    Err(SamvadSetuError::Auth(msg)) => {
        eprintln!("Authentication failed — check your API key: {msg}");
    }
    Err(SamvadSetuError::Provider { error_type, message, code, .. }) => {
        eprintln!("Provider error [{error_type}] {code:?}: {message}");
    }
    Err(SamvadSetuError::Timeout) => {
        eprintln!("Request timed out — increase network_timeout_secs");
    }
    Err(SamvadSetuError::UnsupportedFeature { provider, feature }) => {
        eprintln!("{provider} does not support {feature}");
    }
    Err(e) => eprintln!("Error: {e}"),
}
```

---

## Provider Capability Matrix

| Feature | ChatGPT | DeepSeek | Qwen | Claude | Gemini | Ollama | llama.cpp |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Chat completions | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Tool calling | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Token logprobs | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ |
| JSON object mode | ✓ | ✓ | ✓ | — | ✓ | ✓ | ✓ |
| JSON schema | ✓ | — | — | — | ✓ | ✓ | ✓ |
| Batch API | ✓ | ✓ | ✓ | ✓ | — | — | — |
| Chain-of-thought | — | ✓ | — | ✓ | — | ✓ | — |

> Claude does not return logprobs via its API — `result.logprobs` will be empty
> and `result.mean_probability()` will return `None`.

---

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

See [CONTRIBUTING.md](CONTRIBUTING.md).
