// batch/openai.rs — OpenAI Batch API
//
// Workflow:
//   1. Upload a JSONL file via POST /v1/files (purpose = "batch")
//   2. Create a batch via POST /v1/batches pointing at the file
//   3. Poll GET /v1/batches/{id} until status is terminal
//   4. Download results via GET /v1/files/{output_file_id}/content (JSONL)

use crate::error::SamvadSetuError;
use crate::llm::LLMTextGenerator;
use crate::providers::openai::{prepare_openai_compat_payload, parse_openai_compat_response};
use crate::types::{BatchHandle, BatchRequest, BatchRequestResult, BatchStatus};
use log::debug;
use reqwest::blocking::{multipart, Client};
use serde_json::{json, Value};

/// Derive the OpenAI API base URL (e.g. "https://api.openai.com/v1") from the
/// generator's `svc_base_url` (e.g. "https://api.openai.com/v1/chat/completions").
fn api_base(svc_base_url: &str) -> String {
    if let Some(idx) = svc_base_url.rfind("/chat/completions") {
        return svc_base_url[..idx].to_string();
    }
    // Fallback: strip everything after the version segment.
    if let Some(idx) = svc_base_url.find("/v1/") {
        return format!("{}/v1", &svc_base_url[..idx]);
    }
    svc_base_url.trim_end_matches('/').to_string()
}

/// Upload a JSONL file to the Files API, returning the file ID.
fn upload_batch_file(
    base: &str,
    client: &Client,
    jsonl: String,
) -> Result<String, SamvadSetuError> {
    let url = format!("{base}/files");
    let form = multipart::Form::new()
        .text("purpose", "batch")
        .part(
            "file",
            multipart::Part::bytes(jsonl.into_bytes())
                .file_name("batch.jsonl")
                .mime_str("application/jsonl")
                .map_err(|e| SamvadSetuError::Network(e.to_string()))?,
        );

    let resp = client
        .post(&url)
        .multipart(form)
        .send()
        .map_err(|e| SamvadSetuError::Network(e.to_string()))?;

    let status = resp.status();
    let body = resp.text().unwrap_or_default();

    if !status.is_success() {
        return Err(SamvadSetuError::Http {
            status: status.as_u16(),
            body,
        });
    }

    let json: Value = serde_json::from_str(&body).map_err(|e| SamvadSetuError::Parse {
        message: e.to_string(),
        raw_response: Some(body),
    })?;

    json.get("id")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .ok_or_else(|| SamvadSetuError::Parse {
            message: "No 'id' in file upload response".to_string(),
            raw_response: None,
        })
}

/// Submit a batch and return a `BatchHandle`.
pub fn submit_batch(
    params: &LLMTextGenerator,
    client: &Client,
    requests: Vec<BatchRequest>,
) -> Result<BatchHandle, SamvadSetuError> {
    let base = api_base(&params.svc_base_url);

    // Build JSONL
    let mut lines = Vec::with_capacity(requests.len());
    for req in &requests {
        let body = prepare_openai_compat_payload(
            &req.messages,
            req.tools.as_deref(),
            None,
            params,
        );
        let line = json!({
            "custom_id": req.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        });
        lines.push(serde_json::to_string(&line).map_err(|e| SamvadSetuError::Parse {
            message: e.to_string(),
            raw_response: None,
        })?);
    }
    let jsonl = lines.join("\n");

    let file_id = upload_batch_file(&base, client, jsonl)?;
    debug!("Uploaded batch file: {file_id}");

    // Create batch
    let url = format!("{base}/batches");
    let body = json!({
        "input_file_id": file_id,
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h"
    });

    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .map_err(|e| SamvadSetuError::Network(e.to_string()))?;

    let status = resp.status();
    let body_text = resp.text().unwrap_or_default();

    if !status.is_success() {
        return Err(SamvadSetuError::Http {
            status: status.as_u16(),
            body: body_text,
        });
    }

    let json: Value =
        serde_json::from_str(&body_text).map_err(|e| SamvadSetuError::Parse {
            message: e.to_string(),
            raw_response: Some(body_text),
        })?;

    let batch_id = json
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let status_str = json
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("validating");

    debug!("Created batch {batch_id}, status: {status_str}");

    Ok(BatchHandle {
        batch_id,
        provider: params.llm_service.clone(),
        status: BatchStatus::from_str(status_str),
        output_file_id: None,
    })
}

/// Poll the status of a batch.
pub fn get_batch_status(
    params: &LLMTextGenerator,
    client: &Client,
    handle: &BatchHandle,
) -> Result<BatchHandle, SamvadSetuError> {
    let base = api_base(&params.svc_base_url);
    let url = format!("{base}/batches/{}", handle.batch_id);

    let resp = client
        .get(&url)
        .send()
        .map_err(|e| SamvadSetuError::Network(e.to_string()))?;

    let status = resp.status();
    let body = resp.text().unwrap_or_default();

    if !status.is_success() {
        return Err(SamvadSetuError::Http {
            status: status.as_u16(),
            body,
        });
    }

    let json: Value = serde_json::from_str(&body).map_err(|e| SamvadSetuError::Parse {
        message: e.to_string(),
        raw_response: Some(body),
    })?;

    let batch_status = BatchStatus::from_str(
        json.get("status").and_then(|v| v.as_str()).unwrap_or("in_progress"),
    );
    let output_file_id = json
        .get("output_file_id")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    Ok(BatchHandle {
        batch_id: handle.batch_id.clone(),
        provider: params.llm_service.clone(),
        status: batch_status,
        output_file_id,
    })
}

/// Download and parse the results of a completed batch.
pub fn retrieve_batch_results(
    params: &LLMTextGenerator,
    client: &Client,
    handle: &BatchHandle,
) -> Result<Vec<BatchRequestResult>, SamvadSetuError> {
    if !handle.status.is_terminal() {
        return Err(SamvadSetuError::BatchNotComplete {
            batch_id: handle.batch_id.clone(),
            status: format!("{:?}", handle.status),
        });
    }

    let file_id = handle.output_file_id.as_deref().ok_or_else(|| {
        SamvadSetuError::Parse {
            message: "Batch has no output_file_id".to_string(),
            raw_response: None,
        }
    })?;

    let base = api_base(&params.svc_base_url);
    let url = format!("{base}/files/{file_id}/content");

    let resp = client
        .get(&url)
        .send()
        .map_err(|e| SamvadSetuError::Network(e.to_string()))?;

    let status = resp.status();
    let body = resp.text().unwrap_or_default();

    if !status.is_success() {
        return Err(SamvadSetuError::Http {
            status: status.as_u16(),
            body,
        });
    }

    let mut results = Vec::new();
    for line in body.lines() {
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(line) {
            Ok(entry) => {
                let custom_id = entry
                    .get("custom_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let result = if let Some(response) = entry.get("response") {
                    let status_code = response
                        .get("status_code")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(200) as u16;
                    if status_code == 200 {
                        if let Some(body_val) = response.get("body") {
                            parse_openai_compat_response(body_val)
                        } else {
                            Err(SamvadSetuError::Parse {
                                message: "Missing body in batch result".to_string(),
                                raw_response: Some(line.to_string()),
                            })
                        }
                    } else {
                        let err_body = response
                            .get("body")
                            .map(|v| v.to_string())
                            .unwrap_or_default();
                        Err(crate::error::parse_openai_error_body(status_code, &err_body))
                    }
                } else if let Some(err) = entry.get("error") {
                    Err(SamvadSetuError::Provider {
                        error_type: err
                            .get("code")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string(),
                        message: err
                            .get("message")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Unknown batch error")
                            .to_string(),
                        param: None,
                        code: None,
                    })
                } else {
                    Err(SamvadSetuError::Parse {
                        message: "Unrecognised batch result line".to_string(),
                        raw_response: Some(line.to_string()),
                    })
                };

                results.push(BatchRequestResult { custom_id, result });
            }
            Err(e) => {
                results.push(BatchRequestResult {
                    custom_id: String::new(),
                    result: Err(SamvadSetuError::Parse {
                        message: format!("JSONL parse error: {e}"),
                        raw_response: Some(line.to_string()),
                    }),
                });
            }
        }
    }

    Ok(results)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::LLMTextGenBuilder;
    use crate::types::BatchStatus;

    #[test]
    fn test_api_base_extraction() {
        assert_eq!(
            api_base("https://api.openai.com/v1/chat/completions"),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            api_base("https://api.deepseek.com/chat/completions"),
            "https://api.deepseek.com"
        );
    }

    #[test]
    fn test_batch_not_complete_error() {
        let llm_gen = LLMTextGenBuilder::build("chatgpt", "gpt-4o-mini", 60, None, None).unwrap();
        let handle = BatchHandle {
            batch_id: "batch_123".to_string(),
            provider: "chatgpt".to_string(),
            status: BatchStatus::InProgress,
            output_file_id: None,
        };
        let err = retrieve_batch_results(&llm_gen, &llm_gen.api_client, &handle).unwrap_err();
        match err {
            SamvadSetuError::BatchNotComplete { batch_id, .. } => {
                assert_eq!(batch_id, "batch_123");
            }
            _ => panic!("Expected BatchNotComplete error"),
        }
    }

    #[test]
    fn test_batch_result_parsing() {
        let line = r#"{"id":"br_1","custom_id":"req-1","response":{"status_code":200,"body":{"model":"gpt-4o-mini","choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"Hello"},"logprobs":null}],"usage":{"prompt_tokens":5,"completion_tokens":3}}},"error":null}"#;

        let entry: serde_json::Value = serde_json::from_str(line).unwrap();
        let custom_id = entry["custom_id"].as_str().unwrap();
        assert_eq!(custom_id, "req-1");

        if let Some(resp) = entry.get("response") {
            let body = resp.get("body").unwrap();
            let result = parse_openai_compat_response(body).unwrap();
            assert_eq!(result.generated_text, "Hello");
        }
    }
}
