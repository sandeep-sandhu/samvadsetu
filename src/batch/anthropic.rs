// batch/anthropic.rs — Anthropic Message Batches API
//
// Reference: https://platform.claude.com/docs/en/api/creating-message-batches
//
// Workflow:
//   1. POST /v1/messages/batches  with a `requests` array
//   2. Poll GET /v1/messages/batches/{id} until status is "ended"
//   3. Stream results from GET /v1/messages/batches/{id}/results (JSONL)

use crate::error::{parse_anthropic_error_body, SamvadSetuError};
use crate::llm::LLMTextGenerator;
use crate::providers::anthropic::{parse_anthropic_response, prepare_anthropic_payload};
use crate::types::{BatchHandle, BatchRequest, BatchRequestResult, BatchStatus};
use log::debug;
use reqwest::blocking::Client;
use serde_json::{json, Value};

/// Derive the Anthropic base URL from the generator's `svc_base_url`.
fn batch_base_url(svc: &str) -> String {
    // Strip trailing endpoint segment to get the base (/v1/messages → /v1/messages/batches)
    let url = svc.trim_end_matches('/');
    if url.ends_with("/messages") {
        format!("{url}/batches")
    } else {
        format!("{url}/messages/batches")
    }
}

/// Submit a batch of message requests.
pub fn submit_batch(
    params: &LLMTextGenerator,
    client: &Client,
    requests: Vec<BatchRequest>,
) -> Result<BatchHandle, SamvadSetuError> {
    let url = batch_base_url(&params.svc_base_url);

    let api_requests: Vec<Value> = requests
        .iter()
        .map(|req| {
            let mut req_params = prepare_anthropic_payload(
                &req.messages,
                req.tools.as_deref(),
                params,
            );
            // Override system prompt if provided per-request
            if let Some(sp) = &req.system_prompt {
                req_params["system"] = json!(sp);
            }
            json!({
                "custom_id": req.custom_id,
                "params": req_params
            })
        })
        .collect();

    let body = json!({"requests": api_requests});

    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .map_err(|e| SamvadSetuError::Network(e.to_string()))?;

    let status = resp.status();
    let body_text = resp.text().unwrap_or_default();

    if !status.is_success() {
        return Err(parse_anthropic_error_body(status.as_u16(), &body_text));
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
        .get("processing_status")
        .and_then(|v| v.as_str())
        .unwrap_or("in_progress");

    debug!("Created Anthropic batch {batch_id}, status: {status_str}");

    Ok(BatchHandle {
        batch_id,
        provider: params.llm_service.clone(),
        status: anthropic_status_to_batch_status(status_str),
        output_file_id: None,
    })
}

/// Poll the status of an Anthropic batch.
pub fn get_batch_status(
    params: &LLMTextGenerator,
    client: &Client,
    handle: &BatchHandle,
) -> Result<BatchHandle, SamvadSetuError> {
    let base = batch_base_url(&params.svc_base_url);
    let url = format!("{base}/{}", handle.batch_id);

    let resp = client
        .get(&url)
        .send()
        .map_err(|e| SamvadSetuError::Network(e.to_string()))?;

    let status = resp.status();
    let body = resp.text().unwrap_or_default();

    if !status.is_success() {
        return Err(parse_anthropic_error_body(status.as_u16(), &body));
    }

    let json: Value = serde_json::from_str(&body).map_err(|e| SamvadSetuError::Parse {
        message: e.to_string(),
        raw_response: Some(body),
    })?;

    let status_str = json
        .get("processing_status")
        .and_then(|v| v.as_str())
        .unwrap_or("in_progress");

    // When ended, the results URL is available
    let results_url = json
        .get("results_url")
        .and_then(|v| v.as_str())
        .map(str::to_string);

    Ok(BatchHandle {
        batch_id: handle.batch_id.clone(),
        provider: params.llm_service.clone(),
        status: anthropic_status_to_batch_status(status_str),
        output_file_id: results_url,
    })
}

/// Download and parse JSONL results for a completed Anthropic batch.
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

    // Use results_url if available, otherwise construct from batch id
    let url = handle.output_file_id.clone().unwrap_or_else(|| {
        format!("{}/{}/results", batch_base_url(&params.svc_base_url), handle.batch_id)
    });

    let resp = client
        .get(&url)
        .send()
        .map_err(|e| SamvadSetuError::Network(e.to_string()))?;

    let status = resp.status();
    let body = resp.text().unwrap_or_default();

    if !status.is_success() {
        return Err(parse_anthropic_error_body(status.as_u16(), &body));
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

                let result = match entry.get("result") {
                    Some(result_obj) => {
                        match result_obj.get("type").and_then(|v| v.as_str()) {
                            Some("succeeded") => {
                                if let Some(msg) = result_obj.get("message") {
                                    parse_anthropic_response(msg)
                                } else {
                                    Err(SamvadSetuError::Parse {
                                        message: "Missing message in succeeded result".to_string(),
                                        raw_response: Some(line.to_string()),
                                    })
                                }
                            }
                            Some("errored") => {
                                let err_obj = result_obj.get("error").unwrap_or(result_obj);
                                Err(SamvadSetuError::Provider {
                                    error_type: err_obj
                                        .get("type")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("api_error")
                                        .to_string(),
                                    message: err_obj
                                        .get("message")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("Unknown error")
                                        .to_string(),
                                    param: None,
                                    code: None,
                                })
                            }
                            Some("canceled") | Some("expired") => {
                                Err(SamvadSetuError::BatchNotComplete {
                                    batch_id: handle.batch_id.clone(),
                                    status: result_obj
                                        .get("type")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("unknown")
                                        .to_string(),
                                })
                            }
                            _ => Err(SamvadSetuError::Parse {
                                message: "Unknown batch result type".to_string(),
                                raw_response: Some(line.to_string()),
                            }),
                        }
                    }
                    None => Err(SamvadSetuError::Parse {
                        message: "Missing result in batch line".to_string(),
                        raw_response: Some(line.to_string()),
                    }),
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

fn anthropic_status_to_batch_status(s: &str) -> BatchStatus {
    match s {
        "in_progress" => BatchStatus::InProgress,
        "canceling" => BatchStatus::Cancelling,
        "ended" => BatchStatus::Completed,
        _ => BatchStatus::InProgress,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::LLMTextGenBuilder;
    use crate::types::BatchStatus;

    #[test]
    fn test_batch_base_url_from_messages_endpoint() {
        assert_eq!(
            batch_base_url("https://api.anthropic.com/v1/messages"),
            "https://api.anthropic.com/v1/messages/batches"
        );
    }

    #[test]
    fn test_anthropic_status_mapping() {
        assert_eq!(anthropic_status_to_batch_status("in_progress"), BatchStatus::InProgress);
        assert_eq!(anthropic_status_to_batch_status("ended"), BatchStatus::Completed);
        assert_eq!(anthropic_status_to_batch_status("canceling"), BatchStatus::Cancelling);
    }

    #[test]
    fn test_not_complete_error() {
        let llm_gen = LLMTextGenBuilder::build("claude", "claude-haiku-4-5", 60, None, None).unwrap();
        let handle = BatchHandle {
            batch_id: "msgbatch_abc".to_string(),
            provider: "claude".to_string(),
            status: BatchStatus::InProgress,
            output_file_id: None,
        };
        let err = retrieve_batch_results(&llm_gen, &llm_gen.api_client, &handle).unwrap_err();
        assert!(matches!(err, SamvadSetuError::BatchNotComplete { .. }));
    }

    #[test]
    fn test_succeeded_result_parsed() {
        let line = r#"{"custom_id":"req-1","result":{"type":"succeeded","message":{"type":"message","id":"msg_1","role":"assistant","model":"claude-haiku-4-5","stop_reason":"end_turn","content":[{"type":"text","text":"Hello!"}],"usage":{"input_tokens":5,"output_tokens":2}}}}"#;
        let entry: serde_json::Value = serde_json::from_str(line).unwrap();
        let result_obj = entry.get("result").unwrap();
        let msg = result_obj.get("message").unwrap();
        let result = parse_anthropic_response(msg).unwrap();
        assert_eq!(result.generated_text, "Hello!");
    }
}
