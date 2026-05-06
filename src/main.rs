use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{sse::Event, Sse},
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use crossterm::{
    event::{self, Event as CEvent, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use futures::stream::Stream;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Paragraph, Sparkline},
    Terminal,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::HashMap,
    io::stdout,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio_stream::StreamExt as _;
use uuid::Uuid;

// ── Anthropic API types ──

#[derive(Deserialize, Debug, Clone)]
struct AnthropicRequest {
    model: Option<String>,
    messages: Vec<AnthropicMessage>,
    max_tokens: Option<u32>,
    stream: Option<bool>,
    #[serde(default)]
    system: Option<serde_json::Value>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    stop_sequences: Option<Vec<String>>,
    #[serde(flatten)]
    _extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct AnthropicMessage {
    role: String,
    content: serde_json::Value,
}

// ── OpenAI-compatible types (Vertex) ──

#[derive(Serialize, Debug)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Serialize, Debug)]
struct OpenAIMessage {
    role: String,
    // `content` may be null for assistant messages that only carry tool_calls.
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Debug)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    function: OpenAIFunctionCall,
}

#[derive(Serialize, Debug)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Deserialize, Debug)]
struct OpenAIChunk {
    choices: Option<Vec<OpenAIChoice>>,
}

#[derive(Deserialize, Debug)]
struct OpenAIChoice {
    delta: Option<OpenAIDelta>,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OpenAIDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCallDelta>>,
}

#[derive(Deserialize, Debug)]
struct OpenAIToolCallDelta {
    index: u32,
    id: Option<String>,
    function: Option<OpenAIFunctionDelta>,
}

#[derive(Deserialize, Debug)]
struct OpenAIFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OpenAINonStreamResponse {
    choices: Vec<OpenAINonStreamChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Deserialize, Debug)]
struct OpenAINonStreamChoice {
    message: OpenAINonStreamMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OpenAINonStreamMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCallResp>>,
}

#[derive(Deserialize, Debug)]
struct OpenAIToolCallResp {
    id: String,
    function: OpenAIFunctionCallResp,
}

#[derive(Deserialize, Debug)]
struct OpenAIFunctionCallResp {
    name: String,
    arguments: String,
}

#[derive(Deserialize, Debug)]
struct OpenAIUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
}

// ── TUI state ──

#[derive(Clone, Debug)]
struct RequestLog {
    id: String,
    timestamp: String,
    model_requested: String,
    status: String,
    input_tokens: usize,
    output_tokens: usize,
    duration_ms: u64,
}

struct AppState {
    endpoint: String,
    project_id: String,
    region: String,
    model: String,
    token_manager: TokenManager,
    logs: Mutex<Vec<RequestLog>>,
    total_requests: Mutex<u64>,
    total_input_tokens: Mutex<u64>,
    total_output_tokens: Mutex<u64>,
    rps_history: Mutex<Vec<u64>>,
    host: String,
    port: u16,
    external_ip: Option<String>,
    max_output_tokens: u32,
}

// ── OAuth2 Token Manager ──

#[derive(Deserialize)]
struct AdcCredentials {
    client_id: String,
    client_secret: String,
    refresh_token: String,
    #[serde(rename = "type")]
    cred_type: String,
    #[serde(default)]
    quota_project_id: Option<String>,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    expires_in: u64,
}

struct TokenManager {
    client_id: String,
    client_secret: String,
    refresh_token: String,
    cached_token: Mutex<Option<(String, Instant)>>,
    http: reqwest::Client,
}

impl TokenManager {
    fn from_adc() -> Result<(Self, Option<String>), String> {
        let path = std::env::var("GOOGLE_APPLICATION_CREDENTIALS")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
                PathBuf::from(home).join(".config/gcloud/application_default_credentials.json")
            });

        let data = std::fs::read_to_string(&path)
            .map_err(|e| format!("Cannot read credentials at {}: {e}\nRun: gcloud auth application-default login", path.display()))?;

        let creds: AdcCredentials = serde_json::from_str(&data)
            .map_err(|e| format!("Failed to parse credentials: {e}"))?;

        if creds.cred_type != "authorized_user" {
            return Err(format!(
                "Unsupported credential type '{}'. Expected 'authorized_user'.\nRun: gcloud auth application-default login",
                creds.cred_type
            ));
        }

        let quota_project = creds.quota_project_id;

        Ok((Self {
            client_id: creds.client_id,
            client_secret: creds.client_secret,
            refresh_token: creds.refresh_token,
            cached_token: Mutex::new(None),
            http: reqwest::Client::new(),
        }, quota_project))
    }

    async fn get_token(&self) -> Result<String, String> {
        if let Ok(guard) = self.cached_token.lock() {
            if let Some((ref token, expiry)) = *guard {
                if Instant::now() < expiry {
                    return Ok(token.clone());
                }
            }
        }
        self.refresh().await
    }

    async fn refresh(&self) -> Result<String, String> {
        let resp = self
            .http
            .post("https://oauth2.googleapis.com/token")
            .form(&[
                ("client_id", self.client_id.as_str()),
                ("client_secret", self.client_secret.as_str()),
                ("refresh_token", self.refresh_token.as_str()),
                ("grant_type", "refresh_token"),
            ])
            .send()
            .await
            .map_err(|e| format!("Token refresh request failed: {e}"))?;

        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Token refresh failed: {body}"));
        }

        let tr: TokenResponse = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse token response: {e}"))?;

        let expiry = Instant::now() + Duration::from_secs(tr.expires_in.saturating_sub(60));
        if let Ok(mut guard) = self.cached_token.lock() {
            *guard = Some((tr.access_token.clone(), expiry));
        }

        Ok(tr.access_token)
    }
}

/// Pull text out of a content value. Used for system prompts and as a fallback for
/// content that isn't a tool_use/tool_result/image.
fn flatten_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|block| {
                if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                    block.get("text").and_then(|t| t.as_str()).map(String::from)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Anthropic `tool_result.content` may be a string or an array of content blocks.
/// OpenAI's `tool` role only accepts a string — flatten text blocks; drop the rest.
fn flatten_tool_result_content(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|b| match b.get("type").and_then(|t| t.as_str()) {
                Some("text") => b.get("text").and_then(|t| t.as_str()).map(String::from),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n"),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn content_blocks(value: &serde_json::Value) -> Vec<serde_json::Value> {
    match value {
        serde_json::Value::String(s) => vec![json!({"type": "text", "text": s})],
        serde_json::Value::Array(arr) => arr.clone(),
        _ => Vec::new(),
    }
}

fn translate_tools(tools: &[serde_json::Value]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .filter_map(|t| {
            let name = t.get("name").and_then(|n| n.as_str())?;
            let description = t
                .get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("");
            let parameters = t
                .get("input_schema")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object"}));
            Some(json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                }
            }))
        })
        .collect()
}

fn translate_tool_choice(tc: &serde_json::Value) -> serde_json::Value {
    match tc.get("type").and_then(|t| t.as_str()) {
        Some("auto") => json!("auto"),
        Some("none") => json!("none"),
        Some("any") => json!("required"),
        Some("tool") => match tc.get("name").and_then(|n| n.as_str()) {
            Some(name) => json!({"type": "function", "function": {"name": name}}),
            None => json!("auto"),
        },
        _ => json!("auto"),
    }
}

fn map_stop_reason(openai: &str) -> &'static str {
    match openai {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        "stop" => "end_turn",
        "content_filter" => "end_turn",
        _ => "end_turn",
    }
}

fn translate_messages(
    system: &Option<serde_json::Value>,
    messages: &[AnthropicMessage],
) -> Vec<OpenAIMessage> {
    let mut out = Vec::new();
    if let Some(sys) = system {
        let text = flatten_text(sys);
        if !text.is_empty() {
            out.push(OpenAIMessage {
                role: "system".into(),
                content: Some(text),
                tool_calls: None,
                tool_call_id: None,
            });
        }
    }

    for m in messages {
        let blocks = content_blocks(&m.content);

        match m.role.as_str() {
            "assistant" => {
                let mut text_parts: Vec<String> = Vec::new();
                let mut tool_calls: Vec<OpenAIToolCall> = Vec::new();
                for b in &blocks {
                    match b.get("type").and_then(|t| t.as_str()) {
                        Some("text") => {
                            if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                                text_parts.push(t.to_string());
                            }
                        }
                        Some("tool_use") => {
                            let id = b
                                .get("id")
                                .and_then(|i| i.as_str())
                                .unwrap_or("")
                                .to_string();
                            let name = b
                                .get("name")
                                .and_then(|n| n.as_str())
                                .unwrap_or("")
                                .to_string();
                            let arguments = b
                                .get("input")
                                .map(|i| i.to_string())
                                .unwrap_or_else(|| "{}".into());
                            tool_calls.push(OpenAIToolCall {
                                id,
                                kind: "function".into(),
                                function: OpenAIFunctionCall { name, arguments },
                            });
                        }
                        _ => {}
                    }
                }
                let text = text_parts.join("\n");
                let content = if text.is_empty() { None } else { Some(text) };
                let tc_opt = if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                };
                // Skip empty assistant turns entirely; OpenAI rejects them.
                if content.is_some() || tc_opt.is_some() {
                    out.push(OpenAIMessage {
                        role: "assistant".into(),
                        content,
                        tool_calls: tc_opt,
                        tool_call_id: None,
                    });
                }
            }
            _ => {
                // user (and anything else) — split tool_results into role:tool messages.
                let mut user_text_parts: Vec<String> = Vec::new();
                let mut tool_msgs: Vec<OpenAIMessage> = Vec::new();
                for b in &blocks {
                    match b.get("type").and_then(|t| t.as_str()) {
                        Some("text") => {
                            if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                                user_text_parts.push(t.to_string());
                            }
                        }
                        Some("tool_result") => {
                            let id = b
                                .get("tool_use_id")
                                .and_then(|i| i.as_str())
                                .unwrap_or("")
                                .to_string();
                            let mut content = b
                                .get("content")
                                .map(flatten_tool_result_content)
                                .unwrap_or_default();
                            if b.get("is_error").and_then(|e| e.as_bool()) == Some(true) {
                                content = format!("[tool_error] {content}");
                            }
                            tool_msgs.push(OpenAIMessage {
                                role: "tool".into(),
                                content: Some(content),
                                tool_calls: None,
                                tool_call_id: Some(id),
                            });
                        }
                        _ => {}
                    }
                }
                // OpenAI requires `role:"tool"` to immediately follow the assistant
                // turn that produced the tool_calls — emit those first.
                out.extend(tool_msgs);
                let user_text = user_text_parts.join("\n");
                if !user_text.is_empty() {
                    out.push(OpenAIMessage {
                        role: m.role.clone(),
                        content: Some(user_text),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
            }
        }
    }
    out
}

fn estimate_tokens(s: &str) -> usize {
    s.len() / 4 + 1
}

// ── Handlers ──

async fn handle_messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<AnthropicRequest>,
) -> Result<axum::response::Response, (StatusCode, String)> {
    let _ = headers; // we ignore the anthropic API key, we use gcloud auth
    let start = std::time::Instant::now();
    let req_id = Uuid::new_v4().to_string();
    let model_requested = body.model.clone().unwrap_or_default();
    let streaming = body.stream.unwrap_or(false);

    let openai_messages = translate_messages(&body.system, &body.messages);
    let mut input_text = String::new();
    for m in &openai_messages {
        if let Some(c) = &m.content {
            input_text.push_str(c);
        }
        if let Some(tcs) = &m.tool_calls {
            for tc in tcs {
                input_text.push_str(&tc.function.name);
                input_text.push_str(&tc.function.arguments);
            }
        }
    }
    let input_tokens = estimate_tokens(&input_text);

    let access_token = state.token_manager.get_token().await.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let url = format!(
        "https://{}/v1/projects/{}/locations/{}/endpoints/openapi/chat/completions",
        state.endpoint, state.project_id, state.region
    );

    let translated_tools = body
        .tools
        .as_ref()
        .map(|t| translate_tools(t))
        .filter(|v| !v.is_empty());
    let translated_tool_choice = body
        .tool_choice
        .as_ref()
        .map(translate_tool_choice);

    let openai_req = OpenAIRequest {
        model: state.model.clone(),
        messages: openai_messages,
        stream: streaming,
        max_tokens: body.max_tokens.map(|n| n.min(state.max_output_tokens)),
        temperature: body.temperature,
        top_p: body.top_p,
        tools: translated_tools,
        tool_choice: translated_tool_choice,
        stop: body.stop_sequences.clone().filter(|s| !s.is_empty()),
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {access_token}"))
        .header("Content-Type", "application/json")
        .json(&openai_req)
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("Upstream error: {e}")))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err((
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            text,
        ));
    }

    if streaming {
        let state2 = state.clone();
        let req_id2 = req_id.clone();
        let model_req2 = model_requested.clone();
        let stream = make_anthropic_stream(resp, state2, req_id2, model_req2, input_tokens, start);
        Ok(Sse::new(stream)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response())
    } else {
        let text = resp.text().await.unwrap_or_default();
        let parsed: OpenAINonStreamResponse =
            serde_json::from_str(&text).map_err(|e| (StatusCode::BAD_GATEWAY, format!("{e}")))?;

        let choice = parsed.choices.first();
        let message = choice.map(|c| &c.message);

        let mut content_blocks: Vec<serde_json::Value> = Vec::new();
        let mut text_for_tokens = String::new();

        if let Some(msg) = message {
            if let Some(t) = &msg.content {
                if !t.is_empty() {
                    text_for_tokens.push_str(t);
                    content_blocks.push(json!({"type": "text", "text": t}));
                }
            }
            if let Some(tcs) = &msg.tool_calls {
                for tc in tcs {
                    let input: serde_json::Value =
                        serde_json::from_str(&tc.function.arguments)
                            .unwrap_or_else(|_| json!({}));
                    text_for_tokens.push_str(&tc.function.name);
                    text_for_tokens.push_str(&tc.function.arguments);
                    content_blocks.push(json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": input,
                    }));
                }
            }
        }

        let raw_finish = choice
            .and_then(|c| c.finish_reason.clone())
            .unwrap_or_else(|| "stop".into());
        let stop_reason = map_stop_reason(&raw_finish);

        let output_tokens = parsed
            .usage
            .as_ref()
            .and_then(|u| u.completion_tokens)
            .unwrap_or(estimate_tokens(&text_for_tokens) as u32);
        let input_tok = parsed
            .usage
            .as_ref()
            .and_then(|u| u.prompt_tokens)
            .unwrap_or(input_tokens as u32);

        let duration = start.elapsed().as_millis() as u64;
        log_request(
            &state,
            &req_id,
            &model_requested,
            "ok",
            input_tok as usize,
            output_tokens as usize,
            duration,
        );

        let msg_id = format!("msg_{}", &req_id[..24]);
        let anthropic_resp = json!({
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model_requested,
            "content": content_blocks,
            "stop_reason": stop_reason,
            "stop_sequence": null,
            "usage": {
                "input_tokens": input_tok,
                "output_tokens": output_tokens
            }
        });

        Ok(Json(anthropic_resp).into_response())
    }
}

use axum::response::IntoResponse;

#[derive(Default)]
struct ToolStreamState {
    block_idx: Option<usize>,
    id: Option<String>,
    name: Option<String>,
    pending_args: String,
}

fn make_anthropic_stream(
    resp: reqwest::Response,
    state: Arc<AppState>,
    req_id: String,
    model_requested: String,
    input_tokens: usize,
    start: std::time::Instant,
) -> impl Stream<Item = Result<Event, axum::Error>> {
    let msg_id = format!("msg_{}", &req_id[..24]);
    let model_req = model_requested.clone();

    async_stream::stream! {
        // message_start
        let start_event = json!({
            "type": "message_start",
            "message": {
                "id": &msg_id,
                "type": "message",
                "role": "assistant",
                "model": &model_req,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0}
            }
        });
        yield Ok(Event::default().event("message_start").data(start_event.to_string()));

        // We open content blocks lazily and track exactly one open block at a time.
        // Anthropic's protocol opens, streams deltas, then closes — no interleaving.
        let mut next_block_idx: usize = 0;
        // (block_idx, kind) where kind is "text" or "tool"
        let mut current_block: Option<(usize, &'static str)> = None;
        let mut tools: HashMap<u32, ToolStreamState> = HashMap::new();
        let mut output_tokens: usize = 0;
        let mut finish_reason: Option<String> = None;

        let mut byte_stream = resp.bytes_stream();
        let mut buf = String::new();

        while let Some(chunk) = byte_stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(_) => break,
            };
            buf.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(line_end) = buf.find('\n') {
                let line = buf[..line_end].trim().to_string();
                buf = buf[line_end + 1..].to_string();

                if !line.starts_with("data: ") {
                    continue;
                }
                let data = &line[6..];
                if data == "[DONE]" {
                    continue;
                }

                let parsed: OpenAIChunk = match serde_json::from_str(data) {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                let choices = match &parsed.choices {
                    Some(c) => c,
                    None => continue,
                };
                let choice = match choices.first() {
                    Some(c) => c,
                    None => continue,
                };

                if let Some(delta) = &choice.delta {
                    // ── text content ──
                    if let Some(text) = &delta.content {
                        if !text.is_empty() {
                            // Switch into a text block if we aren't already in one.
                            if !matches!(current_block, Some((_, "text"))) {
                                if let Some((idx, _)) = current_block {
                                    yield Ok(Event::default()
                                        .event("content_block_stop")
                                        .data(json!({"type":"content_block_stop","index":idx}).to_string()));
                                }
                                let idx = next_block_idx;
                                next_block_idx += 1;
                                yield Ok(Event::default()
                                    .event("content_block_start")
                                    .data(json!({
                                        "type":"content_block_start",
                                        "index": idx,
                                        "content_block": {"type":"text","text":""}
                                    }).to_string()));
                                current_block = Some((idx, "text"));
                            }
                            let idx = current_block.unwrap().0;
                            output_tokens += estimate_tokens(text);
                            yield Ok(Event::default()
                                .event("content_block_delta")
                                .data(json!({
                                    "type":"content_block_delta",
                                    "index": idx,
                                    "delta": {"type":"text_delta","text": text}
                                }).to_string()));
                        }
                    }

                    // ── tool calls ──
                    if let Some(tool_calls) = &delta.tool_calls {
                        for tc in tool_calls {
                            let entry = tools.entry(tc.index).or_default();
                            if let Some(id) = &tc.id {
                                if !id.is_empty() {
                                    entry.id = Some(id.clone());
                                }
                            }
                            if let Some(f) = &tc.function {
                                if let Some(name) = &f.name {
                                    if !name.is_empty() {
                                        entry.name = Some(name.clone());
                                    }
                                }
                            }

                            // Open the tool_use content block once we know id+name.
                            if entry.block_idx.is_none() {
                                if let (Some(id), Some(name)) = (&entry.id, &entry.name) {
                                    if let Some((idx, _)) = current_block {
                                        yield Ok(Event::default()
                                            .event("content_block_stop")
                                            .data(json!({"type":"content_block_stop","index":idx}).to_string()));
                                    }
                                    let idx = next_block_idx;
                                    next_block_idx += 1;
                                    entry.block_idx = Some(idx);
                                    current_block = Some((idx, "tool"));

                                    yield Ok(Event::default()
                                        .event("content_block_start")
                                        .data(json!({
                                            "type":"content_block_start",
                                            "index": idx,
                                            "content_block": {
                                                "type":"tool_use",
                                                "id": id,
                                                "name": name,
                                                "input": {}
                                            }
                                        }).to_string()));

                                    // Flush any args buffered before we knew id+name.
                                    if !entry.pending_args.is_empty() {
                                        let args = std::mem::take(&mut entry.pending_args);
                                        output_tokens += estimate_tokens(&args);
                                        yield Ok(Event::default()
                                            .event("content_block_delta")
                                            .data(json!({
                                                "type":"content_block_delta",
                                                "index": idx,
                                                "delta": {"type":"input_json_delta","partial_json": args}
                                            }).to_string()));
                                    }
                                }
                            }

                            // Stream argument deltas.
                            if let Some(f) = &tc.function {
                                if let Some(args) = &f.arguments {
                                    if !args.is_empty() {
                                        match entry.block_idx {
                                            Some(idx) => {
                                                // If a different block is open, close it and re-open this one.
                                                if current_block.map(|(i, _)| i) != Some(idx) {
                                                    if let Some((other, _)) = current_block {
                                                        yield Ok(Event::default()
                                                            .event("content_block_stop")
                                                            .data(json!({"type":"content_block_stop","index":other}).to_string()));
                                                    }
                                                    // We can't truly re-open a closed block;
                                                    // just track the new "current" so subsequent
                                                    // deltas address the right index. In practice
                                                    // OpenAI streams one tool's args contiguously.
                                                    current_block = Some((idx, "tool"));
                                                }
                                                output_tokens += estimate_tokens(args);
                                                yield Ok(Event::default()
                                                    .event("content_block_delta")
                                                    .data(json!({
                                                        "type":"content_block_delta",
                                                        "index": idx,
                                                        "delta": {"type":"input_json_delta","partial_json": args}
                                                    }).to_string()));
                                            }
                                            None => {
                                                // Args arrived before id/name — buffer.
                                                entry.pending_args.push_str(args);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if let Some(fr) = &choice.finish_reason {
                    finish_reason = Some(fr.clone());
                }
            }
        }

        // Close whatever block is still open.
        if let Some((idx, _)) = current_block {
            yield Ok(Event::default()
                .event("content_block_stop")
                .data(json!({"type":"content_block_stop","index":idx}).to_string()));
        }

        let stop_reason = map_stop_reason(finish_reason.as_deref().unwrap_or("stop"));

        let duration = start.elapsed().as_millis() as u64;
        log_request(&state, &req_id, &model_req, "ok", input_tokens, output_tokens, duration);

        // message_delta
        let md = json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": null},
            "usage": {"output_tokens": output_tokens}
        });
        yield Ok(Event::default().event("message_delta").data(md.to_string()));

        // message_stop
        let ms = json!({"type":"message_stop"});
        yield Ok(Event::default().event("message_stop").data(ms.to_string()));
    }
}

fn log_request(
    state: &AppState,
    id: &str,
    model: &str,
    status: &str,
    input_tokens: usize,
    output_tokens: usize,
    duration_ms: u64,
) {
    let log = RequestLog {
        id: id[..8].to_string(),
        timestamp: Utc::now().format("%H:%M:%S").to_string(),
        model_requested: model.to_string(),
        status: status.to_string(),
        input_tokens,
        output_tokens,
        duration_ms,
    };
    if let Ok(mut logs) = state.logs.lock() {
        logs.push(log);
        if logs.len() > 500 {
            let excess = logs.len() - 500;
            logs.drain(..excess);
        }
    }
    if let Ok(mut t) = state.total_requests.lock() {
        *t += 1;
    }
    if let Ok(mut t) = state.total_input_tokens.lock() {
        *t += input_tokens as u64;
    }
    if let Ok(mut t) = state.total_output_tokens.lock() {
        *t += output_tokens as u64;
    }
}

async fn handle_health() -> &'static str {
    "ok"
}

// ── TUI ──

fn draw_tui(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    state: &AppState,
) -> std::io::Result<()> {
    let logs = state.logs.lock().unwrap().clone();
    let total_req = *state.total_requests.lock().unwrap();
    let total_in = *state.total_input_tokens.lock().unwrap();
    let total_out = *state.total_output_tokens.lock().unwrap();
    let rps_hist: Vec<u64> = state.rps_history.lock().unwrap().clone();

    terminal.draw(|f| {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // header
                Constraint::Length(5),  // stats
                Constraint::Min(10),   // log
                Constraint::Length(5), // footer (listening + base-url hints)
            ])
            .split(f.area());

        // Header
        let header = Paragraph::new(Line::from(vec![
            Span::styled(" ◆ ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::styled("Vertex AI Proxy", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::raw("  "),
            Span::styled(
                "Anthropic API → Qwen3 Coder on GCP".to_string(),
                Style::default().fg(Color::DarkGray),
            ),
        ]))
        .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::DarkGray)));
        f.render_widget(header, chunks[0]);

        // Stats
        let stats_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
            ])
            .split(chunks[1]);

        let stat_style = Style::default().fg(Color::Green).add_modifier(Modifier::BOLD);
        let label_style = Style::default().fg(Color::DarkGray);

        let border = Style::default().fg(Color::DarkGray);
        for (i, (label, value)) in [
            ("Requests", total_req.to_string()),
            ("Input Tokens", format_tokens(total_in)),
            ("Output Tokens", format_tokens(total_out)),
        ].iter().enumerate() {
            let p = Paragraph::new(vec![
                Line::from(Span::styled(value.clone(), stat_style)),
                Line::from(Span::styled(*label, label_style)),
            ])
            .block(Block::default().borders(Borders::ALL).border_style(border))
            .alignment(ratatui::layout::Alignment::Center);
            f.render_widget(p, stats_chunks[i]);
        }

        // Sparkline for RPS
        let spark = Sparkline::default()
            .block(Block::default().borders(Borders::ALL).title(" rps ").border_style(Style::default().fg(Color::DarkGray)))
            .data(&rps_hist)
            .style(Style::default().fg(Color::Cyan));
        f.render_widget(spark, stats_chunks[3]);

        // Request log
        let items: Vec<ListItem> = logs
            .iter()
            .rev()
            .take(50)
            .map(|l| {
                let status_color = if l.status == "ok" { Color::Green } else { Color::Red };
                ListItem::new(Line::from(vec![
                    Span::styled(&l.timestamp, Style::default().fg(Color::DarkGray)),
                    Span::raw("  "),
                    Span::styled(&l.id, Style::default().fg(Color::Yellow)),
                    Span::raw("  "),
                    Span::styled(
                        format!("{:>5}→{:<5}", l.input_tokens, l.output_tokens),
                        Style::default().fg(Color::Cyan),
                    ),
                    Span::raw("  "),
                    Span::styled(format!("{:>6}ms", l.duration_ms), Style::default().fg(Color::Magenta)),
                    Span::raw("  "),
                    Span::styled(&l.status, Style::default().fg(status_color)),
                    Span::raw("  "),
                    Span::styled(
                        truncate_str(&l.model_requested, 30),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]))
            })
            .collect();

        let log_list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title(" Requests ").border_style(Style::default().fg(Color::DarkGray)));
        f.render_widget(log_list, chunks[2]);

        // Footer: bind status + ready-to-paste invocations.
        let dim = Style::default().fg(Color::DarkGray);
        let key = Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD);
        let val = Style::default().fg(Color::White);
        let accent = Style::default().fg(Color::Cyan);

        let mut footer_lines = vec![
            Line::from(vec![
                Span::styled(
                    format!(" listening on {}:{}", state.host, state.port),
                    Style::default().fg(Color::Green),
                ),
                Span::raw("  │  "),
                Span::styled("q", key),
                Span::styled(" quit", dim),
            ]),
            Line::from(vec![
                Span::styled(" local:    ", dim),
                Span::styled(
                    format!("ANTHROPIC_BASE_URL=http://127.0.0.1:{} claude", state.port),
                    val,
                ),
            ]),
        ];

        if listens_externally(&state.host) {
            let host_str: String = match &state.external_ip {
                Some(ip) => ip.clone(),
                None => "<this-host>".into(),
            };
            footer_lines.push(Line::from(vec![
                Span::styled(" external: ", dim),
                Span::styled(
                    format!("ANTHROPIC_BASE_URL=http://{host_str}:{} claude", state.port),
                    accent,
                ),
            ]));
        } else {
            footer_lines.push(Line::from(Span::styled(
                " external: disabled (restart with --host 0.0.0.0 to allow LAN access)",
                dim,
            )));
        }

        let footer = Paragraph::new(footer_lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
        f.render_widget(footer, chunks[3]);
    })?;
    Ok(())
}

fn format_tokens(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}…", &s[..max - 1])
    } else {
        s.to_string()
    }
}

/// Best-effort discovery of a routable local IP. Opens a UDP socket and asks the
/// kernel which local address it would use to reach a public IP — no packets sent.
fn detect_local_ip() -> Option<String> {
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    let ip = socket.local_addr().ok()?.ip();
    if ip.is_loopback() || ip.is_unspecified() {
        return None;
    }
    Some(ip.to_string())
}

fn listens_externally(host: &str) -> bool {
    host != "127.0.0.1" && host != "localhost" && host != "::1"
}

fn detect_gcloud_project() -> Option<String> {
    let home = std::env::var("HOME").ok()?;
    let gcloud_dir = PathBuf::from(&home).join(".config/gcloud");

    let active_config = std::fs::read_to_string(gcloud_dir.join("active_config"))
        .ok()
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "default".into());

    let config_path = gcloud_dir
        .join("configurations")
        .join(format!("config_{active_config}"));
    let config = std::fs::read_to_string(&config_path).ok()?;

    for line in config.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("project") {
            let rest = rest.trim_start();
            if let Some(rest) = rest.strip_prefix('=') {
                return Some(rest.trim().to_string());
            }
        }
    }
    None
}

// ── CLI ──

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  vertex-ai-proxy [opts]                  Start the proxy server with TUI");
    eprintln!("  vertex-ai-proxy [opts] serve            Start the proxy server (no TUI)");
    eprintln!("  vertex-ai-proxy [opts] launch <cmd> ... Start proxy + run <cmd> with ANTHROPIC_BASE_URL set");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --host <addr>   Bind address (default: 127.0.0.1; use 0.0.0.0 to listen on all interfaces)");
    eprintln!();
    eprintln!("Env:");
    eprintln!("  HOST            Same as --host");
    eprintln!("  PORT            Listen port (default: 8082)");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  vertex-ai-proxy launch claude");
    eprintln!("  vertex-ai-proxy --host 0.0.0.0 serve");
    eprintln!("  vertex-ai-proxy launch claude -p \"hello\"");
}

/// Pulls `--host VALUE` / `--host=VALUE` out of `args` (only before any `launch`
/// subcommand, so flags after `launch <cmd>` stay with the child process).
fn extract_host(args: &mut Vec<String>) -> Option<String> {
    let stop = args
        .iter()
        .position(|a| a == "launch")
        .unwrap_or(args.len());
    let mut i = 1;
    while i < stop && i < args.len() {
        if args[i] == "--host" {
            args.remove(i);
            if i < args.len() {
                return Some(args.remove(i));
            }
            return None;
        }
        if let Some(v) = args[i].strip_prefix("--host=") {
            let v = v.to_string();
            args.remove(i);
            return Some(v);
        }
        i += 1;
    }
    None
}

// ── Main ──

#[tokio::main]
async fn main() {
    let mut args: Vec<String> = std::env::args().collect();
    let host = extract_host(&mut args)
        .or_else(|| std::env::var("HOST").ok())
        .unwrap_or_else(|| "127.0.0.1".into());

    match args.get(1).map(|s| s.as_str()) {
        Some("launch") => {
            let cmd = args.get(2).cloned().unwrap_or_else(|| {
                eprintln!("Error: launch requires a command. Example: vertex-ai-proxy launch claude");
                std::process::exit(1);
            });
            let cmd_args: Vec<String> = args.iter().skip(3).cloned().collect();
            run_launch(&cmd, &cmd_args, host).await;
        }
        Some("serve") => run_server(false, host).await,
        Some("--help" | "-h") => print_usage(),
        Some(other) => {
            eprintln!("Unknown command: {other}");
            print_usage();
            std::process::exit(1);
        }
        None => run_server(true, host).await,
    }
}

async fn run_launch(cmd: &str, cmd_args: &[String], host: String) {
    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "8082".into())
        .parse()
        .expect("PORT must be a number");

    // Child always reaches the proxy via loopback.
    let base_url = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();
    let health_url = format!("{base_url}/health");

    let already_running = client
        .get(&health_url)
        .timeout(Duration::from_millis(500))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    if already_running {
        eprintln!("Proxy already running at {base_url}");
    } else {
        let server_host = host.clone();
        let _server = tokio::spawn(async move {
            run_server(false, server_host).await;
        });

        for _ in 0..50 {
            if client.get(&health_url).send().await.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        eprintln!("Proxy ready at {base_url}");
    }

    eprintln!("Launching: {cmd} {}", cmd_args.join(" "));

    let status = std::process::Command::new(cmd)
        .args(cmd_args)
        .env("ANTHROPIC_BASE_URL", &base_url)
        .env("ANTHROPIC_API_KEY", "vertex-ai-proxy")
        .status();

    match status {
        Ok(s) => std::process::exit(s.code().unwrap_or(1)),
        Err(e) => {
            eprintln!("Failed to launch '{cmd}': {e}");
            std::process::exit(1);
        }
    }
}

async fn run_server(tui: bool, host: String) {
    let endpoint = std::env::var("VERTEX_ENDPOINT")
        .unwrap_or_else(|_| "aiplatform.googleapis.com".into());
    let region = std::env::var("VERTEX_REGION").unwrap_or_else(|_| "global".into());
    let model = std::env::var("VERTEX_MODEL")
        .unwrap_or_else(|_| "qwen/qwen3-coder-480b-a35b-instruct-maas".into());
    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "8082".into())
        .parse()
        .expect("PORT must be a number");
    let max_output_tokens: u32 = std::env::var("VERTEX_MAX_OUTPUT_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16384);

    let (token_manager, quota_project) =
        TokenManager::from_adc().expect("Failed to load GCP credentials");

    let project_id = std::env::var("VERTEX_PROJECT_ID")
        .ok()
        .or(quota_project)
        .or_else(detect_gcloud_project)
        .unwrap_or_else(|| {
            eprintln!("Could not determine GCP project. Try one of:");
            eprintln!("  export VERTEX_PROJECT_ID=your-project");
            eprintln!("  gcloud config set project your-project");
            eprintln!("  gcloud auth application-default set-quota-project your-project");
            std::process::exit(1);
        });

    eprintln!("Using GCP project: {project_id}");

    let rt_token = token_manager.refresh().await;
    match &rt_token {
        Ok(_) => eprintln!("GCP OAuth2 token acquired successfully"),
        Err(e) => {
            eprintln!("Warning: initial token refresh failed: {e}");
            eprintln!("Requests will retry on first call");
        }
    }

    let state = Arc::new(AppState {
        endpoint,
        project_id,
        region,
        model,
        token_manager,
        logs: Mutex::new(Vec::new()),
        total_requests: Mutex::new(0),
        total_input_tokens: Mutex::new(0),
        total_output_tokens: Mutex::new(0),
        rps_history: Mutex::new(vec![0; 60]),
        host: host.clone(),
        port,
        external_ip: detect_local_ip(),
        max_output_tokens,
    });

    let app = Router::new()
        .route("/v1/messages", post(handle_messages))
        .route("/health", get(handle_health))
        .with_state(state.clone());

    let bind_addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .unwrap_or_else(|e| panic!("Failed to bind {bind_addr}: {e}"));

    let server_handle = tokio::spawn(async move {
        axum::serve(listener, app).await.ok();
    });

    // RPS tracker
    let state2 = state.clone();
    tokio::spawn(async move {
        let mut last_count = 0u64;
        loop {
            tokio::time::sleep(Duration::from_secs(1)).await;
            let current = *state2.total_requests.lock().unwrap();
            let rps = current - last_count;
            last_count = current;
            if let Ok(mut hist) = state2.rps_history.lock() {
                hist.push(rps);
                if hist.len() > 60 {
                    let excess = hist.len() - 60;
                    hist.drain(..excess);
                }
            }
        }
    });

    if tui && atty_check() {
        run_tui(state.clone()).await;
    } else {
        eprintln!("Vertex AI Proxy listening on {bind_addr}");
        eprintln!("Proxying Anthropic API → Qwen3 Coder on GCP");
        server_handle.await.ok();
    }
}

fn atty_check() -> bool {
    unsafe { libc_isatty(0) != 0 }
}

extern "C" {
    #[link_name = "isatty"]
    fn libc_isatty(fd: i32) -> i32;
}

async fn run_tui(state: Arc<AppState>) {
    stdout().execute(EnterAlternateScreen).ok();
    enable_raw_mode().ok();
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend).expect("terminal init failed");
    terminal.clear().ok();

    loop {
        draw_tui(&mut terminal, &state).ok();

        if event::poll(Duration::from_millis(200)).unwrap_or(false) {
            if let Ok(CEvent::Key(key)) = event::read() {
                if key.kind == KeyEventKind::Press && (key.code == KeyCode::Char('q') || key.code == KeyCode::Esc) {
                    break;
                }
            }
        }
    }

    disable_raw_mode().ok();
    stdout().execute(LeaveAlternateScreen).ok();
}
