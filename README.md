# Vertex AI Proxy

A proxy server that translates Anthropic API requests to Google Cloud Vertex AI's Qwen3 model, enabling you to use the `claude` CLI with Google Cloud's AI infrastructure.

## Overview

This proxy acts as a bridge between the Anthropic API format and Google Cloud Vertex AI's Qwen3 model. It allows you to use the standard `claude` CLI tool while leveraging Google Cloud's AI infrastructure instead of Anthropic's API.

The proxy handles:
- API request translation from Anthropic format to Vertex AI format
- OAuth2 authentication with Google Cloud
- Real-time streaming of responses
- Request logging and monitoring
- Token management and rate limiting

## Features

- **Anthropic API Compatibility**: Accepts requests in the Anthropic API format and translates them to Vertex AI
- **Streaming Support**: Full support for streaming responses with proper event handling
- **Interactive TUI**: Built-in terminal user interface for monitoring requests and performance
- **Token Management**: Automatic OAuth2 token refresh using Google Cloud credentials
- **Request Logging**: Detailed logging of all requests with timing and token usage
- **Multi-mode Operation**: Can run as a standalone server, with TUI, or launch commands with environment variables

## Architecture

The proxy follows a clean translation pattern:

1. **Request Reception**: Receives Anthropic-formatted requests at `/v1/messages`
2. **Translation**: Converts Anthropic request format to Vertex AI OpenAPI format
3. **Authentication**: Uses Google Cloud ADC (Application Default Credentials) for OAuth2
4. **Forwarding**: Sends translated request to Vertex AI endpoint
5. **Response Translation**: Converts Vertex AI responses back to Anthropic format
6. **Streaming**: Handles both streaming and non-streaming responses appropriately

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/vertex-ai-proxy.git
cd vertex-ai-proxy

# Build the project
cargo build --release
```

### Configuration

The proxy uses the following environment variables:

- `VERTEX_ENDPOINT`: Vertex AI endpoint (default: `aiplatform.googleapis.com`)
- `VERTEX_REGION`: Region for Vertex AI (default: `global`)
- `VERTEX_MODEL`: Model to use (default: `qwen/qwen3-235b-a22b-instruct-2507-maas`)
- `VERTEX_PROJECT_ID`: GCP project ID
- `PORT`: Port to listen on (default: `8082`)
- `HOST`: Host to bind to (default: `127.0.0.1`)

### Running the Proxy

The proxy supports three modes of operation:

#### 1. Interactive Mode (with TUI)

```bash
# Run with interactive terminal UI
cargo run
```

#### 2. Server Mode (no TUI)

```bash
# Run as a background server
cargo run -- serve
```

#### 3. Command Launch Mode

```bash
# Run proxy and launch a command with proper environment
# This automatically sets ANTHROPIC_BASE_URL
cargo run -- launch claude

# Launch with specific command and arguments
cargo run -- launch claude -p "Hello, world"
```

#### 4. External Access

```bash
# Listen on all interfaces for LAN access
cargo run -- --host 0.0.0.0
```

## Authentication

The proxy uses Google Cloud Application Default Credentials (ADC). You can set this up with:

```bash
# Using gcloud CLI
gcloud auth application-default login

# Or set service account key
gcloud auth application-default login --cred-file=service-account-key.json
```

## Environment Variables for Clients

When the proxy is running, you can use the `claude` CLI by setting:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8082
export ANTHROPIC_API_KEY=vertex-ai-proxy  # Any value works
claude -p "Your prompt here"
```

## API Translation

### Request Translation

The proxy translates Anthropic API requests to Vertex AI format:

**Anthropic Request**:
```json
{
  "model": "claude-3-opus-20240229",
  "messages": [
    {
      "role": "user",
      "content": "Hello, world"
    }
  ],
  "max_tokens": 1024
}
```

**Translated to Vertex AI**:
```json
{
  "model": "qwen/qwen3-235b-a22b-instruct-2507-maas",
  "messages": [
    {
      "role": "user",
      "content": "Hello, world"
    }
  ],
  "max_tokens": 1024,
  "stream": false
}
```

### Response Translation

The proxy translates Vertex AI responses back to Anthropic format, including proper handling of:
- Content blocks (text, tool_use)
- Streaming events (message_start, content_block_start, content_block_delta, etc.)
- Usage statistics
- Stop reasons

## Monitoring

The interactive TUI displays:

- Real-time request log with ID, timing, and token usage
- Total requests, input tokens, and output tokens
- Requests per second (RPS) sparkline
- Connection status and usage instructions

## Development

### Building

```bash
# Debug build
cargo build

# Release build
cargo build --release
```

### Running Tests

```bash
# Run all tests
cargo test
```

## Security

- The proxy uses OAuth2 with Google Cloud for authentication
- No API keys are required - uses ADC credentials
- Supports both local (127.0.0.1) and external (0.0.0.0) binding
- All communication with Vertex AI is encrypted via HTTPS

## Limitations

- Currently supports only the Qwen3 model on Vertex AI
- Tool use functionality is translated but may have limitations
- Rate limiting is handled by Vertex AI, not the proxy

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.