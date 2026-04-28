# Manual Testing: Inference Backends

A hands-on reference for verifying backend behaviour through the SMG gateway. Tests are
organized by backend so you can work through one section at a time with a running stack.
Consistent headings across sections make it easy to compare behaviour side-by-side.

---

## Backend Capability Quick Reference

| Test area | MLX | vLLM | SGLang |
|---|:---:|:---:|:---:|
| Basic chat (non-streaming) | ✓ | ✓ | ✓ |
| Streaming chat | ✓ | ✓ | ✓ |
| `/v1/completions` | ✓ | ✓ | ✓ |
| String stop sequences | ✓† | ✓ | ✓ |
| Stop token IDs | ✓ | ✓ | ✓ |
| `finish_reason = "length"` | ✓ | ✓ | ✓ |
| Sampling params (temp/top_p/top_k) | ✓ | ✓ | ✓ |
| Tool calling | ✓* | ✓* | ✓* |
| Reasoning / thinking extraction | ✓* | ✓* | ✓* |
| Constrained decoding (json_schema) | ✗ | ✓ | ✓ |
| Logprobs (wired) | ✗ | ✗‡ | ✗‡ |
| Multimodal | ✗ | ✓ | ✓ |
| Embeddings | ✗ | ✓ | ✓ |

\* Handled in gateway Rust layer — backend-agnostic  
† Single-token stops only; multi-token strings are skipped with a warning. The constraint is in the SMG↔MLX interface: the proto carries stops as `repeated uint32` (individual token IDs), so only strings that encode to a single token can be expressed. mlx-lm's `SequenceStateMachine` supports multi-token sequences natively — fixing this requires either a new proto field (`repeated bytes stop_strings`) or servicer-side tokenization.  
‡ Proto + servicer produce logprobs, gateway response mapping not yet wired

---

## Prerequisites

All tests assume the gateway is running on port 3000 and the target backend is reachable.

### MLX stack

```bash
# Terminal 1 — servicer
.venv/bin/python3.12 -m smg_grpc_servicer.mlx.server \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit --port 50051
# Wait ~30s for warmup

# Terminal 2 — gateway
./target/debug/smg --worker-urls grpc://localhost:50051 --port 3000
```

Model name in all requests: `mlx-community/Qwen3-4B-Instruct-2507-4bit`

### vLLM stack

```bash
# Terminal 1 — vLLM server (example)
python -m vllm.entrypoints.grpc.api_server \
  --model YOUR_MODEL --port 50051

# Terminal 2 — gateway
./target/debug/smg --worker-urls grpc://localhost:50051 --port 3000
```

Replace `YOUR_MODEL` with the model path used by your vLLM instance.

### SGLang stack

```bash
# Terminal 1 — SGLang server (example)
python -m sglang.launch_server \
  --model-path YOUR_MODEL --port 50051

# Terminal 2 — gateway
./target/debug/smg --worker-urls grpc://localhost:50051 --port 3000
```

---

## MLX (Qwen3-0.6B-4bit)

> **Note:** Always include `"max_tokens"` in test requests. Qwen3's thinking mode generates
> nondeterministically long reasoning chains — without a cap, repeated requests can exhaust
> Apple Silicon unified memory and appear hung.

### Basic generation

```bash
# Non-streaming
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false,
    "max_tokens": 100
  }'
# Expected: choices[0].message.content non-empty; reasoning_content non-null (thinking mode)
#           finish_reason = "stop"; matched_stop = 151645 (<|im_end|> token)

# Streaming
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "max_tokens": 100
  }'
# Expected: series of data: {...} SSE events, ending with data: [DONE]
```

### `/v1/completions`

```bash
curl http://localhost:3000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "prompt": "The capital of France is",
    "stream": false,
    "max_tokens": 10
  }'
# Expected: text continues the prompt (e.g., " Paris")
```

### Stop sequences

**Before this fix** — any request with `stop: [...]` returned HTTP 400:
```json
{"error": {"type": "invalid_request_parameters", "message": "MLX backend does not support string stop sequences"}}
```

**After this fix** — the gateway tokenizes stop strings at the `RequestBuilding` stage and
converts them to `stop_token_ids` before the gRPC call:

| Request | Before | After |
|---|---|---|
| `stop: ["6"]` (single token) | HTTP 400 | `finish_reason = "stop"`, generation halts at "6" |
| `stop: ["<im_end>"]` (single special token) | HTTP 400 | Tokenized to `151645`, stop honored |
| `stop: ["Hello World"]` (multi-token) | HTTP 400 | Request succeeds; stop skipped with gateway warning; generation continues |
| `stop_token_ids: [151645]` | Already worked | Unchanged |
| `stop: ["3"]` + `stop_token_ids: [151645]` | HTTP 400 | Both merged; whichever token fires first wins |

```bash
# Single-token stop string — FIXED on this branch
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "messages": [{"role": "user", "content": "Count from 1 to 10, one number per line"}],
    "stop": ["6"],
    "stream": false,
    "max_tokens": 100
  }'
# Expected: finish_reason = "stop"; output contains 1-5, stops at or before "6"

# Multi-token stop string — gateway skips it with a warning
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "messages": [{"role": "user", "content": "Say: Hello World"}],
    "stop": ["Hello World"],
    "stream": false,
    "max_tokens": 50
  }'
# Expected: request succeeds (no 400); gateway logs warn about multi-token stop being skipped

# stop_token_ids (original proto-native path, unaffected by this fix)
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "messages": [{"role": "user", "content": "Hello"}],
    "stop_token_ids": [151645],
    "stream": false,
    "max_tokens": 100
  }'
# 151645 = <|im_end|> for Qwen3; matched_stop should be 151645

# Combined: stop string + stop_token_ids merged
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "messages": [{"role": "user", "content": "Count: 1 2 3 4 5"}],
    "stop": ["3"],
    "stop_token_ids": [151645],
    "stream": false,
    "max_tokens": 100
  }'
# Expected: stops at "3" (stop string token wins before <|im_end|>)
```

### `finish_reason` variants

```bash
# finish_reason = "length" (hits max_tokens before natural stop)
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "messages": [{"role": "user", "content": "Write a long essay about computing history"}],
    "stream": false,
    "max_tokens": 15
  }'
# Expected: finish_reason = "length"; output truncated mid-sentence
```

### Sampling parameters

```bash
# temperature=0 → greedy, deterministic output
# Run twice and verify identical responses
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "messages": [{"role": "system", "content": "/no_think"}, {"role": "user", "content": "Reply with one word: the color of the sky"}],
    "temperature": 0,
    "stream": false,
    "max_tokens": 10
  }'
# Expected: same single-word response both runs
```

### Thinking mode control

```bash
# Thinking enabled (default) — reasoning_content will be populated
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": false,
    "max_tokens": 300
  }'
# Expected: choices[0].message.reasoning_content contains a <think> block

# Thinking disabled via /no_think system message
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "messages": [{"role": "system", "content": "/no_think"}, {"role": "user", "content": "What is 2+2?"}],
    "stream": false,
    "max_tokens": 50
  }'
# Expected: reasoning_content = null or empty; much shorter response
```

### Tool calling (gateway-layer, backend-agnostic)

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string", "description": "City name"}},
          "required": ["location"]
        }
      }
    }],
    "tool_choice": "auto",
    "stream": false,
    "max_tokens": 200
  }'
# Expected: choices[0].message.tool_calls populated with get_weather call;
#           choices[0].message.content = null; finish_reason = "tool_calls"
```

### Constrained decoding (expected failure)

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "messages": [{"role": "user", "content": "Give me a person with name and age as JSON"}],
    "response_format": {"type": "json_object"},
    "stream": false,
    "max_tokens": 50
  }'
# Expected: HTTP 4xx error — MLX does not support constrained decoding
```

---

## vLLM (Qwen/Qwen2.5-1.5B-Instruct)

> **Model notes:** Qwen2.5-Instruct does not have a built-in thinking mode (that's Qwen3).
> `reasoning_content` will always be null. Tool calling uses the hermes-style format,
> which the gateway's tool parser handles transparently.

### Basic generation

```bash
# Non-streaming
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false,
    "max_tokens": 100
  }'
# Expected: choices[0].message.content non-empty; reasoning_content = null
#           finish_reason = "stop"

# Streaming
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "max_tokens": 100
  }'
# Expected: series of data: {...} SSE events ending with data: [DONE]
```

### `/v1/completions`

```bash
curl http://localhost:3000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "The capital of France is",
    "stream": false,
    "max_tokens": 10
  }'
```

### Stop sequences

```bash
# Single-token stop string
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Count from 1 to 10, one number per line"}],
    "stop": ["6"],
    "stream": false,
    "max_tokens": 100
  }'
# Expected: finish_reason = "stop"; output contains 1-5, stops at or before "6"

# Multi-token stop string — vLLM handles this natively (unlike MLX which skips it)
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Say: Hello World then keep going"}],
    "stop": ["Hello World"],
    "stream": false,
    "max_tokens": 100
  }'
# Expected: stops at "Hello World"; no warning in gateway logs
```

### `finish_reason` variants

```bash
# finish_reason = "length"
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Write a long essay about computing history"}],
    "stream": false,
    "max_tokens": 15
  }'
# Expected: finish_reason = "length"; output truncated mid-sentence
```

### Sampling parameters

```bash
# temperature=0 → greedy, deterministic — run twice for identical output
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Reply with one word: the color of the sky"}],
    "temperature": 0,
    "stream": false,
    "max_tokens": 10
  }'
```

### Tool calling (gateway-layer, backend-agnostic)

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }
    }],
    "tool_choice": "auto",
    "stream": false,
    "max_tokens": 200
  }'
# Expected: choices[0].message.tool_calls populated; finish_reason = "tool_calls"
```

### Constrained decoding (vLLM supports this; MLX does not)

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Give me a person with name and age as JSON"}],
    "response_format": {"type": "json_object"},
    "stream": false,
    "max_tokens": 100
  }'
# Expected: valid JSON in choices[0].message.content; no error (unlike MLX)
```

---

## SGLang

> Replace `YOUR_MODEL` with the actual model served by your SGLang instance.

### Basic generation

```bash
# Non-streaming
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YOUR_MODEL",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false,
    "max_tokens": 100
  }'

# Streaming
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YOUR_MODEL",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "max_tokens": 100
  }'
```

### Stop sequences

```bash
# String stop
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YOUR_MODEL",
    "messages": [{"role": "user", "content": "Count from 1 to 10, one number per line"}],
    "stop": ["6"],
    "stream": false,
    "max_tokens": 100
  }'

# Multi-token stop string — SGLang handles natively
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YOUR_MODEL",
    "messages": [{"role": "user", "content": "Say: Hello World then keep going"}],
    "stop": ["Hello World"],
    "stream": false,
    "max_tokens": 100
  }'
```

### `finish_reason` variants

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YOUR_MODEL",
    "messages": [{"role": "user", "content": "Write a long essay about computing history"}],
    "stream": false,
    "max_tokens": 15
  }'
# Expected: finish_reason = "length"
```

### Sampling parameters

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YOUR_MODEL",
    "messages": [{"role": "user", "content": "Reply with one word: the color of the sky"}],
    "temperature": 0,
    "stream": false,
    "max_tokens": 10
  }'
```

### Tool calling

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YOUR_MODEL",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }
    }],
    "tool_choice": "auto",
    "stream": false,
    "max_tokens": 200
  }'
```

### Constrained decoding (SGLang supports this; MLX does not)

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YOUR_MODEL",
    "messages": [{"role": "user", "content": "Give me a person with name and age as JSON"}],
    "response_format": {"type": "json_object"},
    "stream": false,
    "max_tokens": 100
  }'
# Expected: valid JSON; no error
```

---

## Key Differences Summary

| Behaviour | MLX | vLLM | SGLang |
|---|---|---|---|
| Multi-token stop strings | Skipped with warning | Handled natively | Handled natively |
| `response_format: json_object` | Error (unsupported) | Works | Works |
| Thinking mode | Qwen3 `/no_think` system message | Model-dependent | Model-dependent |
| `matched_stop` in response | Token ID (e.g. `151645`) | Token ID or null | Token ID or null |
| Default `max_tokens` behaviour | Nondeterministic thinking length — always set a cap | Bounded by model context | Bounded by model context |
