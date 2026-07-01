# NotLLAMA v0.3.0 -- Usage Guide & Command Reference

> Vulkan Compute LLM Inference Engine for AMD RDNA4 / RDNA3
> Multi-Model | MTP | Web Server | Distributed Agents | Graphify | MCP

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Building from Source](#building-from-source)
3. [Command-Line Switches](#command-line-switches)
   - [Model Options](#model-options)
   - [Prompt Options](#prompt-options)
   - [Generation / Sampling Options](#generation--sampling-options)
   - [Performance & Memory Options](#performance--memory-options)
   - [Server Mode Options](#server-mode-options)
   - [Multi-Model & Routing Options](#multi-model--routing-options)
   - [Distributed Agent Options](#distributed-agent-options)
   - [External Tool Awareness](#external-tool-awareness)
   - [Model Creation](#model-creation)
   - [Benchmark Options](#benchmark-options)
   - [Format & Display Options](#format--display-options)
4. [Running Modes](#running-modes)
   - [Single-Shot Inference](#single-shot-inference)
   - [Interactive Chat](#interactive-chat)
   - [Server Mode (OpenAI-Compatible API)](#server-mode-openai-compatible-api)
   - [Benchmark Mode](#benchmark-mode)
5. [Multi-Model Loading & Routing](#multi-model-loading--routing)
6. [Memory Tiering & Layer Offloading](#memory-tiering--layer-offloading)
7. [Multi-Token Prediction (MTP)](#multi-token-prediction-mtp)
8. [Distributed Agents](#distributed-agents)
9. [Web API Endpoints](#web-api-endpoints)
10. [Interactive Commands](#interactive-commands)
11. [Environment Variables](#environment-variables)
12. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Single-shot generation
rdna4_llama -m model.gguf -p "Explain quantum computing" -n 128

# Interactive chat
rdna4_llama -m model.gguf -i --temperature 0.7

# Server mode (OpenAI-compatible API)
rdna4_llama -m model.gguf --server --host 0.0.0.0 --port 8080

# Multi-model with routing
rdna4_llama -m code.gguf --model-id coder --model-tags code,python \
            -m chat.gguf --model-id chat --model-tags english,chat \
            --router-mode single -p "Write a Python function" -n 256

# GPU layer offloading (first 20 layers on GPU, rest in system RAM)
rdna4_llama -m model.gguf -ngl 20 -p "Hello" -n 64

# All layers on GPU
rdna4_llama -m model.gguf -ngl -1 -p "Hello" -n 64
```

---

## Building from Source

### Prerequisites

| Requirement | Version | Notes |
|------------|---------|-------|
| CMake | >= 3.25 | Build system |
| C++ Compiler | C++20 | MSVC (Windows), GCC 12+, Clang 15+ |
| Vulkan SDK | >= 1.3 | Required for GPU compute |
| glslc | bundled | SPIR-V compiler (from Vulkan SDK) |
| nlohmann/json | 3.11.3 | Fetched automatically by CMake |

### Windows (Visual Studio)

```powershell
# Set Vulkan SDK path
$env:VULKAN_SDK = "C:\VulkanSDK\1.3.xxx.x"

# Clone and build
git clone https://github.com/Maxritz/NotLLAMA.git -b KIMI-STUFF
cd NotLLAMA
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

# Run
.\Release\rdna4_llama.exe -m model.gguf -p "Hello" -n 32
```

### Linux

```bash
# Install dependencies (Debian/Ubuntu)
sudo apt install cmake g++ vulkan-validationlayers-dev

# Clone and build
git clone https://github.com/Maxritz/NotLLAMA.git -b KIMI-STUFF
cd NotLLAMA
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run
./rdna4_llama -m model.gguf -p "Hello" -n 32
```

### Build Targets

| Target | Description |
|--------|-------------|
| `rdna4_llama` | Main executable |
| `test_turboquant` | Quantization unit tests |
| `test_compression` | Compression unit tests |
| `test_engine` | Engine sanity test |
| `test_shader_compile` | Shader compiler test |
| `test_rms_norm` | RMS norm kernel test |
| `test_rope` | RoPE kernel test |
| `test_silu_mul` | SiLU multiply kernel test |
| `test_embed` | Embedding kernel test |

---

## Command-Line Switches

### Model Options

| Switch | Short | Argument | Default | Description |
|--------|-------|----------|---------|-------------|
| `--model` | `-m` | `FNAME` | (required) | Path to model file (GGUF). Can specify multiple times for multi-model. |
| `--model-id` | | `ID` | filename | Human-readable label for the model. |
| `--model-tags` | | `TAGS` | (none) | Comma-separated capability tags, e.g. `code,python,math`. |
| `--draft-model` | | `FNAME` | (none) | Draft model for speculative decoding (MTP). |
| `--gpu-layers` | `-ngl` | `N` | `-1` | Layers in GPU VRAM. `-1` = all, `0` = CPU only, `N` = first N layers. |
| `--split-mode` | | `MODE` | `layer` | How to split across memory: `layer` or `row`. |
| `--spill-dir` | | `PATH` | (none) | Directory to page layers to when RAM is full. |

### Prompt Options

| Switch | Short | Argument | Default | Description |
|--------|-------|----------|---------|-------------|
| `--prompt` | `-p` | `TEXT` | (none) | Prompt text for single-shot mode. |
| `--file` | `-f` | `FNAME` | (none) | Read prompt from file. |
| `--system-prompt` | | `TEXT` | (none) | System prompt prefix (chat mode). |
| `--chat-template` | | `NAME` | (none) | Chat template name. |
| `--interactive` | `-i` | | false | Run in interactive chat mode. |
| `--multiline-input` | | | false | Allow multi-line input (empty line to submit). |

### Generation / Sampling Options

| Switch | Short | Argument | Default | Description |
|--------|-------|----------|---------|-------------|
| `--n-predict` | `-n` | `N` | `-1` | Tokens to predict. `-1` = infinite (until EOS). |
| `--ctx-size` | `-c` | `N` | `4096` | Context window size in tokens. |
| `--batch-size` | `-b` | `N` | `2048` | Maximum batch size. |
| `--ubatch-size` | | `N` | `512` | Micro-batch size for processing. |
| `--temperature` | | `T` | `0.8` | Sampling temperature. `0` = greedy. |
| `--top-p` | | `P` | `0.95` | Nucleus sampling threshold. |
| `--top-k` | | `K` | `40` | Top-k sampling. `0` = disabled. |
| `--repeat-penalty` | | `P` | `1.1` | Penalty for repeated tokens. `1.0` = none. |
| `--repeat-last-n` | | `N` | `64` | Number of previous tokens to check for repeats. |
| `--frequency-penalty` | | `P` | `0.0` | Decreases repeated token probability. |
| `--presence-penalty` | | `P` | `0.0` | Increases new token probability. |
| `--seed` | | `N` | random | RNG seed. Use for reproducible output. |
| `--mirostat` | | `N` | `0` | Mirostat mode: `0`=off, `1`=v1, `2`=v2. |
| `--mirostat-tau` | | `T` | `5.0` | Mirostat target perplexity. |
| `--mirostat-eta` | | `E` | `0.1` | Mirostat learning rate. |
| `--dynatemp-range` | | `R` | `0.0` | Dynamic temperature range. |
| `--dynatemp-exponent` | | `E` | `1.0` | Dynamic temperature exponent. |
| `--min-p` | | `N` | `0` | Minimum probability threshold for sampling. |
| `--sampler-seq` | | `SEQ` | `default` | Sampler sequence configuration. |

### Performance & Memory Options

| Switch | Short | Argument | Default | Description |
|--------|-------|----------|---------|-------------|
| `--threads` | `-t` | `N` | `-1` | CPU threads. `-1` = auto-detect. |
| `--threads-batch` | | `N` | `-1` | Threads for batch processing. |
| `--flash-attn` | | | true | Enable Flash Attention (default on). |
| `--no-flash-attn` | | | | Disable Flash Attention. |
| `--no-kv-offload` | | | false | Keep KV cache on CPU instead of GPU. |
| `--tensor-split` | | `LIST` | (none) | Comma-split ratios for multi-GPU. |
| `--main-gpu` | | `ID` | (none) | Primary GPU device ID. |

### Server Mode Options

| Switch | Argument | Default | Description |
|--------|----------|---------|-------------|
| `--server` | | false | Run HTTP API server (OpenAI-compatible). |
| `--host` | `ADDR` | `127.0.0.1` | Bind address. Use `0.0.0.0` for LAN. |
| `--port` | `PORT` | `8080` | Listen port. |
| `--timeout` | `SEC` | `600` | Request timeout in seconds. |
| `--api-key` | `KEY` | (none) | API key for authentication. |
| `--embeddings` | | true | Enable `/v1/embeddings` endpoint. |
| `--reranking` | | false | Enable `/v1/rerank` endpoint. |

### Multi-Model & Routing Options

| Switch | Argument | Default | Description |
|--------|----------|---------|-------------|
| `--router-mode` | `MODE` | `single` | Routing strategy: `single`, `ensemble`, `cascade`. |
| `--mtp` | | false | Enable Multi-Token Prediction (speculative decoding). |
| `--mtp-n-draft` | `N` | `4` | Number of draft tokens to generate per step. |

### Distributed Agent Options

| Switch | Argument | Default | Description |
|--------|----------|---------|-------------|
| `--agent-name` | `NAME` | `notllama-agent` | Unique name for this node. |
| `--agent-port` | `PORT` | `0` | Agent HTTP listener port. `0` = disabled. |
| `--agent-peer` | `SPEC` | (none) | Peer specification. Format: `host:port,name,tags`. Repeatable. |
| `--enable-reason-sharing` | | true | Allow agents to share reasoning with peers. |
| `--enable-model-distill` | | false | Enable cross-agent model distillation. |

### External Tool Awareness

| Switch | Argument | Default | Description |
|--------|----------|---------|-------------|
| `--use-graphify` | | false | Enable Graphify integration flag. |
| `--graphify-url` | `URL` | (none) | Graphify endpoint URL (implies `--use-graphify`). |
| `--use-mcp` | | false | Enable MCP (Model Context Protocol) flag. |
| `--mcp-url` | `URL` | (none) | MCP server URL (implies `--use-mcp`). |

> **Note:** Graphify and MCP are **awareness flags only** -- they tell the system that external tools are available. They do not embed Graphify or MCP implementations into NotLLAMA. Your external services must provide the actual functionality.

### Model Creation

| Switch | Argument | Default | Description |
|--------|----------|---------|-------------|
| `--create-model` | `NAME` | (none) | Create a new model with given name. |
| `--create-model-type` | `TYPE` | (none) | Base model type: `llama`, `qwen`, `gemma`. |
| `--create-model-size` | `MB` | `0` | Target model size in megabytes. |
| `--create-model-quant` | `Q` | `Q4_K` | Quantization format: `Q4_0`, `Q8_0`, `Q4_K`, `Q6_K`, `Q8_K`, `F16`. |
| `--create-model-arch` | `ARCH` | `dense` | Architecture: `dense`, `moe`, `mixed`. |

### Benchmark Options

| Switch | Argument | Default | Description |
|--------|----------|---------|-------------|
| `--benchmark` | | false | Run performance benchmark. |
| `--benchmark-iterations` | `N` | `10` | Number of benchmark iterations. |
| `--prompt-benchmark` | | false | Benchmark prompt processing (prefill) instead of decode. |

### Format & Display Options

| Switch | Argument | Default | Description |
|--------|----------|---------|-------------|
| `--verbose-prompt` | | false | Print the prompt before generation. |
| `--no-display-prompt` | | false | Do not echo the prompt in output. |
| `--color` | | true | Enable colored terminal output. |
| `--no-color` | | | Disable colored output. |
| `--special` | | false | Display special/control tokens. |
| `--output-format` | `FMT` | `text` | Output format: `text` or `json`. |
| `--log-disable` | | false | Disable logging. |
| `--log-file` | `PATH` | (none) | Log file path (enables circular 50 MB logging). |

### General Switches

| Switch | Short | Description |
|--------|-------|-------------|
| `--help` | `-h` | Show help message and exit. |
| `--version` | `-v` | Show version information and exit. |

---

## Running Modes

### Single-Shot Inference

Provide a prompt with `-p` and get a single response.

```bash
# Basic generation
rdna4_llama -m model.gguf -p "What is the capital of France?" -n 64

# With custom sampling
rdna4_llama -m model.gguf -p "Write a haiku" -n 32 --temperature 1.2 --top-p 0.9 --top-k 80

# JSON output
rdna4_llama -m model.gguf -p "List 3 colors" -n 20 --output-format json

# Read prompt from file
rdna4_llama -m model.gguf -f prompt.txt -n 128 --temperature 0.6

# System prompt + user prompt
rdna4_llama -m model.gguf \
  --system-prompt "You are a helpful coding assistant." \
  -p "Write a Python function to reverse a string." \
  -n 256 --temperature 0.3
```

### Interactive Chat

Run with `-i` for a back-and-forth chat session.

```bash
# Basic interactive mode
rdna4_llama -m model.gguf -i

# With system prompt and custom sampling
rdna4_llama -m model.gguf -i \
  --system-prompt "You are an expert physicist." \
  --temperature 0.8 --top-p 0.95 --ctx-size 8192

# Multi-line input (empty line to submit)
rdna4_llama -m model.gguf -i --multiline-input

# Interactive with memory tier display
rdna4_llama -m model.gguf -i -ngl 10
```

**Interactive Commands:**

| Command | Description |
|---------|-------------|
| `quit` / `exit` | Exit the session. |
| `/clear` | Clear the conversation context. |
| `/models` | List all loaded models. |
| `/switch <model_id>` | Switch to a different loaded model. |
| `/route <prompt>` | Show which model the router would select. |
| `/memory` | Show VRAM/RAM tier statistics. |
| `/peers` | List connected agent peers. |
| `/ask <peer_name> <prompt>` | Ask a specific peer agent. |
| `/askall <prompt>` | Broadcast question to all peers. |
| `/graphify` | Check Graphify awareness status. |
| `/mcp` | Check MCP awareness status. |

### Server Mode (OpenAI-Compatible API)

Run as an HTTP server with OpenAI-compatible endpoints.

```bash
# Basic server
rdna4_llama -m model.gguf --server --port 8080

# Public-facing server
rdna4_llama -m model.gguf --server --host 0.0.0.0 --port 8080

# With API key authentication
rdna4_llama -m model.gguf --server --port 8080 --api-key "sk-mysecret"

# Multi-model server
rdna4_llama -m code.gguf --model-id coder --model-tags code \
            -m chat.gguf --model-id chat --model-tags english \
            --server --host 0.0.0.0 --port 8080

# Server with embeddings endpoint
rdna4_llama -m model.gguf --server --port 8080 --embeddings
```

### Benchmark Mode

Measure inference performance.

```bash
# Standard decode benchmark (token generation speed)
rdna4_llama -m model.gguf --benchmark --benchmark-iterations 20

# Prefill benchmark (prompt processing speed)
rdna4_llama -m model.gguf --benchmark --prompt-benchmark --benchmark-iterations 10

# With specific GPU layer count
rdna4_llama -m model.gguf -ngl 20 --benchmark --benchmark-iterations 50
```

---

## Multi-Model Loading & Routing

Load multiple models simultaneously and route prompts to the best one.

### Loading Multiple Models

```bash
rdna4_llama \
  -m /models/code-qwen.gguf --model-id coder --model-tags code,python,cpp \
  -m /models/chat-llama.gguf --model-id chat --model-tags english,chat,general \
  -m /models/math-gemma.gguf --model-id math --model-tags math,logic,reasoning \
  -i
```

### Router Modes

| Mode | Description |
|------|-------------|
| `single` | Use the model tagged as best match, or primary model. |
| `ensemble` | Query all models and return the best response. |
| `cascade` | Try models in order of capability until one succeeds. |

```bash
# Cascade routing: try coder first, then chat
rdna4_llama -m code.gguf --model-id coder --model-tags code \
            -m chat.gguf --model-id chat --model-tags english \
            --router-mode cascade -p "Write a Python class" -n 128

# Ensemble routing: aggregate from all models
rdna4_llama -m model1.gguf --model-id m1 \
            -m model2.gguf --model-id m2 \
            --router-mode ensemble -p "Explain gravity" -n 128
```

---

## Memory Tiering & Layer Offloading

NotLLAMA manages three memory tiers: **VRAM** (GPU), **System RAM** (CPU), and **Disk** (storage).

### GPU Layer Offloading (`-ngl`)

| Value | Behavior |
|-------|----------|
| `-1` | All layers in VRAM (fastest, most VRAM used) |
| `0` | All layers in system RAM (slowest, no VRAM used) |
| `N` | First N layers in VRAM, rest in RAM |

```bash
# All layers on GPU (requires enough VRAM)
rdna4_llama -m model.gguf -ngl -1 -p "Hello" -n 64

# First 20 layers on GPU, rest in system RAM
rdna4_llama -m model.gguf -ngl 20 -p "Hello" -n 64

# CPU only (all layers in system RAM)
rdna4_llama -m model.gguf -ngl 0 -p "Hello" -n 64

# With disk spill for very large models
rdna4_llama -m 70b_model.gguf -ngl 10 --spill-dir /tmp/notllama_spill -p "Hello" -n 64
```

### Split Mode

```bash
# Split by layer (default) -- entire layers go to GPU or RAM
rdna4_llama -m model.gguf -ngl 15 --split-mode layer

# Split by row -- individual weight rows distributed
rdna4_llama -m model.gguf --split-mode row
```

### Memory Tier Statistics

In interactive mode, type `/memory` to see:

```
[MemoryTier] VRAM: 4.2 GB / 16.0 GB (26%) | RAM: 2.1 GB / 64.0 GB (3%) | Disk: 0 B
  gpu_layers=20 split_mode=layer
```

---

## Multi-Token Prediction (MTP)

Speculative decoding using a draft model to predict multiple tokens per forward pass.

```bash
# Enable MTP with default settings (4 draft tokens)
rdna4_llama -m target_model.gguf --draft-model draft_model.gguf --mtp -p "Hello" -n 64

# Custom draft count
rdna4_llama -m target.gguf --draft-model draft.gguf --mtp --mtp-n-draft 8 \
  -p "Write a function" -n 128

# MTP in interactive mode
rdna4_llama -m target.gguf --draft-model draft.gguf --mtp -i
```

**How it works:**
1. The draft model quickly generates N candidate tokens.
2. The target model verifies all N tokens in a single forward pass.
3. Accepted tokens are emitted; rejected tokens trigger a re-generation.
4. Typical speedup: **1.5x - 2.5x** depending on model compatibility.

**Requirements:**
- Draft model should be smaller and faster than the target model.
- Both models must share the same tokenizer/vocabulary.

---

## Distributed Agents

Multiple NotLLAMA instances can form a mesh network and collaborate on reasoning tasks.

### Starting an Agent Node

```bash
# Node 1: Code expert
rdna4_llama -m code_model.gguf --model-id coder --model-tags code,python \
  --agent-name "coder-node" --agent-port 9001 -i

# Node 2: General chat expert
rdna4_llama -m chat_model.gguf --model-id chat --model-tags english,general \
  --agent-name "chat-node" --agent-port 9002 \
  --agent-peer "localhost,9001,coder-node,code,python" \
  -i

# Node 3: Math expert
rdna4_llama -m math_model.gguf --model-id math --model-tags math,logic \
  --agent-name "math-node" --agent-port 9003 \
  --agent-peer "localhost,9001,coder-node,code,python" \
  --agent-peer "localhost,9002,chat-node,english,general" \
  -i
```

### Peer Specification Format

```
--agent-peer "HOST,PORT,NAME,TAG1,TAG2,..."
```

Example:
```bash
--agent-peer "192.168.1.10,9001,office-coder,code,python,cpp"
--agent-peer "192.168.1.11,9002,office-chat,english,writing"
```

### Interactive Agent Commands

| Command | Description |
|---------|-------------|
| `/peers` | List all peers with status and capabilities. |
| `/ask <peer_name> <prompt>` | Send a reasoning query to a specific peer. |
| `/askall <prompt>` | Broadcast to all peers, returns best answer. |

### Agent HTTP Endpoints (when server mode is enabled)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/agent/reason` | Submit a reasoning query to this agent. |
| `GET` | `/agent/status` | Get agent status, loaded model, peers. |
| `POST` | `/agent/model/create` | Request distributed model creation. |

**Example: Query an agent via HTTP:**
```bash
curl -X POST http://localhost:8080/agent/reason \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "req-001",
    "from_agent": "user-client",
    "query": "Optimize this Python loop: for i in range(n): result += i*i",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

---

## Web API Endpoints

When running with `--server`, the following OpenAI-compatible endpoints are available:

### Core Endpoints

| Method | Path | OpenAI Equivalent | Description |
|--------|------|-------------------|-------------|
| `POST` | `/v1/completions` | ` completions` | Text completion. |
| `POST` | `/v1/chat/completions` | `chat.completions` | Chat completion with message history. |
| `GET` | `/v1/models` | `models` | List loaded models. |
| `GET` | `/health` | -- | Health check. |
| `GET` | `/props` | -- | Server properties. |
| `POST` | `/tokenize` | -- | Tokenize text to token IDs. |
| `POST` | `/detokenize` | -- | Convert token IDs to text. |

### Optional Endpoints

| Method | Path | Flag Required | Description |
|--------|------|---------------|-------------|
| `POST` | `/v1/embeddings` | `--embeddings` | Text embeddings. |
| `POST` | `/v1/rerank` | `--reranking` | Document reranking. |
| `POST` | `/agent/reason` | `--agent-peer` or `--agent-port` | Distributed reasoning. |
| `GET` | `/agent/status` | `--agent-peer` or `--agent-port` | Agent status. |
| `POST` | `/agent/model/create` | `--agent-peer` or `--agent-port` | Model creation. |

### API Examples

**Text Completion:**
```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-mysecret" \
  -d '{
    "model": "coder",
    "prompt": "def fibonacci(n):",
    "max_tokens": 128,
    "temperature": 0.3,
    "top_p": 0.95,
    "seed": 42
  }'
```

**Chat Completion:**
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chat",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain neural networks."}
    ],
    "max_tokens": 256,
    "temperature": 0.8
  }'
```

**List Models:**
```bash
curl http://localhost:8080/v1/models
```

**Tokenize:**
```bash
curl -X POST http://localhost:8080/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}'
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VULKAN_SDK` | Path to Vulkan SDK (required for build). |
| `NOTLLAMA_SHADERS` | Override shader directory path. |
| `NOTLLAMA_LOG_LEVEL` | Log verbosity: `0`=error, `1`=info, `2`=debug, `3`=trace. |
| `NOTLLAMA_DISABLE_WAVE32` | Set to `1` to force wave64 mode. |

---

## Troubleshooting

### Vulkan Initialization Fails

```
[FATAL] Vulkan initialization failed
```

- Ensure your GPU drivers are up to date.
- Install the Vulkan Runtime: `sudo apt install vulkan-tools` (Linux) or install the Vulkan SDK (Windows).
- Run `vulkaninfo` to verify your GPU supports Vulkan 1.3.
- AMD GPUs: ensure `amdgpu` driver is loaded (Linux) or Adrenalin drivers are installed (Windows).

### Model Loading Fails

```
[WARN] Failed to load model: model.gguf
```

- Verify the file path is correct.
- Ensure the model is in GGUF format.
- Check that you have read permissions on the file.
- Try loading with `-ngl 0` to rule out VRAM issues.

### Out of VRAM

```
[StepBatch] AllocScratchBuffers failed
```

- Reduce `-ngl` value to offload fewer layers to GPU.
- Enable disk spill with `--spill-dir /path`.
- Close other GPU applications.
- Reduce `--ctx-size` to use less KV cache memory.

### Slow Inference

- Increase `-ngl` to put more layers on GPU.
- Enable MTP with `--draft-model` for speculative decoding.
- Ensure Flash Attention is enabled (default on).
- Check `/memory` in interactive mode to verify layer placement.

### Shader Not Found

```
[Init] Shader dir: (not found)
```

- Ensure the `shaders/` directory is next to the executable.
- Set `NOTLLAMA_SHADERS` environment variable to the shader path.
- Re-run CMake build to install shaders.

### Windows-Specific Issues

| Issue | Solution |
|-------|----------|
| `ssize_t` errors | Fixed in current build -- ensure you have the latest KIMI-STUFF branch. |
| Socket errors | Build with MSVC and ensure `ws2_32.lib` is linked (handled by CMake). |
| `glslc not found` | Set `VULKAN_SDK` environment variable to your Vulkan SDK path. |

---

## Version History

| Version | Changes |
|---------|---------|
| 0.3.0 | Real Vulkan inference engine (InferenceRunner), distributed agents, memory tiering, model creation, Graphify/MCP awareness |
| 0.2.0 | Multi-model loading, MTP, web server, CLI parser, model router |
| 0.1.0 | Initial Vulkan compute pipeline, shader kernels, GGUF loader |

---

*NotLLAMA -- Vulkan Compute LLM Inference Engine*
*Copyright (c) 2024-2025 Maxritz (Ritesh Nair)*
