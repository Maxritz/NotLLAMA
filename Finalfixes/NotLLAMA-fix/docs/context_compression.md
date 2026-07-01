# Context Compression Strategies

## Overview

Context compression reduces the number of tokens in the KV cache and token window when the sequence approaches the model's maximum context length. Compression is triggered at a configurable threshold (default: 85% of max context) and compacts to a configurable target (default: 50%).

## Strategies

### SLIDING_WINDOW
Keep only the last `window_size` tokens. Drop everything before the window.

- **Pros**: Fastest strategy. O(1) cost. Works well for streaming chat where only recent context matters.
- **Cons**: Drops all early context including system prompts (unless `preserveSystemPrompt` is enabled).
- **Config**: `window_size` (default 1024), `stride` (default 512).

### HALF_SLIDE
Drop the first half of the context, preserve the most recent half.

- **Pros**: Simple, no scoring needed. Ensures the most recent tokens are always kept.
- **Cons**: May lose important early context that isn't protected by system prompt preservation.
- **Config**: `preserve_recent` (default 512) — number of most recent tokens to always keep.

### FIFO
First-in-first-out eviction. Drop the oldest tokens first, one by one, until the target is reached.

- **Pros**: Predictable behavior. Drops tokens in a deterministic order.
- **Cons**: Does not consider semantic importance. May drop critical early context (system prompt, user query).
- **Config**: `drop_percent` (default 50) — percentage of oldest tokens to drop.

### IMPORTANCE
Score each token position by attention entropy (how much each token attends to others). Keep the highest-scoring positions.

- **Pros**: Best quality retention. Keeps tokens that the model actually attends to.
- **Cons**: Requires attention scores from the forward pass. More complex than sliding window approaches.
- **Scorers**:
  - `attention_entropy`: Measure how uniformly a token attends to others. Low entropy = important.
  - `gradient_magnitude`: Track gradient norms during training (offline scoring only).
  - `token_frequency`: Keep rare/memorable tokens, drop frequent filler tokens.
- **Config**: `scorer` (default: `attention_entropy`), `keep_top_percent` (default: 50).

### SUMMARY
Run a lightweight summarization on the tokens to be dropped, then insert summary tokens at the compression boundary.

- **Pros**: Best retention of semantic content. Enables much longer effective context by compressing meaning.
- **Cons**: Most complex. Requires a running summarization model on CPU. Increased latency during compression.
- **Modes**:
  - `internal`: Use the same model for summarization (triggers brief forward pass).
  - `external`: Offload summarization to a separate lightweight model or API call.
- **Config**: `summarizer_model` (default: `internal`), `compress_ratio` (default: 0.3).

## Integration

Context compression integrates with the KV cache manager via `KvShiftCallback`. After compaction, the callback notifies the inference engine about the new token positions, so the KV cache can be rearranged accordingly via the `compress_context.comp` shader.

### Lifecycle
1. Token is appended to the sequence.
2. `ContextManager::appendTokens()` checks usage ratio.
3. If threshold exceeded, `ContextManager::maybeCompact()` is called.
4. The selected strategy computes which tokens to keep.
5. `KvShiftCallback` fires with old and new base positions.
6. The inference engine dispatches `compress_context.comp` to rearrange the KV cache.
7. Token list is updated. Generation continues.
