"""
debug_forward.py — CPU reference forward pass for debugging GPU divergence.

Reads model weights and runs layer-by-layer forward pass in numpy,
printing intermediate tensor statistics after each stage.

Usage:
    python debug_forward.py model/acrux-500m-q6k.weights.json model/acrux-500m-q6k.weights.bin
"""

import sys, json, struct, math
import numpy as np

# Quant format mapping
Q8_0 = 8
Q6_K = 14
F32 = 0

def read_tensor(bin_data, offset, shape, fmt):
    n = int(np.prod(shape))
    if fmt == F32:
        vals = struct.unpack(f'{n}f', bin_data[offset:offset + n*4])
        return np.array(vals, dtype=np.float32).reshape(shape)
    elif fmt == Q8_0:
        # Q8_0: 32 elements per block, 34 bytes per block
        n_blocks = (n + 31) // 32
        out = np.zeros(n, dtype=np.float32)
        for b in range(n_blocks):
            block_off = offset + b * 34
            delta = np.frombuffer(bin_data[block_off:block_off+2], dtype=np.float16)[0]
            qs = np.frombuffer(bin_data[block_off+2:block_off+34], dtype=np.uint8)
            for i in range(32):
                idx = b * 32 + i
                if idx >= n:
                    break
                q = int(qs[i])
                if q > 127:
                    q -= 256
                out[idx] = delta * q
        return out.reshape(shape)
    elif fmt == Q6_K:
        # Q6_K: 256 elements per block, 210 bytes per block
        n_blocks = (n + 255) // 256
        out = np.zeros(n, dtype=np.float32)
        for b in range(n_blocks):
            block_off = offset + b * 210
            d = np.frombuffer(bin_data[block_off+208:block_off+210], dtype=np.float16)[0]
            ql = np.frombuffer(bin_data[block_off+0:block_off+128], dtype=np.uint8)
            qh = np.frombuffer(bin_data[block_off+128:block_off+192], dtype=np.uint8)
            scales = np.frombuffer(bin_data[block_off+192:block_off+208], dtype=np.uint8)
            for i in range(256):
                idx = b * 256 + i
                if idx >= n:
                    break
                sub_block = i // 16
                sc = int(scales[sub_block])
                if sc >= 128:
                    sc -= 256
                ql_byte_idx = i // 2
                q4raw = ql[ql_byte_idx]
                q4 = (q4raw & 0xF) if (i & 1) == 0 else (q4raw >> 4)
                qh_byte_idx = i // 4
                qh_byte = qh[qh_byte_idx]
                qh_shift = (i & 3) * 2
                qh_bits = (qh_byte >> qh_shift) & 3
                val = int((q4 | (qh_bits << 4))) - 32
                out[idx] = d * sc * val
        return out.reshape(shape)
    else:
        raise ValueError(f"Unsupported format {fmt}")

def rms_norm(x, weight, eps=1e-6):
    inv_rms = 1.0 / np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps)
    return x * inv_rms * weight

def silu(x):
    return x / (1.0 + np.exp(-x))

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def main():
    json_path = sys.argv[1]
    bin_path = sys.argv[2]

    with open(json_path) as f:
        meta = json.load(f)

    with open(bin_path, 'rb') as f:
        bin_data = f.read()

    tensors = {t['name']: t for t in meta['tensors']}
    model_cfg = meta['model']
    dim = model_cfg['embedding_length']
    hidden_dim = model_cfg.get('feed_forward_length', dim * 4)
    n_layers = model_cfg['block_count']
    n_heads = model_cfg['attention.head_count']
    n_kv_heads = model_cfg['attention.head_count_kv']
    head_dim = dim // n_heads
    vocab_size = model_cfg['vocab_size']

    print(f"Model: dim={dim} hidden_dim={hidden_dim} layers={n_layers} heads={n_heads}/{n_kv_heads} head_dim={head_dim} vocab={vocab_size}")

    # Embedding (token 1 for test)
    token_id = 1
    emb_t = tensors['token_embd.weight']
    emb = read_tensor(bin_data, emb_t['bin_offset'], emb_t['shape'], emb_t['dtype_id'])
    hidden = emb[token_id].astype(np.float32).copy()
    print(f"[embed] hidden shape={hidden.shape} min={hidden.min():.4f} max={hidden.max():.4f} mean={hidden.mean():.4f}")

    # KV cache
    k_cache = [np.zeros((1, n_kv_heads, head_dim), dtype=np.float32) for _ in range(n_layers)]
    v_cache = [np.zeros((1, n_kv_heads, head_dim), dtype=np.float32) for _ in range(n_layers)]

    for layer in range(n_layers):
        prefix = f"blk.{layer}"
        print(f"\n=== Layer {layer} ===")

        # Attention norm
        norm_w = read_tensor(bin_data, tensors[prefix + '.attn_norm.weight']['bin_offset'],
                             tensors[prefix + '.attn_norm.weight']['shape'],
                             tensors[prefix + '.attn_norm.weight']['dtype_id']).reshape(-1)
        attn_in = rms_norm(hidden, norm_w)
        print(f"[attn_norm] min={attn_in.min():.4f} max={attn_in.max():.4f} mean={attn_in.mean():.4f}")

        # Q, K, V projections
        qw = read_tensor(bin_data, tensors[prefix + '.attn_q.weight']['bin_offset'],
                         tensors[prefix + '.attn_q.weight']['shape'],
                         tensors[prefix + '.attn_q.weight']['dtype_id'])
        kw = read_tensor(bin_data, tensors[prefix + '.attn_k.weight']['bin_offset'],
                         tensors[prefix + '.attn_k.weight']['shape'],
                         tensors[prefix + '.attn_k.weight']['dtype_id'])
        vw = read_tensor(bin_data, tensors[prefix + '.attn_v.weight']['bin_offset'],
                         tensors[prefix + '.attn_v.weight']['shape'],
                         tensors[prefix + '.attn_v.weight']['dtype_id'])

        q = attn_in @ qw.T
        k = attn_in @ kw.T
        v = attn_in @ vw.T
        print(f"[q_proj] min={q.min():.4f} max={q.max():.4f} mean={q.mean():.4f}")
        print(f"[k_proj] min={k.min():.4f} max={k.max():.4f} mean={k.mean():.4f}")
        print(f"[v_proj] min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}")

        # RoPE
        def apply_rope(x, seq_len):
            x = x.reshape(n_heads, head_dim)
            for h in range(n_heads):
                for d in range(0, head_dim, 2):
                    theta = 10000.0 ** (-float(d) / head_dim)
                    angle = seq_len * theta
                    cos_a = math.cos(angle)
                    sin_a = math.sin(angle)
                    x0 = x[h, d]
                    x1 = x[h, d+1]
                    x[h, d] = x0 * cos_a - x1 * sin_a
                    x[h, d+1] = x1 * cos_a + x0 * sin_a
            return x.reshape(-1)

        q = apply_rope(q, 0)
        k = apply_rope(k, 0)
        print(f"[rope_q] min={q.min():.4f} max={q.max():.4f} mean={q.mean():.4f}")
        print(f"[rope_k] min={k.min():.4f} max={k.max():.4f} mean={k.mean():.4f}")

        # Store in KV cache
        k_cache[layer][0] = k.reshape(n_kv_heads, head_dim)
        v_cache[layer][0] = v.reshape(n_kv_heads, head_dim)

        # Attention
        q_heads = q.reshape(n_heads, head_dim)
        k_heads = k.reshape(n_kv_heads, head_dim)
        v_heads = v.reshape(n_kv_heads, head_dim)

        attn_out = np.zeros(dim, dtype=np.float32)
        for h in range(n_heads):
            kv_h = h // (n_heads // n_kv_heads)
            scores = np.dot(q_heads[h], k_cache[layer][0, kv_h].T) / math.sqrt(head_dim)
            weights = softmax(scores)
            attn_out[h*head_dim:(h+1)*head_dim] = weights * v_cache[layer][0, kv_h]

        print(f"[attn_out] min={attn_out.min():.4f} max={attn_out.max():.4f} mean={attn_out.mean():.4f}")

        # Output projection
        ow = read_tensor(bin_data, tensors[prefix + '.attn_output.weight']['bin_offset'],
                         tensors[prefix + '.attn_output.weight']['shape'],
                         tensors[prefix + '.attn_output.weight']['dtype_id'])
        attn_proj = attn_out @ ow.T
        hidden = hidden + attn_proj
        print(f"[attn_residual] min={hidden.min():.4f} max={hidden.max():.4f} mean={hidden.mean():.4f}")

        # FFN norm
        ffn_norm_w = read_tensor(bin_data, tensors[prefix + '.ffn_norm.weight']['bin_offset'],
                                 tensors[prefix + '.ffn_norm.weight']['shape'],
                                 tensors[prefix + '.ffn_norm.weight']['dtype_id']).reshape(-1)
        ffn_in = rms_norm(hidden, ffn_norm_w)
        print(f"[ffn_norm] min={ffn_in.min():.4f} max={ffn_in.max():.4f} mean={ffn_in.mean():.4f}")

        # FFN gate/up/down
        gw = read_tensor(bin_data, tensors[prefix + '.ffn_gate.weight']['bin_offset'],
                         tensors[prefix + '.ffn_gate.weight']['shape'],
                         tensors[prefix + '.ffn_gate.weight']['dtype_id'])
        uw = read_tensor(bin_data, tensors[prefix + '.ffn_up.weight']['bin_offset'],
                         tensors[prefix + '.ffn_up.weight']['shape'],
                         tensors[prefix + '.ffn_up.weight']['dtype_id'])
        dw = read_tensor(bin_data, tensors[prefix + '.ffn_down.weight']['bin_offset'],
                         tensors[prefix + '.ffn_down.weight']['shape'],
                         tensors[prefix + '.ffn_down.weight']['dtype_id'])

        gate = ffn_in @ gw.T
        up = ffn_in @ uw.T
        silu_gate = silu(gate)
        ffn_hidden = silu_gate * up
        ffn_out = ffn_hidden @ dw.T
        hidden = hidden + ffn_out
        print(f"[ffn_residual] min={hidden.min():.4f} max={hidden.max():.4f} mean={hidden.mean():.4f}")

        if layer == 0:
            print("\n--- Stopping after layer 0 for debug ---")
            break

    # Final norm
    out_norm_w = read_tensor(bin_data, tensors['output_norm.weight']['bin_offset'],
                             tensors['output_norm.weight']['shape'],
                             tensors['output_norm.weight']['dtype_id']).reshape(-1)
    final = rms_norm(hidden, out_norm_w)
    print(f"\n[final_norm] min={final.min():.4f} max={final.max():.4f} mean={final.mean():.4f}")

    # LM head
    if 'output.weight' in tensors:
        lm_w = read_tensor(bin_data, tensors['output.weight']['bin_offset'],
                           tensors['output.weight']['shape'],
                           tensors['output.weight']['dtype_id'])
    else:
        lm_w = emb  # weight tying
    logits = final @ lm_w.T
    print(f"[logits] min={logits.min():.4f} max={logits.max():.4f} mean={logits.mean():.4f}")
    print(f"[argmax] token={int(np.argmax(logits))} val={logits.max():.4f}")

if __name__ == '__main__':
    main()
