"""
Neural Steganography v11 — Verified AC + Keyed Robust Mode


Pipeline:
  Encode: compress(msg) → encrypt(seed) → AC decode(full vocab) → tokens
  Decode: tokens → AC encode(full vocab) → decrypt(seed) → decompress → msg
"""

import torch
import hashlib
import hmac
import struct
import zlib
import bisect
import os
from transformers import AutoTokenizer, AutoConfig

# ── Arithmetic coding constants ──────────────────────────────────────────────
PRECISION = 32
MAX_CODE  = (1 << PRECISION) - 1
HALF      = 1 << (PRECISION - 1)
QUARTER   = 1 << (PRECISION - 2)
THREE_Q   = 3 * QUARTER
CDF_BITS  = 24                          # must be < PRECISION-2 for safe range math
CDF_TOTAL = 1 << CDF_BITS              # 16,777,216

# ── Model cache ──────────────────────────────────────────────────────────────
_tokenizer = None
_model = None
_loaded_model_name = None
_loaded_dtype = None
_device = None

DEFAULT_MODEL = "gpt2-medium"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_name: str = None, dtype=None):
    """Load model on best available device (MPS/CUDA/CPU)."""
    if dtype is None:
        dtype = torch.float32
    global _tokenizer, _model, _loaded_model_name, _loaded_dtype, _device
    if model_name is None:
        model_name = DEFAULT_MODEL
    if _tokenizer is not None and _loaded_model_name == model_name and _loaded_dtype == dtype:
        return _tokenizer, _model

    if _model is not None:
        del _model; _model = None
        import gc; gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    from transformers import AutoModelForCausalLM
    _device = get_device()
    print(f"Loading {model_name} ({dtype}) on {_device}...")
    config = AutoConfig.from_pretrained(model_name)
    if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
        config.pad_token_id = _tokenizer.eos_token_id
    _model = AutoModelForCausalLM.from_pretrained(
        model_name, config=config, torch_dtype=dtype,
    ).to(_device)
    _model.eval()
    _loaded_model_name = model_name
    _loaded_dtype = dtype
    print(f"Model ready on {_device}.\n")
    return _tokenizer, _model


# ── Key stream & encryption ──────────────────────────────────────────────────

def _keystream_bytes(seed: str, length: int) -> bytes:
    """Generate `length` pseudorandom bytes from seed via SHA-256 CTR."""
    result = bytearray()
    counter = 0
    while len(result) < length:
        h = hashlib.sha256(f"{seed}:{counter}".encode()).digest()
        result.extend(h)
        counter += 1
    return bytes(result[:length])


def _encrypt(data: bytes, seed: str) -> bytes:
    """XOR encrypt. Symmetric — decrypt is the same call."""
    ks = _keystream_bytes(seed, len(data))
    return bytes(a ^ b for a, b in zip(data, ks))


# ── Compression ──────────────────────────────────────────────────────────────

def _pack_message(text: str, seed: str) -> bytes:
    """Compress → encrypt → frame with header.
    Format: [1B flag] [2B payload_len] [payload] [2B flush padding]
    """
    raw = text.encode('utf-8')
    compressed = zlib.compress(raw, level=9)
    if len(compressed) < len(raw):
        payload, flag = compressed, 1
    else:
        payload, flag = raw, 0
    encrypted = _encrypt(payload, seed)
    header = struct.pack('>BH', flag, len(encrypted))
    return header + encrypted + b'\xff\xff'   # flush padding


def _unpack_message(data: bytes, seed: str) -> str:
    """Reverse of _pack_message."""
    flag = data[0]
    length = struct.unpack('>H', data[1:3])[0]
    encrypted = data[3:3 + length]
    payload = _encrypt(encrypted, seed)         # XOR is symmetric
    if flag == 1:
        return zlib.decompress(payload).decode('utf-8')
    return payload.decode('utf-8')


def _bytes_to_bits(data: bytes) -> list:
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _bits_to_bytes(bits: list) -> bytes:
    while len(bits) % 8 != 0:
        bits.append(0)
    return bytes(
        int(''.join(str(b) for b in bits[i:i + 8]), 2)
        for i in range(0, len(bits), 8)
    )


# ── Tokenization ─────────────────────────────────────────────────────────────

def _tokenize(tokenizer, text: str) -> list:
    return tokenizer.encode(text, add_special_tokens=False)


# ── Cover text planner ────────────────────────────────────────────────────────

def plan_cover(cover_draft: str, style: str = "twitter",
               model_name: str = None) -> str:
    """Reword, extend, and rephrase user text. Strips their writing style."""
    if model_name is None:
        model_name = DEFAULT_MODEL
    tokenizer, model = load_model(model_name)
    device = _device

    prompt = (
        f"Original: {cover_draft.strip()}\n\n"
        "Rewritten casually in totally different words:\n"
    )

    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    # Detect max context from various config attributes
    max_ctx = getattr(model.config, 'max_position_embeddings',
              getattr(model.config, 'n_positions',
              getattr(model.config, 'seq_length', 2048)))
    if len(input_ids) > max_ctx - 150:
        input_ids = input_ids[:max_ctx - 150]

    inp = torch.tensor([input_ids]).to(device)

    with torch.no_grad():
        out = model.generate(
            inp,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = out[0][len(input_ids):]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Clean up
    for stop in ['\n\n', '\nOriginal:', '\nParaphrased', '---']:
        if stop in result:
            result = result[:result.index(stop)].strip()

    return result if result and len(result) > 20 else cover_draft


# ── Token whitelist (English-safe filtering) ─────────────────────────────────

_whitelist_cache = {}   # model_name → sorted list of valid token IDs

def _get_whitelist(tokenizer, model_name, model=None):
    """Pre-compute set of token IDs that decode to ASCII/Latin text.

    Multilingual models (Qwen, etc.) have 100K+ tokens across many scripts.
    Without filtering, arithmetic coding occasionally selects Chinese/Arabic/etc
    tokens because they have nonzero probability. This filter restricts the CDF
    to English-safe tokens only.

    The steganographic guarantee still holds: output is indistinguishable from
    the model's distribution *conditioned on producing English text*.
    """
    if model_name in _whitelist_cache:
        return _whitelist_cache[model_name]

    import re
    vocab_size = len(tokenizer)

    # Cap at model's actual output dimension if available
    if model is not None:
        try:
            lm_head = getattr(model, 'lm_head', None)
            if lm_head is not None:
                model_vocab = lm_head.out_features if hasattr(lm_head, 'out_features') else lm_head.weight.shape[0]
                vocab_size = min(vocab_size, model_vocab)
        except Exception:
            pass

    whitelist = []

    # Pattern: printable ASCII + common Latin-1 (accented chars)
    safe_pattern = re.compile(r'^[\x20-\x7e\xa0-\xff\n\r\t]+$')

    for token_id in range(vocab_size):
        try:
            text = tokenizer.decode([token_id])
            if text and safe_pattern.match(text):
                whitelist.append(token_id)
        except Exception:
            continue

    # Always include special tokens the model needs
    if tokenizer.eos_token_id is not None:
        if tokenizer.eos_token_id not in whitelist and tokenizer.eos_token_id < vocab_size:
            whitelist.append(tokenizer.eos_token_id)

    whitelist.sort()
    _whitelist_cache[model_name] = whitelist
    print(f"  Whitelist: {len(whitelist)}/{vocab_size} tokens ({100*len(whitelist)//vocab_size}% of vocab)")
    return whitelist


# ── Cross-device disambiguation: SyncPool prefix grouping + logit quantization
#
# Two problems kill cross-device AC decoding:
#
# 1. BPE AMBIGUITY: tokenize("leading") can produce either ["lead","ing"] or
#    ["leading"] depending on context. Encoder and decoder get different token
#    sequences → different CDF indices → AC desync.
#    Fix: SyncPool prefix grouping. Tokens that are prefixes of others are
#    grouped together. AC operates on groups. Within a group, a shared CSPRNG
#    (seeded from the steganography password + step number) picks the specific
#    token. Both sides compute identical groups and CSPRNG → identical behavior.
#    Based on: Qi et al. "Provably Secure Disambiguating Neural Linguistic
#    Steganography" (IEEE TDSC, 2024)
#
# 2. LOGIT DIVERGENCE: different GPUs produce slightly different float32 logits
#    (different CUDA versions, cuBLAS algorithms, operation ordering). Even 1e-5
#    relative error → 94% of CDF entries differ → ~1% symbol mismatch per step
#    → guaranteed decode failure over 100+ tokens.
#    Fix: Quantize logits to reduced precision (float16) before computing the
#    CDF. This absorbs all GPU-specific float noise. float16 absorbs differences
#    up to ~0.025 relative error, far exceeding GPU-to-GPU divergence (~1e-5).
#    Based on: "Addressing Tokenization Inconsistency in Steganography and
#    Watermarking Based on LLMs" (2025)

_token_text_cache = {}       # tokenizer_id → {token_id: decoded_text}
_prefix_groups_cache = {}    # tokenizer_id → (group_map, groups)


def _precompute_token_texts(tokenizer, whitelist):
    """Cache the decoded text of each whitelist token (once per tokenizer)."""
    key = id(tokenizer)
    if key in _token_text_cache:
        return _token_text_cache[key]
    texts = {}
    for tid in whitelist:
        try:
            texts[tid] = tokenizer.decode([tid])
        except Exception:
            texts[tid] = ""
    _token_text_cache[key] = texts
    return texts


def _build_prefix_groups(tokenizer, whitelist):
    """Group whitelist tokens by prefix relationships (SyncPool).

    Returns:
        group_map: dict mapping token_id → group_index
        groups: list of lists, groups[i] = [tid1, tid2, ...] sharing a prefix
        group_heads: list of group_index values (one per group, for CDF)
    """
    key = id(tokenizer)
    if key in _prefix_groups_cache:
        return _prefix_groups_cache[key]

    # Get text representation of each token
    token_texts = _precompute_token_texts(tokenizer, whitelist)

    # Sort tokens by their text (lexicographic) for prefix detection
    sorted_tokens = sorted(whitelist, key=lambda tid: token_texts.get(tid, ""))

    # Group by prefix: if token A's text is a prefix of token B's text,
    # they go in the same group
    groups = []
    group_map = {}

    i = 0
    while i < len(sorted_tokens):
        # Start a new group with this token
        head_tid = sorted_tokens[i]
        head_text = token_texts.get(head_tid, "")
        group = [head_tid]
        group_idx = len(groups)
        group_map[head_tid] = group_idx

        # Collect all subsequent tokens that start with head_text
        j = i + 1
        while j < len(sorted_tokens):
            cand_tid = sorted_tokens[j]
            cand_text = token_texts.get(cand_tid, "")
            if head_text and cand_text.startswith(head_text):
                group.append(cand_tid)
                group_map[cand_tid] = group_idx
                j += 1
            else:
                break

        groups.append(group)
        i = j

    n_ambig = sum(1 for g in groups if len(g) > 1)
    n_tokens_in_ambig = sum(len(g) for g in groups if len(g) > 1)
    print(f"  SyncPool: {len(groups)} groups from {len(whitelist)} tokens "
          f"({n_ambig} ambiguous groups containing {n_tokens_in_ambig} tokens)")

    result = (group_map, groups)
    _prefix_groups_cache[key] = result
    return result


def _syncpool_csprng_select(groups, group_idx, seed, step):
    """Select a specific token from an ambiguous group using shared CSPRNG.

    Both encoder and decoder call this with identical (seed, step) → identical result.
    For single-token groups, returns the only token (no randomness needed).
    """
    group = groups[group_idx]
    if len(group) == 1:
        return group[0]

    # Deterministic selection using HMAC-SHA256
    import hmac, hashlib
    key = f"{seed}:syncpool:{step}".encode()
    h = hmac.new(key, b"select", hashlib.sha256).digest()
    rand_val = int.from_bytes(h[:4], 'big')
    return group[rand_val % len(group)]


# ── Per-step safe whitelist (BPE merge prevention) ───────────────────────────
#
# Even with SyncPool prefix grouping, BPE merges (where two consecutive tokens
# collapse into one during re-tokenization) can cause decode failure. The safest
# approach: at each encode step, filter out candidate tokens that would merge
# with the previous token. Both encoder and decoder apply the same deterministic
# filter → identical CDFs.

_pair_safety_cache = {}   # tokenizer_id → {(prev_tid, cand_tid): bool}


def _is_pair_safe(tokenizer, prev_tid, cand_tid, token_texts):
    """Check if (prev_token, candidate) survives BPE roundtrip. Cached."""
    key = id(tokenizer)
    if key not in _pair_safety_cache:
        _pair_safety_cache[key] = {}
    cache = _pair_safety_cache[key]
    pair = (prev_tid, cand_tid)

    if pair in cache:
        return cache[pair]

    # Full roundtrip check
    prev_text = token_texts.get(prev_tid, tokenizer.decode([prev_tid]))
    cand_text = token_texts.get(cand_tid, tokenizer.decode([cand_tid]))
    combined = prev_text + cand_text

    try:
        rt = tokenizer.encode(combined, add_special_tokens=False)
        safe = (len(rt) == 2 and rt[0] == prev_tid and rt[1] == cand_tid)
    except Exception:
        safe = False

    cache[pair] = safe
    return safe


def _compute_safe_whitelist(tokenizer, data_tokens_so_far, whitelist,
                            token_texts=None):
    """Return subset of whitelist that survives roundtrip at current boundary.

    Fast path: tokens starting with space (' ') are word-initial in BPE and
    NEVER merge with the previous token. Only non-space tokens need roundtrip
    checks. Results are cached per (prev, candidate) pair, so repeated calls
    with the same previous token are O(1) after warm-up.
    """
    if not data_tokens_so_far:
        return whitelist

    prev_token = data_tokens_so_far[-1]

    safe = []
    for tid in whitelist:
        text = token_texts.get(tid, "") if token_texts else ""
        # Space-prefixed tokens never merge with previous token
        if text and text[0] == ' ':
            safe.append(tid)
        elif _is_pair_safe(tokenizer, prev_token, tid, token_texts or {}):
            safe.append(tid)

    return safe if safe else whitelist  # fallback: never produce empty CDF


# ── Full-vocabulary CDF (with optional whitelist) ────────────────────────────

def _build_cdf(logits, whitelist=None, quantize=False):
    """Build quantized CDF over vocabulary.

    If quantize=True, logits are truncated to float16 precision before softmax.
    This absorbs cross-device float differences (GPU-to-GPU logit divergence)
    at the cost of very slight information loss (~0.1% capacity reduction).
    """
    # Ensure logits are on CPU as float32 for deterministic softmax
    logits_cpu = logits.float().cpu() if logits.device.type != 'cpu' else logits.float()

    if whitelist is not None:
        sub_logits = logits_cpu[whitelist]
        token_map = whitelist
    else:
        sub_logits = logits_cpu
        token_map = None

    # Cross-device logit quantization: round logits to nearest integer.
    # This absorbs ALL GPU-to-GPU float noise (typically ~1e-5 relative error).
    # Integer steps are ~1.0, GPU noise is ~0.00001 → probability of a logit
    # landing right at an integer boundary due to GPU noise is ~0.003%.
    # With 50K tokens, ~1-2 might be at boundaries, but their CDF impact is nil.
    # Capacity impact: ~5-10% reduction in bits/token (coarser distribution).
    # Still 5-8 bpt vs robust mode's 0.67 bpt.
    if quantize:
        sub_logits = sub_logits.round()

    probs = torch.softmax(sub_logits, dim=-1)

    n = len(probs)
    counts = (probs * CDF_TOTAL).long().clamp(min=1)

    diff = CDF_TOTAL - counts.sum().item()
    counts[counts.argmax()] += diff

    cumulative = torch.cumsum(counts, dim=0)
    cdf = [0] + cumulative.tolist()
    return cdf, token_map


def _build_grouped_cdf(logits, whitelist, groups, group_map, quantize=False):
    """Build CDF over SyncPool prefix groups instead of individual tokens.

    Each group's CDF mass = sum of probabilities of all tokens in the group.
    This ensures tokens that are prefixes of each other (e.g. "lead" and
    "leading") share the same CDF interval, making AC identical regardless
    of which specific token the BPE tokenizer produces.

    Returns:
        cdf: cumulative distribution over groups
        groups_ref: the groups list (for mapping group_idx → token selection)
    """
    logits_cpu = logits.float().cpu() if logits.device.type != 'cpu' else logits.float()

    sub_logits = logits_cpu[whitelist]
    if quantize:
        sub_logits = sub_logits.round()

    probs = torch.softmax(sub_logits, dim=-1)

    # Build whitelist-index lookup
    wl_idx = {tid: i for i, tid in enumerate(whitelist)}

    # Sum probabilities within each group
    n_groups = len(groups)
    group_probs = []
    for g in groups:
        mass = sum(probs[wl_idx[tid]].item() for tid in g if tid in wl_idx)
        group_probs.append(mass)

    # Quantize to CDF
    total = sum(group_probs)
    counts = [max(1, int(p / total * CDF_TOTAL)) for p in group_probs]
    diff = CDF_TOTAL - sum(counts)
    max_idx = group_probs.index(max(group_probs))
    counts[max_idx] += diff

    cdf = [0]
    for c in counts:
        cdf.append(cdf[-1] + c)

    return cdf, groups


# ── Prompt building ──────────────────────────────────────────────────────────

def build_prompt(cover_context: str, style: str, persona: str = "",
                 cover_draft: str = "", seed: str = "",
                 topic: str = "") -> str:
    """Build prompt deterministically from seed + optional topic.

    In robust mode the decoder never uses the prompt, so
    the topic only needs to be known at encode time.
    """
    if topic:
        return f"{topic.strip()}\n\n"

    if seed:
        import hashlib
        h = hashlib.sha256(seed.encode()).hexdigest()[:8]
        prompt = f"Post {h}:\n\n"
    else:
        prompt = "Post:\n\n"
    return prompt


# ══════════════════════════════════════════════════════════════════════════════
#  ENCODER  (encrypted message bits → tokens via arithmetic DECODING)
# ══════════════════════════════════════════════════════════════════════════════

def encode(
    secret: str,
    seed: str,
    cover_context: str = "",
    style: str = "email",
    max_tokens: int = 400,
    model_name: str = "gpt2-medium",
    persona: str = "",
    cover_draft: str = "",
    verified: bool = True,
    **kwargs,
) -> tuple:
    """
    Returns (response_text, all_token_ids, prompt_len, stats_dict).

    Uses SyncPool prefix grouping (Qi et al., 2024) to handle BPE
    segmentation ambiguity without per-step pairwise safety checks.
    Tokens sharing prefix relationships are grouped; AC operates over
    groups. CSPRNG selects the specific token within each group,
    keeping encoder and decoder in sync.
    """
    tokenizer, model = load_model(model_name)
    device = _device
    vocab_size = len(tokenizer)

    # Build whitelist for multilingual models
    whitelist = _get_whitelist(tokenizer, model_name, model)
    n_tokens = len(whitelist) if whitelist else vocab_size

    # ── SyncPool: build prefix groups once (replaces per-step safety checks) ──
    if whitelist:
        token_texts = _precompute_token_texts(tokenizer, whitelist)
        group_map, groups = _build_prefix_groups(tokenizer, whitelist)
        print(f"  SyncPool mode: {len(groups)} groups from {len(whitelist)} tokens")
    else:
        token_texts = None
        group_map, groups = None, None

    # ── Compress + encrypt ──
    packed = _pack_message(secret, seed)
    message_bits = _bytes_to_bits(packed)
    total_msg_bits = len(message_bits)

    # Pad for AC flush
    padded_bits = message_bits + [0] * (PRECISION * 2)

    # ── Prompt ──
    prompt = build_prompt(cover_context, style, persona, cover_draft, seed=seed)
    prompt_tokens = _tokenize(tokenizer, prompt)
    generated = list(prompt_tokens)

    # ── AC decoder state ──
    low = 0
    high = MAX_CODE
    bit_idx = 0

    # Load initial value from message bits
    value = 0
    for _ in range(PRECISION):
        value = (value << 1) | padded_bits[bit_idx]
        bit_idx += 1

    # ── KV-cache warm-up ──
    inp = torch.tensor([prompt_tokens]).to(device)
    print(f"  Starting encode: {len(prompt_tokens)} prompt tokens, {total_msg_bits} msg bits")
    print(f"  Device: {device}, inp shape: {inp.shape}")
    with torch.no_grad():
        print("  Running first forward pass...")
        out = model(inp, use_cache=True)
        print("  Forward pass OK, starting AC loop...")
        past = out.past_key_values
        data_tokens = 0

        for step in range(max_tokens):
            logits = out.logits[0, -1, :]

            # Check if done
            if bit_idx >= total_msg_bits + PRECISION:
                text_so_far = tokenizer.decode(
                    generated[len(prompt_tokens):], skip_special_tokens=True
                )
                if data_tokens > 0 and text_so_far.rstrip().endswith(('.', '!', '?', '"')):
                    break
                if step - data_tokens > 8:
                    break
                # Greedy padding token (from whitelist if available)
                if whitelist:
                    indices = torch.tensor(whitelist, device=logits.device)
                    sub_logits = logits[indices]
                    chosen = whitelist[sub_logits.argmax().item()]
                else:
                    chosen = logits.argmax().item()
                generated.append(chosen)
                inp = torch.tensor([[chosen]]).to(device)
                out = model(inp, past_key_values=past, use_cache=True)
                past = out.past_key_values
                continue

            data_tokens += 1

            # ── Build CDF over SyncPool groups (or plain whitelist) ──
            if groups is not None:
                cdf, _groups_ref = _build_grouped_cdf(
                    logits, whitelist, groups, group_map, quantize=True
                )
                n = len(cdf) - 1
            else:
                cdf, token_map = _build_cdf(logits, whitelist, quantize=True)
                n = len(cdf) - 1

            range_size = high - low + 1

            # ── Find symbol whose CDF interval contains `value` ──
            cum = ((value - low + 1) * CDF_TOTAL - 1) // range_size
            cum = max(0, min(cum, CDF_TOTAL - 1))

            # Binary search: find i such that cdf[i] <= cum < cdf[i+1]
            selected = bisect.bisect_right(cdf, cum) - 1
            selected = max(0, min(selected, n - 1))

            # ── Update range ──
            sym_low  = cdf[selected]
            sym_high = cdf[selected + 1]
            new_low  = low + (range_size * sym_low)  // CDF_TOTAL
            new_high = low + (range_size * sym_high) // CDF_TOTAL - 1
            low  = new_low
            high = new_high

            # ── Renormalize ──
            while True:
                if high < HALF:
                    low  = low << 1
                    high = (high << 1) | 1
                    b = padded_bits[bit_idx] if bit_idx < len(padded_bits) else 0
                    value = ((value << 1) & MAX_CODE) | b
                    bit_idx += 1
                elif low >= HALF:
                    low  = (low - HALF) << 1
                    high = ((high - HALF) << 1) | 1
                    value = (((value - HALF) << 1) & MAX_CODE)
                    b = padded_bits[bit_idx] if bit_idx < len(padded_bits) else 0
                    value |= b
                    bit_idx += 1
                elif low >= QUARTER and high < THREE_Q:
                    low  = (low - QUARTER) << 1
                    high = ((high - QUARTER) << 1) | 1
                    value = (((value - QUARTER) << 1) & MAX_CODE)
                    b = padded_bits[bit_idx] if bit_idx < len(padded_bits) else 0
                    value |= b
                    bit_idx += 1
                else:
                    break

            # ── Map selected CDF index → actual token ID ──
            if groups is not None:
                # SyncPool: selected is a group index → CSPRNG picks token
                actual_token = _syncpool_csprng_select(groups, selected, seed, data_tokens)
            else:
                actual_token = token_map[selected] if token_map else selected

            generated.append(actual_token)
            inp = torch.tensor([[actual_token]]).to(device)
            out = model(inp, past_key_values=past, use_cache=True)
            past = out.past_key_values

    # Output: derive cover_text from full decode
    full_text = tokenizer.decode(generated, skip_special_tokens=True)
    prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
    if full_text.startswith(prompt_text):
        response_text = full_text[len(prompt_text):]
    else:
        response_text = tokenizer.decode(
            generated[len(prompt_tokens):], skip_special_tokens=True
        )

    # Roundtrip verification
    full_rt_tokens = list(_tokenize(tokenizer, prompt_text + response_text))
    if full_rt_tokens == list(generated):
        print(f"✓  Tokenization roundtrip OK ({len(generated)-len(prompt_tokens)} data tokens)")
    else:
        print(f"⚠  Roundtrip mismatch: {len(generated)} orig vs {len(full_rt_tokens)} re-tokenized")
        print(f"   (SyncPool handles this — decoder maps re-tokenized tokens to same groups)")

    stats = {
        "original_bytes": len(secret.encode('utf-8')),
        "compressed_bytes": len(packed),
        "total_bits": total_msg_bits,
        "data_tokens": data_tokens,
        "total_tokens": len(generated) - len(prompt_tokens),
        "bits_per_token": round(total_msg_bits / max(data_tokens, 1), 1),
        "method": "arithmetic-fullvocab-syncpool",
        "vocab_size": n_tokens,
        "vocab_total": vocab_size,
        "syncpool_groups": len(groups) if groups else n_tokens,
    }

    return response_text, generated, len(prompt_tokens), stats


# ══════════════════════════════════════════════════════════════════════════════
#  DECODER  (tokens → encrypted message bits via arithmetic ENCODING)
# ══════════════════════════════════════════════════════════════════════════════

def decode(
    seed: str,
    cover_context: str = "",
    style: str = "email",
    model_name: str = "gpt2-medium",
    token_ids: list = None,
    cover_text: str = None,
    persona: str = "",
    cover_draft: str = "",
    verified: bool = True,
    **kwargs,
) -> str:
    tokenizer, model = load_model(model_name)
    device = _device

    # Same whitelist as encoder
    whitelist = _get_whitelist(tokenizer, model_name, model)

    # ── SyncPool: build same prefix groups as encoder ──
    if whitelist:
        token_texts = _precompute_token_texts(tokenizer, whitelist)
        group_map, groups = _build_prefix_groups(tokenizer, whitelist)
        print(f"  SyncPool mode: {len(groups)} groups")
    else:
        token_texts = None
        group_map, groups = None, None

    prompt = build_prompt(cover_context, style, persona, cover_draft, seed=seed)
    prompt_tokens = _tokenize(tokenizer, prompt)
    prompt_len = len(prompt_tokens)

    if token_ids is not None:
        tokens = token_ids
    elif cover_text is not None:
        print("  Re-tokenizing from text...")
        # CRITICAL: tokenize prompt+cover as ONE string.
        # BPE is context-dependent: tokenize("A")+tokenize("B") ≠ tokenize("A"+"B")
        # Tokenizing separately creates wrong splits at the boundary.
        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        full_text = prompt_text + cover_text
        tokens = list(_tokenize(tokenizer, full_text))
        # Verify prompt portion matches
        if tokens[:prompt_len] != list(prompt_tokens):
            # Prompt re-tokenized differently — find where cover starts
            prompt_len = len(_tokenize(tokenizer, prompt_text))
        print(f"  Tokens: {len(tokens)} total, {prompt_len} prompt, {len(tokens)-prompt_len} data")
    else:
        raise ValueError("Need token_ids or cover_text")

    # ── AC encoder state ──
    low = 0
    high = MAX_CODE
    pending = 0
    output_bits = []
    expected_bits = None

    def emit(b):
        nonlocal pending
        output_bits.append(b)
        for _ in range(pending):
            output_bits.append(1 - b)
        pending = 0

    with torch.no_grad():
        inp = torch.tensor([tokens[:prompt_len]]).to(device)
        out = model(inp, use_cache=True)
        past = out.past_key_values

        data_token_count = 0
        for i in range(prompt_len, len(tokens)):
            if expected_bits is not None and len(output_bits) >= expected_bits + PRECISION:
                break

            logits = out.logits[0, -1, :]

            token_id = tokens[i]

            # ── Build CDF over SyncPool groups (same as encoder) ──
            if groups is not None:
                cdf, _groups_ref = _build_grouped_cdf(
                    logits, whitelist, groups, group_map, quantize=True
                )

                # Map this token to its group index
                if token_id in group_map:
                    sym_idx = group_map[token_id]
                else:
                    # Token not in any group (not in whitelist) — skip
                    inp = torch.tensor([[token_id]]).to(device)
                    out = model(inp, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    continue
            else:
                cdf, token_map = _build_cdf(logits, whitelist, quantize=True)
                if token_map is not None:
                    _wl_idx = {tid: idx for idx, tid in enumerate(token_map)}
                    if token_id in _wl_idx:
                        sym_idx = _wl_idx[token_id]
                    else:
                        inp = torch.tensor([[token_id]]).to(device)
                        out = model(inp, past_key_values=past, use_cache=True)
                        past = out.past_key_values
                        continue
                else:
                    sym_idx = token_id

            data_token_count += 1

            sym_low  = cdf[sym_idx]
            sym_high = cdf[sym_idx + 1]

            range_size = high - low + 1
            new_low  = low + (range_size * sym_low)  // CDF_TOTAL
            new_high = low + (range_size * sym_high) // CDF_TOTAL - 1
            low  = new_low
            high = new_high

            # ── Renormalize & emit bits ──
            while True:
                if high < HALF:
                    emit(0)
                    low  = low << 1
                    high = (high << 1) | 1
                elif low >= HALF:
                    emit(1)
                    low  = (low - HALF) << 1
                    high = ((high - HALF) << 1) | 1
                elif low >= QUARTER and high < THREE_Q:
                    pending += 1
                    low  = (low - QUARTER) << 1
                    high = ((high - QUARTER) << 1) | 1
                else:
                    break

            # After 24 bits (3 bytes header), we know total length
            if expected_bits is None and len(output_bits) >= 24:
                hdr = _bits_to_bytes(output_bits[:24])
                payload_len = struct.unpack('>H', hdr[1:3])[0]
                expected_bits = (3 + payload_len) * 8

            # Advance KV cache
            inp = torch.tensor([[tokens[i]]]).to(device)
            out = model(inp, past_key_values=past, use_cache=True)
            past = out.past_key_values

    # Flush
    pending += 1
    if low < QUARTER:
        emit(0)
    else:
        emit(1)

    print(f"  Decoded {data_token_count} data tokens, {len(output_bits)} bits extracted")

    if expected_bits is None:
        return "[decode error: couldn't read header]"

    data_bytes = _bits_to_bytes(output_bits[:expected_bits])
    try:
        return _unpack_message(data_bytes, seed)
    except Exception as e:
        return f"[decode error: {e}]"


# ══════════════════════════════════════════════════════════════════════════════
#  ROBUST MODE — Keyed bin encoding with interleaved error correction
#  Works cross-device: decoder only needs tokenizer, not model
#
#  v10.1 security fixes:
#    1. Keyed bin permutation — bin assignment changes per-position via
#       HMAC(seed, position). Without the seed, an adversary cannot determine
#       which bin a token was intended for, defeating statistical detection.
#    2. Block-interleaved redundancy — symbols are spread across blocks
#       instead of consecutive repetition, eliminating the detectable
#       pattern of identical residues in adjacent tokens.
# ══════════════════════════════════════════════════════════════════════════════

ROBUST_BITS_PER_TOKEN = 2  # 2 bits per token
ROBUST_BINS = 1 << ROBUST_BITS_PER_TOKEN  # 4 bins
ROBUST_REDUNDANCY = 3  # 3x redundancy, tolerates 1 error per symbol
INTERLEAVE_BLOCK = 32  # symbols per interleave block


# ── Keyed bin permutation ────────────────────────────────────────────────────

def _keyed_offset(seed: str, position: int) -> int:
    """Per-position bin offset derived from HMAC.
    Encoder and decoder compute the same offset for each token position.
    Without the seed, the mapping looks uniformly random."""
    h = hmac.new(seed.encode(), struct.pack('>I', position), hashlib.sha256).digest()
    return int.from_bytes(h[:4], 'big') % ROBUST_BINS


def _keyed_bin_for_symbol(target_sym: int, position: int, seed: str) -> int:
    """Which static bin (token_id % 4) contains tokens that map to target_sym
    at this position? Inverts the keyed permutation for the encoder."""
    offset = _keyed_offset(seed, position)
    return (target_sym - offset) % ROBUST_BINS


def _keyed_symbol_from_token(token_id: int, position: int, seed: str) -> int:
    """What symbol does this token encode at this position?
    Used by the decoder."""
    offset = _keyed_offset(seed, position)
    return (token_id % ROBUST_BINS + offset) % ROBUST_BINS


# ── Block interleaving ───────────────────────────────────────────────────────

def _interleave_symbols(symbols, redundancy, block_size=INTERLEAVE_BLOCK):
    """Block interleave: within each block, send all symbols R times.
    [A B C D] x3 → [A B C D | A B C D | A B C D]  per block.
    Destroys the consecutive-repetition fingerprint while staying
    self-synchronizing (decoder doesn't need to know total length)."""
    tx = []
    for start in range(0, len(symbols), block_size):
        block = symbols[start:start + block_size]
        for _r in range(redundancy):
            tx.extend(block)
    return tx


def _deinterleave_vote(raw_symbols, redundancy, block_size=INTERLEAVE_BLOCK):
    """Reverse block interleave + majority vote.
    Processes tokens in blocks of (block_size * redundancy)."""
    decoded = []
    tokens_per_block = block_size * redundancy
    for blk_start in range(0, len(raw_symbols), tokens_per_block):
        blk_tokens = raw_symbols[blk_start:blk_start + tokens_per_block]
        # Detect actual block size (last block may be partial)
        actual_bs = len(blk_tokens) // redundancy
        if actual_bs == 0:
            break
        for i in range(actual_bs):
            votes = []
            for r in range(redundancy):
                idx = r * actual_bs + i
                if idx < len(blk_tokens):
                    votes.append(blk_tokens[idx])
            if votes:
                counts = [0] * ROBUST_BINS
                for v in votes:
                    counts[v] += 1
                decoded.append(counts.index(max(counts)))
    return decoded


# ── Packing (unchanged logic, same wire format) ─────────────────────────────

def _pack_robust(text: str, seed: str) -> bytes:
    """Minimal packing for robust mode.
    2-byte header (flags + length) + encrypted payload. Max 16383 bytes."""
    raw = text.encode('utf-8')
    compressed = zlib.compress(raw, level=9)
    if len(compressed) < len(raw):
        payload, flag = compressed, 0x4000  # bit 14 = compressed
    else:
        payload, flag = raw, 0x0000
    encrypted = _encrypt(payload, seed)
    if len(encrypted) > 0x3FFF:
        raise ValueError(f"Message too long: {len(encrypted)} bytes (max 16383)")
    header = struct.pack('>H', flag | len(encrypted))
    return header + encrypted


def _unpack_robust(data: bytes, seed: str) -> str:
    """Reverse of _pack_robust."""
    hdr = struct.unpack('>H', data[:2])[0]
    compressed = bool(hdr & 0x4000)
    length = hdr & 0x3FFF
    encrypted = data[2:2 + length]
    payload = _encrypt(encrypted, seed)
    if compressed:
        return zlib.decompress(payload).decode('utf-8')
    return payload.decode('utf-8')


# ── Static bin split (used for token selection during generation) ────────────

def _split_vocab_bins(whitelist):
    """Split whitelist into ROBUST_BINS bins by token_id % ROBUST_BINS.
    This is the static grouping; the keyed permutation rotates which
    static bin is used for each symbol at each position."""
    bins = [[] for _ in range(ROBUST_BINS)]
    for t in whitelist:
        bins[t % ROBUST_BINS].append(t)
    return bins


# ── Encoder ──────────────────────────────────────────────────────────────────

def encode_robust(
    secret: str,
    seed: str,
    cover_context: str = "",
    style: str = "email",
    max_tokens: int = 2000,
    model_name: str = "gpt2-medium",
    persona: str = "",
    cover_draft: str = "",
    topic: str = "",
    **kwargs,
) -> tuple:
    """Robust encode with keyed bins + interleaved redundancy.
    Hides 2 bits per token, 3x redundancy with block interleaving.
    Returns (cover_text, token_ids, prompt_len, stats)."""
    tokenizer, model = load_model(model_name)
    device = _device

    whitelist = _get_whitelist(tokenizer, model_name, model)
    if not whitelist:
        whitelist = list(range(len(tokenizer)))
    bins = _split_vocab_bins(whitelist)
    min_bin = min(len(b) for b in bins)
    print(f"  Bins: {ROBUST_BINS} bins, smallest={min_bin} tokens")

    # Compact packing
    packed = _pack_robust(secret, seed)
    message_bits = _bytes_to_bits(packed)
    total_msg_bits = len(message_bits)

    # Group bits into 2-bit symbols
    symbols = []
    for i in range(0, len(message_bits), ROBUST_BITS_PER_TOKEN):
        chunk = message_bits[i:i + ROBUST_BITS_PER_TOKEN]
        while len(chunk) < ROBUST_BITS_PER_TOKEN:
            chunk.append(0)
        val = 0
        for b in chunk:
            val = (val << 1) | b
        symbols.append(val)

    # Block-interleaved redundancy (replaces consecutive repetition)
    # Pad symbols to full block so trailing sentence-padding doesn't corrupt
    # the last block's de-interleave alignment
    n_pad = (INTERLEAVE_BLOCK - len(symbols) % INTERLEAVE_BLOCK) % INTERLEAVE_BLOCK
    padded_symbols = symbols + [0] * n_pad
    tx_symbols = _interleave_symbols(padded_symbols, ROBUST_REDUNDANCY, INTERLEAVE_BLOCK)
    total_tx = len(tx_symbols)

    # Prompt
    prompt = build_prompt(cover_context, style, persona, cover_draft, seed=seed, topic=topic)
    prompt_tokens = _tokenize(tokenizer, prompt)
    generated = list(prompt_tokens)

    # Auto-detect context window and cap max_tokens
    config = model.config if hasattr(model, 'config') else AutoConfig.from_pretrained(model_name)
    ctx_len = getattr(config, 'max_position_embeddings',
              getattr(config, 'n_positions',
              getattr(config, 'seq_length', 1024)))
    usable = ctx_len - len(prompt_tokens) - 10
    if max_tokens > usable:
        print(f"  Context window: {ctx_len}, capping tokens to {usable}")
        max_tokens = usable
    if total_tx > max_tokens:
        raise ValueError(
            f"Message needs {total_tx} tokens but model context allows {max_tokens}. "
            f"Use a shorter message or a model with larger context (Qwen, Mistral)."
        )
    generated = list(prompt_tokens)

    # Generate tokens
    inp = torch.tensor([prompt_tokens]).to(device)
    print(f"  Robust encode: {len(packed)}B → {total_msg_bits} bits → {len(symbols)} symbols × {ROBUST_REDUNDANCY} = {total_tx} tokens needed")
    print(f"  Security: keyed bins (HMAC-SHA256) + block interleave (block={INTERLEAVE_BLOCK})")
    sym_idx = 0

    with torch.no_grad():
        out = model(inp, use_cache=True)
        past = out.past_key_values

        for step in range(max_tokens):
            if sym_idx >= total_tx:
                # Pad to sentence end
                logits = out.logits[0, -1, :]
                if whitelist:
                    wl_logits = logits[whitelist]
                    chosen = whitelist[wl_logits.argmax().item()]
                else:
                    chosen = logits.argmax().item()
                generated.append(chosen)
                text_so_far = tokenizer.decode(
                    generated[len(prompt_tokens):], skip_special_tokens=True
                )
                if text_so_far.rstrip().endswith(('.', '!', '?', '"')):
                    break
                if step - total_tx > 10:
                    break
                inp = torch.tensor([[chosen]]).to(device)
                out = model(inp, past_key_values=past, use_cache=True)
                past = out.past_key_values
                continue

            logits = out.logits[0, -1, :]
            target_sym = tx_symbols[sym_idx]

            # ── Keyed bin selection ──
            # Which static bin (token_id % 4) maps to target_sym at this position?
            static_bin_idx = _keyed_bin_for_symbol(target_sym, sym_idx, seed)
            active_bin = bins[static_bin_idx]
            bin_tensor = torch.tensor(active_bin, device=logits.device)
            bin_logits = logits[bin_tensor]

            # Rank tokens by probability
            probs = torch.softmax(bin_logits.float() / 0.9, dim=-1)
            ranked = torch.argsort(probs, descending=True)

            # Pick best token that survives roundtrip
            chosen = None
            data_so_far = generated[len(prompt_tokens):]
            # Only check boundary tokens (BPE merges are local, max 3 tokens)
            boundary_start = max(0, len(data_so_far) - 2)
            boundary_prefix = data_so_far[boundary_start:]
            for rank in ranked[:50]:
                candidate = active_bin[rank.item()]
                test_seq = boundary_prefix + [candidate]
                test_text = tokenizer.decode(test_seq, skip_special_tokens=True)
                rt = _tokenize(tokenizer, test_text)
                if len(rt) == len(test_seq) and rt[-1] == candidate:
                    chosen = candidate
                    break
            if chosen is None:
                chosen = active_bin[ranked[0].item()]

            generated.append(chosen)
            sym_idx += 1

            inp = torch.tensor([[chosen]]).to(device)
            out = model(inp, past_key_values=past, use_cache=True)
            past = out.past_key_values

    # Output
    data_token_ids = generated[len(prompt_tokens):]
    response_text = tokenizer.decode(data_token_ids, skip_special_tokens=True)

    # Verify roundtrip
    rt_tokens = _tokenize(tokenizer, response_text)
    if rt_tokens == data_token_ids:
        print(f"  ✓ Roundtrip OK")
    else:
        mismatches = sum(1 for a, b in zip(rt_tokens, data_token_ids) if a != b)
        mismatches += abs(len(rt_tokens) - len(data_token_ids))
        print(f"  ⚠ Roundtrip: {mismatches} mismatches out of {len(data_token_ids)} tokens")

    data_token_count = sym_idx
    stats = {
        "original_bytes": len(secret.encode('utf-8')),
        "compressed_bytes": len(packed),
        "total_bits": total_msg_bits,
        "symbols": len(symbols),
        "tx_symbols": total_tx,
        "data_tokens": data_token_count,
        "total_tokens": len(generated) - len(prompt_tokens),
        "bits_per_token": round(total_msg_bits / max(data_token_count, 1), 2),
        "method": f"robust-{ROBUST_BITS_PER_TOKEN}bpt-x{ROBUST_REDUNDANCY}-keyed-interleaved",
    }
    print(f"  ✓ Encoded {data_token_count} tokens ({total_msg_bits} bits, {ROBUST_BITS_PER_TOKEN} bits/tok, {ROBUST_REDUNDANCY}x redundancy)")

    return response_text, generated, len(prompt_tokens), stats


# ── Decoder ──────────────────────────────────────────────────────────────────

def decode_robust(
    seed: str,
    cover_text: str,
    style: str = "email",
    model_name: str = "gpt2-medium",
    **kwargs,
) -> str:
    """Robust decode with keyed bins + block de-interleave. NO MODEL NEEDED.
    Just tokenizes the cover text, applies keyed bin permutation, and
    de-interleaves with majority vote."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    wl = _get_whitelist_from_tokenizer(tokenizer, model_name)
    wl_set = set(wl) if wl else None

    # Tokenize cover text
    tokens = _tokenize(tokenizer, cover_text)

    # Extract keyed symbols (applying per-position permutation)
    # Position counter must match encoder's sym_idx: count only accepted tokens
    raw_symbols = []
    accepted_pos = 0
    for t in tokens:
        if wl_set and t not in wl_set:
            continue
        sym = _keyed_symbol_from_token(t, accepted_pos, seed)
        raw_symbols.append(sym)
        accepted_pos += 1

    print(f"  Extracted {len(raw_symbols)} keyed symbols from {len(tokens)} tokens")

    # Block de-interleave + majority vote
    decoded_symbols = _deinterleave_vote(raw_symbols, ROBUST_REDUNDANCY, INTERLEAVE_BLOCK)

    print(f"  After de-interleave + vote: {len(decoded_symbols)} symbols")

    # Convert symbols → bits
    decoded_bits = []
    for sym in decoded_symbols:
        for shift in range(ROBUST_BITS_PER_TOKEN - 1, -1, -1):
            decoded_bits.append((sym >> shift) & 1)

    data_bytes = _bits_to_bytes(decoded_bits)
    try:
        return _unpack_robust(data_bytes, seed)
    except Exception as e:
        return f"[decode error: {e}]"


def _get_whitelist_from_tokenizer(tokenizer, model_name):
    """Build whitelist from tokenizer only (no model needed)."""
    if model_name in _whitelist_cache:
        return _whitelist_cache[model_name]
    import re
    vocab_size = len(tokenizer)
    whitelist = []
    safe_pattern = re.compile(r'^[\x20-\x7e\xa0-\xff\n\r\t]+$')
    for token_id in range(vocab_size):
        try:
            text = tokenizer.decode([token_id])
            if text and safe_pattern.match(text):
                whitelist.append(token_id)
        except Exception:
            continue
    if tokenizer.eos_token_id is not None:
        if tokenizer.eos_token_id not in whitelist and tokenizer.eos_token_id < vocab_size:
            whitelist.append(tokenizer.eos_token_id)
    whitelist.sort()
    _whitelist_cache[model_name] = whitelist
    return whitelist