"""
Neural Steganography — Core Engine
====================================

Hides encrypted messages inside AI-generated text using robust mode:

  5 bits/token (32 bins, 1× redundancy), only needs the tokenizer
  for decoding (no GPU required on receiver side).
  Context-dependent bin permutation (KGW-style): bin assignment
  depends on (seed, prev_token, position) via HMAC-SHA256 →
  Fisher-Yates shuffle of 32 bins. No static frequency fingerprint.

Based on:
  - Ziegler et al., "Neural Linguistic Steganography" (Harvard, 2019)
  - Kirchenbauer et al., "A Watermark for Large Language Models" (2023)
    — KGW-style context-dependent hashing (adapted for steganography)
"""

import gc
import hashlib
import hmac
import logging
import os
import re
import struct

import torch
import zstandard as zstd
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

ROBUST_BITS_PER_TOKEN = 5
ROBUST_BINS           = 1 << ROBUST_BITS_PER_TOKEN  # 32
ROBUST_REDUNDANCY     = 1
INTERLEAVE_BLOCK      = 32

DEFAULT_MODEL = "gpt2-medium"

_SAFE_TOKEN_PATTERN = re.compile(r"^[\x20-\x7e\xa0-\xff\n\r\t]+$")

# Knuth multiplicative hash constant for bin assignment
_BIN_HASH_MULT = 2654435761


# ═══════════════════════════════════════════════════════════════════════════════
#  Encryption (stateless)
# ═══════════════════════════════════════════════════════════════════════════════

def _derive_key(seed: str) -> bytes:
    """Derive a 256-bit key from a seed string using HKDF-SHA256."""
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"neural-stego-v2",
        info=b"chacha20-poly1305",
    ).derive(seed.encode("utf-8"))


def _encrypt(data: bytes, seed: str) -> bytes:
    """Encrypt with ChaCha20-Poly1305 (AEAD). Returns nonce || ciphertext+tag."""
    key = _derive_key(seed)
    nonce = os.urandom(12)
    ct = ChaCha20Poly1305(key).encrypt(nonce, data, None)
    return nonce + ct


def _decrypt(data: bytes, seed: str) -> bytes:
    """Decrypt ChaCha20-Poly1305. Input: nonce || ciphertext+tag."""
    if len(data) < 12 + 16:
        raise ValueError("Ciphertext too short (need at least nonce + tag)")
    key = _derive_key(seed)
    nonce, ct = data[:12], data[12:]
    return ChaCha20Poly1305(key).decrypt(nonce, ct, None)


# ═══════════════════════════════════════════════════════════════════════════════
#  Robust mode message framing (zstd)
# ═══════════════════════════════════════════════════════════════════════════════

_zstd_compressor = zstd.ZstdCompressor(level=19)
_zstd_decompressor = zstd.ZstdDecompressor()


def _pack_robust(text: str, seed: str) -> bytes:
    """Compact framing for robust mode. Max payload: 16383 bytes.

    Wire format: [2B header (flags | length)] [encrypted payload]
    Bit 14 of header indicates zstd compression.
    """
    raw = text.encode("utf-8")
    compressed = _zstd_compressor.compress(raw)
    if len(compressed) < len(raw):
        payload, flag = compressed, 0x4000
    else:
        payload, flag = raw, 0x0000
    encrypted = _encrypt(payload, seed)
    if len(encrypted) > 0x3FFF:
        raise ValueError(f"Message too long: {len(encrypted)} bytes (max 16383)")
    return struct.pack(">H", flag | len(encrypted)) + encrypted


def _unpack_robust(data: bytes, seed: str) -> str:
    """Reverse of _pack_robust."""
    if len(data) < 2:
        raise ValueError(f"Robust frame too short: {len(data)} bytes (need >= 2)")
    hdr = struct.unpack(">H", data[:2])[0]
    compressed = bool(hdr & 0x4000)
    length = hdr & 0x3FFF
    if len(data) < 2 + length:
        raise ValueError(
            f"Robust frame truncated: header says {length} bytes but only "
            f"{len(data) - 2} available"
        )
    payload = _decrypt(data[2:2 + length], seed)
    if compressed:
        return _zstd_decompressor.decompress(payload).decode("utf-8")
    return payload.decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
#  Bit/Byte Conversion (stateless)
# ═══════════════════════════════════════════════════════════════════════════════

def _bytes_to_bits(data: bytes) -> list:
    """Convert bytes to a list of individual bits (MSB first)."""
    return [(b >> (7 - i)) & 1 for b in data for i in range(8)]


def _bits_to_bytes(bits: list) -> bytes:
    """Convert a list of bits back to bytes, zero-padding the last byte."""
    bits = bits + [0] * (-len(bits) % 8)
    return bytes(
        int("".join(str(b) for b in bits[i:i + 8]), 2)
        for i in range(0, len(bits), 8)
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Tokenization Helpers (stateless)
# ═══════════════════════════════════════════════════════════════════════════════

def _tokenize(tokenizer, text: str) -> list:
    """Tokenize text without adding special tokens."""
    return tokenizer.encode(text, add_special_tokens=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  Prompt Construction (stateless)
# ═══════════════════════════════════════════════════════════════════════════════

def build_prompt(seed: str = "", topic: str = "") -> str:
    """Build a deterministic prompt from seed and optional topic."""
    if topic:
        return f"{topic.strip()}\n\n"
    if seed:
        h = hashlib.sha256(seed.encode()).hexdigest()[:8]
        return f"Post {h}:\n\n"
    return "Post:\n\n"


# ═══════════════════════════════════════════════════════════════════════════════
#  Bin Assignment & Context-Dependent Permutation (stateless)
# ═══════════════════════════════════════════════════════════════════════════════

def _token_to_bin(token_id):
    """Map token ID to static bin using multiplicative hashing.

    Much better distribution than bare modulo — virtually eliminates
    empty bins even with unusual tokenizer vocabularies.
    """
    return ((token_id * _BIN_HASH_MULT) >> 16) % ROBUST_BINS


def _step_permutation(seed: str, position: int, prev_token: int) -> list:
    """Full permutation of [0..15] seeded by (seed, prev_token, position).

    Fisher-Yates shuffle driven by HMAC-SHA256 bytes.
    Returns perm where perm[static_bin] = symbol.
    """
    data = struct.pack(">II", position, prev_token)
    h = hmac.new(seed.encode(), data, hashlib.sha256).digest()

    perm = list(range(ROBUST_BINS))
    for i in range(ROBUST_BINS - 1, 0, -1):
        j = h[ROBUST_BINS - 1 - i] % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def _step_inv_permutation(perm: list) -> list:
    """Inverse of a permutation: inv[perm[i]] = i."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def _symbol_from_token(token_id: int, position: int, prev_token: int,
                       seed: str) -> int:
    """Decoder: what symbol does this token encode at this position?"""
    perm = _step_permutation(seed, position, prev_token)
    return perm[_token_to_bin(token_id)]


def _bin_for_symbol(target_sym: int, position: int, prev_token: int,
                    seed: str) -> int:
    """Encoder: which static bin (by hash) holds target_sym?"""
    perm = _step_permutation(seed, position, prev_token)
    inv = _step_inv_permutation(perm)
    return inv[target_sym]


def _split_vocab_bins(whitelist):
    """Split whitelist into 16 static bins by multiplicative hash.

    Uses _token_to_bin for even distribution.
    Raises ValueError only if the whitelist is too small to fill all bins.
    """
    bins = [[] for _ in range(ROBUST_BINS)]
    for t in whitelist:
        bins[_token_to_bin(t)].append(t)
    empty = [i for i, b in enumerate(bins) if not b]
    if empty:
        raise ValueError(
            f"Empty bins {empty} — whitelist ({len(whitelist)} tokens) "
            f"cannot fill all {ROBUST_BINS} bins. "
            f"This tokenizer may not be compatible with robust mode."
        )
    return bins


# ═══════════════════════════════════════════════════════════════════════════════
#  Block Interleaving (stateless)
# ═══════════════════════════════════════════════════════════════════════════════

def _interleave_symbols(symbols, redundancy, block_size=INTERLEAVE_BLOCK):
    """Block interleave: repeat each block R times."""
    tx = []
    for start in range(0, len(symbols), block_size):
        block = symbols[start:start + block_size]
        for _ in range(redundancy):
            tx.extend(block)
    return tx


def _deinterleave_vote(raw_symbols, redundancy, block_size=INTERLEAVE_BLOCK):
    """Reverse block interleave and apply majority vote."""
    decoded = []
    tokens_per_block = block_size * redundancy

    for blk_start in range(0, len(raw_symbols), tokens_per_block):
        blk_tokens = raw_symbols[blk_start:blk_start + tokens_per_block]
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


# ═══════════════════════════════════════════════════════════════════════════════
#  StegoEngine — Stateful core holding model, tokenizer, and caches
# ═══════════════════════════════════════════════════════════════════════════════

class StegoEngine:
    """Encapsulates model state and caches for neural steganography."""

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._loaded_model_name = None
        self._loaded_dtype = None
        self._device = None
        self._whitelist_cache = {}

    # ── Device management ─────────────────────────────────────────────────

    @staticmethod
    def get_device() -> torch.device:
        """Select the best available compute device (MPS > CUDA > CPU)."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # ── Model loading ─────────────────────────────────────────────────────

    def load_model(self, model_name: str = None, dtype=None):
        """Load a causal language model and its tokenizer, with caching."""
        model_name = model_name or DEFAULT_MODEL
        dtype = dtype or torch.float16

        if (self._tokenizer is not None
                and self._loaded_model_name == model_name
                and self._loaded_dtype == dtype):
            return self._tokenizer, self._model

        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._device = self.get_device()
        log.info("Loading %s (%s) on %s...", model_name, dtype, self._device)

        config = AutoConfig.from_pretrained(model_name)
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = self._tokenizer.eos_token_id

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, torch_dtype=dtype
        ).to(self._device)
        self._model.eval()

        self._loaded_model_name = model_name
        self._loaded_dtype = dtype
        log.info("Model ready on %s.", self._device)
        return self._tokenizer, self._model

    @staticmethod
    def _get_max_context(model) -> int:
        """Read the model's maximum context window from its config."""
        cfg = model.config
        return getattr(cfg, "max_position_embeddings",
               getattr(cfg, "n_positions",
               getattr(cfg, "seq_length", 1024)))

    # ── Whitelist construction ────────────────────────────────────────────

    def _get_whitelist(self, tokenizer, model_name, model=None):
        """Return sorted list of token IDs that decode to ASCII/Latin-1 text."""
        if model_name in self._whitelist_cache:
            return self._whitelist_cache[model_name]

        vocab_size = len(tokenizer)

        if model is not None:
            try:
                lm_head = getattr(model, "lm_head", None)
                if lm_head is not None:
                    head_size = (lm_head.out_features if hasattr(lm_head, "out_features")
                                 else lm_head.weight.shape[0])
                    vocab_size = min(vocab_size, head_size)
            except Exception:
                pass

        whitelist = []
        for token_id in range(vocab_size):
            try:
                text = tokenizer.decode([token_id])
                if text and _SAFE_TOKEN_PATTERN.match(text):
                    whitelist.append(token_id)
            except Exception:
                continue

        if (tokenizer.eos_token_id is not None
                and tokenizer.eos_token_id not in whitelist
                and tokenizer.eos_token_id < vocab_size):
            whitelist.append(tokenizer.eos_token_id)

        whitelist.sort()
        self._whitelist_cache[model_name] = whitelist
        log.info("Whitelist: %d/%d tokens (%d%% of vocab)",
                 len(whitelist), vocab_size,
                 100 * len(whitelist) // vocab_size)
        return whitelist

    def _get_whitelist_from_tokenizer(self, tokenizer, model_name):
        """Build whitelist from tokenizer only (no model needed). Used by decoder."""
        if model_name in self._whitelist_cache:
            return self._whitelist_cache[model_name]

        vocab_size = len(tokenizer)
        whitelist = []
        for token_id in range(vocab_size):
            try:
                text = tokenizer.decode([token_id])
                if text and _SAFE_TOKEN_PATTERN.match(text):
                    whitelist.append(token_id)
            except Exception:
                continue

        if (tokenizer.eos_token_id is not None
                and tokenizer.eos_token_id not in whitelist
                and tokenizer.eos_token_id < vocab_size):
            whitelist.append(tokenizer.eos_token_id)

        whitelist.sort()
        self._whitelist_cache[model_name] = whitelist
        return whitelist

    # ══════════════════════════════════════════════════════════════════════
    #  Robust Mode — Encoder
    # ══════════════════════════════════════════════════════════════════════

    def encode_robust(
        self,
        secret: str,
        seed: str,
        max_tokens: int = 1500,
        model_name: str = "gpt2-medium",
        topic: str = "",
        **kwargs,
    ) -> tuple:
        """Encode using robust mode: 5 bits/token with 32 bins.

        Context-dependent bin permutation (KGW-style).
        Returns (cover_text, token_ids, prompt_length, stats_dict).
        """
        tokenizer, model = self.load_model(model_name)
        device = self._device

        whitelist = self._get_whitelist(tokenizer, model_name, model)
        if not whitelist:
            whitelist = list(range(len(tokenizer)))
        bins = _split_vocab_bins(whitelist)

        packed = _pack_robust(secret, seed)
        message_bits = _bytes_to_bits(packed)
        total_msg_bits = len(message_bits)

        # Group bits into 5-bit symbols
        symbols = []
        for i in range(0, len(message_bits), ROBUST_BITS_PER_TOKEN):
            chunk = message_bits[i:i + ROBUST_BITS_PER_TOKEN]
            while len(chunk) < ROBUST_BITS_PER_TOKEN:
                chunk.append(0)
            val = 0
            for b in chunk:
                val = (val << 1) | b
            symbols.append(val)

        # Block-interleaved redundancy
        n_pad = (INTERLEAVE_BLOCK - len(symbols) % INTERLEAVE_BLOCK) % INTERLEAVE_BLOCK
        padded_symbols = symbols + [0] * n_pad
        tx_symbols = _interleave_symbols(padded_symbols, ROBUST_REDUNDANCY, INTERLEAVE_BLOCK)
        total_tx = len(tx_symbols)

        prompt = build_prompt(seed=seed, topic=topic)
        prompt_tokens = _tokenize(tokenizer, prompt)
        generated = list(prompt_tokens)

        ctx_len = self._get_max_context(model)
        usable = ctx_len - len(prompt_tokens) - 10
        max_tokens = min(max_tokens, usable, total_tx + 30)
        if total_tx > usable:
            raise ValueError(
                f"Message needs {total_tx} tokens but model context allows {usable}. "
                f"Use a shorter message or a model with larger context."
            )

        log.info("Robust: %d B -> %d bits -> %d symbols -> %d tokens needed",
                 len(packed), total_msg_bits, len(symbols), total_tx)

        inp = torch.tensor([prompt_tokens]).to(device)
        sym_idx = 0

        with torch.no_grad():
            out = model(inp, use_cache=True)
            past = out.past_key_values

            for step in range(max_tokens):
                if sym_idx >= total_tx:
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
                    if text_so_far.rstrip().endswith((".", "!", "?", '"')):
                        break
                    if step - total_tx > 10:
                        break
                    inp = torch.tensor([[chosen]]).to(device)
                    out = model(inp, past_key_values=past, use_cache=True)
                    past = out.past_key_values
                    continue

                logits = out.logits[0, -1, :]

                # Repetition penalty — dampen tokens already generated
                # so the output doesn't degrade into loops
                if generated[len(prompt_tokens):]:
                    seen = torch.tensor(
                        list(set(generated[len(prompt_tokens):])),
                        device=logits.device,
                    )
                    penalties = logits[seen]
                    logits[seen] = torch.where(
                        penalties > 0, penalties / 1.3, penalties * 1.3
                    )

                target_sym = tx_symbols[sym_idx]

                # Context-dependent bin selection
                prev_token = generated[-1] if len(generated) > len(prompt_tokens) else 0
                static_bin_idx = _bin_for_symbol(target_sym, sym_idx, prev_token, seed)
                active_bin = bins[static_bin_idx]
                bin_tensor = torch.tensor(active_bin, device=logits.device)
                bin_logits = logits[bin_tensor]

                # Top-p (nucleus) sampling within the bin:
                # only keep the most probable tokens covering 92% of mass,
                # then temperature-sample from those. Prevents picking
                # extremely unlikely tokens when the bin is large.
                temperature = 0.7
                scaled = bin_logits.float() / temperature
                probs = torch.softmax(scaled, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                cutoff = (cumulative - sorted_probs) > 0.92
                sorted_probs[cutoff] = 0.0
                sorted_probs /= sorted_probs.sum()

                # Sample from the nucleus
                n_candidates = min(50, int((~cutoff).sum().item()))
                n_candidates = max(n_candidates, 1)
                sampled = torch.multinomial(sorted_probs, n_candidates, replacement=False)
                # Map back to bin indices
                sample_order = sorted_idx[sampled]

                chosen = None
                data_so_far = generated[len(prompt_tokens):]
                boundary_start = max(0, len(data_so_far) - 2)
                boundary_prefix = data_so_far[boundary_start:]

                for idx in sample_order:
                    candidate = active_bin[idx.item()]
                    test_seq = boundary_prefix + [candidate]
                    test_text = tokenizer.decode(test_seq, skip_special_tokens=True)
                    rt = _tokenize(tokenizer, test_text)
                    if len(rt) == len(test_seq) and rt[-1] == candidate:
                        chosen = candidate
                        break

                if chosen is None:
                    # Fallback: highest-prob BPE-safe token (no randomness)
                    for idx in sorted_idx:
                        candidate = active_bin[idx.item()]
                        test_seq = boundary_prefix + [candidate]
                        test_text = tokenizer.decode(test_seq, skip_special_tokens=True)
                        rt = _tokenize(tokenizer, test_text)
                        if len(rt) == len(test_seq) and rt[-1] == candidate:
                            chosen = candidate
                            break

                if chosen is None:
                    chosen = active_bin[sorted_idx[0].item()]

                generated.append(chosen)
                sym_idx += 1

                inp = torch.tensor([[chosen]]).to(device)
                out = model(inp, past_key_values=past, use_cache=True)
                past = out.past_key_values

        data_token_ids = generated[len(prompt_tokens):]
        response_text = tokenizer.decode(data_token_ids, skip_special_tokens=True)

        rt_tokens = _tokenize(tokenizer, response_text)
        if rt_tokens != data_token_ids:
            mismatches = sum(1 for a, b in zip(rt_tokens, data_token_ids) if a != b)
            mismatches += abs(len(rt_tokens) - len(data_token_ids))
            log.warning("Roundtrip: %d mismatches out of %d tokens",
                        mismatches, len(data_token_ids))

        stats = {
            "original_bytes": len(secret.encode("utf-8")),
            "compressed_bytes": len(packed),
            "total_bits": total_msg_bits,
            "symbols": len(symbols),
            "tx_symbols": total_tx,
            "data_tokens": sym_idx,
            "total_tokens": len(generated) - len(prompt_tokens),
            "bits_per_token": round(total_msg_bits / max(sym_idx, 1), 2),
            "effective_bpt": round(total_msg_bits / max(sym_idx, 1), 2),
            "method": f"robust-{ROBUST_BITS_PER_TOKEN}bpt-x{ROBUST_REDUNDANCY}-ctx-perm",
        }
        log.info("Encoded %d tokens (%d bits, ~%s effective bpt)",
                 sym_idx, total_msg_bits, stats['effective_bpt'])

        return response_text, generated, len(prompt_tokens), stats

    # ══════════════════════════════════════════════════════════════════════
    #  Robust Mode — Decoder
    # ══════════════════════════════════════════════════════════════════════

    def decode_robust(
        self,
        seed: str,
        cover_text: str,
        model_name: str = "gpt2-medium",
        **kwargs,
    ) -> str:
        """Decode a message from robust-mode cover text.

        Only loads the tokenizer — no model or GPU required.
        Context-dependent permutation using (seed, prev_token, position).
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        wl = self._get_whitelist_from_tokenizer(tokenizer, model_name)
        wl_set = set(wl) if wl else None

        tokens = _tokenize(tokenizer, cover_text)

        # Extract symbols using context-dependent permutation
        raw_symbols = []
        accepted_pos = 0
        prev_token = 0

        for t in tokens:
            if wl_set and t not in wl_set:
                continue
            sym = _symbol_from_token(t, accepted_pos, prev_token, seed)
            raw_symbols.append(sym)
            prev_token = t
            accepted_pos += 1

        decoded_symbols = _deinterleave_vote(raw_symbols, ROBUST_REDUNDANCY, INTERLEAVE_BLOCK)

        decoded_bits = []
        for sym in decoded_symbols:
            for shift in range(ROBUST_BITS_PER_TOKEN - 1, -1, -1):
                decoded_bits.append((sym >> shift) & 1)

        data_bytes = _bits_to_bytes(decoded_bits)
        try:
            return _unpack_robust(data_bytes, seed)
        except Exception as e:
            return f"[decode error: {e}]"


# ═══════════════════════════════════════════════════════════════════════════════
#  Default instance & module-level convenience wrappers
# ═══════════════════════════════════════════════════════════════════════════════

_default_engine = StegoEngine()

get_device = StegoEngine.get_device
load_model = _default_engine.load_model
encode_robust = _default_engine.encode_robust
decode_robust = _default_engine.decode_robust
