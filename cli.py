#!/usr/bin/env python3
"""
Neural Steganography — Interactive CLI v10
"""

import logging
import sys, os

logging.basicConfig(level=logging.INFO, format="  %(message)s")

MODELS = [
    ("gpt2",           "GPT-2 Small (124M) — fastest, ~0.5GB"),
    ("gpt2-medium",    "GPT-2 Medium (355M) — balanced, ~1.4GB"),
    ("gpt2-large",     "GPT-2 Large (774M) — better text, ~3GB"),
    ("gpt2-xl",        "GPT-2 XL (1.5B) — high quality, ~6GB"),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama 1.1B — chat model, ~4.4GB"),
    ("Qwen/Qwen2.5-0.5B", "Qwen 2.5 0.5B — multilingual, ~2GB"),
    ("Qwen/Qwen2.5-1.5B", "Qwen 2.5 1.5B — multilingual, ~6GB"),
    ("Qwen/Qwen2.5-3B",   "Qwen 2.5 3B — high quality, ~12GB"),
    ("Qwen/Qwen2.5-7B",   "Qwen 2.5 7B — best quality, ~28GB"),
    ("mistralai/Mistral-7B-v0.1", "Mistral 7B — excellent text, ~28GB"),
]

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def banner():
    print("""
  ╔═══════════════════════════════════════╗
  ║     🔐  Neural Steganography v10     ║
  ║   Hide secrets in plain text         ║
  ╚═══════════════════════════════════════╝
    """)

def prompt_choice(question, options):
    print(f"\n  {question}\n")
    for i, opt in enumerate(options, 1):
        print(f"    [{i}] {opt}")
    while True:
        try:
            pick = int(input("\n  → "))
            if 1 <= pick <= len(options):
                return pick
        except (ValueError, EOFError):
            pass
        print("    Invalid choice, try again.")

def pick_model():
    print("\n  ── MODEL SELECT ──────────────────────\n")
    for i, (name, desc) in enumerate(MODELS, 1):
        print(f"    [{i}] {desc}")
    print(f"\n    Bigger = better text, slower, more RAM")
    print(f"    Encoder needs the model, decoder does NOT\n")
    while True:
        try:
            raw = input("  Model [1]: ").strip() or "1"
            pick = int(raw)
            if 1 <= pick <= len(MODELS):
                name, desc = MODELS[pick - 1]
                print(f"  ✓ {desc}")
                return name
        except (ValueError, EOFError):
            pass
        print("    Invalid choice.")

def do_encode():
    print("\n  ── ENCODE ─────────────────────────────\n")

    secret = input("  Secret message: ").strip()
    if not secret:
        print("  ✗ No message entered."); return

    password = input("  Password: ").strip()
    if not password:
        print("  ✗ No password entered."); return

    model = pick_model()

    print("\n  ── TOPIC (steers the cover text) ──────\n")
    print("    Examples:")
    print('    • "Write a blog post about cooking Italian food"')
    print('    • "Dear hiring manager, I am writing to apply"')
    print('    • "The latest developments in quantum computing"')
    print('    • (blank = random topic)\n')
    topic = input("  Topic: ").strip()

    print(f"\n  Generating cover text...\n")

    from stego import encode_robust
    try:
        cover, token_ids, prompt_len, stats = encode_robust(
            secret=secret, seed=password,
            model_name=model, topic=topic,
        )
    except ValueError as e:
        print(f"\n  ✗ {e}")
        return

    print("\n  ═══════════════════════════════════════")
    print("  📄  COVER TEXT (copy this):\n")
    print(cover)
    print("\n  ═══════════════════════════════════════")
    print(f"  📊  {stats['original_bytes']}B message → {stats['data_tokens']} tokens")
    print(f"      Method: {stats['method']}")

    save = input("\n  Save to file? [y/N]: ").strip().lower()
    if save == 'y':
        fname = input("  Filename [cover.txt]: ").strip() or "cover.txt"
        with open(fname, 'w') as f:
            f.write(cover)
        print(f"  ✓ Saved to {fname}")

    print(f"\n  ℹ️  To decode, the recipient needs:")
    print(f"    • This text (or the file)")
    print(f"    • The password (share separately!)")
    print(f"    • This program + tokenizer for: {model}")

def do_decode():
    print("\n  ── DECODE ─────────────────────────────\n")

    source = prompt_choice("Where is the cover text?", [
        "Paste it here",
        "Read from file",
    ])

    if source == 1:
        print("\n  Paste the cover text, then press Enter twice:\n")
        lines = []
        empty_count = 0
        while True:
            try:
                line = input()
                if line == "":
                    empty_count += 1
                    if empty_count >= 2:
                        break
                    lines.append(line)
                else:
                    empty_count = 0
                    lines.append(line)
            except EOFError:
                break
        cover_text = "\n".join(lines)
    else:
        fname = input("  Filename [cover.txt]: ").strip() or "cover.txt"
        if not os.path.exists(fname):
            print(f"  ✗ File not found: {fname}"); return
        with open(fname) as f:
            cover_text = f.read()

    if not cover_text.strip():
        print("  ✗ No text provided."); return

    password = input("\n  Password: ").strip()
    if not password:
        print("  ✗ No password entered."); return

    model = pick_model()

    print(f"\n  Decoding... (only loading tokenizer, no GPU needed)\n")

    from stego import decode_robust
    recovered = decode_robust(
        seed=password, cover_text=cover_text,
        model_name=model,
    )

    print("\n  ═══════════════════════════════════════")
    if recovered.startswith("[decode error"):
        print(f"  ✗ {recovered}")
        print("  Check: wrong password? wrong model? wrong text?")
    else:
        print(f"  🔓  SECRET MESSAGE:\n")
        print(f"  {recovered}")
    print("\n  ═══════════════════════════════════════")

def main():
    clear()
    banner()

    while True:
        choice = prompt_choice("What would you like to do?", [
            "Hide a secret message",
            "Extract a hidden message",
            "Quit",
        ])

        if choice == 1:
            do_encode()
        elif choice == 2:
            do_decode()
        elif choice == 3:
            print("\n  Bye! 👋\n")
            sys.exit(0)

        input("\n  Press Enter to continue...")
        clear()
        banner()

if __name__ == '__main__':
    main()
