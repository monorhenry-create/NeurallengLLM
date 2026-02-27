# Neural Steganography

Hide secret messages inside normal-looking AI-generated text. You give it a secret and a password, and it spits out a paragraph that looks totally ordinary — but the secret is baked into which words the model chose. Only someone with the password and this tool can pull the message back out.

## How it works

When a language model generates text, it picks from thousands of possible next words at every step. Normally that choice is random (weighted by probability). This tool rigs those choices so each token quietly encodes a couple bits of your secret message. The text still reads naturally because we only pick from high-probability tokens — it just so happens that the specific token we pick also carries hidden data. The message gets compressed and encrypted before encoding, so even if someone suspects something, they can't extract anything without the password. Inspired by [Neural Linguistic Steganography](https://arxiv.org/abs/1909.01496) (Ziegler, Deng & Rush, 2019).

## Usage

Install dependencies:

```
pip install torch transformers
```

Run the CLI:

```
python cli.py
```

Pick option 1 to hide a message or option 2 to extract one. It'll walk you through everything — just follow the prompts. You'll choose a secret message, a password, and a model (GPT-2 Small is fine to start). The recipient needs the same password and model to decode.

Encoding needs the full model (uses GPU if available). Decoding only loads the tokenizer, so it's fast and runs on anything.

Try decoding example text first with password AIGOD using Qwen 2.5 0.5B model. 