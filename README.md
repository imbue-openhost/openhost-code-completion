# Code Completion

A self-hosted code-completion server for Cloud in a Bottle. It runs
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` under the
hood and puts a model-management web UI in front of it, so you can download a
GGUF model, pick which one is active, and point your editor at an
OpenAI-compatible completion API — all on your own zone, with no code leaving
your machine.

**Who it's for:** developers who want local, private AI code completion (FIM /
autocomplete) wired into their editor without sending source to a third-party
service.

## What it does

- **Model management UI.** Search Hugging Face for GGUF models, download one,
  and select the active model. The UI is reachable only by the zone owner.
- **Runs models on CPU** via a source-built `llama.cpp` (GPU layers are
  auto-offloaded if an NVIDIA GPU is present). Concurrency and context size are
  configurable (see below).
- **OpenAI-compatible + native endpoints.** Serves `/v1/*` (OpenAI-compatible)
  plus llama.cpp's `/infill`, `/completions`, `/tokenize`, `/detokenize`, and
  `/embedding` for editor integrations.

## Access & auth

- **Owner:** browsing through the zone, the owner reaches the management UI and
  the inference API with no login (the app trusts the router's
  `X-OpenHost-Is-Owner` signal).
- **Editors / tools:** the inference endpoints are public paths so external
  tools can reach them directly, gated by an **app-issued API token** shown in
  the UI. Send it as an `Authorization: Bearer <token>` or `X-API-Key` header.
  You can regenerate the token from the UI.

## Using it from an editor

Point any OpenAI-compatible completion client at:

```
https://<your-app>.<your-zone>/v1
```

and set the API key to the token from the app's UI. For editors that speak
llama.cpp's FIM endpoint directly (e.g. the `llama.vim` / `llama.vscode`
plugins), use `/infill`.

## Configuration

Set via environment variables (all optional):

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLM_THREADS` | CPU count − 2 | Inference threads |
| `LLM_CTX_SIZE` | 4096 | Context window |
| `LLM_SLOTS` | 2 | Concurrent request slots |
| `LLM_GPU_LAYERS` | 0 (99 if a GPU is detected) | Layers offloaded to GPU |

## Data

- **Active-model state** persists under the app data dir (backed up).
- **Downloaded model weights** live in temp/scratch storage — recreatable on
  demand and excluded from backups so they don't bloat backup size.

## Links

- llama.cpp (inference engine) — https://github.com/ggml-org/llama.cpp
- GGUF models — https://huggingface.co/models?library=gguf
