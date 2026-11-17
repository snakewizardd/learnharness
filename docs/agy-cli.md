# Google Antigravity CLI (`agy`)

This fork adds a zero-dependency local OpenAI-compatible `/v1` shim ([`scripts/agy-openai-shim.mjs`](../scripts/agy-openai-shim.mjs)) so **Google AI / Antigravity subscribers** can run LearnHarness end-to-end using an installed **`agy` CLI**—without modifying LearnHarness's core server, UI, or persistence layers.

## Prerequisites

1. Install and sign in to the **Google Antigravity CLI (`agy`)** on your machine (`agy` / `agy.exe` available on `PATH`).
2. Node.js 24+ (or 26) and npm.

## Quick start

```bash
npm ci
npm run plugins:setup
npm run dev:agy
```

`npm run dev:agy` launches both processes concurrently:
- **Antigravity `/v1` bridge:** `http://127.0.0.1:11435/v1` (`npm run agy:shim` to run standalone)
- **LearnHarness web UI:** `http://localhost:3000`

### Connect inside LearnHarness

1. Open `http://localhost:3000`.
2. Go to **Settings → Providers → + Add a provider → Custom endpoint** (or use the first-run setup dialog).
3. Enter:
   - **Base URL:** `http://127.0.0.1:11435/v1`
   - **Authentication:** **Keyless**
4. Click **Connect**, then assign **`agy-default`** to **Strong**, **Cheap**, and **Vision**.

## How it works

- **Zero core edits:** LearnHarness talks to the shim through its standard [Custom OpenAI-compatible endpoint](custom-providers.md).
- **Structured tool-call bridge:** Each model call runs `agy --output-format=json --disable-slash-commands --json-schema=<temp-schema>` in an isolated temporary directory so workspace `AGENTS.md` files never leak into lessons, and LearnHarness tools (`probe`, `recall`, `ask_user_question`, `probe_done`, `plan_init`, `quiz`, `explain`, `problem`, `conclude`, extraction, etc.) map cleanly to OpenAI `tool_calls`.
- **Fast default model (`gemini-4-hydrogen-low`):** By default, `agy-default` pins `gemini-4-hydrogen-low` (`--effort=low`) for responsive ~8–15s turns across Probing, Planning, and Teaching. Selecting **High** reasoning effort in LearnHarness escalates automatically to `gemini-4-hydrogen-high`.
- **Windows command-line safety & concurrency gating:** Prompts exceeding Windows's 32 KB `CreateProcess` limit automatically spill into bounded `<30 KB` temporary files inside the isolated working directory, and a built-in FIFO semaphore (`AGY_MAX_CONCURRENCY=2` by default) prevents background Knowledge Graph extraction from spawning too many simultaneous `agy` language-server processes while keeping SSE `: keep-alive` heartbeats active.

## Optional environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `AGY_SHIM_HOST` | `127.0.0.1` | Host interface to bind |
| `AGY_SHIM_PORT` | `11435` | Port to listen on |
| `AGY_BIN` | `agy.exe` (Windows) / `agy` | Path or binary name for Antigravity CLI |
| `AGY_MODEL` | *(auto: `gemini-4-hydrogen-low`)* | Pin any `agy` model ID (e.g. `gemini-4-hydrogen-medium`, `gemini-4-carbon-medium`, `claude-sonnet-5-5-medium`) |
| `AGY_EFFORT` | *(auto from model/picker)* | Force `--effort` (`low`, `medium`, `high`, `xhigh`, `max`) |
| `AGY_MAX_CONCURRENCY` | `2` | Maximum simultaneous `agy` subprocesses |
| `AGY_DEBUG` | `0` (`--debug` flag also supported) | Set to `1` to log prompt sizes, model flags, and raw `agy` envelopes |

## Limitations

- Each LLM turn spawns a fresh headless `agy` invocation (`--print`), adding a few seconds of CLI startup overhead compared to direct HTTP inference APIs.
- Billing, quota, and model availability are governed entirely by your signed-in `agy` account; LearnHarness displays custom-endpoint dollar costs as unknown/unpriced.
