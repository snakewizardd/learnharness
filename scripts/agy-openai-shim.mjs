#!/usr/bin/env node
/**
 * OpenAI-compatible `/v1` HTTP shim backed by Google Antigravity CLI (`agy`).
 *
 * Zero edits to LearnHarness required. Start this server in a terminal:
 *   node scripts/agy-openai-shim.mjs
 *
 * Then in LearnHarness:
 *   Settings -> Providers -> + Add a provider -> Custom endpoint
 *   - Base URL:  http://127.0.0.1:11435/v1
 *   - Auth:      Keyless
 *   - Click Connect, then select `agy-default` for Strong, Cheap, and Vision.
 *
 * Optional environment variables:
 *   AGY_SHIM_HOST   Bind host (default: 127.0.0.1)
 *   AGY_SHIM_PORT   Bind port (default: 11435)
 *   AGY_BIN         Path/name of Antigravity CLI executable (default: agy)
 *   AGY_MODEL       Pin a specific `agy --model <id>` (default: agy's active default)
 *   AGY_EFFORT      Override `agy --effort <low|medium|high|xhigh|max>`
 *   AGY_DEBUG       Set to "1" to log full prompt sizes and raw agy envelopes
 */

import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import fs from "node:fs/promises";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const HOST = process.env.AGY_SHIM_HOST || "127.0.0.1";
const PORT = Number.parseInt(process.env.AGY_SHIM_PORT || "11435", 10);
const AGY_BIN = process.env.AGY_BIN || (process.platform === "win32" ? "agy.exe" : "agy");
const AGY_MODEL_ENV = process.env.AGY_MODEL?.trim() || "";
const AGY_EFFORT_ENV = process.env.AGY_EFFORT?.trim() || "";
const DEBUG = process.env.AGY_DEBUG === "1" || process.argv.includes("--debug");

const DEFAULT_MODEL_ID = "agy-default";
const MAX_BODY_BYTES = 32 * 1024 * 1024; // 32 MiB (accommodates image attachments)
const SSE_HEARTBEAT_MS = 5_000;
const MAX_CONCURRENCY = Math.max(1, Number.parseInt(process.env.AGY_MAX_CONCURRENCY || "2", 10) || 2);

let activePermits = 0;
const permitQueue = [];

/** FIFO concurrency gate so background workers never stampede `agy.exe` processes. */
async function withConcurrencyPermit(abortSignal, fn) {
	if (abortSignal?.aborted) throw new Error("Request aborted before acquiring worker permit");
	if (activePermits >= MAX_CONCURRENCY) {
		await new Promise((resolve, reject) => {
			const entry = {
				grant() {
					abortSignal?.removeEventListener("abort", onAbort);
					resolve();
				},
			};
			const onAbort = () => {
				const idx = permitQueue.indexOf(entry);
				if (idx !== -1) permitQueue.splice(idx, 1);
				reject(new Error("Request aborted while queued"));
			};
			abortSignal?.addEventListener("abort", onAbort, { once: true });
			permitQueue.push(entry);
		});
	} else {
		activePermits += 1;
	}
	try {
		return await fn();
	} finally {
		const next = permitQueue.shift();
		if (next) {
			next.grant();
		} else {
			activePermits -= 1;
		}
	}
}

/** Advertise context, max output, vision, and reasoning so LearnHarness enables
 *  Strong + Cheap + Vision slots, generous compaction budgets, and effort controls. */
function buildModelsPayload() {
	const ids = [DEFAULT_MODEL_ID];
	if (AGY_MODEL_ENV && AGY_MODEL_ENV !== DEFAULT_MODEL_ID) {
		ids.push(AGY_MODEL_ENV);
	}
	return {
		object: "list",
		data: ids.map((id) => ({
			id,
			object: "model",
			created: 1700000000,
			owned_by: "antigravity-cli",
			context_window: 200000,
			max_output_tokens: 16384,
			reasoning: true,
			capabilities: ["completion", "vision", "reasoning"],
			architecture: {
				input_modalities: ["text", "image"],
				output_modalities: ["text", "reasoning"],
			},
		})),
	};
}

/** Resolve `agy --model` and `--effort`:
 *  Defaults `agy-default` to `gemini-4-hydrogen-low` for snappy turns unless
 *  the user explicitly selects High/XHigh effort in LearnHarness (or sets AGY_MODEL / AGY_EFFORT). */
function resolveModelAndEffort(requestedModel, requestEffort) {
	const cleanModel = typeof requestedModel === "string" ? requestedModel.trim() : "";
	const customModel = AGY_MODEL_ENV || (cleanModel && cleanModel !== DEFAULT_MODEL_ID ? cleanModel : "");

	if (customModel) {
		const effort = AGY_EFFORT_ENV || (
			requestEffort === "minimal" ? "low"
				: ["low", "medium", "high", "xhigh", "max"].includes(requestEffort) ? requestEffort
					: ""
		);
		return { model: customModel, effort };
	}

	if (AGY_EFFORT_ENV) {
		const tier = AGY_EFFORT_ENV === "max" ? "xhigh" : AGY_EFFORT_ENV;
		return {
			model: ["low", "medium", "high", "xhigh"].includes(tier) ? `gemini-4-hydrogen-${tier}` : "gemini-4-hydrogen-low",
			effort: AGY_EFFORT_ENV,
		};
	}

	if (requestEffort === "high") {
		return { model: "gemini-4-hydrogen-high", effort: "high" };
	}
	if (requestEffort === "xhigh" || requestEffort === "max") {
		return { model: "gemini-4-hydrogen-xhigh", effort: "xhigh" };
	}
	return { model: "gemini-4-hydrogen-low", effort: "low" };
}

/** Build the JSON schema passed to `agy --json-schema` for this request. */
function buildTurnSchema(tools, toolChoice) {
	const allowTools = Array.isArray(tools) && tools.length > 0 && toolChoice !== "none";
	if (!allowTools) {
		return {
			type: "object",
			properties: {
				content: {
					type: "string",
					description: "Your complete assistant response to the conversation.",
				},
			},
			required: ["content"],
		};
	}

	const allNames = tools
		.map((t) => (t?.type === "function" ? t.function?.name : t?.name))
		.filter((n) => typeof n === "string" && n.length > 0);

	const forcedName =
		typeof toolChoice === "object" &&
		toolChoice !== null &&
		toolChoice.type === "function" &&
		typeof toolChoice.function?.name === "string"
			? toolChoice.function.name
			: null;

	const allowedNames = forcedName ? [forcedName] : allNames;
	const requireToolCall = toolChoice === "required" || Boolean(forcedName);

	const nameProp = {
		type: "string",
		description: "Exact function name from <AVAILABLE_TOOLS>.",
		...(allowedNames.length > 0 ? { enum: allowedNames } : {}),
	};

	const toolCallsProp = {
		type: "array",
		description:
			"Tool calls to invoke from <AVAILABLE_TOOLS>. Leave empty ([]) if responding with plain text only.",
		items: {
			type: "object",
			properties: {
				name: nameProp,
				arguments_json: {
					type: "string",
					description:
						"Valid JSON object string matching that tool's parameters schema (for example: '{\"question\":\"...\"}'). Use '{}' when a tool takes no arguments.",
				},
			},
			required: ["name", "arguments_json"],
		},
		...(requireToolCall ? { minItems: 1 } : {}),
	};

	return {
		type: "object",
		properties: {
			content: {
				type: "string",
				description:
					"Assistant text shown to the learner before any tool calls. Use an empty string (\"\") if emitting only tool call(s) with no accompanying prose.",
			},
			tool_calls: toolCallsProp,
		},
		required: ["content", "tool_calls"],
	};
}

/** Extract text (and materialize base64 data-URI images into `tmpDir`) from OpenAI content. */
async function renderContentParts(content, tmpDir, imageState) {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return content == null ? "" : JSON.stringify(content);

	const chunks = [];
	for (const part of content) {
		if (!part || typeof part !== "object") continue;
		if (part.type === "text" && typeof part.text === "string") {
			chunks.push(part.text);
			continue;
		}
		if (part.type === "image_url" && typeof part.image_url?.url === "string") {
			const url = part.image_url.url;
			const match = /^data:image\/([a-zA-Z0-9.+-]+);base64,(.+)$/s.exec(url);
			if (match) {
				imageState.count += 1;
				const rawExt = match[1].toLowerCase().replace(/[^a-z0-9]/g, "") || "png";
				const ext = rawExt === "jpeg" ? "jpg" : rawExt;
				const filename = `attachment-${imageState.count}.${ext}`;
				const filePath = path.join(tmpDir, filename);
				await fs.writeFile(filePath, Buffer.from(match[2], "base64"));
				chunks.push(`[Attached image saved in current directory as ./${filename} — use read_file on ./${filename} if you need to inspect it visually]`);
			} else {
				chunks.push(`[Attached image URL: ${url}]`);
			}
			continue;
		}
		chunks.push(JSON.stringify(part));
	}
	return chunks.join("\n");
}

/** Serialize OpenAI `messages` + `tools` into a clean, unambiguous prompt for `agy`. */
async function buildStdinPrompt(body, tmpDir) {
	const messages = Array.isArray(body.messages) ? body.messages : [];
	const tools = Array.isArray(body.tools) && body.tool_choice !== "none" ? body.tools : [];
	const imageState = { count: 0 };

	const systemBlocks = [];
	const historyBlocks = [];

	for (const msg of messages) {
		if (!msg || typeof msg !== "object") continue;
		const role = String(msg.role || "user");
		const text = await renderContentParts(msg.content, tmpDir, imageState);

		if (role === "system" || role === "developer") {
			if (text.trim()) systemBlocks.push(text.trim());
			continue;
		}

		if (role === "user") {
			historyBlocks.push(`<MESSAGE role="user">\n${text}\n</MESSAGE>`);
			continue;
		}

		if (role === "assistant") {
			const parts = [];
			if (text) parts.push(text);
			if (Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
				const callsText = msg.tool_calls
					.map((tc) => {
						const fn = tc?.function || {};
						return `  - id=${tc?.id || "call"} name=${fn.name || "unknown"} arguments=${fn.arguments ?? "{}"}`;
					})
					.join("\n");
				parts.push(`<ASSISTANT_TOOL_CALLS>\n${callsText}\n</ASSISTANT_TOOL_CALLS>`);
			}
			historyBlocks.push(`<MESSAGE role="assistant">\n${parts.join("\n")}\n</MESSAGE>`);
			continue;
		}

		if (role === "tool") {
			const attrs = [
				`role="tool"`,
				msg.tool_call_id ? `tool_call_id="${msg.tool_call_id}"` : "",
				msg.name ? `name="${msg.name}"` : "",
			]
				.filter(Boolean)
				.join(" ");
			historyBlocks.push(`<MESSAGE ${attrs}>\n${text}\n</MESSAGE>`);
			continue;
		}

		historyBlocks.push(`<MESSAGE role="${role}">\n${text}\n</MESSAGE>`);
	}

	const sections = [];

	sections.push(
		[
			"<BACKEND_BRIDGE_RULES>",
			"You are acting directly as the LLM inference engine behind an OpenAI Chat Completions endpoint for LearnHarness.",
			"CRITICAL RULES:",
			imageState.count > 0
				? "1. Do NOT invoke any built-in Antigravity CLI tools (run_command, write_file, edit_file, ask_question, invoke_subagent, etc.), EXCEPT read_file if you need to view an ./attachment-* image listed in a message."
				: "1. Do NOT invoke any built-in Antigravity CLI tools (read_file, run_command, write_file, edit_file, ask_question, invoke_subagent, etc.). Everything you need is in this prompt.",
			tools.length > 0
				? "2. Whenever the system instructions or task call for using a tool (e.g. probe, recall, ask_user_question, probe_done, plan_init, quiz, explain, problem, conclude, research, visual, demo, etc.), DO NOT try to execute it as a CLI tool. Instead, emit it inside the `tool_calls` array of your final JSON output, with `name` matching `<AVAILABLE_TOOLS>` and `arguments_json` containing a valid JSON object string matching that tool's `parameters` schema."
				: "2. Return your complete response in the `content` field of the final JSON output.",
			"3. Follow `<SYSTEM_INSTRUCTIONS>` faithfully and continue cleanly from the last entry in `<CONVERSATION_HISTORY>`.",
			"</BACKEND_BRIDGE_RULES>",
		].join("\n"),
	);

	if (systemBlocks.length > 0) {
		sections.push(`<SYSTEM_INSTRUCTIONS>\n${systemBlocks.join("\n\n---\n\n")}\n</SYSTEM_INSTRUCTIONS>`);
	}

	if (tools.length > 0) {
		const toolDefs = tools.map((t) => JSON.stringify(t?.type === "function" ? t.function : t));
		sections.push(`<AVAILABLE_TOOLS>\n[\n${toolDefs.join(",\n")}\n]\n</AVAILABLE_TOOLS>`);
		if (body.tool_choice && body.tool_choice !== "auto") {
			sections.push(`<TOOL_CHOICE>\n${JSON.stringify(body.tool_choice)}\n</TOOL_CHOICE>`);
		}
	}

	if (body.response_format && body.response_format.type && body.response_format.type !== "text") {
		sections.push(
			`<REQUESTED_RESPONSE_FORMAT>\nEnsure your \`content\` string conforms to: ${JSON.stringify(body.response_format)}\n</REQUESTED_RESPONSE_FORMAT>`,
		);
	}

	sections.push(`<CONVERSATION_HISTORY>\n${historyBlocks.join("\n\n")}\n</CONVERSATION_HISTORY>`);

	return { stdinText: sections.join("\n\n"), hasImages: imageState.count > 0 };
}

/** Try to parse JSON from raw stdout, tolerating leading diagnostic lines if any. */
function parseJsonStdout(raw) {
	const trimmed = raw.trim();
	if (!trimmed) throw new Error("agy returned empty stdout");
	try {
		return JSON.parse(trimmed);
	} catch {
		// Scan lines from bottom up in case diagnostic lines preceded the JSON object
		const lines = trimmed.split(/\r?\n/);
		for (let i = lines.length - 1; i >= 0; i--) {
			const candidate = lines.slice(i).join("\n").trim();
			if (candidate.startsWith("{")) {
				try {
					return JSON.parse(candidate);
				} catch {
					// keep scanning
				}
			}
		}
		throw new Error(`Unable to parse JSON from agy stdout: ${trimmed.slice(0, 300)}`);
	}
}

/** Normalize `arguments_json` into a guaranteed JSON object string. */
function normalizeArgumentsJson(rawArgs) {
	if (rawArgs && typeof rawArgs === "object" && !Array.isArray(rawArgs)) {
		return JSON.stringify(rawArgs);
	}
	if (typeof rawArgs !== "string") {
		return "{}";
	}
	let trimmed = rawArgs.trim();
	const fenceMatch = /^```(?:json)?\s*([\s\S]*?)\s*```$/i.exec(trimmed);
	if (fenceMatch) {
		trimmed = fenceMatch[1].trim();
	}
	if (!trimmed) return "{}";
	try {
		const parsed = JSON.parse(trimmed);
		if (typeof parsed === "string") {
			// Heal accidental double-encoding
			const inner = JSON.parse(parsed);
			if (inner && typeof inner === "object" && !Array.isArray(inner)) {
				return JSON.stringify(inner);
			}
		}
		if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
			return JSON.stringify(parsed);
		}
		return trimmed;
	} catch {
		return trimmed;
	}
}

/** Extract `{ content, toolCalls, usage }` from `agy --output-format json --json-schema` output. */
function unwrapAgyOutput(parsed, stderr = "") {
	let usage = { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
	if (parsed && typeof parsed === "object" && parsed.usage && typeof parsed.usage === "object") {
		const inTok = Number(parsed.usage.input_tokens ?? parsed.usage.prompt_tokens ?? 0) || 0;
		const outTok = Number(parsed.usage.output_tokens ?? parsed.usage.completion_tokens ?? 0) || 0;
		usage = { prompt_tokens: inTok, completion_tokens: outTok, total_tokens: inTok + outTok };
	}

	if (parsed && typeof parsed === "object" && typeof parsed.status === "string") {
		const upper = parsed.status.toUpperCase();
		if (upper === "ERROR" || upper === "FAILED") {
			throw new Error(`agy reported status=${parsed.status}: ${parsed.error || parsed.response || "unknown error"}`);
		}
	}

	// Locate structured payload (`structured_output`, direct `{content, tool_calls}`, or JSON inside `response`/`result`)
	let payload = null;
	let hasStructuredPayload = false;
	if (parsed && typeof parsed === "object") {
		if (parsed.structured_output && typeof parsed.structured_output === "object") {
			payload = parsed.structured_output;
			hasStructuredPayload = true;
		} else if ("content" in parsed || "tool_calls" in parsed) {
			payload = parsed;
			hasStructuredPayload = true;
		} else {
			const textField =
				typeof parsed.response === "string"
					? parsed.response
					: typeof parsed.result === "string"
						? parsed.result
						: null;
			if (textField !== null) {
				try {
					const inner = JSON.parse(textField);
					if (inner && typeof inner === "object") {
						payload = inner;
						hasStructuredPayload = "content" in inner || "tool_calls" in inner;
					}
				} catch {
					payload = { content: textField, tool_calls: [] };
				}
			}
		}
	}

	if (!payload || typeof payload !== "object") {
		throw new Error(`Unrecognized agy JSON structure: ${JSON.stringify(parsed).slice(0, 300)}`);
	}

	const content = typeof payload.content === "string" ? payload.content : "";
	const rawCalls = Array.isArray(payload.tool_calls) ? payload.tool_calls : [];
	const toolCalls = rawCalls
		.filter((tc) => tc && typeof tc === "object" && typeof tc.name === "string" && tc.name.trim())
		.map((tc, idx) => ({
			id: `call_${randomUUID().replace(/-/g, "").slice(0, 20)}_${idx}`,
			type: "function",
			function: {
				name: tc.name.trim(),
				arguments: normalizeArgumentsJson(tc.arguments_json ?? tc.arguments),
			},
		}));

	// Intentional silent stops (`structured_output: { content: "", tool_calls: [] }`) are valid
	// (e.g. after `plan_init`, `probe_done`, or `conclude`). Only reject when no structured output arrived.
	if (!hasStructuredPayload && !content.trim() && toolCalls.length === 0) {
		throw new Error(
			`agy returned empty output${stderr.trim() ? `: ${stderr.trim().slice(0, 400)}` : ""}`,
		);
	}

	return { content, toolCalls, usage };
}

/** Run a single non-interactive `agy -p` turn in an isolated temporary directory. */
async function runAgyTurn(body, abortSignal) {
	const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "lh-agy-shim-"));
	try {
		const schema = buildTurnSchema(body.tools, body.tool_choice);
		const schemaPath = path.join(tmpDir, "response-schema.json");
		await fs.writeFile(schemaPath, JSON.stringify(schema), "utf8");

		const { stdinText, hasImages } = await buildStdinPrompt(body, tmpDir);

		// Windows CreateProcessW caps total command line at 32,767 UTF-16 chars (including quote escapes).
		// Use fast inline `--print=<prompt>` when it fits safely; otherwise split into <=30KB / <=500-line
		// workspace files inside `tmpDir` where `read_file` is auto-approved with zero truncation.
		const escapedLen = stdinText.length + (stdinText.match(/["\\]/g)?.length ?? 0);
		let printPrompt = stdinText;
		let deliveryMode = "inline";
		if (escapedLen > 26_000) {
			deliveryMode = "files";
			const safeLines = [];
			for (const rawLine of stdinText.split("\n")) {
				if (Buffer.byteLength(rawLine, "utf8") <= 8_000) {
					safeLines.push(rawLine);
				} else {
					for (let i = 0; i < rawLine.length; i += 6_000) {
						safeLines.push(rawLine.slice(i, i + 6_000));
					}
				}
			}
			const partPaths = [];
			let chunkLines = [];
			let chunkBytes = 0;
			const flushPart = async () => {
				if (chunkLines.length === 0) return;
				const idx = partPaths.length + 1;
				const partPath = path.join(tmpDir, `prompt-part-${idx}.md`);
				await fs.writeFile(partPath, chunkLines.join("\n"), "utf8");
				partPaths.push(partPath);
				chunkLines = [];
				chunkBytes = 0;
			};
			for (const line of safeLines) {
				const lineBytes = Buffer.byteLength(line, "utf8") + 1;
				if (chunkLines.length > 0 && (chunkLines.length >= 500 || chunkBytes + lineBytes > 30_000)) {
					await flushPart();
				}
				chunkLines.push(line);
				chunkBytes += lineBytes;
			}
			await flushPart();
			printPrompt = [
				`Read the following prompt file(s) in exact order using ONLY the read_file tool (do NOT call run_command or any other tool):`,
				...partPaths.map((p, i) => `${i + 1}. ${p}`),
				`After reading all ${partPaths.length} file(s), follow the <BACKEND_BRIDGE_RULES>, <SYSTEM_INSTRUCTIONS>, <AVAILABLE_TOOLS>, and <CONVERSATION_HISTORY> inside to produce your response strictly conforming to the JSON schema.`,
			].join("\n");
		}

		const args = [
			"--output-format=json",
			"--disable-slash-commands",
			`--json-schema=${schemaPath}`,
		];

		if (hasImages) {
			args.push("--dangerously-skip-permissions");
		}

		const { model: effectiveModel, effort } = resolveModelAndEffort(body.model, body.reasoning_effort);
		if (effectiveModel) {
			args.push(`--model=${effectiveModel}`);
		}
		if (effort) {
			args.push(`--effort=${effort}`);
		}

		args.push(`--print=${printPrompt}`);

		if (DEBUG) {
			console.log(
				`[agy-shim] spawning ${AGY_BIN} (mode=${deliveryMode}, model=${effectiveModel || "default"}, effort=${effort || "default"}, prompt=${stdinText.length} chars, tools=${Array.isArray(body.tools) ? body.tools.length : 0})`,
			);
		}

		const { stdout, stderr } = await new Promise((resolve, reject) => {
			if (abortSignal?.aborted) {
				reject(new Error("Request aborted before agy spawn"));
				return;
			}

			const isNodeScript = /\.(m?js|cjs)$/i.test(AGY_BIN);
			const spawnCmd = isNodeScript ? process.execPath : AGY_BIN;
			const spawnArgs = isNodeScript ? [AGY_BIN, ...args] : args;

			const child = spawn(spawnCmd, spawnArgs, {
				cwd: tmpDir,
				stdio: ["pipe", "pipe", "pipe"],
				windowsHide: true,
			});

			const outChunks = [];
			const errChunks = [];

			const onAbort = () => {
				try {
					child.kill();
				} catch {
					// ignore
				}
			};
			abortSignal?.addEventListener("abort", onAbort, { once: true });

			child.stdout.on("data", (chunk) => outChunks.push(chunk));
			child.stderr.on("data", (chunk) => errChunks.push(chunk));

			child.on("error", (err) => {
				abortSignal?.removeEventListener("abort", onAbort);
				reject(new Error(`Failed to launch '${AGY_BIN}': ${err.message}`));
			});

			child.on("close", (code) => {
				abortSignal?.removeEventListener("abort", onAbort);
				const outStr = Buffer.concat(outChunks).toString("utf8");
				const errStr = Buffer.concat(errChunks).toString("utf8");
				if (abortSignal?.aborted) {
					reject(new Error("Request aborted by client"));
					return;
				}
				if (code !== 0) {
					reject(
						new Error(
							`agy exited with code ${code}: ${(errStr || outStr).trim().slice(0, 600)}`,
						),
					);
					return;
				}
				resolve({ stdout: outStr, stderr: errStr });
			});

			child.stdin.on("error", () => {
				// Ignore EPIPE if child exits early; `close` handler surfaces stderr/exit code.
			});
			child.stdin.end();
		});

		if (DEBUG && stderr.trim()) {
			console.log(`[agy-shim] stderr: ${stderr.trim().slice(0, 400)}`);
		}

		const parsed = parseJsonStdout(stdout);
		if (DEBUG) {
			console.log(`[agy-shim] raw envelope: ${JSON.stringify(parsed).slice(0, 500)}`);
		}
		return unwrapAgyOutput(parsed, stderr);
	} finally {
		await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {});
	}
}

/** Read bounded JSON request body. */
async function readJsonBody(req) {
	const chunks = [];
	let total = 0;
	for await (const chunk of req) {
		total += chunk.length;
		if (total > MAX_BODY_BYTES) {
			throw new Error(`Request body exceeded ${MAX_BODY_BYTES} bytes`);
		}
		chunks.push(chunk);
	}
	const raw = Buffer.concat(chunks).toString("utf8");
	return raw ? JSON.parse(raw) : {};
}

function sendJson(res, status, payload) {
	const body = JSON.stringify(payload);
	res.writeHead(status, {
		"Content-Type": "application/json; charset=utf-8",
		"Content-Length": Buffer.byteLength(body),
	});
	res.end(body);
}

export async function handleShimRequest(req, res) {
	const method = req.method || "GET";
	const url = new URL(req.url || "/", `http://${req.headers.host || `${HOST}:${PORT}`}`);
	const pathname = url.pathname.replace(/\/+$/, "") || "/";

	// Health / model discovery
	if (method === "GET" && (pathname === "/v1/models" || pathname === "/models")) {
		sendJson(res, 200, buildModelsPayload());
		return;
	}

	if (method === "GET" && (pathname === "/" || pathname === "/health")) {
		sendJson(res, 200, { ok: true, service: "agy-openai-shim", models: buildModelsPayload().data.map((m) => m.id) });
		return;
	}

	if (method === "POST" && (pathname === "/v1/chat/completions" || pathname === "/chat/completions")) {
		const startedAt = Date.now();
		let body;
		try {
			body = await readJsonBody(req);
		} catch (err) {
			sendJson(res, 400, { error: { message: err.message, type: "invalid_request_error" } });
			return;
		}

		const isStream = Boolean(body.stream);
		const model = typeof body.model === "string" && body.model ? body.model : DEFAULT_MODEL_ID;
		const completionId = `chatcmpl-${randomUUID().replace(/-/g, "")}`;
		const created = Math.floor(Date.now() / 1000);

		const abortController = new AbortController();
		res.on("close", () => {
			if (!res.writableEnded) {
				abortController.abort();
			}
		});

		let heartbeatTimer = null;
		if (isStream) {
			res.writeHead(200, {
				"Content-Type": "text/event-stream; charset=utf-8",
				"Cache-Control": "no-cache, no-transform",
				Connection: "keep-alive",
				"X-Accel-Buffering": "no",
			});
			// Immediate comment flushes headers and keeps HTTP clients from timing out while agy thinks
			res.write(": agy-shim stream open\n\n");
			heartbeatTimer = setInterval(() => {
				if (!res.writableEnded) {
					res.write(": keep-alive\n\n");
				}
			}, SSE_HEARTBEAT_MS);
		}

		try {
			const { content, toolCalls, usage } = await withConcurrencyPermit(
				abortController.signal,
				() => runAgyTurn(body, abortController.signal),
			);
			const finishReason = toolCalls.length > 0 ? "tool_calls" : "stop";
			const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
			console.log(
				`[agy-shim] OK (${elapsed}s) finish=${finishReason} text=${content.length}c tools=[${toolCalls.map((t) => t.function.name).join(",")}]`,
			);

			if (!isStream) {
				sendJson(res, 200, {
					id: completionId,
					object: "chat.completion",
					created,
					model,
					choices: [
						{
							index: 0,
							message: {
								role: "assistant",
								content: content || (toolCalls.length > 0 ? null : ""),
								...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
							},
							finish_reason: finishReason,
						},
					],
					usage,
				});
				return;
			}

			if (heartbeatTimer) clearInterval(heartbeatTimer);
			if (res.writableEnded) return;

			const delta = { role: "assistant" };
			if (content) delta.content = content;
			if (toolCalls.length > 0) {
				delta.tool_calls = toolCalls.map((tc, idx) => ({
					index: idx,
					id: tc.id,
					type: "function",
					function: {
						name: tc.function.name,
						arguments: tc.function.arguments,
					},
				}));
			}

			const contentChunk = {
				id: completionId,
				object: "chat.completion.chunk",
				created,
				model,
				choices: [{ index: 0, delta, finish_reason: null }],
			};
			const stopChunk = {
				id: completionId,
				object: "chat.completion.chunk",
				created,
				model,
				choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
				usage,
			};

			res.write(`data: ${JSON.stringify(contentChunk)}\n\n`);
			res.write(`data: ${JSON.stringify(stopChunk)}\n\n`);
			res.write("data: [DONE]\n\n");
			res.end();
		} catch (err) {
			if (heartbeatTimer) clearInterval(heartbeatTimer);
			const message = err instanceof Error ? err.message : String(err);
			console.error(`[agy-shim] ERROR: ${message}`);

			if (!isStream && !res.headersSent) {
				sendJson(res, 502, {
					error: { message, type: "upstream_agy_error", code: "agy_execution_failed" },
				});
				return;
			}

			if (!res.writableEnded) {
				// Surface an explicit OpenAI error event on the stream so pi-ai reports the failure cleanly
				res.write(
					`data: ${JSON.stringify({ error: { message, type: "upstream_agy_error" } })}\n\n`,
				);
				res.write("data: [DONE]\n\n");
				res.end();
			}
		}
		return;
	}

	sendJson(res, 404, { error: { message: `Route not found: ${method} ${pathname}`, type: "not_found" } });
}

const isMain = process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url;
if (isMain) {
	const server = http.createServer(handleShimRequest);
	server.listen(PORT, HOST, () => {
		console.log(`[agy-shim] Listening on http://${HOST}:${PORT}/v1`);
		console.log(`[agy-shim] CLI binary: ${AGY_BIN}${AGY_MODEL_ENV ? ` (model=${AGY_MODEL_ENV})` : ""}${AGY_EFFORT_ENV ? ` (effort=${AGY_EFFORT_ENV})` : ""}`);
		console.log(`[agy-shim] In LearnHarness -> Settings -> Custom endpoint -> http://${HOST}:${PORT}/v1 (Keyless)`);
	});
}
