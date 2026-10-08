# PHASE 0 — FUNCTIONALITY DIAGNOSIS (pre-fix)

Date: 2026-10-06
Scope: functionality/reliability only. No personality/Phase-1 changes.

## Architecture as actually implemented

There is **no** `AgentOrchestrator`, no specialist agents, no goal executor, no
ReAct loop, no WebSocket, and no router in this repository. Verified by grep over
`jarvis/`, `api_server.py`, and `frontend/src/`. The only occurrence of the word
"orchestrator" is a docstring in `jarvis/brain.py`.

The real pipeline is:

```
InputArea (frontend/src/components/InputArea.jsx)
  -> useJarvisState.sendMessage (frontend/src/hooks/useJarvisState.js)
  -> axios POST /api/chat {message}
  -> api_server.py :: chat()  -> single global JarvisBrain
  -> JarvisBrain.ask(user_text)
  -> Gemini chats.create(system_instruction=JARVIS_SYSTEM_PROMPT + memory)
  -> chat.send_message(user_text)   <-- user text is passed through verbatim
  -> if response.function_calls: execute tools, feed FunctionResponse back
  -> return response.text
  -> frontend ChatPanel
```

Message-role assignment in this pipeline is **correct**: the user's text is sent
as a plain `user` message. There is no goal extraction, no specialist
instruction, and no template that replaces it. So the "user input replaced by an
extracted goal" class of bug does not exist here.

## Confirmed bugs

### B1 — P0 — Configured model is unavailable
- File: `config.yaml` (`llm.model: gemini-3.5-flash`)
- Function: `jarvis/config.py::LLM_MODEL`, `JarvisBrain._get_chat`
- Root cause: `gemini-3.5-flash` returns HTTP 503 UNAVAILABLE on every request.
  Measured live: 4/4 attempts 503. Same for `gemini-3.8-flash`, `gemini-3.7-flash`,
  `gemini-flash-latest`, `gemini-3-flash-preview`. `gemini-3.6-flash` and
  `gemini-3.5-flash-lite` respond. There is no retry and no fallback anywhere.
- Impact: every user message returns
  `I encountered an error communicating with my brain: 503 ...`. 100% failure.
  This is the dominant cause of "the agent ignores me".
- Fix: retry + ordered model fallback inside the brain, driven by config.

### B2 — P0 — User messages sent while busy are silently discarded
- File: `frontend/src/hooks/useJarvisState.js`
- Function: `sendMessage`
- Root cause: `if (!trimmed || thinkingRef.current) return`. A message submitted
  during an in-flight request is dropped on the floor — never sent, never queued,
  never surfaced. `App.jsx` also passes `disabled={isThinking}` to `InputArea`,
  and `InputArea.handleSubmit` returns early when disabled.
- Impact: rapid consecutive messages are lost. Direct violation of "the exact
  user message must reach the model".
- Fix: queue messages and drain in order instead of dropping.

### B3 — P0 — One shared conversation for all users
- File: `api_server.py`
- Function: module-level `brain = JarvisBrain()`, `chat`, `clear_conversation`
- Root cause: a single global brain instance, no session/conversation id anywhere
  in the codebase (verified by grep). `/api/clear` wipes every user's history.
- Impact: conversation A and conversation B share Gemini history, so A's facts
  leak into B and vice versa. `clearChat` in the frontend also never calls
  `/api/clear`, so the UI "clean slate" is a lie.
- Fix: per-`conversation_id` brain registry.

### B4 — P1 — Multi-step tool chains are dropped
- File: `jarvis/brain.py`
- Function: `JarvisBrain.ask`
- Root cause: `for function_call in response.function_calls:` iterates the
  function calls of the **first** response only. `response` is reassigned inside
  the loop, but the loop is never re-entered. If the model requests a tool again
  after receiving the first result, that request is discarded and the method
  falls through to `response.text or "Standing by, Boss."`
- Impact: Reproduced with a fake chat — chain of 2 tool calls returns
  `"Standing by, Boss."` and the second tool never runs. The user's actual
  request is answered with a non-answer.
- Fix: iterate until the model stops requesting tools, with a bounded budget.

### B5 — P1 — Silent fallback text masks real failures
- File: `jarvis/brain.py`, `api_server.py`, `frontend/src/utils/api.js`
- Functions: `JarvisBrain.ask` (returns error text as if it were an answer),
  `chat()` (returns HTTP 200 with `error=True`), `sendJarvisMessage`
  (converts any non-2xx into a canned in-persona string)
- Root cause: transport, auth, quota, and model failures are indistinguishable
  from genuine model replies in the UI.
- Impact: a 429 quota error is displayed to the user as if the assistant had
  said it. Makes every other bug invisible.
- Fix: propagate a real error flag; do not invent assistant text for failures.

### B6 — P1 — Shutdown keyword matching discards legitimate messages
- File: `jarvis/main.py`
- Function: `run_assistant`, `run_wake_word_mode.on_wake`
- Root cause: `if any(trigger in query.lower() for trigger in [..., "exit",
  "quit", "shutdown"])`. Unanchored substring match.
- Impact: "Explain the exit code in Python" or "Quite interesting, tell me
  about recursion" terminate the process without answering.
- Fix: normalized exact-match on the whole utterance only.

### B7 — P2 — History "trimming" wipes the entire conversation
- File: `jarvis/brain.py`
- Function: `_trim_history`
- Root cause: on reaching `MAX_HISTORY_MESSAGES` it calls `reset_conversation()`,
  discarding **all** turns instead of keeping a recent window.
- Impact: "My name is Alex." followed by ~20 later turns, then "What is my
  name?" loses the answer. Context is destroyed at an arbitrary boundary.
- Fix: rebuild the chat from a recent window, preserving the system prompt.

### B8 — P2 — Tools block the API server on microphone/stdin
- File: `jarvis/tools/general.py`, `media.py`, `dictionary_tool.py`,
  `email_tool.py`, `nasa.py`
- Functions: `search_wikipedia`, `search_wikihow`, `repeat_words`,
  `play_music`, `take_screenshot`, `lookup_dictionary`, `send_email`,
  `get_nasa_apod`
- Root cause: these tools call `listen()` when an argument is empty. `listen()`
  falls back to `input()` when `sys.stdin.isatty()`, which inside a uvicorn
  worker blocks the event loop on stdin. Tools also call `speak()`, printing
  `[JARVIS]: ...` into the server process.
- Impact: a model tool call can hang the HTTP server and steal terminal input.
- Fix: tools must never prompt for input; return an instruction for the model
  to ask the user in its reply instead.

### B9 — P2 — Empty/invalid input is not handled end to end
- File: `api_server.py`, `frontend/src/utils/api.js`
- Root cause: `/api/chat` raises HTTP 400 for blank input; the frontend's axios
  interceptor turns that 400 into the canned string "Something glitched on my
  end, Boss." instead of a validation message.
- Impact: user gets a misleading assistant reply for a trivial input error.
- Fix: reject blank input in the hook before the request; surface real HTTP
  error details.

## Deliberately NOT changed
- System prompt strength / persona (Phase 1 scope). `JARVIS_SYSTEM_PROMPT` is not
  the cause of any bug above; B1–B9 are.
- Frontend visuals, components, animation, framework.
- No new agents, memory subsystems, or orchestration layer — none are missing.

---

# PHASE 0 — FIX REPORT (post-fix)

All nine bugs above are fixed. Each fix has a regression test that was verified
to **fail** when the fix is reverted, so the tests are proven to detect the
original defects rather than merely passing.

## Verification

- `python3 -m pytest tests/` — 76 passed.
- `npm run build` (frontend) — clean build.
- `python3 -m compileall jarvis api_server.py start.py` — clean.
- `python3 -m jarvis.main --test` now reports real pass/fail per check instead of
  unconditionally printing PASS, and includes two Phase 0 checks:
  "User input forwarded verbatim" and "Minimal direct-model path".

## Live model evidence

- `gemini-3.5-flash` (previously configured): 503 UNAVAILABLE on every attempt.
- `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.0-flash`: 404 retired.
- Working: `gemini-3.6-flash`, `gemini-3.5-flash-lite`.
- `config.yaml` now points at `gemini-3.6-flash` with `gemini-3.5-flash-lite` and
  `gemini-3.1-flash-lite-preview` as ordered fallbacks.
- The minimal direct path returned `4` for "What is 2 + 2?", confirming the base
  model interaction is sound and that the historical failures were in the
  orchestration/state layer.

**Note:** the Gemini free tier on this key is capped at 20 requests/day and is now
exhausted, so live tool-calling checks return HTTP 429. This is an account quota
limit, not a code defect. All logic-level behaviour is covered by offline tests.
---

# PHASE 0.5 — MODEL VERIFICATION RECORD

Recorded 2026-10-06. Every claim below was checked against a live provider
catalog, not assumed.

## Gemini (via `google-genai`, `models.list`)

| Model | Result |
|---|---|
| `gemini-3.6-flash` | responds; **429 free-tier quota** on this key |
| `gemini-3.5-flash-lite` | responds — live verified, `2+2` -> `4` |
| `gemini-3.5-flash` | **503 UNAVAILABLE on every attempt** (was the Phase 0 default) |
| `gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-flash-latest` | 503 |
| `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.0-flash`, `gemini-2.5-pro` | **404 retired** |

## OpenRouter (`GET /api/v1/models`, 465 models, 16 free)

Verified ids, context length and `tools` support from the catalog:

- `nvidia/nemotron-3.5-lightning:free` — 1,000,000 ctx, tools yes
- `nvidia/nemotron-3-ultra-550b-a55b:free` — 1,000,000 ctx, tools yes
- `nvidia/nemotron-3-super-120b-a12b:free` — 262,144 ctx, tools yes
- `inclusionai/ling-3.0-flash-sante:free` — 262,144 ctx, tools yes

**Live calls unverified: no `OPENROUTER_API_KEY` configured on this machine.**

## Space Bunny / Big Pickle — provider correction

A catalog search of OpenRouter for `bunny`, `pickle`, `space` and `big-pickle`
returned **no matches**. These are not OpenRouter models.

They are **OpenCode Zen** models, confirmed against `GET
https://opencode.ai/zen/v1/models` (an OpenAI-compatible endpoint):

- `space-bunny-free` — present
- `big-pickle` — present
- also present: `nemotron-3.5-lightning-free`, `nemotron-3-ultra-free`,
  `ling-3.1-flash-free`, `mimo-v2.6-flash-free`, `fledge-alpha-free`

So a third provider (`opencode_zen`) was added rather than mis-filing these ids
under OpenRouter. The Zen catalog does **not** advertise tool calling, so
`tool_calling: false` is recorded for both models — which means the router will
never send them a tool-required request. That is a conservative, honest
annotation, not a claim that they cannot ever do it.

**Live calls unverified: no `OPENCODE_ZEN_API_KEY` configured on this machine.**

## Honest summary

| Acceptance criterion | Status |
|---|---|
| Gemini still works | **Verified live** |
| OpenRouter works | Adapter implemented + unit-tested; **live unverified** (no key) |
| >=1 verified free OpenRouter model | 4 verified in the catalog; **live unverified** |
| Space Bunny / Big Pickle | Ids verified against OpenCode Zen; **live unverified** |

The three provider integrations that could not be exercised live are covered by
transport-level unit tests (message translation, tool-schema generation, HTTP
error classification, redaction) plus a `probe()` test that skips cleanly when
credentials are absent. No model id in `config.yaml` was invented.
