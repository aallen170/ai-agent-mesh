"""
prompt.py — System prompt, user message template, and response parser for the
AIMESH task complexity classifier.

The classifier asks a small local model (llama3.2:3b) to read an incoming task
and output a JSON object that names the lowest compute tier capable of handling
it well.  Forcing JSON output avoids fragile free-text parsing.

Tier reference (kept in sync with architecture.md)
---------------------------------------------------
Tier 0 — Edge (phones/tablets, 3 B models, MLX/MLC)
Tier 1 — iGPU laptop (7–8 B models, Ollama)
Tier 2 — dGPU desktop/laptop (13–70 B models, Ollama)
Tier 3 — RunPod serverless (70 B+ models, on-demand)
Tier 4 — Claude Sonnet/Opus (executive review, high-stakes)
"""
from __future__ import annotations

import json
import re

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a routing classifier for a distributed LLM inference mesh.
Your only job is to read an incoming task and decide which compute tier should
handle it — always choosing the LOWEST tier that can do the job well.

Tier definitions
----------------
Tier 0 (edge — phones/tablets, ~3 B model):
  Simple factual Q&A, single-sentence answers, basic unit conversion,
  short keyword extraction, yes/no questions, simple data formatting.

Tier 1 (iGPU laptop — ~8 B model):
  Multi-step reasoning over short context, code explanation (<50 lines),
  medium-length summarisation (<2 k tokens), translation, entity extraction,
  basic sentiment analysis, short creative writing (<200 words).

Tier 2 (dGPU desktop/laptop — 13 B–70 B model):
  Complex code generation/refactoring, long document analysis (>2 k tokens),
  multi-document synthesis, detailed technical explanations, structured data
  extraction from free text, longer creative writing.

Tier 3 (RunPod serverless — very large model, on-demand):
  Tasks requiring a very large context window (>32 k tokens), complex multi-hop
  reasoning chains, large-scale batch processing, specialised domain models.

Tier 4 (Claude Sonnet/Opus — cloud API):
  Tasks that require executive judgment, extremely high-stakes accuracy
  (medical, legal, financial decisions), final review/validation of lower-tier
  output, or where repeated failures at lower tiers suggest higher capability
  is needed.

Rules
-----
1. Always prefer lower tiers — only escalate if genuinely necessary.
2. Base your decision on the task content and context, not just the task type.
3. If the task type is unknown, infer from the content.
4. Respond with a JSON object and nothing else — no markdown, no explanation
   outside the JSON.

Response format (strict JSON, no extra keys)
--------------------------------------------
{
  "tier": <integer 0–4>,
  "reasoning": "<one sentence explaining the choice>"
}

Examples
--------
Task type: question_answer
Content: What is the capital of France?
{"tier": 0, "reasoning": "Simple factual lookup; a 3B edge model handles this easily."}

Task type: summarise
Content: Summarise this 5-page research paper on transformer attention mechanisms.
{"tier": 1, "reasoning": "Medium-length summarisation within 8B model capability."}

Task type: llm_inference
Content: Refactor this 800-line Python service to use async/await throughout and add type annotations.
{"tier": 2, "reasoning": "Large-scale code refactoring needs a 13B+ model for reliable output."}

Task type: llm_inference
Content: Analyse these 200 customer support transcripts (total ~80k tokens) and identify systemic issues.
{"tier": 3, "reasoning": "Context window exceeds what local models support; RunPod large model required."}

Task type: review
Content: Review this legal contract draft for compliance with GDPR and flag any liability clauses.
{"tier": 4, "reasoning": "High-stakes legal review requires Claude-level accuracy."}
"""

_USER_TEMPLATE = """\
Task type: {task_type}
Content: {prompt}
"""


def build_messages(prompt: str, task_type: str | None = None) -> list[dict[str, str]]:
    """
    Build the OpenAI-format messages list for a single classification call.

    Parameters
    ----------
    prompt      The task content / user prompt to classify.
    task_type   Optional application-defined task type label.  Falls back to
                "unknown" if not provided.
    """
    user_content = _USER_TEMPLATE.format(
        task_type=task_type or "unknown",
        prompt=prompt.strip(),
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_response(text: str) -> tuple[int, str]:
    """
    Extract ``(tier, reasoning)`` from the model's raw text response.

    The model is prompted to return strict JSON, but small models sometimes
    wrap it in markdown fences or add preamble.  This parser is tolerant:
    it extracts the first ``{...}`` block and parses it.

    Parameters
    ----------
    text    Raw string returned by the completion API.

    Returns
    -------
    (tier, reasoning) — tier is guaranteed to be in [0, 4].

    Raises
    ------
    ValueError  If no valid JSON object is found or required keys are missing.
    """
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object found in model response: {text!r}")

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from response: {text!r}") from exc

    if "tier" not in data:
        raise ValueError(f"Response JSON missing 'tier' key: {data}")

    raw_tier = int(data["tier"])
    if raw_tier not in range(5):
        raise ValueError(f"Tier {raw_tier!r} out of valid range 0–4")

    reasoning = str(data.get("reasoning", "")).strip() or "No reasoning provided."
    return raw_tier, reasoning
