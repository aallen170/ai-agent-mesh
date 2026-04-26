"""
gateway_worker.py — GatewayWorker: the generic AIMESH worker agent.

This is the single Python worker implementation that runs on every device in
the mesh — tier 0 phones/tablets through tier 4 cloud.  It receives tasks from
the Redis stream for its tier, calls the LiteLLM gateway to run inference, and
publishes the result back to the control plane.

How it fits in the mesh
-----------------------
                   ┌─────────────────┐
                   │  Control Plane  │
                   │  (TaskRouter)   │
                   └────────┬────────┘
                            │ Redis Stream (tier-N)
               ┌────────────▼────────────┐
               │       GatewayWorker     │  ← runs on every device
               │  BaseWorker lifecycle   │
               │  + GatewayClient call   │
               └────────────┬────────────┘
                            │ HTTPS
               ┌────────────▼────────────┐
               │     LiteLLM Gateway     │  ← routes to the right backend
               └─────────────────────────┘

Task payload contract
---------------------
GatewayWorker handles tasks of type ``"llm_inference"``.

Required payload fields:
    messages    list[dict]  OpenAI-format message list.
                            e.g. [{"role": "user", "content": "Hello"}]

Optional payload fields:
    model       str         LiteLLM model name to use for this task.
                            e.g. "tier2/llama3.1-70b-desktop"
                            Defaults to the first model_id in device_config.yaml.
    temperature float       Sampling temperature (default: 0.7).
    max_tokens  int         Max output tokens (default: 1024).
    stream      bool        Must be False — streaming not supported over Redis.

Any other task_type is handled with a descriptive error result so the worker
loop keeps running rather than crashing.

Result payload
--------------
On success:
    text        str         The model's response text.
    model       str         The LiteLLM model name that was used.
    usage       dict        Token counts: prompt_tokens, completion_tokens, total_tokens.
    finish_reason str       OpenAI finish reason (e.g. "stop", "length").

On error:
    ResultEnvelope.error is set; result dict is empty.

Usage
-----
Typically started via scripts/run_worker.py:

    python scripts/run_worker.py --config config/my_device.yaml

Or directly in code (useful for testing):

    config = DeviceConfig.from_dict({
        "device_id": "test-device",
        "name": "Test",
        "tier": 2,
        "models": ["tier2/llama3.1-70b-desktop"],
    })
    worker = GatewayWorker(config)
    worker.run()   # blocks until stop() is called
"""
from __future__ import annotations

import logging
from typing import Any

from .config import DeviceConfig
from .contract import BaseWorker, ResultEnvelope, TaskEnvelope
from ..control_plane.gateway import GatewayClient

logger = logging.getLogger(__name__)

# Task type this worker handles
_TASK_TYPE_INFERENCE = "llm_inference"

# Payload defaults
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_MAX_TOKENS = 1024


class GatewayWorker(BaseWorker):
    """
    Generic AIMESH worker — routes every task through the LiteLLM gateway.

    One codebase, every device.  The gateway handles the per-device routing
    so this worker never needs to know whether it's talking to Ollama,
    MLX, RunPod, or Claude.

    Parameters
    ----------
    config      DeviceConfig loaded from the device's YAML file.
    redis_url   Override the Redis URL from config (useful in tests).
    """

    def __init__(
        self,
        config: DeviceConfig,
        redis_url: str | None = None,
    ) -> None:
        super().__init__(config, redis_url=redis_url)
        self._gateway = GatewayClient(base_url=config.litellm_url)
        logger.info(
            "GatewayWorker initialised (device=%r, tier=%d, gateway=%s)",
            config.device_id, config.tier, config.litellm_url,
        )

    # ------------------------------------------------------------------
    # BaseWorker interface
    # ------------------------------------------------------------------

    def process_task(self, task: TaskEnvelope) -> ResultEnvelope:
        """
        Handle a single task from the Redis stream.

        Dispatches by task_type:
          - ``"llm_inference"`` → calls the LiteLLM gateway.
          - anything else       → returns an error result (worker keeps running).
        """
        if task.task_type == _TASK_TYPE_INFERENCE:
            return self._handle_inference(task)

        logger.warning(
            "GatewayWorker received unknown task_type %r (task_id=%s) — skipping",
            task.task_type, task.task_id,
        )
        return ResultEnvelope(
            task_id=task.task_id,
            result={},
            error=f"Unsupported task_type: {task.task_type!r}. "
                  f"GatewayWorker only handles {_TASK_TYPE_INFERENCE!r}.",
        )

    # ------------------------------------------------------------------
    # Inference handler
    # ------------------------------------------------------------------

    def _handle_inference(self, task: TaskEnvelope) -> ResultEnvelope:
        """
        Call the LiteLLM gateway and return the response as a ResultEnvelope.

        Picks the model from the payload if provided, otherwise falls back to
        the first model_id declared in the device config.
        """
        payload = task.payload

        # Resolve model: payload takes priority, then device config default
        model = payload.get("model") or self._default_model()
        if not model:
            return ResultEnvelope(
                task_id=task.task_id,
                result={},
                error=(
                    "No model specified in task payload and device config has no "
                    "model_ids. Set 'model' in the task payload or add 'models' "
                    "to device_config.yaml."
                ),
            )

        messages: list[dict[str, Any]] = payload.get("messages", [])
        if not messages:
            return ResultEnvelope(
                task_id=task.task_id,
                result={},
                error="Task payload is missing required field 'messages'.",
            )

        # Forward optional generation params; unknown keys are dropped by LiteLLM
        kwargs: dict[str, Any] = {}
        if "temperature" in payload:
            kwargs["temperature"] = float(payload["temperature"])
        if "max_tokens" in payload:
            kwargs["max_tokens"] = int(payload["max_tokens"])

        logger.debug(
            "Calling gateway: model=%r, messages=%d, task_id=%s",
            model, len(messages), task.task_id,
        )

        try:
            response = self._gateway.complete(
                model=model,
                messages=messages,
                **kwargs,
            )
        except Exception as exc:
            logger.exception(
                "Gateway call failed for task %s (model=%r)", task.task_id, model
            )
            return ResultEnvelope(
                task_id=task.task_id,
                result={},
                error=f"Gateway error ({type(exc).__name__}): {exc}",
            )

        choice = response.choices[0]
        usage = response.usage

        result: dict[str, Any] = {
            "text": choice.message.content or "",
            "model": model,
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
        }

        logger.debug(
            "Gateway response received: task_id=%s, tokens=%s",
            task.task_id,
            result["usage"]["total_tokens"],
        )

        return ResultEnvelope(task_id=task.task_id, result=result)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _default_model(self) -> str | None:
        """Return the first model_id from the device config, or None."""
        return self.config.model_ids[0] if self.config.model_ids else None
