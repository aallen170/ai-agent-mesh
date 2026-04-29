"""
classifier.py — TaskClassifier: complexity-based tier classifier for AIMESH.

The classifier calls a small local model (default: tier2/llama3.2-3b-desktop)
through the LiteLLM gateway and returns the recommended compute tier for an
incoming task.

Fallback behaviour
------------------
If the gateway is unreachable or the model returns an unparseable response,
the classifier logs a warning and returns ``fallback_tier`` (default: 2) so
that task submission can continue without interruption.

Usage
-----
    from src.control_plane.gateway import GatewayClient
    from src.control_plane.classifier import TaskClassifier

    classifier = TaskClassifier(GatewayClient())
    result = classifier.classify("Summarise this paper in three bullet points.")
    print(result.tier, result.reasoning)

    # Or with an explicit task type:
    result = classifier.classify(
        prompt="def fib(n): ...",
        task_type="code_review",
    )
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from opentelemetry import trace

from ...telemetry import get_meter, get_tracer
from ..gateway.client import GatewayClient
from .prompt import build_messages, parse_response

logger = logging.getLogger(__name__)

_tracer = get_tracer(__name__)
_meter = get_meter(__name__)

# Metrics
_classify_counter = _meter.create_counter(
    "aimesh.classifier.calls",
    unit="1",
    description="Total classifier invocations",
)
_classify_duration = _meter.create_histogram(
    "aimesh.classifier.duration",
    unit="ms",
    description="Wall-clock time of the classifier model call in milliseconds",
)
_tier_counter = _meter.create_counter(
    "aimesh.classifier.tier_assigned",
    unit="1",
    description="Tiers assigned by the classifier",
)

# Model registered in infra/litellm_config.yaml (AIMESH-12).
# Small and fast — intended for low-latency classification at ingestion time.
_DEFAULT_MODEL = "tier2/llama3.2-3b-desktop"
_DEFAULT_FALLBACK_TIER = 2


@dataclass
class ClassificationResult:
    """
    The outcome of a single TaskClassifier.classify() call.

    Fields
    ------
    tier        Recommended compute tier (0–4).
    reasoning   One-sentence explanation from the model (or fallback note).
    model       The LiteLLM model name that produced the result.
    elapsed_ms  Wall-clock time of the model call in milliseconds.
    fallback    True if the classifier fell back to the default tier because
                the model call failed or returned an unparseable response.
    """
    tier: int
    reasoning: str
    model: str
    elapsed_ms: float
    fallback: bool = False


class TaskClassifier:
    """
    Classifies an incoming task prompt into a compute tier (0–4).

    Parameters
    ----------
    gateway_client  A connected GatewayClient pointing at the LiteLLM proxy.
    model           LiteLLM model name to use for classification.
                    Defaults to ``tier2/llama3.2-3b-desktop``.
                    Override via the AIMESH_CLASSIFIER_MODEL env var or pass
                    directly here.
    fallback_tier   Tier returned when the model is unreachable or the response
                    cannot be parsed.  Defaults to 2 (dGPU desktop — safe bet).
    temperature     Sampling temperature.  Low (default 0.1) keeps output
                    deterministic and JSON-shaped.
    max_tokens      Max tokens to generate.  The JSON response is short;
                    128 is generous.
    """

    def __init__(
        self,
        gateway_client: GatewayClient,
        model: str = _DEFAULT_MODEL,
        fallback_tier: int = _DEFAULT_FALLBACK_TIER,
        temperature: float = 0.1,
        max_tokens: int = 128,
    ) -> None:
        self._client = gateway_client
        self.model = model
        self.fallback_tier = fallback_tier
        self._temperature = temperature
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def classify(
        self,
        prompt: str,
        task_type: str | None = None,
    ) -> ClassificationResult:
        """
        Classify *prompt* and return the recommended compute tier.

        Parameters
        ----------
        prompt      The task content to classify — typically the user's prompt
                    or a brief description of the work to be done.
        task_type   Optional application-defined label
                    (e.g. "llm_inference", "summarise", "code_review").
                    Included in the prompt to help the model route correctly.

        Returns
        -------
        ClassificationResult with ``fallback=False`` on success, or
        ``fallback=True`` if the model was unreachable or its response could
        not be parsed (tier is set to ``self.fallback_tier`` in that case).
        """
        messages = build_messages(prompt=prompt, task_type=task_type)

        with _tracer.start_as_current_span(
            "aimesh.classifier.classify",
            attributes={
                "classifier.model": self.model,
                "task.type": task_type or "unknown",
            },
        ) as span:
            t0 = time.monotonic()
            try:
                response = self._client.complete(
                    model=self.model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                elapsed_ms = (time.monotonic() - t0) * 1000

                raw_text = response.choices[0].message.content or ""
                tier, reasoning = parse_response(raw_text)

                span.set_attributes({
                    "classifier.tier": tier,
                    "classifier.fallback": False,
                    "classifier.elapsed_ms": elapsed_ms,
                })

                _classify_counter.add(1, {"model": self.model, "fallback": "false"})
                _classify_duration.record(elapsed_ms, {"model": self.model})
                _tier_counter.add(1, {"tier": str(tier), "fallback": "false"})

                logger.debug(
                    "Classified as tier-%d in %.0fms (model=%s): %s",
                    tier, elapsed_ms, self.model, reasoning,
                )
                return ClassificationResult(
                    tier=tier,
                    reasoning=reasoning,
                    model=self.model,
                    elapsed_ms=elapsed_ms,
                    fallback=False,
                )

            except Exception as exc:
                elapsed_ms = (time.monotonic() - t0) * 1000
                span.set_attributes({
                    "classifier.tier": self.fallback_tier,
                    "classifier.fallback": True,
                    "classifier.elapsed_ms": elapsed_ms,
                })
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, str(exc))

                _classify_counter.add(1, {"model": self.model, "fallback": "true"})
                _classify_duration.record(elapsed_ms, {"model": self.model})
                _tier_counter.add(1, {"tier": str(self.fallback_tier), "fallback": "true"})

                logger.warning(
                    "Classifier failed (model=%s, %.0fms) — defaulting to tier %d. Error: %s",
                    self.model, elapsed_ms, self.fallback_tier, exc,
                )
                return ClassificationResult(
                    tier=self.fallback_tier,
                    reasoning=f"Classifier unavailable — defaulted to tier {self.fallback_tier}.",
                    model=self.model,
                    elapsed_ms=elapsed_ms,
                    fallback=True,
                )
