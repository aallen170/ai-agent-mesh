"""
smoke_test_classifier.py — Offline unit tests for AIMESH-12.

Runs without Redis or a live LiteLLM gateway by mocking the GatewayClient.
Tests three scenarios:

  1. Direct classify() — model returns valid JSON → correct tier extracted.
  2. Auto-classify path through TaskRouter — tier=None → classifier called,
     record dispatched with the classified tier.
  3. Fallback — model call raises an exception → tier defaults to 2,
     ClassificationResult.fallback == True.

Run with:
    python scripts/smoke_test_classifier.py
"""
from __future__ import annotations

import json
import sys
import types
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal stubs so we can import the classifier without a real Redis / gateway
# ---------------------------------------------------------------------------

def _make_completion_response(content: str):
    """Produce a minimal fake ChatCompletion object."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

# We import after stubs are in place so that openai / redis aren't required.
sys.path.insert(0, ".")

from src.control_plane.classifier.prompt import build_messages, parse_response
from src.control_plane.classifier.classifier import ClassificationResult, TaskClassifier


class TestPromptBuilder(unittest.TestCase):
    """Verify build_messages shapes the messages list correctly."""

    def test_system_message_present(self):
        msgs = build_messages("What is 2+2?", task_type="math")
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("Tier 0", msgs[0]["content"])

    def test_user_message_contains_prompt(self):
        msgs = build_messages("Explain transformers", task_type="llm_inference")
        user_msg = msgs[1]["content"]
        self.assertIn("Explain transformers", user_msg)
        self.assertIn("llm_inference", user_msg)

    def test_unknown_task_type_fallback(self):
        msgs = build_messages("Hello world")
        self.assertIn("unknown", msgs[1]["content"])


class TestParseResponse(unittest.TestCase):
    """Verify the JSON parser handles clean and messy model output."""

    def test_clean_json(self):
        raw = '{"tier": 1, "reasoning": "Medium length summarisation."}'
        tier, reasoning = parse_response(raw)
        self.assertEqual(tier, 1)
        self.assertIn("summarisation", reasoning)

    def test_json_wrapped_in_markdown(self):
        raw = '```json\n{"tier": 2, "reasoning": "Complex code task."}\n```'
        tier, reasoning = parse_response(raw)
        self.assertEqual(tier, 2)

    def test_json_with_preamble(self):
        raw = 'Sure! Here is the JSON:\n{"tier": 0, "reasoning": "Simple Q&A."}'
        tier, reasoning = parse_response(raw)
        self.assertEqual(tier, 0)

    def test_all_valid_tiers(self):
        for t in range(5):
            raw = json.dumps({"tier": t, "reasoning": f"Tier {t} task."})
            tier, _ = parse_response(raw)
            self.assertEqual(tier, t)

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            parse_response("I cannot classify this.")

    def test_out_of_range_tier_raises(self):
        with self.assertRaises(ValueError):
            parse_response('{"tier": 5, "reasoning": "oops"}')

    def test_missing_tier_key_raises(self):
        with self.assertRaises(ValueError):
            parse_response('{"reasoning": "No tier here."}')


class TestTaskClassifier(unittest.TestCase):
    """Verify TaskClassifier.classify() against a mocked GatewayClient."""

    def _make_classifier(self, response_content: str) -> TaskClassifier:
        gateway = MagicMock()
        gateway.complete.return_value = _make_completion_response(response_content)
        return TaskClassifier(gateway_client=gateway)

    # --- Happy path ---

    def test_classify_returns_correct_tier(self):
        clf = self._make_classifier('{"tier": 1, "reasoning": "8B model sufficient."}')
        result = clf.classify("Summarise this article", task_type="summarise")
        self.assertEqual(result.tier, 1)
        self.assertFalse(result.fallback)
        self.assertGreater(result.elapsed_ms, 0)
        self.assertEqual(result.model, "tier2/llama3.2-3b-desktop")

    def test_classify_tier4(self):
        clf = self._make_classifier('{"tier": 4, "reasoning": "Legal contract review."}')
        result = clf.classify("Review this NDA for GDPR compliance.", task_type="review")
        self.assertEqual(result.tier, 4)
        self.assertFalse(result.fallback)

    def test_classify_passes_task_type_to_gateway(self):
        gateway = MagicMock()
        gateway.complete.return_value = _make_completion_response(
            '{"tier": 0, "reasoning": "Simple lookup."}'
        )
        clf = TaskClassifier(gateway_client=gateway)
        clf.classify("What is the capital of France?", task_type="question_answer")
        call_kwargs = gateway.complete.call_args
        messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][1]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        self.assertIn("question_answer", user_msg)

    # --- Fallback on model failure ---

    def test_fallback_on_connection_error(self):
        gateway = MagicMock()
        gateway.complete.side_effect = ConnectionError("gateway unreachable")
        clf = TaskClassifier(gateway_client=gateway, fallback_tier=2)
        result = clf.classify("Do something complex.")
        self.assertEqual(result.tier, 2)
        self.assertTrue(result.fallback)

    def test_fallback_on_unparseable_response(self):
        clf = self._make_classifier("I don't know what tier this should be.")
        result = clf.classify("Some task.")
        self.assertEqual(result.tier, 2)
        self.assertTrue(result.fallback)

    def test_custom_fallback_tier(self):
        gateway = MagicMock()
        gateway.complete.side_effect = RuntimeError("model not loaded")
        clf = TaskClassifier(gateway_client=gateway, fallback_tier=1)
        result = clf.classify("Anything.")
        self.assertEqual(result.tier, 1)
        self.assertTrue(result.fallback)


class TestTaskRouterAutoClassify(unittest.TestCase):
    """
    Verify TaskRouter.submit() auto-classifies when tier is None.
    Uses fully mocked Redis and streams so no live infrastructure is needed.
    """

    def _make_router(self, classified_tier: int):
        """Return a TaskRouter with mocked Redis, streams, and a real classifier stub."""
        from src.control_plane.queue.router import TaskRouter
        from src.control_plane.queue.task import TaskRequest

        # Mock classifier
        mock_clf = MagicMock()
        mock_clf.classify.return_value = ClassificationResult(
            tier=classified_tier,
            reasoning="Auto-classified.",
            model="tier2/llama3.2-3b-desktop",
            elapsed_ms=42.0,
            fallback=False,
        )

        # Mock Redis internals
        mock_redis = MagicMock()
        mock_redis.r = MagicMock()
        mock_redis.r.pipeline.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_redis.r.pipeline.return_value.__exit__ = MagicMock(return_value=False)
        pipe = MagicMock()
        mock_redis.r.pipeline.return_value = pipe
        pipe.execute.return_value = [1, 1]

        mock_streams = MagicMock()
        mock_streams.enqueue_task.return_value = "1234-0"

        router = TaskRouter(
            redis_client=mock_redis,
            streams_client=mock_streams,
            classifier=mock_clf,
        )
        return router, mock_clf, mock_streams

    def test_auto_classify_selects_tier(self):
        from src.control_plane.queue.task import TaskRequest

        router, mock_clf, mock_streams = self._make_router(classified_tier=1)
        req = TaskRequest(
            task_type="summarise",
            payload={"prompt": "Summarise this document."},
            tier=None,
        )
        record = router.submit(req)

        # Classifier was called with the prompt text
        mock_clf.classify.assert_called_once()
        call_kwargs = mock_clf.classify.call_args[1]
        self.assertIn("Summarise", call_kwargs["prompt"])

        # Record was dispatched with the classified tier
        self.assertEqual(record.tier, 1)
        mock_streams.enqueue_task.assert_called_once_with(
            tier=1,
            task_type="summarise",
            payload={"prompt": "Summarise this document."},
            task_id=req.task_id,
        )

    def test_explicit_tier_skips_classifier(self):
        from src.control_plane.queue.task import TaskRequest

        router, mock_clf, _ = self._make_router(classified_tier=0)
        req = TaskRequest(
            task_type="question_answer",
            payload={"prompt": "What is 2+2?"},
            tier=0,
        )
        router.submit(req)
        mock_clf.classify.assert_not_called()

    def test_no_classifier_falls_back_to_tier2(self):
        from src.control_plane.queue.router import TaskRouter
        from src.control_plane.queue.task import TaskRequest

        mock_redis = MagicMock()
        pipe = MagicMock()
        mock_redis.r.pipeline.return_value = pipe
        pipe.execute.return_value = [1, 1]
        mock_streams = MagicMock()
        mock_streams.enqueue_task.return_value = "1234-0"

        router = TaskRouter(
            redis_client=mock_redis,
            streams_client=mock_streams,
            classifier=None,
        )
        req = TaskRequest(
            task_type="llm_inference",
            payload={"prompt": "Do something."},
            tier=None,
        )
        record = router.submit(req)
        self.assertEqual(record.tier, 2)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestPromptBuilder,
        TestParseResponse,
        TestTaskClassifier,
        TestTaskRouterAutoClassify,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
