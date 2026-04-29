"""
contract.py — The formal interface between the AIMESH control plane and worker agents.

This module defines:

  TaskEnvelope    The shape of a task as a worker receives it from the Redis stream.
  ResultEnvelope  The shape of a result a worker must return after processing.
  BaseWorker      Abstract base class that every device-specific worker extends.

How it fits together
--------------------
The control plane writes tasks to Redis Streams (one stream per tier).
A BaseWorker subclass running on each device:

  1. Registers its DeviceInfo with the control-plane registry.
  2. Starts a background heartbeat loop.
  3. Loops over its tier stream, unwraps each message into a TaskEnvelope,
     and calls the subclass's process_task() method.
  4. Wraps the returned ResultEnvelope and publishes it to the results stream.
  5. ACKs the stream message so it isn't redelivered.

To onboard a new device, subclass BaseWorker, implement process_task(), and
call worker.run() in your entrypoint script.  Everything else is handled here.

Example
-------
    class OllamaWorker(BaseWorker):
        def process_task(self, task: TaskEnvelope) -> ResultEnvelope:
            prompt = task.payload.get("prompt", "")
            response = ollama.chat(model=task.payload["model"], messages=[...])
            return ResultEnvelope(task_id=task.task_id, result={"text": response})

    if __name__ == "__main__":
        config = DeviceConfig.from_yaml("device_config.yaml")
        OllamaWorker(config).run()
"""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import trace

from ..control_plane.redis.client import RedisClient
from ..control_plane.redis.pubsub import PubSubClient
from ..control_plane.redis.streams import StreamsClient
from ..control_plane.registry.device import Capabilities, DeviceInfo, DeviceStatus, Tier
from ..control_plane.registry.registry import DeviceRegistry
from ..telemetry import get_meter, get_tracer
from .config import DeviceConfig

logger = logging.getLogger(__name__)

_tracer = get_tracer(__name__)
_meter = get_meter(__name__)

_worker_tasks = _meter.create_counter(
    "aimesh.worker.tasks_processed",
    unit="1",
    description="Tasks processed by this worker agent",
)
_worker_duration = _meter.create_histogram(
    "aimesh.worker.task_duration",
    unit="s",
    description="Time spent processing a single task on this worker in seconds",
)


# ---------------------------------------------------------------------------
# Message types — shared between control plane and workers
# ---------------------------------------------------------------------------

@dataclass
class TaskEnvelope:
    """
    A task as delivered to a worker from the Redis stream.

    Fields
    ------
    task_id     Globally unique task identifier (UUID string).
    task_type   Application-defined string describing what to do
                (e.g. "llm_inference", "embedding", "summarise").
    payload     Arbitrary dict containing task-specific data
                (e.g. {"prompt": "...", "model": "llama3:8b"}).
    msg_id      Internal Redis stream message ID — the worker uses this
                to ACK the message after processing.  Do not modify.
    """
    task_id: str
    task_type: str
    payload: dict[str, Any]
    msg_id: str  # Redis stream message ID for ACK


@dataclass
class ResultEnvelope:
    """
    The result a worker returns after processing a TaskEnvelope.

    Fields
    ------
    task_id   Must match the task_id from the corresponding TaskEnvelope.
    result    Dict of output data (e.g. {"text": "...", "tokens": 128}).
              Use an empty dict if there is no meaningful output.
    error     Human-readable error message if processing failed.
              Leave as None for success.
    """
    task_id: str
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def success(self) -> bool:
        """True if the task completed without an error."""
        return self.error is None


# ---------------------------------------------------------------------------
# BaseWorker — the template every device-specific worker extends
# ---------------------------------------------------------------------------

class BaseWorker(ABC):
    """
    Abstract base class for all AIMESH worker agents.

    Subclasses must implement:
        process_task(task: TaskEnvelope) -> ResultEnvelope

    Everything else — registration, heartbeating, consuming the Redis stream,
    publishing results, and ACKing messages — is handled here.

    Usage
    -----
        class MyWorker(BaseWorker):
            def process_task(self, task: TaskEnvelope) -> ResultEnvelope:
                # run inference, return result
                return ResultEnvelope(task_id=task.task_id, result={...})

        MyWorker(config).run()

    Parameters
    ----------
    config      DeviceConfig loaded from device_config.yaml.
    redis_url   Override the Redis URL from config (useful in tests).
    """

    def __init__(
        self,
        config: DeviceConfig,
        redis_url: str | None = None,
    ) -> None:
        self.config = config
        url = redis_url or config.redis_url
        self._redis = RedisClient(url)
        self._streams = StreamsClient(self._redis)
        self._pubsub = PubSubClient(self._redis)
        self._registry = DeviceRegistry(self._redis, self._pubsub)
        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Abstract interface — subclasses implement this
    # ------------------------------------------------------------------

    @abstractmethod
    def process_task(self, task: TaskEnvelope) -> ResultEnvelope:
        """
        Run inference (or any other work) for the given task.

        Called once per task, synchronously in the consume loop.
        Raise any exception to signal failure — BaseWorker will catch it,
        publish an error result, and continue to the next task.

        Parameters
        ----------
        task    The TaskEnvelope received from the Redis stream.

        Returns
        -------
        ResultEnvelope with task_id matching task.task_id.
        Set ResultEnvelope.error to a string on failure rather than raising
        if you want more control over the error message.
        """
        ...

    # ------------------------------------------------------------------
    # DeviceInfo construction — override to add dynamic capabilities
    # ------------------------------------------------------------------

    def build_device_info(self) -> DeviceInfo:
        """
        Construct the DeviceInfo that will be registered with the control plane.

        Override this if you need to add runtime-detected capabilities
        (e.g. dynamically queried VRAM, loaded model list from Ollama API).
        """
        return DeviceInfo(
            device_id=self.config.device_id,
            tier=Tier(self.config.tier),
            name=self.config.name,
            capabilities=Capabilities(
                model_ids=self.config.model_ids,
                ram_gb=self.config.ram_gb,
                vram_gb=self.config.vram_gb,
                gpu_name=self.config.gpu_name,
                cpu_cores=self.config.cpu_cores,
                os=self.config.os,
            ),
        )

    # ------------------------------------------------------------------
    # Lifecycle — call run() to start; call stop() to shut down cleanly
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Start the worker.  Blocks until stop() is called or a fatal error occurs.

        Sequence:
          1. Ensure tier streams exist on Redis.
          2. Register this device with the control-plane registry.
          3. Start the background heartbeat thread.
          4. Enter the task consume loop.
          5. On exit: deregister from the registry and clean up.
        """
        logger.info(
            "Worker %r starting (tier=%d, models=%s)",
            self.config.device_id, self.config.tier, self.config.model_ids,
        )

        # Bootstrap streams in case control plane hasn't started yet
        self._streams.ensure_streams()

        device_info = self.build_device_info()
        self._registry.register(device_info)

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"heartbeat-{self.config.device_id}",
            daemon=True,
        )
        self._heartbeat_thread.start()

        try:
            self._consume_loop()
        finally:
            self._stop_event.set()
            self._registry.deregister(self.config.device_id)
            if self._heartbeat_thread:
                self._heartbeat_thread.join(timeout=5)
            self._redis.close()
            logger.info("Worker %r stopped cleanly", self.config.device_id)

    def stop(self) -> None:
        """
        Signal the worker to stop after the current task finishes.
        The run() call will return within one consume-loop iteration.
        """
        logger.info("Stop requested for worker %r", self.config.device_id)
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    def _heartbeat_loop(self) -> None:
        """
        Runs in a daemon thread.  Sends a heartbeat to the registry every
        heartbeat_interval seconds.  If the control plane has lost our
        record (e.g. it restarted), re-registers automatically.
        """
        logger.debug(
            "Heartbeat loop started (interval=%.1fs)", self.config.heartbeat_interval
        )
        while not self._stop_event.wait(timeout=self.config.heartbeat_interval):
            try:
                known = self._registry.heartbeat(
                    self.config.device_id, status=DeviceStatus.ONLINE
                )
                if not known:
                    logger.warning(
                        "Heartbeat rejected — re-registering %r", self.config.device_id
                    )
                    self._registry.register(self.build_device_info())
            except Exception:
                logger.exception("Error in heartbeat loop for %r", self.config.device_id)

    def _consume_loop(self) -> None:
        """
        Main task-processing loop.

        Reads tasks from this device's tier stream one at a time,
        calls process_task(), and publishes the result.  Blocks for
        2 seconds on each read if the stream is empty, then re-checks
        the stop event.
        """
        logger.info(
            "Consume loop started for tier-%d stream (consumer=%r)",
            self.config.tier, self.config.device_id,
        )
        while not self._stop_event.is_set():
            try:
                raw_tasks = self._streams.read_tasks(
                    tier=self.config.tier,
                    consumer_name=self.config.device_id,
                    count=1,
                    block_ms=2000,
                )
            except Exception:
                logger.exception("Error reading from task stream")
                time.sleep(1)
                continue

            for raw in raw_tasks:
                envelope = TaskEnvelope(
                    task_id=raw["task_id"],
                    task_type=raw["task_type"],
                    payload=raw["payload"],
                    msg_id=raw["_msg_id"],
                )
                self._handle_task(envelope)

    def _handle_task(self, task: TaskEnvelope) -> None:
        """
        Process a single task: call process_task(), publish result, ACK.

        Exceptions from process_task() are caught and turned into error
        results so the worker loop keeps running.
        """
        logger.info(
            "Task %s received (type=%s)", task.task_id, task.task_type
        )

        with _tracer.start_as_current_span(
            "aimesh.worker.process_task",
            attributes={
                "task.id": task.task_id,
                "task.type": task.task_type,
                "worker.device_id": self.config.device_id,
                "worker.tier": self.config.tier,
            },
        ) as span:
            start = time.time()

            # Mark device as busy while processing
            self._registry.heartbeat(self.config.device_id, status=DeviceStatus.BUSY)

            try:
                result_envelope = self.process_task(task)
            except Exception as exc:
                logger.exception("Task %s raised an exception", task.task_id)
                span.record_exception(exc)
                span.set_status(trace.StatusCode.ERROR, str(exc))
                result_envelope = ResultEnvelope(
                    task_id=task.task_id,
                    result={},
                    error=f"{type(exc).__name__}: {exc}",
                )

            elapsed = time.time() - start
            status_label = "completed" if result_envelope.success else "failed"
            span.set_attributes({
                "task.success": result_envelope.success,
                "task.duration_s": elapsed,
            })

            _worker_tasks.add(
                1,
                {
                    "device_id": self.config.device_id,
                    "tier": str(self.config.tier),
                    "status": status_label,
                },
            )
            _worker_duration.record(
                elapsed,
                {"device_id": self.config.device_id, "tier": str(self.config.tier)},
            )

            logger.info(
                "Task %s finished in %.2fs (success=%s)",
                task.task_id, elapsed, result_envelope.success,
            )

            try:
                self._streams.publish_result(
                    task_id=result_envelope.task_id,
                    tier=self.config.tier,
                    device_id=self.config.device_id,
                    result=result_envelope.result,
                    error=result_envelope.error,
                )
            except Exception:
                logger.exception("Failed to publish result for task %s", task.task_id)

            # Always ACK — even on error — so the message doesn't get redelivered
            try:
                self._streams.ack_task(self.config.tier, task.msg_id)
            except Exception:
                logger.exception("Failed to ACK task %s (msg_id=%s)", task.task_id, task.msg_id)

            # Return to online status
            self._registry.heartbeat(self.config.device_id, status=DeviceStatus.ONLINE)
