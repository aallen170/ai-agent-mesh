"""
telemetry.py — Central OpenTelemetry SDK setup for AIMESH.

Call ``configure_telemetry()`` once at process startup (before importing any
instrumented modules).  After that, every module can call ``get_tracer()`` and
``get_meter()`` to obtain its own named tracer/meter without touching the
providers directly.

Architecture
------------
Python app  →  OTLP/gRPC (port 4317)  →  OTel Collector
                                               │
                                    Prometheus exporter (port 8889)
                                               │
                                          Prometheus scrape
                                               │
                                            Grafana

Configuration (via environment variables)
-----------------------------------------
OTEL_EXPORTER_OTLP_ENDPOINT   gRPC endpoint for the OTel Collector.
                               Default: "http://localhost:4317"
OTEL_SERVICE_NAME              Service name attached to all telemetry.
                               Default: "aimesh-control-plane"
AIMESH_TELEMETRY_ENABLED       Set to "false" to disable telemetry entirely
                               (useful in unit tests).  Default: "true".

Usage
-----
    # At process startup (e.g. run_worker.py or a future main.py):
    from src.telemetry import configure_telemetry
    configure_telemetry()

    # In any module:
    from src.telemetry import get_tracer, get_meter
    tracer = get_tracer(__name__)
    meter  = get_meter(__name__)
"""
from __future__ import annotations

import logging
import os

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

_configured = False


def configure_telemetry(
    service_name: str | None = None,
    otlp_endpoint: str | None = None,
    enabled: bool | None = None,
) -> None:
    """
    Initialise the OTel SDK and register global providers.

    Safe to call multiple times — subsequent calls are no-ops.

    Parameters
    ----------
    service_name    Override OTEL_SERVICE_NAME env var.
    otlp_endpoint   Override OTEL_EXPORTER_OTLP_ENDPOINT env var.
    enabled         Override AIMESH_TELEMETRY_ENABLED env var.
    """
    global _configured
    if _configured:
        return
    _configured = True

    # --- Resolve config -------------------------------------------------------
    if enabled is None:
        enabled = os.environ.get("AIMESH_TELEMETRY_ENABLED", "true").lower() != "false"

    if not enabled:
        logger.info("Telemetry disabled (AIMESH_TELEMETRY_ENABLED=false)")
        return

    svc_name = service_name or os.environ.get("OTEL_SERVICE_NAME", "aimesh-control-plane")
    endpoint = otlp_endpoint or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
    )

    resource = Resource.create({"service.name": svc_name})

    # --- Traces ---------------------------------------------------------------
    span_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)

    # --- Metrics --------------------------------------------------------------
    metric_exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
    metric_reader = PeriodicExportingMetricReader(
        metric_exporter,
        export_interval_millis=15_000,  # push every 15 s
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    logger.info(
        "Telemetry configured (service=%r, endpoint=%r)", svc_name, endpoint
    )


def get_tracer(name: str) -> trace.Tracer:
    """Return a named tracer from the global provider."""
    return trace.get_tracer(name)


def get_meter(name: str) -> metrics.Meter:
    """Return a named meter from the global provider."""
    return metrics.get_meter(name)
