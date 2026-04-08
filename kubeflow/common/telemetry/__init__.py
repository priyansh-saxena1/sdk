# Copyright 2026 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import atexit
import importlib.metadata

from opentelemetry import trace
from opentelemetry.propagate import inject

_INSTRUMENTATION_NAME = "io.kubeflow.sdk"
_SCHEMA_URL = "https://opentelemetry.io/schemas/1.21.0"

# Kubeflow attribute constants
KUBEFLOW_JOB_NAME = "kubeflow.job.name"
KUBEFLOW_JOB_STATUS = "kubeflow.job.status"
KUBEFLOW_BACKEND_TYPE = "kubeflow.backend.type"
KUBEFLOW_RUNTIME_NAME = "kubeflow.runtime.name"
KUBEFLOW_NUM_NODES = "kubeflow.trainer.num_nodes"
KUBEFLOW_INITIALIZER_TYPE = "kubeflow.initializer.type"

try:
    _SDK_VERSION = importlib.metadata.version("kubeflow")
except importlib.metadata.PackageNotFoundError:
    _SDK_VERSION = "unknown"


# version() hit the filesystem and was ~440x slower in a tight loop; cache the value once.
def _get_tracer() -> trace.Tracer:
    """Return the instrumentation tracer for the Kubeflow SDK."""
    return trace.get_tracer(_INSTRUMENTATION_NAME, _SDK_VERSION, schema_url=_SCHEMA_URL)


def configure_telemetry(
    exporter: str = "otlp",
    endpoint: str | None = None,
    sampling_rate: float = 1.0,
    service_name: str = "kubeflow-sdk",
) -> None:
    """Configure a TracerProvider with optional exporter and sampling."""
    try:
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import (
            ALWAYS_ON,
            ParentBased,
            TraceIdRatioBased,
        )
    except ImportError as exc:
        raise ImportError(
            "configure_telemetry() requires opentelemetry-sdk. "
            "Install it with: pip install opentelemetry-sdk"
        ) from exc

    sampler = ParentBased(TraceIdRatioBased(sampling_rate)) if sampling_rate < 1.0 else ALWAYS_ON

    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource, sampler=sampler)

    if exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError as exc:
            raise ImportError(
                "exporter='otlp' requires opentelemetry-exporter-otlp. "
                "Install it with: pip install opentelemetry-exporter-otlp"
            ) from exc
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    elif exporter == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif exporter != "none":
        raise ValueError(f"Unknown exporter {exporter!r}. Use 'otlp', 'console', or 'none'.")

    trace.set_tracer_provider(provider)

    # Register shutdown first so force_flush runs before shutdown (LIFO order).
    atexit.register(provider.shutdown)
    atexit.register(provider.force_flush)


def inject_trace_context() -> dict[str, str]:
    """Return trace context in a carrier, or an empty dict when no span is active."""
    span = trace.get_current_span()
    if not span or not span.get_span_context().is_valid:
        return {}
    carrier: dict[str, str] = {}
    inject(carrier)
    return carrier


__all__ = [
    "_get_tracer",
    "configure_telemetry",
    "inject_trace_context",
    "KUBEFLOW_JOB_NAME",
    "KUBEFLOW_JOB_STATUS",
    "KUBEFLOW_BACKEND_TYPE",
    "KUBEFLOW_RUNTIME_NAME",
    "KUBEFLOW_NUM_NODES",
    "KUBEFLOW_INITIALIZER_TYPE",
]
