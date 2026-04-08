from __future__ import annotations

from datetime import datetime
import logging

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
import pytest

from kubeflow.common.telemetry import KUBEFLOW_JOB_NAME, _get_tracer, configure_telemetry
from kubeflow.trainer.api.trainer_client import TrainerClient
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types


class _StubBackend:
    def __init__(self, job_name: str = "job-123", exc: Exception | None = None):
        self._job_name = job_name
        self._exc = exc

    def train(self, *args, **kwargs) -> str:
        if self._exc:
            raise self._exc
        return self._job_name


class _StatusBackend:
    def wait_for_job_status(
        self,
        name: str,
        status: set[str],
        timeout: int,
        polling_interval: int,
        callbacks=None,
    ) -> types.TrainJob:
        runtime = types.Runtime(
            name="dummy",
            trainer=types.RuntimeTrainer(
                trainer_type=types.TrainerType.CUSTOM_TRAINER,
                framework="dummy",
                image="dummy",
            ),
        )
        last_job = None
        for job_status in [
            constants.TRAINJOB_CREATED,
            constants.TRAINJOB_RUNNING,
            constants.TRAINJOB_COMPLETE,
        ]:
            last_job = types.TrainJob(
                name=name,
                runtime=runtime,
                steps=[],
                num_nodes=1,
                creation_timestamp=datetime.now(),
                status=job_status,
            )
            if callbacks:
                for callback in callbacks:
                    callback(last_job)
        return last_job


# NOTE: This test must run before any test that sets a global TracerProvider.
# It relies on the default no-op provider behavior.


def test_no_op_before_configure():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = _get_tracer()
    with tracer.start_as_current_span("no-op-span"):
        pass

    assert not exporter.get_finished_spans()


def _make_client() -> TrainerClient:
    client = TrainerClient.__new__(TrainerClient)
    return client


def test_span_name_convention(exporter: InMemorySpanExporter):
    client = _make_client()
    client.backend = _StubBackend(job_name="job-123")

    client.train()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "TrainerClient.train"


def test_job_name_attribute(exporter: InMemorySpanExporter):
    client = _make_client()
    client.backend = _StubBackend(job_name="job-456")

    client.train()

    span = exporter.get_finished_spans()[0]
    assert span.attributes.get(KUBEFLOW_JOB_NAME) == "job-456"


def test_exception_recording(exporter: InMemorySpanExporter):
    client = _make_client()
    client.backend = _StubBackend(exc=ValueError("boom"))

    with pytest.raises(ValueError, match="boom"):
        client.train()

    span = exporter.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    exception_events = [event for event in span.events if event.name == "exception"]
    assert len(exception_events) == 1


def test_status_change_events(exporter: InMemorySpanExporter):
    client = _make_client()
    client.backend = _StatusBackend()

    def _otel_cb(job: types.TrainJob) -> None:
        trace.get_current_span().add_event("status_change")

    tracer = _get_tracer()
    with tracer.start_as_current_span("TrainerClient.wait_for_job_status"):
        client.wait_for_job_status(
            name="job-evt",
            status={constants.TRAINJOB_COMPLETE},
            timeout=10,
            polling_interval=1,
            callbacks=[_otel_cb],
        )

    span = exporter.get_finished_spans()[0]
    assert [event.name for event in span.events] == [
        "status_change",
        "status_change",
        "status_change",
    ]


def test_sampling_rate_zero(exporter: InMemorySpanExporter, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.WARNING, logger="opentelemetry.trace")

    configure_telemetry(sampling_rate=0.0)

    tracer = _get_tracer()
    with tracer.start_as_current_span("sampling-zero"):
        pass

    assert len(exporter.get_finished_spans()) == 1
    assert any(
        "Overriding of current TracerProvider is not allowed" in record.message
        for record in caplog.records
    )
