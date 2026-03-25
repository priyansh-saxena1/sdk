# rough telemetry helper -- wip, not final design
# TODO: figure out if this should be a proper package or just stay here

try:
    from opentelemetry import trace
    _otel_available = True
except ImportError:
    _otel_available = False


def get_tracer():
    if _otel_available:
        return trace.get_tracer("kubeflow.sdk.trainer")
    return _NoOpTracer()


class _NoOpSpan:
    # needed because callers do span.set_attribute() etc on the result
    # nullcontext() returns None by default which would crash
    def set_attribute(self, key, value): pass
    def set_status(self, status, description=None): pass
    def record_exception(self, exc): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass


class _NoOpTracer:
    def start_as_current_span(self, name, **kwargs):
        return _NoOpSpan()
