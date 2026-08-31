from datetime import datetime, timezone

from repro_scripts import caa_completion_audit as engine
from repro_scripts import caa_completion_audit_v2 as audit


def test_v2_is_authoritative_datetime_safe_entry_point() -> None:
    value = datetime(2025, 1, 1, tzinfo=timezone.utc)

    assert engine._jsonable(value) == "2025-01-01T00:00:00Z"
    assert audit.audit_completion is engine.audit_completion
    assert audit.run is audit.audit_completion
    assert audit.AUTHORITATIVE_MODULE == "caa_rahc.candidates_nested"
