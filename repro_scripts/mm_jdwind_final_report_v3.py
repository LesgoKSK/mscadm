from __future__ import annotations

import json

import numpy as np

from repro_scripts import mm_jdwind_final_report as report
from repro_scripts.mm_jdwind_final_report_v2 import (
    adjacency_vs_per_day,
    per_day,
)


_json_dumps = json.dumps


def numpy_safe_dumps(value, *args, **kwargs):
    kwargs.setdefault(
        "default",
        lambda item: item.item()
        if isinstance(item, np.generic)
        else (_ for _ in ()).throw(
            TypeError(f"Object of type {type(item).__name__} is not JSON serializable")
        ),
    )
    return _json_dumps(value, *args, **kwargs)


if __name__ == "__main__":
    report.adjacency_vs_per_day = adjacency_vs_per_day
    report.per_day = per_day
    report.json.dumps = numpy_safe_dumps
    report.main()
