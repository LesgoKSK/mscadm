from __future__ import annotations

import json

import numpy as np

from repro_scripts import stgf_final_report as report
from repro_scripts.stgf_final_report_v2 import adjacency_vs_per_day
from repro_scripts.stgf_final_report_v3 import per_day


_JSON_DUMPS = json.dumps


def numpy_safe_dumps(value, *args, **kwargs):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"not JSON serializable: {type(item)!r}")

    kwargs.setdefault("default", convert)
    return _JSON_DUMPS(value, *args, **kwargs)


if __name__ == "__main__":
    report.adjacency_vs_per_day = adjacency_vs_per_day
    report.per_day = per_day
    report.json.dumps = numpy_safe_dumps
    report.main()
