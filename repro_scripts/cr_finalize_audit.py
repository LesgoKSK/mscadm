from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    root = Path("outputs/cr_mscadm")
    audit_path = root / "completion_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    xml_root = ET.parse(root / "pytest.xml").getroot()
    suite = xml_root.find("testsuite") if xml_root.tag == "testsuites" else xml_root
    if suite is None:
        raise RuntimeError("JUnit XML contains no testsuite")
    detail = dict(suite.attrib)
    passed = (
        int(detail.get("tests", 0)) == 26
        and int(detail.get("failures", 0)) == 0
        and int(detail.get("errors", 0)) == 0
    )
    audit["checks"]["full_test_suite"] = {"passed": passed, "detail": detail}
    audit["passed"] = all(value["passed"] for value in audit["checks"].values())
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": audit["passed"],
                "checks": {name: value["passed"] for name, value in audit["checks"].items()},
            },
            indent=2,
        )
    )
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
