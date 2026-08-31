"""Launch a hidden Python module and forward arguments after ``--``."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True)
    parser.add_argument("--stdout", required=True)
    parser.add_argument("--stderr", required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = list(args.arguments)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    stdout_path = Path(args.stdout)
    stderr_path = Path(args.stderr)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    flags = 0
    if os.name == "nt":
        flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                args.module,
                *forwarded,
            ],
            cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            creationflags=flags,
        )
    print(
        json.dumps(
            {
                "pid": process.pid,
                "module": args.module,
                "arguments": forwarded,
                "stdout": str(stdout_path.resolve()),
                "stderr": str(stderr_path.resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()
