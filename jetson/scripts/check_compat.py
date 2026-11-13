#!/usr/bin/env python3
"""Check that jetson/ still runs on the oldest Python we promise (3.8) and on JetPack 6's (3.10).

    python jetson/scripts/check_compat.py                       # static checks, any Python >= 3.8
    python jetson/scripts/check_compat.py --run-tests jetson/.venv38/bin/python jetson/.venv310/bin/python

Static checks (no imports of our code, so they work without OpenCV/numpy):
  1. every .py parses with the 3.8 grammar (ast feature_version), e.g. no `match`, no
     parenthesised context managers;
  2. modules that write `list[int]` / `X | None` style annotations start with
     `from __future__ import annotations` (otherwise they crash at import on 3.8/3.9);
  3. no calls to APIs newer than 3.8 that we have tripped on before (str.removeprefix,
     functools.cache, zip(strict=), math.lcm, int.bit_count, dataclass(slots=/kw_only=)).

--run-tests runs the whole pytest suite under each interpreter you give it. That is the real
check; the static part only catches mistakes early on a laptop that has a single Python.

What this does NOT prove: that JetPack's own builds (OpenCV with GStreamer, TensorRT, NVIDIA's
onnxruntime-gpu, pycuda) behave like the pip wheels the tests use. Only the bench checklist on a
real Jetson does that (docs/BRINGUP_CHECKLIST.md).
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
from typing import List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
REPO = os.path.dirname(PKG)
MIN_VERSION = (3, 8)

NEWER_APIS = [
    (re.compile(r"\.removeprefix\(|\.removesuffix\("), "str.removeprefix/removesuffix is 3.9+"),
    (re.compile(r"functools\.cache\b|from functools import [^\n]*\bcache\b(?!_)"), "functools.cache is 3.9+"),
    (re.compile(r"\bzip\([^)]*strict\s*="), "zip(strict=) is 3.10+"),
    (re.compile(r"math\.lcm\("), "math.lcm is 3.9+"),
    (re.compile(r"\.bit_count\(\)"), "int.bit_count is 3.10+"),
    (re.compile(r"dataclass\([^)]*(slots|kw_only)\s*="), "dataclass(slots=/kw_only=) is 3.10+"),
]
NEW_ANNOTATION = re.compile(r"(:|->)\s*[^=\n]*\b(list|dict|tuple|set|type)\[|(:|->)\s*[^=\n#]*\w\s*\|\s*None")


def python_files(root: str = PKG) -> List[str]:
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if not x.startswith(".") and x not in ("__pycache__", "models", "media", "out")]
        out += [os.path.join(d, f) for f in files if f.endswith(".py")
                and os.path.join(d, f) != os.path.abspath(__file__)]  # our regexes would match themselves
    return sorted(out)


def check_file(path: str, version=MIN_VERSION) -> List[str]:
    rel = os.path.relpath(path, REPO)
    src = open(path, encoding="utf-8").read()
    problems = []
    try:
        tree = ast.parse(src, filename=path, feature_version=version)
    except SyntaxError as e:
        return [f"{rel}:{e.lineno}: not valid Python {version[0]}.{version[1]} syntax: {e.msg}"]
    has_future = any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
                     and any(a.name == "annotations" for a in n.names) for n in tree.body)
    code_lines = [ln for ln in src.splitlines()]
    in_doc = False
    for i, line in enumerate(code_lines, 1):
        s = line.strip()
        if s.count('"""') == 1:
            in_doc = not in_doc
            continue
        if in_doc or s.startswith("#"):
            continue
        code = line.split("#", 1)[0]
        for rx, why in NEWER_APIS:
            if rx.search(code):
                problems.append(f"{rel}:{i}: {why}")
        if not has_future and (s.startswith("def ") or re.match(r"^\s*\w+\s*:", line)) \
                and NEW_ANNOTATION.search(code):
            problems.append(f"{rel}:{i}: 3.9+/3.10+ annotation without `from __future__ import annotations`")
    return problems


def run_tests(pythons: Sequence[str]) -> int:
    worst = 0
    for py in pythons:
        ver = subprocess.run([py, "-c", "import sys; print(sys.version.split()[0])"], capture_output=True,
                             text=True).stdout.strip()
        print(f"== {py} (Python {ver or '?'})")
        r = subprocess.run([py, "-m", "pytest", os.path.join(PKG, "tests"), "-q", "-p", "no:cacheprovider"],
                           cwd=REPO, capture_output=True, text=True)
        tail = [ln for ln in r.stdout.splitlines() if ln.strip()][-1:] or ["(no output)"]
        print("   " + tail[0])
        worst = max(worst, r.returncode)
    return worst


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-tests", nargs="+", metavar="PYTHON", help="also run pytest under these interpreters")
    args = ap.parse_args(argv)
    files = python_files()
    problems = [p for f in files for p in check_file(f)]
    for p in problems:
        print("PROBLEM: " + p)
    print(f"static: {len(files)} files checked against Python {MIN_VERSION[0]}.{MIN_VERSION[1]}: "
          f"{len(problems)} problem(s)")
    rc = 1 if problems else 0
    if args.run_tests:
        rc = max(rc, run_tests(args.run_tests))
    return rc


if __name__ == "__main__":
    sys.exit(main())
