"""Deployment-readiness scripts: Python-compat checker, TensorRT export dry run, Jetson setup files."""

import importlib.util
import os
import shutil
import subprocess
import sys

import pytest

from .conftest import JETSON_DIR, MODELS_DIR

SCRIPTS = os.path.join(JETSON_DIR, "scripts")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_whole_package_passes_static_compat_check():
    cc = _load("check_compat")
    problems = [p for f in cc.python_files() for p in cc.check_file(f)]
    assert problems == []


@pytest.mark.parametrize("src, needle", [
    ("match x:\n    case 1:\n        pass\n", "syntax"),
    ("def f(a: list[int]) -> None:\n    pass\n", "annotation"),
    ("x = 'ab'.removeprefix('a')\n", "3.9+"),
    ("y = list(zip([1], [2], strict=True))\n", "3.10+"),
])
def test_compat_checker_catches_newer_python(tmp_path, src, needle):
    cc = _load("check_compat")
    f = tmp_path / "bad.py"
    f.write_text(src)
    probs = cc.check_file(str(f))
    assert probs and needle in probs[0]


def test_compat_checker_accepts_future_annotations(tmp_path):
    cc = _load("check_compat")
    f = tmp_path / "ok.py"
    f.write_text("from __future__ import annotations\n\ndef f(a: list[int] | None) -> dict[str, int]:\n    return {}\n")
    assert cc.check_file(str(f)) == []


def test_trt_build_command_per_tensorrt_version():
    ex = _load("export_trt")
    c10 = ex.build_command("trtexec", "m.onnx", "m.engine", "input.1", (3, 480, 640), major=10)
    assert "--fp16" in c10 and "--shapes=input.1:1x3x480x640" in c10 and "--memPoolSize=workspace:1024M" in c10
    c8 = ex.build_command("trtexec", "m.onnx", "m.engine", "images", None, precision="fp32", major=8)
    assert "--fp16" not in c8 and "--workspace=1024" in c8 and not any(a.startswith("--shapes") for a in c8)


def test_trt_shape_checks():
    ex = _load("export_trt")
    assert ex.check_shape("x", ("in", [1, 3, 416, 416]), None) == []
    assert ex.check_shape("x", ("in", [1, 3, "?", "?"]), (3, 480, 640)) == []
    assert ex.check_shape("x", ("in", [1, 3, "?", "?"]), None)
    assert ex.check_shape("x", ("in", [1, 3, 416, 416]), (3, 480, 640))


def test_trt_engine_names_match_backend_defaults():
    ex = _load("export_trt")
    from jetson.detectors.person import YOLOX_ENGINE
    from jetson.detectors.trt import DEFAULT_ENGINE

    assert ex.SPECS["yunet"][1] + "_fp16.engine" == DEFAULT_ENGINE
    assert ex.SPECS["yolox"][1] + "_fp16.engine" == YOLOX_ENGINE


def test_export_trt_dry_run_on_fetched_models(capsys):
    if not os.path.exists(os.path.join(MODELS_DIR, "yolox_nano.onnx")):
        pytest.skip("models missing: python -m jetson.fetch models")
    ex = _load("export_trt")
    assert ex.main(["yunet", "yolox", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "yolox_nano_416_fp16.engine" in out and "PROBLEM" not in out


def test_export_trt_without_trtexec_says_where_it_lives(capsys, monkeypatch):
    ex = _load("export_trt")
    monkeypatch.setattr(ex, "find_trtexec", lambda explicit=None: None)
    assert ex.main(["yolox"]) == 2
    assert "/usr/src/tensorrt/bin/trtexec" in capsys.readouterr().out


def test_jetson_setup_script_is_valid_bash():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    path = os.path.join(SCRIPTS, "setup_jetson_jp6.sh")
    assert subprocess.run([bash, "-n", path]).returncode == 0
    text = open(path).read()
    assert "set -euo pipefail" in text and "--system-site-packages" in text


def test_dockerfile_targets_jetpack6():
    text = open(os.path.join(JETSON_DIR, "docker", "Dockerfile.jp6")).read()
    assert "l4t" in text and "r36" in text and "requirements-jetson-jp6.txt" in text
