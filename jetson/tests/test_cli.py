"""Smoke tests: every command-line entry point runs end to end on this machine (no drone)."""

import csv
import os

import cv2
import numpy as np
import pytest

from jetson import bench, demo, doctor, follow
from jetson.bench import parse_spec


def test_demo_toy_with_oracle_writes_video_and_csv(tmp_path, capsys):
    out, log = tmp_path / "toy.mp4", tmp_path / "toy.csv"
    assert demo.main(["--source", "toy", "--backend", "oracle", "--seconds", "6", "--out", str(out),
                      "--csv", str(log), "--log-every", "100"]) == 0
    assert out.stat().st_size > 10_000
    rows = list(csv.DictReader(open(log)))
    assert len(rows) >= 80 and rows[-1]["mode"] == "TRACKING"
    assert "final: distance" in capsys.readouterr().out


def test_demo_on_a_video_file(tmp_path, yunet_model, media_image):
    clip = str(tmp_path / "clip.avi")
    img = cv2.resize(cv2.imread(media_image), (640, 512))
    vw = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"MJPG"), 15, (640, 512))
    for _ in range(12):
        vw.write(img)
    vw.release()
    log = tmp_path / "clip.csv"
    assert demo.main(["--source", clip, "--csv", str(log), "--log-every", "100"]) == 0
    rows = list(csv.DictReader(open(log)))
    assert len(rows) == 12 and int(rows[-1]["faces"]) >= 3 and rows[-1]["mode"] == "TRACKING"


def test_demo_bad_source_is_a_clean_error(capsys):
    assert demo.main(["--source", "nope://camera"]) == 2
    assert "camera error" in capsys.readouterr().out


def test_follow_dry_run_on_a_file(tmp_path, yunet_model, media_image):
    clip = str(tmp_path / "clip.avi")
    img = cv2.resize(cv2.imread(media_image), (640, 512))
    vw = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"MJPG"), 15, (640, 512))
    for _ in range(20):
        vw.write(img)
    vw.release()
    out = tmp_path / "follow.mp4"
    assert follow.main(["--source", clip, "--out", str(out), "--max-seconds", "5"]) == 0
    assert out.stat().st_size > 10_000


def test_bench_runs_and_saves(tmp_path, monkeypatch, media_image, yunet_model):
    monkeypatch.setattr(bench, "RESULTS_DIR", str(tmp_path))
    assert bench.main(["--backends", "yunet,haar,nosuch", "--widths", "320", "--frames", "2", "--warmup", "1",
                       "--image", media_image, "--save"]) == 0
    files = os.listdir(tmp_path)
    assert len(files) == 1
    text = open(tmp_path / files[0]).read()
    assert "not Jetson" in text or "Jetson" in text
    assert "| yunet |" in text and "Load average" in text


def test_bench_spec_parsing():
    assert parse_spec("yunet") == ("yunet", None, None)
    assert parse_spec("onnx:scrfd") == ("onnx", "scrfd", None)
    assert parse_spec("onnx:yunet@tensorrt") == ("onnx", "yunet", "tensorrt")


def test_doctor_runs(capsys):
    assert doctor.main([]) in (0, 1)
    out = capsys.readouterr().out
    assert "== Python" in out and "backend yunet" in out


def test_demo_toy_with_person_detector_logs_source(tmp_path):
    log = tmp_path / "toy_body.csv"
    assert demo.main(["--source", "toy", "--backend", "oracle", "--seconds", "2", "--person", "hog",
                      "--person-every", "2", "--csv", str(log), "--log-every", "100"]) == 0
    rows = list(csv.DictReader(open(log)))
    assert rows and {"source", "bodies", "person_ms"} <= set(rows[0])
    assert rows[-1]["source"] in ("face", "body")


def test_follow_missing_person_model_fails_before_arming(tmp_path, monkeypatch, capsys):
    from jetson.detectors import base

    monkeypatch.setattr("jetson.detectors.person.model_path", lambda f: str(tmp_path / f))
    with pytest.raises(base.BackendUnavailable, match="jetson.fetch models"):
        follow.main(["--source", "nope://camera", "--backend", "haar", "--person", "yolox"])
