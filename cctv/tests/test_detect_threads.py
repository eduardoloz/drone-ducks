"""CPU thread cap for the tracker (ultralytics resets torch threads; we put the cap back)."""

from cctv.detect_track import cpu_threads


def test_cpu_threads_precedence(monkeypatch):
    monkeypatch.delenv("CCTV_THREADS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    assert cpu_threads() is None
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    assert cpu_threads() == 2
    monkeypatch.setenv("CCTV_THREADS", "3")
    assert cpu_threads() == 3
    assert cpu_threads(1) == 1
    monkeypatch.setenv("CCTV_THREADS", "")
    assert cpu_threads() == 2
