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


def test_apply_thread_cap_sets_opencv_and_torch(monkeypatch):
    """OpenCV's own pool must be capped too: torch alone still sampled ~250% CPU (2026-11-17)."""
    import sys
    import types

    from cctv.detect_track import apply_thread_cap

    seen = {}
    fake_cv2 = types.SimpleNamespace(setNumThreads=lambda n: seen.__setitem__("cv2", n))
    fake_torch = types.SimpleNamespace(set_num_threads=lambda n: seen.__setitem__("torch", n))
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    apply_thread_cap(None)
    assert seen == {}
    apply_thread_cap(2)
    assert seen == {"cv2": 2, "torch": 2}
