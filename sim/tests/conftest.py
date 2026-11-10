"""Shared pytest setup.

Unit tests (tests/unit) need nothing but the venv. SITL tests (tests/sitl, marker `sitl`) need PX4
SITL running; they are skipped automatically when nothing answers on the MAVLink port, so
`pytest` is always safe to run.

    pytest tests/unit                  # seconds, no simulator
    scripts/start_sim.sh && pytest -m sitl -v -s tests/sitl     # minutes, needs SITL
"""
import os
import sys

# Let `pytest` work from sim/ without `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
