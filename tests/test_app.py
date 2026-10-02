"""Renders app.py headlessly with Streamlit's AppTest. Skipped if Streamlit isn't installed."""
from pathlib import Path

import pytest

testing = pytest.importorskip("streamlit.testing.v1")


def test_dashboard_renders_without_errors():
    at = testing.AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=180).run()
    assert not at.exception
    assert len(at.tabs) == 6
