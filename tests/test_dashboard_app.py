"""Headless smoke tests of the Streamlit app using streamlit.testing (no browser, no network).

These prove the pages import and render without exceptions and that button callbacks create / advance /
reset a real simulation. They do NOT replace a manual look at the UI (see README checklist).
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest

import utils.weather as W

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
PAGES = ["dashboard", "building_simulation", "floor_controls", "weather_page", "ppo_model", "analytics",
         "evaluation", "settings_page"]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Never call the real weather API from tests: simulate an outage."""
    def boom(*a, **k):
        raise ConnectionError("network disabled in tests")
    monkeypatch.setattr(W, "_default_get", boom)


def page_script(name):
    import sys
    from pathlib import Path as P
    root = P(sys.modules["__main__"].__file__).resolve().parents[0] if False else None
    import importlib
    from dashboard import state
    state.init_state()
    importlib.import_module(f"dashboard.views.{name}").render()


@pytest.mark.parametrize("name", PAGES)
def test_every_page_renders(name):
    at = AppTest.from_function(page_script, args=(name,), default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]


def test_full_app_entry_point_loads_and_labels_weather_outage():
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "SIMULATED fallback" in text


def test_start_advance_reset_cycle():
    at = AppTest.from_file(APP, default_timeout=90).run()
    [b for b in at.button if b.key == "dash_start"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    sim = at.session_state["sim"]
    if sim is None:      # no trained model in this environment: a clear error must show, not a crash
        assert any("PPO model not ready" in e.value for e in at.error)
        return
    assert sim.state == "running"
    before = sim.current_step              # the sidebar clock fragment already ran once on this rerun
    assert before >= 1 and sim.env.current_step == before
    sim.advance(3)
    assert sim.current_step == before + 3 and sim.env.current_step == before + 3
    [b for b in at.button if b.key == "dash_reset"][0].click().run()
    assert at.session_state["sim"] is None


def test_baseline_start_without_ppo_model(monkeypatch):
    """Baseline controller works even when no PPO model exists, and is labelled as such."""
    from dashboard import state
    from rl.model_io import ModelStatus, STATE_MISSING
    monkeypatch.setattr(state, "_cached_model", lambda key: (None, ModelStatus(STATE_MISSING, "missing")))
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.session_state["settings"]["controller"] = "baseline"
    [b for b in at.button if b.key == "dash_start"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    sim = at.session_state["sim"]
    assert sim is not None and sim.config.controller == "baseline"
    assert "not RL" in at.session_state["sim_meta"]["controller"]
