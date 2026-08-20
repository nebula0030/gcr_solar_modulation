"""Shared test fixtures.

``--events`` defaults on, so every ``process_data.main`` run now fetches DONKI
solar events. Stub that fetch out by default so the suite never touches the
network; tests that exercise the events path override this with their own
``monkeypatch.setattr`` (which runs after this autouse fixture and therefore
wins for the duration of that test).
"""
import pytest


@pytest.fixture(autouse=True)
def _no_network_solar_events(request, monkeypatch):
    # test_external_sources.py exercises fetch_solar_events itself and manages
    # its own network isolation (it monkeypatches http_get), so leave the real
    # function in place there.
    if "test_external_sources" in request.node.nodeid:
        return
    import external_sources
    monkeypatch.setattr(external_sources, "fetch_solar_events",
                        lambda *a, **k: [])
