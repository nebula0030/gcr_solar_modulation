from __future__ import annotations

import json

import numpy as np
import pytest

import external_sources as es

START = np.datetime64("2026-08-01T00:00:00")
END = np.datetime64("2026-08-15T00:00:00")


def test_parse_flares_keeps_x_class_at_peak():
    payload = json.dumps([
        {"classType": "X1.5", "peakTime": "2026-08-11T14:24Z"},
        {"classType": "M2.0", "peakTime": "2026-08-11T09:00Z"},   # dropped
        {"classType": "X9.0", "peakTime": "2026-09-01T00:00Z"},   # out of window
    ]).encode()
    evs = es.parse_donki_flares(payload, START, END)
    assert [e.label for e in evs] == ["X1.5"]
    assert evs[0].kind == "flare"
    assert evs[0].utc == np.datetime64("2026-08-11T14:24")


def test_parse_cme_keeps_earth_directed_arrival():
    payload = json.dumps([
        {"cmeAnalyses": [
            {"isMostAccurate": True, "enlilList": [
                {"isEarthGB": True,
                 "estimatedShockArrivalTime": "2026-08-12T06:00Z"}]}]},
        {"cmeAnalyses": [
            {"isMostAccurate": True, "enlilList": [
                {"isEarthGB": False, "estimatedShockArrivalTime": None}]}]},  # dropped
    ]).encode()
    evs = es.parse_donki_cme(payload, START, END)
    assert len(evs) == 1
    assert evs[0].kind == "cme" and evs[0].label == "CME"
    assert evs[0].utc == np.datetime64("2026-08-12T06:00")


def test_fetch_solar_events_propagates_fetcherror(monkeypatch):
    def boom(*a, **k):
        raise es.FetchError("offline")
    monkeypatch.setattr(es, "http_get", boom)
    with pytest.raises(es.FetchError):
        es.fetch_solar_events(START, END)
