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


def _cme(onset, runs):
    # runs: list of (isEarthGB, isEarthMinorImpact, arrival, isMostAccurate, level)
    return {"startTime": onset, "cmeAnalyses": [
        {"isMostAccurate": ma, "levelOfData": lvl, "submissionTime": "2026-01-01T00:00Z",
         "enlilList": [{"isEarthGB": gb, "isEarthMinorImpact": mi,
                        "estimatedShockArrivalTime": arr}]}
        for (gb, mi, arr, ma, lvl) in runs]}


def test_parse_cme_keeps_direct_hit():
    payload = json.dumps([_cme("2026-08-11T00:00Z",
        [(False, False, "2026-08-12T06:00Z", True, 1)])]).encode()
    evs = es.parse_donki_cme(payload, START, END)
    assert len(evs) == 1 and evs[0].kind == "cme" and evs[0].label == "CME"
    assert evs[0].utc == np.datetime64("2026-08-12T06:00")


def test_parse_cme_drops_glancing_blow():
    payload = json.dumps([_cme("2026-08-11T00:00Z",
        [(True, False, "2026-08-12T06:00Z", True, 1)])]).encode()
    assert es.parse_donki_cme(payload, START, END) == []


def test_parse_cme_drops_minor_impact():
    payload = json.dumps([_cme("2026-08-11T00:00Z",
        [(False, True, "2026-08-12T06:00Z", True, 1)])]).encode()
    assert es.parse_donki_cme(payload, START, END) == []


def test_parse_cme_drops_no_earth_arrival():
    payload = json.dumps([_cme("2026-08-11T00:00Z",
        [(False, False, None, True, 1)])]).encode()
    assert es.parse_donki_cme(payload, START, END) == []


def test_parse_cme_best_run_prefers_highest_level():
    # two most-accurate analyses, level 0 (arrival A) and level 1 (arrival B);
    # the level-1 arrival must win (mirrors the real 2026-01-18 record).
    payload = json.dumps([{"startTime": "2026-08-11T00:00Z", "cmeAnalyses": [
        {"isMostAccurate": True, "levelOfData": 0, "submissionTime": "2026-08-11T06:00Z",
         "enlilList": [{"isEarthGB": False, "isEarthMinorImpact": False,
                        "estimatedShockArrivalTime": "2026-08-12T00:00Z"}]},
        {"isMostAccurate": True, "levelOfData": 1, "submissionTime": "2026-08-11T14:00Z",
         "enlilList": [{"isEarthGB": False, "isEarthMinorImpact": False,
                        "estimatedShockArrivalTime": "2026-08-12T06:00Z"}]},
    ]}]).encode()
    evs = es.parse_donki_cme(payload, START, END)
    assert len(evs) == 1
    assert evs[0].utc == np.datetime64("2026-08-12T06:00")  # level-1 arrival


def test_parse_cme_level_dominates_submissiontime():
    payload = json.dumps([{"startTime": "2026-08-11T00:00Z", "cmeAnalyses": [
        # level 1, LATER submission, arrival A
        {"isMostAccurate": True, "levelOfData": 1, "submissionTime": "2026-08-11T18:00Z",
         "enlilList": [{"isEarthGB": False, "isEarthMinorImpact": False,
                        "estimatedShockArrivalTime": "2026-08-12T00:00Z"}]},
        # level 2, EARLIER submission, arrival B  -> higher level must win
        {"isMostAccurate": True, "levelOfData": 2, "submissionTime": "2026-08-11T06:00Z",
         "enlilList": [{"isEarthGB": False, "isEarthMinorImpact": False,
                        "estimatedShockArrivalTime": "2026-08-12T09:00Z"}]},
    ]}]).encode()
    evs = es.parse_donki_cme(payload, START, END)
    assert len(evs) == 1
    assert evs[0].utc == np.datetime64("2026-08-12T09:00")  # level-2 arrival, not the later-submitted level-1


def test_parse_cme_glancing_best_run_blocks_fallback_to_lower_direct():
    payload = json.dumps([{"startTime": "2026-08-11T00:00Z", "cmeAnalyses": [
        # top priority: most-accurate, level 2 -> GLANCING blow (with arrival)
        {"isMostAccurate": True, "levelOfData": 2, "submissionTime": "2026-08-11T12:00Z",
         "enlilList": [{"isEarthGB": True, "isEarthMinorImpact": False,
                        "estimatedShockArrivalTime": "2026-08-12T06:00Z"}]},
        # lower priority: level 1 -> DIRECT hit (with arrival)
        {"isMostAccurate": True, "levelOfData": 1, "submissionTime": "2026-08-11T06:00Z",
         "enlilList": [{"isEarthGB": False, "isEarthMinorImpact": False,
                        "estimatedShockArrivalTime": "2026-08-12T10:00Z"}]},
    ]}]).encode()
    assert es.parse_donki_cme(payload, START, END) == []  # dropped: best run is glancing, no fallback


def test_parse_flares_window_boundaries_inclusive():
    before_start = START - np.timedelta64(1, "s")
    payload = json.dumps([
        {"classType": "X1.0", "peakTime": str(START) + "Z"},
        {"classType": "X2.0", "peakTime": str(END) + "Z"},
        {"classType": "X3.0", "peakTime": str(before_start) + "Z"},
    ]).encode()
    evs = es.parse_donki_flares(payload, START, END)
    assert [e.label for e in evs] == ["X1.0", "X2.0"]


def test_fetch_solar_events_propagates_fetcherror(monkeypatch):
    def boom(*a, **k):
        raise es.FetchError("offline")
    monkeypatch.setattr(es, "http_get", boom)
    with pytest.raises(es.FetchError):
        es.fetch_solar_events(START, END)
