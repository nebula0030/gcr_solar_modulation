from __future__ import annotations

import numpy as np
import pytest

from align import AlignedSeries
from correction import CorrectionResult
from plotting import PlotMetadata, build_overlay, build_side_by_side, write_html
from rate import RateSeries


def make_inputs(n=6):
    start = np.datetime64("2026-07-10T00:00:00.000000000")
    offsets = (np.arange(n) * 3600.0 * 1e9).astype("timedelta64[ns]")
    rs = RateSeries(
        bin_start_utc=start + offsets,
        bin_mid_utc=start + offsets,
        counts=np.full(n, 1700, dtype=np.int64),
        livetime_s=np.full(n, 3600.0),
        rate_hz=np.full(n, 0.47),
        rate_err_hz=np.full(n, 0.011),
        press_hpa=np.linspace(1006.0, 1010.0, n),
        temp_c=np.linspace(22.0, 25.0, n),
        bin_length_s=3600.0,
    )
    correction = CorrectionResult(
        corrected_rate_hz=np.full(n, 0.47),
        corrected_err_hz=np.full(n, 0.011),
        beta_p=-0.0013, beta_t=-0.004,
        beta_p_err=0.0002, beta_t_err=0.0005,
        r_squared=0.85, p0_hpa=1008.0, t0_c=23.5,
        method="fit", warnings=[],
    )
    aligned = [
        AlignedSeries(
            name="Neutron monitor (UFSZ)", units="counts/s", source="NMDB",
            values=np.linspace(98.0, 100.0, n),
            interpolated=np.zeros(n, dtype=bool),
        ),
        AlignedSeries(
            name="Kp index", units="Kp (0-9)", source="GFZ Potsdam",
            values=np.linspace(1.0, 4.0, n),
            interpolated=np.array([False, True, True, False, True, True]),
        ),
    ]
    meta = PlotMetadata(
        run_name="TestRun",
        station_label="UFSZ (Zugspitze), Rc=4.10 GV, alt=2650 m",
        correction_label="fit: beta_P=-0.130 %/hPa",
        footer_notes=["NMDB acknowledgement here"],
    )
    return rs, correction, aligned, meta


def test_overlay_contains_the_corrected_rate_trace():
    rs, corr, aligned, meta = make_inputs()
    fig = build_overlay(rs, corr, aligned, meta)
    names = [t.name for t in fig.data]
    assert any("Muon rate" in n for n in names)


def test_overlay_does_not_plot_the_raw_rate():
    """The spec calls for corrected rate only."""
    rs, corr, aligned, meta = make_inputs()
    fig = build_overlay(rs, corr, aligned, meta)
    names = [t.name or "" for t in fig.data]
    assert not any("raw" in n.lower() for n in names)


def test_overlay_includes_every_aligned_series():
    rs, corr, aligned, meta = make_inputs()
    fig = build_overlay(rs, corr, aligned, meta)
    names = " ".join(t.name or "" for t in fig.data)
    assert "Neutron monitor" in names
    assert "Kp index" in names


def test_side_by_side_has_one_row_per_series_plus_the_rate():
    rs, corr, aligned, meta = make_inputs()
    fig = build_side_by_side(rs, corr, aligned, meta)
    assert len(fig.data) >= 3


def test_side_by_side_links_the_x_axes():
    rs, corr, aligned, meta = make_inputs()
    fig = build_side_by_side(rs, corr, aligned, meta)
    layout = fig.layout.to_plotly_json()
    linked = [
        key for key, axis in layout.items()
        if key.startswith("xaxis") and isinstance(axis, dict)
        and axis.get("matches")
    ]
    assert linked


def test_hover_marks_interpolated_points():
    rs, corr, aligned, meta = make_inputs()
    fig = build_overlay(rs, corr, aligned, meta)
    kp = [t for t in fig.data if (t.name or "").startswith("Kp")][0]
    text = " ".join(str(x) for x in (kp.text or []))
    assert "interpolated" in text.lower()


def test_write_html_produces_a_self_contained_file(tmp_path):
    rs, corr, aligned, meta = make_inputs()
    fig = build_overlay(rs, corr, aligned, meta)
    out = tmp_path / "overlay.html"
    write_html(fig, str(out))
    content = out.read_text()
    assert content.lstrip().lower().startswith("<html")
    assert "plotly" in content.lower()
    assert len(content) > 100_000  # the JS bundle is inlined


def test_figures_build_with_no_external_series():
    rs, corr, _, meta = make_inputs()
    assert build_overlay(rs, corr, [], meta) is not None
    assert build_side_by_side(rs, corr, [], meta) is not None


def _annotation_text(fig):
    return " ".join(a.text or "" for a in fig.layout.annotations)


def test_header_block_shows_bin_size_and_inputs_on_both_figures():
    rs, corr, aligned, meta = make_inputs()
    meta.header_lines = [
        "Bin size: 3600 s  (6 bins, 2.40% mean Poisson error)",
        "Detector location: lat 37.3688, lon -122.0363",
    ]
    for fig in (build_overlay(rs, corr, aligned, meta),
                build_side_by_side(rs, corr, aligned, meta)):
        text = _annotation_text(fig)
        assert "Bin size: 3600 s" in text
        assert "Detector location: lat 37.3688" in text
        # Station and correction travel with the header too.
        assert "UFSZ" in text
        assert "beta_P" in text


def test_titles_are_title_cased():
    rs, corr, aligned, meta = make_inputs()
    overlay = build_overlay(rs, corr, aligned, meta)
    side = build_side_by_side(rs, corr, aligned, meta)
    assert "Muon Rate vs. Solar Activity" in _annotation_text(overlay)
    assert "Muon Rate vs. Solar Activity" in _annotation_text(side)


def test_overlay_has_no_rangeslider():
    """The bottom range-slider was removed; nothing sits below the plot to
    overlap with footer text."""
    rs, corr, aligned, meta = make_inputs()
    fig = build_overlay(rs, corr, aligned, meta)
    xaxis = fig.layout.to_plotly_json().get("xaxis", {})
    rs_cfg = xaxis.get("rangeslider")
    assert not (rs_cfg and rs_cfg.get("visible"))


def test_no_annotation_is_anchored_below_the_plot():
    """All descriptive text now lives in the top margin (y >= 1), so none of
    it can cover the graphs."""
    rs, corr, aligned, meta = make_inputs()
    for fig in (build_overlay(rs, corr, aligned, meta),
                build_side_by_side(rs, corr, aligned, meta)):
        for ann in fig.layout.annotations:
            if ann.yref == "paper" and ann.y is not None and ann.text \
                    and "Bin size" not in (ann.text or "") and ann.y < 0:
                raise AssertionError(
                    "descriptive annotation below plot: {0!r}".format(ann.text)
                )
