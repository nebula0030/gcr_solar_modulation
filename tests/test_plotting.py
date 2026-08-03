from __future__ import annotations

import json

import numpy as np
import pytest

import anomaly
import plotting
from align import AlignedSeries
from correction import CorrectionResult
from plotting import (
    DetectorSeries,
    PlotMetadata,
    build_overlay,
    build_side_by_side,
    write_html,
)
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


def make_detector(name, n=6, base=0.47):
    start = np.datetime64("2026-07-10T00:00:00.000000000")
    off = (np.arange(n) * 3600.0 * 1e9).astype("timedelta64[ns]")
    rs = RateSeries(start + off, start + off, np.full(n, 1700, dtype=np.int64),
                    np.full(n, 3600.0), np.full(n, base), np.full(n, 0.01),
                    np.linspace(1006, 1010, n), np.linspace(22, 25, n), 3600.0)
    cr = CorrectionResult(np.full(n, base), np.full(n, 0.01), -0.0013, -0.004,
                          2e-4, 5e-4, 0.85, 1008.0, 23.5, "fit", [])
    return DetectorSeries(name=name, rs=rs, correction=cr)


def test_overlay_contains_the_corrected_rate_trace():
    rs, corr, aligned, meta = make_inputs()
    fig = build_overlay([DetectorSeries("Muon rate", rs, corr)], aligned, meta, rs.bin_mid_utc)
    names = [t.name for t in fig.data]
    assert any("Muon rate" in n for n in names)


def test_overlay_does_not_plot_the_raw_rate():
    """The spec calls for corrected rate only."""
    rs, corr, aligned, meta = make_inputs()
    fig = build_overlay([DetectorSeries("Muon rate", rs, corr)], aligned, meta, rs.bin_mid_utc)
    names = [t.name or "" for t in fig.data]
    assert not any("raw" in n.lower() for n in names)


def test_overlay_includes_every_aligned_series():
    rs, corr, aligned, meta = make_inputs()
    fig = build_overlay([DetectorSeries("Muon rate", rs, corr)], aligned, meta, rs.bin_mid_utc)
    names = " ".join(t.name or "" for t in fig.data)
    assert "Neutron monitor" in names
    assert "Kp index" in names


def test_side_by_side_has_one_row_per_series_plus_the_rate():
    rs, corr, aligned, meta = make_inputs()
    fig = build_side_by_side([DetectorSeries("Muon rate", rs, corr)], aligned, meta, rs.bin_mid_utc)
    assert len(fig.data) >= 3


def test_side_by_side_links_the_x_axes():
    rs, corr, aligned, meta = make_inputs()
    fig = build_side_by_side([DetectorSeries("Muon rate", rs, corr)], aligned, meta, rs.bin_mid_utc)
    layout = fig.layout.to_plotly_json()
    linked = [
        key for key, axis in layout.items()
        if key.startswith("xaxis") and isinstance(axis, dict)
        and axis.get("matches")
    ]
    assert linked


def test_hover_marks_interpolated_points():
    rs, corr, aligned, meta = make_inputs()
    fig = build_overlay([DetectorSeries("Muon rate", rs, corr)], aligned, meta, rs.bin_mid_utc)
    kp = [t for t in fig.data if (t.name or "").startswith("Kp")][0]
    text = " ".join(str(x) for x in (kp.text or []))
    assert "interpolated" in text.lower()


def test_write_html_produces_a_self_contained_file(tmp_path):
    rs, corr, aligned, meta = make_inputs()
    fig = build_overlay([DetectorSeries("Muon rate", rs, corr)], aligned, meta, rs.bin_mid_utc)
    out = tmp_path / "overlay.html"
    write_html(fig, str(out))
    content = out.read_text()
    assert content.lstrip().lower().startswith("<!doctype html")
    assert "plotly" in content.lower()
    assert len(content) > 100_000  # the JS bundle is inlined


def test_page_has_a_checkbox_per_series(tmp_path):
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    n_primary = len(dets) + len(aligned)
    for kind, fig, stacked in (
        ("overlay", build_overlay(dets, aligned, meta, rs.bin_mid_utc), False),
        ("side", build_side_by_side(dets, aligned, meta, rs.bin_mid_utc), True),
    ):
        out = tmp_path / (kind + ".html")
        write_html(fig, str(out), stacked=stacked)
        content = out.read_text()
        n_boxes = content.count('type="checkbox"')
        # Checkboxes are built from primary (detector + external) traces only;
        # the hidden per-detector anomaly traces appended after them must not
        # get their own checkbox (see test_series_checkboxes_exclude_anomaly_traces).
        assert n_boxes == n_primary, (kind, n_boxes, n_primary)
        assert "Neutron monitor" in content and "Kp index" in content


def test_page_has_a_theme_toggle(tmp_path):
    rs, corr, aligned, meta = make_inputs()
    out = tmp_path / "overlay.html"
    write_html(build_overlay([DetectorSeries("Muon rate", rs, corr)], aligned, meta, rs.bin_mid_utc), str(out))
    content = out.read_text()
    assert 'id="theme-toggle"' in content
    assert "Switch to dark mode" in content
    # dark-mode palette steps must be present, not an automatic flip
    assert "#3987e5" in content and "#1a1a19" in content


def test_page_has_matplotlib_style_line_spec_for_the_muon_rate(tmp_path):
    rs, corr, aligned, meta = make_inputs()
    out = tmp_path / "overlay.html"
    write_html(build_overlay([DetectorSeries("Muon rate", rs, corr)], aligned, meta, rs.bin_mid_utc), str(out))
    content = out.read_text()
    assert 'id="line-spec"' in content
    for spec in ("o-", "-", "o", "--", "."):
        assert 'value="{0}"'.format(spec) in content
    # markers-only must really drop the connecting line
    assert '"o": {"mode": "markers"' in content.replace("'", '"')


def test_stacked_page_collapses_deselected_panels(tmp_path):
    """Side-by-side checkboxes must recompute panel domains, not just hide a
    trace and leave an empty panel behind."""
    rs, corr, aligned, meta = make_inputs()
    out = tmp_path / "side.html"
    write_html(build_side_by_side([DetectorSeries("Muon rate", rs, corr)], aligned, meta, rs.bin_mid_utc), str(out), stacked=True)
    content = out.read_text()
    assert '"stacked": true' in content
    assert ".domain" in content
    assert '"rowOfTrace": [1, 2, 3]' in content


def test_figures_build_with_no_external_series():
    rs, corr, _, meta = make_inputs()
    assert build_overlay([DetectorSeries("Muon rate", rs, corr)], [], meta, rs.bin_mid_utc) is not None
    assert build_side_by_side([DetectorSeries("Muon rate", rs, corr)], [], meta, rs.bin_mid_utc) is not None


def test_header_block_shows_bin_size_and_inputs_on_both_figures(tmp_path):
    rs, corr, aligned, meta = make_inputs()
    meta.header_lines = [
        "Bin size: 3600 s  (6 bins, 2.40% mean Poisson error)",
        "Detector location: lat 37.3688, lon -122.0363",
    ]
    for kind, fig, stacked in (
        ("overlay", build_overlay([DetectorSeries("Muon rate", rs, corr)], aligned, meta, rs.bin_mid_utc), False),
        ("side", build_side_by_side([DetectorSeries("Muon rate", rs, corr)], aligned, meta, rs.bin_mid_utc), True),
    ):
        out = tmp_path / (kind + ".html")
        write_html(fig, str(out), stacked=stacked)
        content = out.read_text()
        assert "Bin size: 3600 s" in content
        assert "Detector location: lat 37.3688" in content
        # Station and correction travel with the header too.
        assert "UFSZ" in content
        assert "beta_P" in content
        # The header is an HTML block, not baked into the figure margin.
        assert 'class="run-header"' in content


def test_titles_are_title_cased(tmp_path):
    rs, corr, aligned, meta = make_inputs()
    for kind, fig, stacked in (
        ("overlay", build_overlay([DetectorSeries("Muon rate", rs, corr)], aligned, meta, rs.bin_mid_utc), False),
        ("side", build_side_by_side([DetectorSeries("Muon rate", rs, corr)], aligned, meta, rs.bin_mid_utc), True),
    ):
        out = tmp_path / (kind + ".html")
        write_html(fig, str(out), stacked=stacked)
        assert "Muon Rate vs. Solar Activity" in out.read_text()


def test_header_is_not_baked_into_the_figure_margin():
    """The metadata header must be HTML, not a tall figure top-margin that
    steals vertical space from the graphs."""
    rs, corr, aligned, meta = make_inputs()
    side = build_side_by_side([DetectorSeries("Muon rate", rs, corr)], aligned, meta, rs.bin_mid_utc)
    # top margin only needs to clear the first subplot title now.
    assert side.layout.margin.t <= 80
    # bin-size text is stashed for the HTML page, not added as an annotation.
    ann = " ".join(a.text or "" for a in side.layout.annotations)
    assert "Bin size" not in ann


def test_overlay_has_no_rangeslider():
    """The bottom range-slider was removed; nothing sits below the plot to
    overlap with footer text."""
    rs, corr, aligned, meta = make_inputs()
    fig = build_overlay([DetectorSeries("Muon rate", rs, corr)], aligned, meta, rs.bin_mid_utc)
    xaxis = fig.layout.to_plotly_json().get("xaxis", {})
    rs_cfg = xaxis.get("rangeslider")
    assert not (rs_cfg and rs_cfg.get("visible"))


def test_no_annotation_is_anchored_below_the_plot():
    """All descriptive text now lives in the top margin (y >= 1), so none of
    it can cover the graphs."""
    rs, corr, aligned, meta = make_inputs()
    for fig in (build_overlay([DetectorSeries("Muon rate", rs, corr)], aligned, meta, rs.bin_mid_utc),
                build_side_by_side([DetectorSeries("Muon rate", rs, corr)], aligned, meta, rs.bin_mid_utc)):
        for ann in fig.layout.annotations:
            if ann.yref == "paper" and ann.y is not None and ann.text \
                    and "Bin size" not in (ann.text or "") and ann.y < 0:
                raise AssertionError(
                    "descriptive annotation below plot: {0!r}".format(ann.text)
                )


def test_overlay_has_one_trace_per_detector():
    dets = [make_detector("DetA", base=0.47), make_detector("DetB", base=0.30)]
    aligned = [AlignedSeries("Neutron monitor (OULU)", "counts/s", "NMDB",
                             np.linspace(98, 100, 6), np.zeros(6, bool))]
    meta = PlotMetadata("run", "OULU", "fit", [], [])
    master_utc = dets[0].rs.bin_mid_utc
    fig = build_overlay(dets, aligned, meta, master_utc)
    names = " ".join(t.name or "" for t in fig.data)
    assert "DetA" in names and "DetB" in names


def test_side_by_side_single_rate_panel_holds_all_detectors():
    dets = [make_detector("DetA"), make_detector("DetB")]
    aligned = [AlignedSeries("Kp index", "Kp (0-9)", "GFZ Potsdam",
                             np.linspace(1, 4, 6), np.zeros(6, bool))]
    meta = PlotMetadata("run", "OULU", "fit", [], [])
    master_utc = dets[0].rs.bin_mid_utc
    fig = build_side_by_side(dets, aligned, meta, master_utc)
    # rate panel (row 1) has both detector traces; then 1 external panel.
    # (Match the "(corrected)" suffix, not a bare "Det" prefix -- the hidden
    # per-detector anomaly traces are also named "DetA mean" etc.)
    rate_traces = [t for t in fig.data if (t.name or "").endswith("(corrected)")]
    assert len(rate_traces) == 2


def test_single_detector_gaps_are_shaded():
    import numpy as np
    from align import AlignedSeries
    from plotting import PlotMetadata, build_overlay
    det = make_detector("DetA")
    aligned = [AlignedSeries("Kp index", "Kp (0-9)", "GFZ Potsdam",
                             np.linspace(1, 4, 6), np.zeros(6, bool))]
    gaps = [(np.datetime64("2026-07-10T02:00:00"),
             np.datetime64("2026-07-10T03:00:00"))]
    meta = PlotMetadata("run", "OULU", "fit", [], [])
    fig = build_overlay([det], aligned, meta, det.rs.bin_mid_utc, gaps=gaps)
    shapes = fig.layout.to_plotly_json().get("shapes", [])
    rects = [s for s in shapes if s.get("type") == "rect"]
    assert rects
    # Guard the datetime64-scalar serialization bug: a bare `np.datetime64`
    # passed straight to add_vrect gets JSON-encoded via `.item()`, which at
    # nanosecond precision yields a plain integer (ns since epoch) rather
    # than a date string, landing the band off the date axis entirely. x0/x1
    # must be ISO-8601 date strings, not ints.
    rect = rects[0]
    assert isinstance(rect["x0"], str) and "2026-07-10T02" in rect["x0"]
    assert isinstance(rect["x1"], str) and "2026-07-10T03" in rect["x1"]


def test_no_gaps_no_shapes():
    import numpy as np
    from align import AlignedSeries
    from plotting import PlotMetadata, build_overlay
    det = make_detector("DetA")
    aligned = []
    meta = PlotMetadata("run", "OULU", "fit", [], [])
    fig = build_overlay([det], aligned, meta, det.rs.bin_mid_utc, gaps=None)
    shapes = fig.layout.to_plotly_json().get("shapes", [])
    assert not any(s.get("type") == "rect" for s in shapes)


def test_gap_label_is_classified_muted_so_theme_toggle_does_not_recolor_it(tmp_path):
    """The 'no data' gap annotation must keep its own role so the theme-toggle
    JS applies the muted token to it instead of the full-strength subplot
    text color on the first toggle (see applyTheme's annotationRoles loop)."""
    import re
    import json
    from align import AlignedSeries
    from plotting import PlotMetadata, build_overlay, write_html

    det = make_detector("DetA")
    aligned = [AlignedSeries("Kp index", "Kp (0-9)", "GFZ Potsdam",
                             np.linspace(1, 4, 6), np.zeros(6, bool))]
    gaps = [(np.datetime64("2026-07-10T02:00:00"),
             np.datetime64("2026-07-10T03:00:00"))]
    meta = PlotMetadata("run", "OULU", "fit", [], [])
    fig = build_overlay([det], aligned, meta, det.rs.bin_mid_utc, gaps=gaps)

    out = tmp_path / "overlay.html"
    write_html(fig, str(out))
    content = out.read_text()

    match = re.search(r"var CFG = (\{.*?\});", content)
    assert match is not None, "CFG config block not found in page"
    cfg = json.loads(match.group(1))

    assert "muted" in cfg["annotationRoles"]
    assert "no data" in content


def test_figure_config_shapes_for_overlay_and_side():
    from plotting import _figure_config
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    ov = build_overlay(dets, aligned, meta, rs.bin_mid_utc)
    sb = build_side_by_side(dets, aligned, meta, rs.bin_mid_utc)

    ov_cfg = _figure_config(ov, stacked=False)
    sb_cfg = _figure_config(sb, stacked=True)

    # Overlay: not stacked, one detector, secondary axis, all traces "row 1".
    # rowOfTrace covers only the primary traces (1 detector + 2 external);
    # the hidden per-detector anomaly traces appended after them are excluded.
    assert ov_cfg["stacked"] is False
    assert ov_cfg["nDetectors"] == 1
    assert ov_cfg["hasSecondaryAxis"] is True
    n_primary = len(dets) + len(aligned)
    assert ov_cfg["rowOfTrace"] == [1] * n_primary

    # Side-by-side: stacked, rate panel + external panels, rowOfTrace maps all
    # detectors to row 1 then externals to 2..; nRows = 1 + n_external.
    assert sb_cfg["stacked"] is True
    assert sb_cfg["nDetectors"] == 1
    assert sb_cfg["hasSecondaryAxis"] is False
    assert sb_cfg["rowOfTrace"] == [1, 2, 3]  # 1 detector + 2 external panels
    assert sb_cfg["nRows"] == 3

    # traceColors present for both modes, one entry per trace.
    assert len(ov_cfg["traceColors"]["light"]) == len(ov.data)
    assert len(sb_cfg["traceColors"]["dark"]) == len(sb.data)


def test_write_combined_html_one_file_two_figs(tmp_path):
    from plotting import write_combined_html
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    ov = build_overlay(dets, aligned, meta, rs.bin_mid_utc)
    sb = build_side_by_side(dets, aligned, meta, rs.bin_mid_utc)
    out = tmp_path / "combined.html"
    write_combined_html(ov, sb, str(out))
    content = out.read_text()

    assert content.lstrip().lower().startswith("<!doctype html")
    # Two plot divs.
    assert 'id="viz-overlay"' in content and 'id="viz-side"' in content
    # Tab bar with both buttons.
    assert 'id="tab-overlay"' in content and 'id="tab-side"' in content
    # One checkbox per primary series (3 traces here: 1 detector + 2 external),
    # excluding the hidden per-detector anomaly traces appended after them.
    n_primary = len(dets) + len(aligned)
    assert content.count('type="checkbox"') == n_primary
    # Both figures' configs are embedded.
    assert '"overlay"' in content and '"side"' in content
    # Self-contained and non-trivial (JS bundle inlined once).
    assert len(content) > 100_000


def test_combined_inlines_plotly_once(tmp_path):
    import re
    from plotting import write_combined_html
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    ov = build_overlay(dets, aligned, meta, rs.bin_mid_utc)
    sb = build_side_by_side(dets, aligned, meta, rs.bin_mid_utc)
    out = tmp_path / "combined.html"
    write_combined_html(ov, sb, str(out))
    content = out.read_text()
    # Two figures are initialised (one Plotly.newPlot call per div id)...
    # NOTE: a plain `content.count("Plotly.newPlot")` is not reliable here —
    # the vendored plotly.min.js bundle itself contains the literal text
    # "Plotly.newPlot(gd, data, layout, ...)" inside a Mapbox-token help
    # string, so a naive substring count over-reports by one. Match the
    # actual init calls by the div id argument instead.
    init_calls = re.findall(r'Plotly\.newPlot\(\s*"plot-(overlay|side)"', content)
    assert sorted(init_calls) == ["overlay", "side"]
    # ...but the ~4-5 MB library bundle is inlined only once (with the overlay);
    # inlining it twice would push the file past ~8 MB.
    assert len(content) < 8_000_000


# --- exact-Poisson anomaly traces + shared payload -------------------------


def test_build_anomaly_payload_shape_and_units():
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    payload = plotting.build_anomaly_payload(dets)
    assert len(payload) == len(dets)
    p0 = payload[0]
    assert {"name", "mu", "t", "counts", "livetime", "adj", "rate",
            "good"}.issubset(p0)
    assert p0["mu"] == pytest.approx(
        anomaly.baseline_mean(dets[0].correction.corrected_rate_hz))
    assert len(p0["t"]) == len(p0["counts"]) == len(p0["rate"])
    json.dumps(payload)  # must be JSON-able


def test_overlay_appends_four_hidden_anomaly_traces_per_detector():
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    fig = build_overlay(dets, aligned, meta, rs.bin_mid_utc)
    # the four anomaly-band roles (the marginal hist/poisson traces are also
    # meta.anomaly-tagged and are exercised separately)
    band_roles = ("mean", "lower", "upper", "outlier")
    roles = [t.meta.get("anomaly") for t in fig.data
             if isinstance(t.meta, dict) and t.meta.get("anomaly") in band_roles]
    assert roles == list(band_roles)  # one detector
    for t in fig.data:
        if isinstance(t.meta, dict) and "anomaly" in t.meta:
            assert t.visible is False


def test_side_by_side_anomaly_mean_is_mu_in_hz():
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    mu = anomaly.baseline_mean(dets[0].correction.corrected_rate_hz)
    fig = build_side_by_side(dets, aligned, meta, rs.bin_mid_utc)
    mean_trace = next(t for t in fig.data
                      if isinstance(t.meta, dict)
                      and t.meta.get("anomaly") == "mean")
    assert float(mean_trace.y[0]) == pytest.approx(mu)


def test_overlay_anomaly_mean_is_zero_percent():
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    fig = build_overlay(dets, aligned, meta, rs.bin_mid_utc)
    mean_trace = next(t for t in fig.data
                      if isinstance(t.meta, dict)
                      and t.meta.get("anomaly") == "mean")
    assert float(mean_trace.y[0]) == pytest.approx(0.0)


def test_figure_config_reports_units_and_anomaly_indices():
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    fig = build_overlay(dets, aligned, meta, rs.bin_mid_utc)
    cfg = plotting._figure_config(fig, stacked=False)
    assert cfg["units"] == "pct"
    assert len(cfg["anomalyByDetector"]) == len(dets)
    idx = cfg["anomalyByDetector"][0]
    assert set(idx) == {"mean", "lower", "upper", "outlier"}
    # existing role bookkeeping unchanged
    assert cfg["nDetectors"] == len(dets)


def test_side_by_side_units_are_hz():
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    fig = build_side_by_side(dets, aligned, meta, rs.bin_mid_utc)
    cfg = plotting._figure_config(fig, stacked=True)
    assert cfg["units"] == "hz"


def test_series_checkboxes_exclude_anomaly_traces(tmp_path):
    """The combined page's checkbox list must span only the primary
    (detector + external) traces; the hidden per-detector anomaly traces
    (mean/lower/upper/flagged) must not get their own checkbox, or an
    unrelated toggle would restyle them via applyVisibility's index-aligned
    Plotly.restyle call."""
    import re

    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    n_primary = len(dets) + len(aligned)
    ov = build_overlay(dets, aligned, meta, rs.bin_mid_utc)
    sb = build_side_by_side(dets, aligned, meta, rs.bin_mid_utc)

    out = tmp_path / "combined.html"
    plotting.write_combined_html(ov, sb, str(out))
    content = out.read_text()

    n_boxes = content.count('type="checkbox"')
    assert n_boxes == n_primary

    labels = re.findall(r'</span>([^<]*)</label>', content)
    assert len(labels) == n_primary
    for label in labels:
        low = label.lower()
        for word in ("mean", "lower", "upper", "flagged"):
            assert word not in low, (label, word)


def test_anomaly_by_detector_indices_point_to_the_right_traces():
    """cfg["anomalyByDetector"][d][role] must be the actual fig.data index of
    that role's trace, not just a key that happens to exist."""
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    fig = build_overlay(dets, aligned, meta, rs.bin_mid_utc)
    cfg = plotting._figure_config(fig, stacked=False)
    for role_map in cfg["anomalyByDetector"]:
        for role, idx in role_map.items():
            assert fig.data[idx].meta["anomaly"] == role


def test_payload_none_substitutes_non_finite_rate():
    det = make_detector("DetA")
    det.correction.corrected_rate_hz[1] = np.nan
    payload = plotting.build_anomaly_payload([det])
    assert payload[0]["rate"][1] is None
    assert payload[0]["good"][1] is False
    # untouched bins stay real numbers / good
    assert payload[0]["rate"][0] is not None
    assert payload[0]["good"][0] is True


def test_outlier_trace_contains_exactly_the_flagged_bins():
    """A gross count outlier (way outside the exact-Poisson band implied by
    the run's mean rate) must show up in the outlier trace, and nowhere
    else."""
    det = make_detector("DetA")
    det.rs.counts[2] = 50_000  # ~30x the baseline count for this bin
    aligned = []
    meta = PlotMetadata("run", "OULU", "fit", [], [])
    fig = build_overlay([det], aligned, meta, det.rs.bin_mid_utc)

    outlier = next(t for t in fig.data
                   if isinstance(t.meta, dict) and t.meta.get("anomaly") == "outlier")

    mu = anomaly.baseline_mean(det.correction.corrected_rate_hz)
    raw = det.rs.rate_hz
    corrected = det.correction.corrected_rate_hz
    adj = np.where((raw != 0) & np.isfinite(raw), corrected / raw, 1.0)
    T = det.rs.livetime_s
    k_lo = np.full(det.rs.counts.shape, -1)
    k_hi = np.full(det.rs.counts.shape, np.iinfo(np.int64).max)
    for i in range(len(det.rs.counts)):
        lam_i = anomaly.lambda_per_bin(mu, T[i], adj[i])
        k_lo[i], k_hi[i] = anomaly.threshold_counts(float(lam_i), 0.05)
    expected_flags = anomaly.flag_bins(det.rs.counts, k_lo, k_hi)

    # Plotly normalizes datetime64 -> python datetime on trace construction,
    # so compare via numpy datetime64 conversion rather than raw equality.
    got_x = np.array([np.datetime64(v) for v in outlier.x])
    expected_x = np.array([det.rs.bin_mid_utc[i] for i in np.where(expected_flags)[0]])
    assert len(got_x) == int(np.sum(expected_flags))
    assert list(got_x) == list(expected_x)
    assert np.datetime64(det.rs.bin_mid_utc[2]) in got_x


def test_existing_controls_unaffected_by_anomaly_traces():
    """Appended anomaly traces must not disturb the pre-existing side-by-side
    row map / row count / detector count used by the checkbox JS."""
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    fig = build_side_by_side(dets, aligned, meta, rs.bin_mid_utc)
    cfg = plotting._figure_config(fig, stacked=True)
    # nDetectors counts only the real detector rate traces, not anomaly traces
    assert cfg["nDetectors"] == len(dets)
    assert cfg["nRows"] == 1 + len(aligned)
    n_primary = 1 + len(aligned)
    assert len(cfg["rowOfTrace"]) == n_primary


# --- series-checkbox restyle must not sweep the hidden anomaly traces ------


def _find_chrome_binary():
    """Best-effort discovery of a local Chrome/Chromium binary for the
    headless runtime check below. Returns None if nothing usable is found."""
    import shutil

    candidates = [
        "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]
    for c in candidates:
        if "/" in c:
            import os
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c
        else:
            found = shutil.which(c)
            if found:
                return found
    return None


def test_series_visibility_restyle_targets_primary_trace_indices_only():
    """Both applyVisibility() implementations (single-page and combined-page
    templates) must restyle an explicit primary-trace index list, never a
    bare `{ visible: vis }` with no index argument -- Plotly applies an
    unindexed restyle's array cyclically (modulo) across ALL traces in `gd`,
    which would flip the hidden anomaly traces (appended after the primary
    traces) whenever an unrelated series checkbox is toggled.

    This assertion runs unconditionally (no browser required); the actual
    runtime behavior is verified by the headless-Chrome test below."""
    import pathlib
    import re

    text = pathlib.Path(plotting.__file__).read_text()

    # The explicit-index form must be present for both applyVisibility sites.
    explicit_index_calls = re.findall(
        r"vis\.map\(function \(_, i\) \{+ ?return i; ?\}+\)", text)
    assert len(explicit_index_calls) >= 2, (
        "expected an explicit-index vis.map(...) restyle argument in both "
        "the single-page and combined-page applyVisibility() implementations"
    )

    # No remaining unqualified `{ visible: vis }` restyle (no 3rd/index arg)
    # anywhere in the module -- that is exactly the leak this test guards.
    unqualified = re.findall(
        r"Plotly\.restyle\(\s*(?:gd|GD\.\w+)\s*,\s*\{+\s*visible:\s*vis\s*\}+\s*\)",
        text)
    assert unqualified == [], unqualified


# --- y-axis marginal: observed histogram + exact-Poisson overlay ----------


def test_payload_has_marginal():
    det = make_detector("DetA")
    payload = plotting.build_anomaly_payload([det])
    m = payload[0]["marginal"]
    assert set(m) == {"edges", "observed", "expected"}
    assert len(m["edges"]) == len(m["observed"]) + 1
    json.dumps(payload)  # still JSON-able with the marginal embedded


def test_overlay_has_marginal_axis_and_traces():
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    fig = build_overlay(dets, aligned, meta, rs.bin_mid_utc)
    # main x-axis domain shrank to leave the right margin for the marginal
    assert fig.layout.xaxis.domain[1] == pytest.approx(0.82, abs=1e-6)
    # a dedicated marginal x-axis occupies the right strip, sharing y
    assert fig.layout.xaxis3.domain[0] == pytest.approx(0.85, abs=1e-6)
    assert fig.layout.xaxis3.domain[1] == pytest.approx(1.0, abs=1e-6)
    assert fig.layout.xaxis3.anchor == "y"
    roles = [t.meta.get("anomaly") for t in fig.data
             if isinstance(t.meta, dict) and t.meta.get("anomaly") in ("hist", "poisson")]
    assert roles == ["hist", "poisson"]  # one detector
    for t in fig.data:
        if isinstance(t.meta, dict) and t.meta.get("anomaly") in ("hist", "poisson"):
            assert t.visible is False
            assert t.xaxis == "x3" and t.yaxis == "y"
    cfg = plotting._figure_config(fig, stacked=False)
    assert len(cfg["marginalByDetector"]) == len(dets)
    idx = cfg["marginalByDetector"][0]
    assert set(idx) == {"hist", "poisson"}
    for role, i in idx.items():
        assert fig.data[i].meta["anomaly"] == role


def test_side_by_side_rows_share_shrunk_domain():
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    fig = build_side_by_side(dets, aligned, meta, rs.bin_mid_utc)
    # every stacked time panel keeps the same shrunk x-domain (zoom-synced)
    assert fig.layout.xaxis.domain[1] == pytest.approx(0.82, abs=1e-6)
    assert fig.layout.xaxis2.domain[1] == pytest.approx(0.82, abs=1e-6)
    # marginal x-axis anchored to row-1 y, own independent range (no matches)
    marg = getattr(fig.layout, "xaxis4")
    assert marg.domain[0] == pytest.approx(0.85, abs=1e-6)
    assert marg.domain[1] == pytest.approx(1.0, abs=1e-6)
    assert marg.anchor == "y"
    assert marg.matches is None
    roles = [t.meta.get("anomaly") for t in fig.data
             if isinstance(t.meta, dict) and t.meta.get("anomaly") in ("hist", "poisson")]
    assert roles == ["hist", "poisson"]
    for t in fig.data:
        if isinstance(t.meta, dict) and t.meta.get("anomaly") in ("hist", "poisson"):
            assert t.xaxis == "x4" and t.yaxis == "y"


def test_marginal_traces_do_not_change_row_bookkeeping():
    """Adding the two marginal traces per detector must not shift nRows,
    rowOfTrace, or nDetectors (they are meta.anomaly-tagged, hence excluded)."""
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    fig = build_side_by_side(dets, aligned, meta, rs.bin_mid_utc)
    cfg = plotting._figure_config(fig, stacked=True)
    assert cfg["nDetectors"] == len(dets)
    assert cfg["nRows"] == 1 + len(aligned)
    assert len(cfg["rowOfTrace"]) == len(dets) + len(aligned)


def test_side_by_side_marginal_y_is_hz():
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    mu = anomaly.baseline_mean(dets[0].correction.corrected_rate_hz)
    fig = build_side_by_side(dets, aligned, meta, rs.bin_mid_utc)
    hist = next(t for t in fig.data if isinstance(t.meta, dict)
                and t.meta.get("anomaly") == "hist")
    # bucket midpoints are native Hz, so they bracket the run mean rate
    ys = [float(v) for v in hist.y]
    assert min(ys) <= mu <= max(ys)


def test_headless_toggle_leaves_anomaly_traces_hidden(tmp_path):
    """Runtime regression test for the restyle-index leak: load the combined
    page in headless Chrome, toggle an unrelated (external-series) checkbox
    to fire applyVisibility(), then read back gd.data[k].visible for every
    trace tagged meta.anomaly. All of them must stay False -- if the restyle
    call regresses to an unindexed `{ visible: vis }`, Plotly's cyclic
    application will flip some of these to true/"legendonly".

    Skipped (not failed) when no local Chrome/Chromium binary is available,
    since the pytest suite must stay runnable in environments without one;
    test_series_visibility_restyle_targets_primary_trace_indices_only above
    provides the non-headless fallback guard."""
    import json as _json
    import re
    import subprocess

    chrome = _find_chrome_binary()
    if chrome is None:
        pytest.skip("no local Chrome/Chromium binary found for headless check")

    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    ov = build_overlay(dets, aligned, meta, rs.bin_mid_utc)
    sb = build_side_by_side(dets, aligned, meta, rs.bin_mid_utc)
    n_primary = len(dets) + len(aligned)
    assert n_primary >= 2, "need at least one non-detector checkbox to toggle"

    out = tmp_path / "combined.html"
    plotting.write_combined_html(ov, sb, str(out))

    probe_js = """
<script>
(function () {
  function run() {
    var boxes = document.querySelectorAll("#series-list input[type=checkbox]");
    var target = boxes[1];
    target.checked = false;
    target.dispatchEvent(new Event("change", { bubbles: true }));

    var results = [];
    ["overlay", "side"].forEach(function (view) {
      var gd = document.getElementById("viz-" + view).querySelector(".plotly-graph-div");
      gd.data.forEach(function (t, i) {
        if (t.meta && t.meta.anomaly) {
          results.push(view + ":" + i + ":" + t.meta.anomaly + ":" + String(t.visible));
        }
      });
    });
    var pre = document.createElement("pre");
    pre.id = "probe";
    pre.textContent = results.join("\\n");
    document.body.appendChild(pre);
  }
  if (document.readyState === "complete") { setTimeout(run, 500); }
  else { window.addEventListener("load", function () { setTimeout(run, 500); }); }
})();
</script>
"""
    content = out.read_text()
    content = content.replace("</body>", probe_js + "\n</body>")
    out.write_text(content)

    result = subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
         "--virtual-time-budget=5000", "--dump-dom", f"file://{out}"],
        capture_output=True, text=True, timeout=60,
    )
    dom = result.stdout
    m = re.search(r'<pre id="probe">(.*?)</pre>', dom, re.S)
    assert m, (
        f"probe output not found in dumped DOM "
        f"(chrome rc={result.returncode}, stderr={result.stderr[-2000:]})"
    )
    lines = [ln for ln in m.group(1).splitlines() if ln.strip()]
    # 6 meta.anomaly traces per view (4 anomaly-band + 2 marginal), 2 views.
    assert len(lines) == 2 * 6, f"expected 12 anomaly-trace rows, got: {lines}"
    for line in lines:
        assert line.endswith(":false"), (
            f"anomaly trace leaked visible after unrelated toggle: {line}"
        )
