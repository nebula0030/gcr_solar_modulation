from __future__ import annotations

import json

import numpy as np
import pytest

import anomaly
import external_sources as es
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


def test_annotation_roles_tags_events_and_keeps_gap_label_muted():
    """Event annotations must be tagged "event" even when their (unvalidated,
    external DONKI) label text happens to collide with the gap-label's "no
    data" text -- the event- name check must win over the muted fallback."""
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    gaps = [(np.datetime64("2026-07-10T02:00:00"),
             np.datetime64("2026-07-10T03:00:00"))]
    events = [es.SolarEvent("cme", np.datetime64("2026-07-10T12:00:00"), "no data"),
              es.SolarEvent("flare", np.datetime64("2026-07-10T15:00:00"), "X1.5")]
    fig = build_overlay(dets, aligned, meta, rs.bin_mid_utc, gaps=gaps, events=events)
    roles = plotting._figure_config(fig, stacked=False)["annotationRoles"]

    assert len(roles) == len(fig.layout.annotations)  # index-parallel
    for role, ann in zip(roles, fig.layout.annotations):
        if (ann.name or "").startswith("event-"):
            assert role == "event"
        elif ann.name == "gap-label":
            assert role == "muted"
    assert roles.count("event") == 2
    assert "muted" in roles  # the gap label is still muted


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


def test_overlay_secondary_axis_only_present_with_native_unit_series():
    """The 'Other indices (native units)' secondary axis must appear only when
    a non-modulation external series actually uses it. Otherwise Plotly draws
    an empty, SI-prefixed ('15u') axis that clutters the plot and sits
    confusingly beside the y-axis marginal."""
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]

    # No external series at all -> no secondary axis.
    ov_none = build_overlay(dets, [], meta, rs.bin_mid_utc)
    y2 = getattr(ov_none.layout, "yaxis2", None)
    assert getattr(y2, "overlaying", None) != "y"
    assert plotting._figure_config(ov_none, stacked=False)["hasSecondaryAxis"] is False

    # With a non-modulation native-units series (Kp) it is present.
    ov_ext = build_overlay(dets, aligned, meta, rs.bin_mid_utc)
    assert ov_ext.layout.yaxis2.overlaying == "y"
    assert plotting._figure_config(ov_ext, stacked=False)["hasSecondaryAxis"] is True


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
    # One series checkbox per primary series (3 traces here: 1 detector + 2
    # external), excluding the hidden per-detector anomaly traces appended
    # after them. Series checkboxes carry a data-i attribute; the standalone
    # anomaly-toggle checkbox does not, so counting data-i isolates them.
    n_primary = len(dets) + len(aligned)
    assert content.count('data-i="') == n_primary
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

    # Series checkboxes carry data-i; the anomaly-toggle checkbox does not.
    n_boxes = content.count('data-i="')
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


def test_headless_anomaly_toggle_shows_then_hides_with_payload(tmp_path):
    """Populated-payload runtime test: unlike the empty-payload leak test
    above, this renders with a real anomaly payload so applyAnomaly's
    per-detector restyle loop actually runs. Toggling #anomaly-toggle ON must
    make anomaly/marginal traces visible in BOTH views; toggling it OFF must
    hide every one of them again (index-scoped restyle, no leftover)."""
    import re
    import subprocess

    chrome = _find_chrome_binary()
    if chrome is None:
        pytest.skip("no local Chrome/Chromium binary found for headless check")

    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    ov = build_overlay(dets, aligned, meta, rs.bin_mid_utc)
    sb = build_side_by_side(dets, aligned, meta, rs.bin_mid_utc)
    payload = plotting.build_anomaly_payload(dets)

    out = tmp_path / "combined_anom.html"
    plotting.write_combined_html(ov, sb, str(out), anomaly=payload)

    probe_js = """
<script>
(function () {
  function visRows() {
    var rows = [];
    ["overlay", "side"].forEach(function (view) {
      var gd = document.getElementById("viz-" + view).querySelector(".plotly-graph-div");
      gd.data.forEach(function (t) {
        if (t.meta && t.meta.anomaly) {
          rows.push(view + ":" + t.meta.anomaly + ":" + String(t.visible));
        }
      });
    });
    return rows;
  }
  function run() {
    var box = document.getElementById("anomaly-toggle");
    box.checked = true; box.dispatchEvent(new Event("change", { bubbles: true }));
    var onRows = visRows();
    box.checked = false; box.dispatchEvent(new Event("change", { bubbles: true }));
    var offRows = visRows();
    var pre = document.createElement("pre");
    pre.id = "probe";
    pre.textContent = JSON.stringify({ on: onRows, off: offRows });
    document.body.appendChild(pre);
  }
  if (document.readyState === "complete") { setTimeout(run, 700); }
  else { window.addEventListener("load", function () { setTimeout(run, 700); }); }
})();
</script>
"""
    content = out.read_text().replace("</body>", probe_js + "\n</body>")
    out.write_text(content)

    result = subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
         "--virtual-time-budget=6000", "--dump-dom", f"file://{out}"],
        capture_output=True, text=True, timeout=60,
    )
    m = re.search(r'<pre id="probe">(.*?)</pre>', result.stdout, re.S)
    assert m, (f"probe not found (rc={result.returncode}, "
               f"stderr={result.stderr[-2000:]})")
    data = json.loads(m.group(1))
    # ON: at least one anomaly trace visible in EACH view (applyAnomaly ran).
    assert any(r.startswith("overlay:") and r.endswith(":true")
               for r in data["on"]), data["on"]
    assert any(r.startswith("side:") and r.endswith(":true")
               for r in data["on"]), data["on"]
    # OFF: every anomaly trace hidden again in both views (no leftover).
    for r in data["off"]:
        assert r.endswith(":false"), "anomaly trace visible after off: " + r


# --- embedded JS exact-Poisson numerics, verified against Python -----------


def _eval_js_result(chrome, tmp_path, page_html, result_expr, *, name="probe"):
    """Render ``page_html`` in headless Chrome, evaluate ``result_expr`` in a
    load handler, serialize it into a ``<pre id="probe">`` node, then
    ``--dump-dom`` the rendered page and parse the JSON back out.

    This mirrors the deterministic dump-dom pattern used by
    test_headless_toggle_leaves_anomaly_traces_hidden above -- reading
    console output proved flaky across Chrome versions, so results are
    smuggled through the DOM instead."""
    import re
    import subprocess

    page = tmp_path / f"{name}.html"
    probe_js = """
<script>
(function () {
  function run() {
    var result = (%s);
    var pre = document.createElement("pre");
    pre.id = "probe";
    pre.textContent = JSON.stringify(result);
    document.body.appendChild(pre);
  }
  if (document.readyState === "complete") { setTimeout(run, 100); }
  else { window.addEventListener("load", function () { setTimeout(run, 100); }); }
})();
</script>
""" % result_expr
    page.write_text(page_html + probe_js)

    result = subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
         "--virtual-time-budget=5000", "--dump-dom", f"file://{page}"],
        capture_output=True, text=True, timeout=60,
    )
    dom = result.stdout
    m = re.search(r'<pre id="probe">(.*?)</pre>', dom, re.S)
    assert m, (
        f"probe output not found in dumped DOM "
        f"(chrome rc={result.returncode}, stderr={result.stderr[-2000:]})"
    )
    return json.loads(m.group(1))


def test_anomaly_js_numerics_fragment_has_expected_shape():
    """Static shape check that runs unconditionally (no browser required):
    the fragment defines window.__anom with poissonCdf/poissonThresholds,
    and its gser/gcf use the adaptive iteration cap (matching anomaly.py's
    _gser/_gcf), not a fixed ITMAX -- the fixed cap silently returns wrong
    CDF values for large lambda (e.g. daily binning, lambda ~ 43000)."""
    frag = plotting._anomaly_js_numerics()
    assert "window.__anom" in frag
    assert "poissonCdf" in frag and "poissonThresholds" in frag
    assert "Math.max(1000, Math.floor(4*(a+x)))" in frag, (
        "expected the adaptive iteration cap (mirroring anomaly._gser/_gcf); "
        "a fixed ITMAX cap silently under-converges for large lambda"
    )


def test_anomaly_js_numerics_included_in_combined_page():
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    ov = build_overlay(dets, aligned, meta, rs.bin_mid_utc)
    sb = build_side_by_side(dets, aligned, meta, rs.bin_mid_utc)
    out_path = "/tmp/_cw_combined_js_numerics_probe.html"
    import os
    try:
        plotting.write_combined_html(ov, sb, out_path)
        content = open(out_path, encoding="utf-8").read()
        assert "window.__anom" in content
        assert "poissonThresholds" in content
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


def test_js_poisson_matches_python(tmp_path):
    """Headless-Chrome verification that the embedded JS numerics agree
    exactly with the Python reference (anomaly.threshold_counts /
    anomaly.poisson_cdf) -- including a large-lambda case (daily binning,
    lambda ~ 43200) that only the adaptive iteration cap converges on."""
    chrome = _find_chrome_binary()
    if chrome is None:
        pytest.skip("no local Chrome/Chromium binary found for headless check")

    frag = plotting._anomaly_js_numerics()
    html_body = "<!doctype html><html><body>" + frag

    cases = [(1000.0, 0.05), (1700.0, 0.01), (50.0, 0.1), (43200.0, 0.05)]
    expected_thresholds = [anomaly.threshold_counts(lam, p) for (lam, p) in cases]

    cases_js = json.dumps([[lam, p] for (lam, p) in cases])
    got_thresholds = _eval_js_result(
        chrome, tmp_path, html_body,
        "%s.map(function(c){"
        "var r=window.__anom.poissonThresholds(c[0],c[1]);"
        "return [r.kLo,r.kHi];})" % cases_js,
        name="thresholds",
    )
    for (lam, p), (e_lo, e_hi), (g_lo, g_hi) in zip(
        cases, expected_thresholds, got_thresholds
    ):
        assert (g_lo, g_hi) == (e_lo, e_hi), (
            f"threshold mismatch at lam={lam}, p={p}: "
            f"python={e_lo, e_hi} js={g_lo, g_hi}"
        )

    # CDF agreement at a sample point per case, including the large-lambda one.
    cdf_points = [(1000, 1000.0), (1700, 1700.0), (50, 50.0), (43200, 43200.0)]
    expected_cdf = [anomaly.poisson_cdf(k, lam) for (k, lam) in cdf_points]
    cdf_points_js = json.dumps([[k, lam] for (k, lam) in cdf_points])
    got_cdf = _eval_js_result(
        chrome, tmp_path, html_body,
        "%s.map(function(c){return window.__anom.poissonCdf(c[0],c[1]);})"
        % cdf_points_js,
        name="cdf",
    )
    for (k, lam), e_cdf, g_cdf in zip(cdf_points, expected_cdf, got_cdf):
        assert g_cdf == pytest.approx(e_cdf, abs=1e-9), (
            f"cdf mismatch at k={k}, lam={lam}: python={e_cdf} js={g_cdf}"
        )


# --- interactive anomaly control: markup + headless toggle behaviour -------


def _detector_with_one_outlier(name="DetA", n=6, base=0.47, outlier_bin=3,
                               outlier_counts=6000):
    """One detector whose bin ``outlier_bin`` carries an obviously anomalous
    (far above the exact-Poisson upper threshold) raw count, so toggling the
    control flags exactly that bin."""
    start = np.datetime64("2026-07-10T00:00:00.000000000")
    off = (np.arange(n) * 3600.0 * 1e9).astype("timedelta64[ns]")
    counts = np.full(n, 1700, dtype=np.int64)
    counts[outlier_bin] = outlier_counts
    rs = RateSeries(start + off, start + off, counts,
                    np.full(n, 3600.0), np.full(n, base), np.full(n, 0.01),
                    np.linspace(1006, 1010, n), np.linspace(22, 25, n), 3600.0)
    cr = CorrectionResult(np.full(n, base), np.full(n, 0.01), -0.0013, -0.004,
                          2e-4, 5e-4, 0.85, 1008.0, 23.5, "fit", [])
    return DetectorSeries(name=name, rs=rs, correction=cr)


def test_combined_page_has_anomaly_control_and_list(tmp_path):
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    ov = build_overlay(dets, aligned, meta, rs.bin_mid_utc)
    sb = build_side_by_side(dets, aligned, meta, rs.bin_mid_utc)
    out = tmp_path / "c.html"
    plotting.write_combined_html(
        ov, sb, str(out), anomaly=plotting.build_anomaly_payload(dets))
    html = out.read_text()
    assert 'id="anomaly-toggle"' in html
    assert 'id="anomaly-p"' in html
    assert 'id="anomaly-readout"' in html
    assert 'id="flagged-list"' in html
    assert "detectorsAnomaly" in html
    assert "applyAnomaly" in html
    # the number input defaults to p=0.05 and starts disabled
    assert 'value="0.05"' in html and "disabled" in html


def test_headless_toggle_shows_lines_and_flags(tmp_path):
    """Render a one-detector run with one obvious outlier bin, click
    #anomaly-toggle on load, and read back (via a probe DOM node): the outlier
    trace has >=1 point, #flagged-list names the detector, and at least one
    anomaly trace is visible in BOTH the overlay and side graph divs."""
    import re
    import subprocess

    chrome = _find_chrome_binary()
    if chrome is None:
        pytest.skip("no local Chrome/Chromium binary found for headless check")

    _rs, _corr, aligned, meta = make_inputs()
    det = _detector_with_one_outlier(name="DetA")
    dets = [det]
    master = det.rs.bin_mid_utc
    ov = build_overlay(dets, aligned, meta, master)
    sb = build_side_by_side(dets, aligned, meta, master)

    out = tmp_path / "combined.html"
    plotting.write_combined_html(
        ov, sb, str(out), anomaly=plotting.build_anomaly_payload(dets))

    probe_js = """
<script>
(function () {
  function run() {
    var toggle = document.getElementById("anomaly-toggle");
    toggle.checked = true;
    toggle.dispatchEvent(new Event("change", { bubbles: true }));

    function anyVisible(view) {
      var gd = document.getElementById("viz-" + view)
        .querySelector(".plotly-graph-div");
      return gd.data.some(function (t) {
        return t.meta && t.meta.anomaly && t.visible === true; });
    }
    var ovgd = document.getElementById("viz-overlay")
      .querySelector(".plotly-graph-div");
    var out = ovgd.data.filter(function (t) {
      return t.meta && t.meta.anomaly === "outlier"; })[0];
    var res = {
      outlierPoints: (out && out.x) ? out.x.length : 0,
      listText: document.getElementById("flagged-list").textContent,
      overlayVisible: anyVisible("overlay"),
      sideVisible: anyVisible("side")
    };
    var pre = document.createElement("pre");
    pre.id = "probe";
    pre.textContent = JSON.stringify(res);
    document.body.appendChild(pre);
  }
  if (document.readyState === "complete") { setTimeout(run, 500); }
  else { window.addEventListener("load", function () { setTimeout(run, 500); }); }
})();
</script>
"""
    content = out.read_text().replace("</body>", probe_js + "\n</body>")
    out.write_text(content)

    result = subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
         "--virtual-time-budget=5000", "--dump-dom", f"file://{out}"],
        capture_output=True, text=True, timeout=60,
    )
    m = re.search(r'<pre id="probe">(.*?)</pre>', result.stdout, re.S)
    assert m, (
        f"probe output not found (chrome rc={result.returncode}, "
        f"stderr={result.stderr[-2000:]})"
    )
    res = json.loads(m.group(1))
    assert res["outlierPoints"] >= 1, res
    assert res["listText"].strip(), "flagged-list should be populated"
    assert "DetA" in res["listText"], res["listText"]
    assert res["overlayVisible"] is True, res
    assert res["sideVisible"] is True, res


def _two_events():
    return [
        es.SolarEvent("cme", np.datetime64("2026-07-10T12:00:00"), "CME"),
        es.SolarEvent("flare", np.datetime64("2026-07-11T03:00:00"), "X1.5"),
    ]


def test_overlay_adds_hidden_event_shapes_and_annotations():
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    fig = build_overlay(dets, aligned, meta, rs.bin_mid_utc, events=_two_events())
    ev_shapes = [s for s in fig.layout.shapes
                 if (s.name or "").startswith("event-")]
    ev_anns = [a for a in fig.layout.annotations
               if (a.name or "").startswith("event-")]
    assert len(ev_shapes) == 2 and len(ev_anns) == 2
    assert all(s.visible is False for s in ev_shapes)
    assert all(s.yref == "paper" for s in ev_shapes)


def test_events_none_adds_nothing():
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    fig = build_overlay(dets, aligned, meta, rs.bin_mid_utc)  # events default None
    assert not any((s.name or "").startswith("event-") for s in fig.layout.shapes)


def test_events_requested_meta_merges_with_header_meta():
    """eventsRequested must not clobber the header_title/header_meta stashed
    by _apply_common_layout -- update_layout(meta=...) replaces the dict
    wholesale rather than merging it, so the builder must merge by hand."""
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    fig = build_overlay(dets, aligned, meta, rs.bin_mid_utc, events=_two_events())
    assert fig.layout.meta.get("eventsRequested") is True
    assert fig.layout.meta.get("header_title")

    fig_none = build_overlay(dets, aligned, meta, rs.bin_mid_utc)
    assert not fig_none.layout.meta.get("eventsRequested")

    sb = build_side_by_side(dets, aligned, meta, rs.bin_mid_utc,
                            events=_two_events())
    assert sb.layout.meta.get("eventsRequested") is True
    assert sb.layout.meta.get("header_title")


def test_side_by_side_event_line_spans_panels():
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    fig = build_side_by_side(dets, aligned, meta, rs.bin_mid_utc,
                             events=_two_events())
    ev = [s for s in fig.layout.shapes if (s.name or "").startswith("event-")]
    assert len(ev) == 2
    assert all(s.yref == "paper" and s.y0 == 0 and s.y1 == 1 for s in ev)


# --- "Show events" checkbox: presence + config + headless toggle -----------


def test_figure_config_collects_event_indices():
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    ov = build_overlay(dets, aligned, meta, rs.bin_mid_utc, events=_two_events())
    cfg = plotting._figure_config(ov, stacked=False)
    # Two events -> two event shapes + two event annotations, indices valid.
    assert len(cfg["eventShapes"]) == 2
    assert len(cfg["eventAnnotations"]) == 2
    for i in cfg["eventShapes"]:
        assert (ov.layout.shapes[i].name or "").startswith("event-")
    for j in cfg["eventAnnotations"]:
        assert (ov.layout.annotations[j].name or "").startswith("event-")
    # No events -> empty index lists.
    ov0 = build_overlay(dets, aligned, meta, rs.bin_mid_utc)
    cfg0 = plotting._figure_config(ov0, stacked=False)
    assert cfg0["eventShapes"] == [] and cfg0["eventAnnotations"] == []


def test_combined_page_has_events_toggle_only_with_events(tmp_path):
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    ov = build_overlay(dets, aligned, meta, rs.bin_mid_utc, events=_two_events())
    sb = build_side_by_side(dets, aligned, meta, rs.bin_mid_utc, events=_two_events())
    out = tmp_path / "c.html"
    plotting.write_combined_html(ov, sb, str(out))
    assert 'id="events-toggle"' in out.read_text()

    ov0 = build_overlay(dets, aligned, meta, rs.bin_mid_utc)   # no events
    sb0 = build_side_by_side(dets, aligned, meta, rs.bin_mid_utc)
    out0 = tmp_path / "c0.html"
    plotting.write_combined_html(ov0, sb0, str(out0))
    assert 'id="events-toggle"' not in out0.read_text()


def test_combined_page_events_requested_but_none_shows_note(tmp_path):
    """When events were requested but the window held none, the checkbox still
    renders (with the empty-window note) so the viewer knows the query ran."""
    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    ov = build_overlay(dets, aligned, meta, rs.bin_mid_utc)  # no events drawn
    sb = build_side_by_side(dets, aligned, meta, rs.bin_mid_utc)
    ov.layout.meta = dict(ov.layout.meta or {})
    ov.layout.meta["eventsRequested"] = True
    out = tmp_path / "c.html"
    plotting.write_combined_html(ov, sb, str(out))
    content = out.read_text()
    assert 'id="events-toggle"' in content
    assert "no CME/X-flare events in this window" in content


def test_headless_events_toggle_shows_lines(tmp_path):
    """Render the combined page with two events, click #events-toggle on, and
    read back that every event-* shape reports visible===true in BOTH graph
    divs; toggling off returns them all to visible===false. Skip if no Chrome."""
    import re
    import subprocess

    chrome = _find_chrome_binary()
    if chrome is None:
        pytest.skip("no local Chrome/Chromium binary found for headless check")

    rs, corr, aligned, meta = make_inputs()
    dets = [DetectorSeries("Muon rate", rs, corr)]
    ov = build_overlay(dets, aligned, meta, rs.bin_mid_utc, events=_two_events())
    sb = build_side_by_side(dets, aligned, meta, rs.bin_mid_utc, events=_two_events())

    out = tmp_path / "combined_events.html"
    plotting.write_combined_html(ov, sb, str(out))

    probe_js = """
<script>
(function () {
  function evStates() {
    var rows = [];
    var annRows = [];
    ["overlay", "side"].forEach(function (view) {
      var gd = document.getElementById("viz-" + view)
        .querySelector(".plotly-graph-div");
      (gd.layout.shapes || []).forEach(function (s) {
        if (s.name && s.name.indexOf("event-") === 0) {
          rows.push(view + ":" + String(s.visible));
        }
      });
      (gd.layout.annotations || []).forEach(function (a) {
        if (a.name && a.name.indexOf("event-") === 0) {
          annRows.push(view + ":" + String(a.visible));
        }
      });
    });
    return { shapes: rows, annotations: annRows };
  }
  function run() {
    var box = document.getElementById("events-toggle");
    box.checked = true; box.dispatchEvent(new Event("change", { bubbles: true }));
    var onStates = evStates();
    box.checked = false; box.dispatchEvent(new Event("change", { bubbles: true }));
    var offStates = evStates();
    var pre = document.createElement("pre");
    pre.id = "probe";
    pre.textContent = JSON.stringify({
      on: onStates.shapes, off: offStates.shapes,
      onAnn: onStates.annotations, offAnn: offStates.annotations
    });
    document.body.appendChild(pre);
  }
  if (document.readyState === "complete") { setTimeout(run, 700); }
  else { window.addEventListener("load", function () { setTimeout(run, 700); }); }
})();
</script>
"""
    content = out.read_text().replace("</body>", probe_js + "\n</body>")
    out.write_text(content)

    result = subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
         "--virtual-time-budget=6000", "--dump-dom", f"file://{out}"],
        capture_output=True, text=True, timeout=60,
    )
    m = re.search(r'<pre id="probe">(.*?)</pre>', result.stdout, re.S)
    assert m, (f"probe not found (rc={result.returncode}, "
               f"stderr={result.stderr[-2000:]})")
    data = json.loads(m.group(1))
    # Two events x two views = four event shapes, all visible when toggled on.
    assert len(data["on"]) == 4, data
    assert all(r.endswith(":true") for r in data["on"]), data["on"]
    assert any(r.startswith("overlay:") for r in data["on"]), data["on"]
    assert any(r.startswith("side:") for r in data["on"]), data["on"]
    # Toggling off returns every event shape to hidden in both views.
    assert len(data["off"]) == 4, data
    for r in data["off"]:
        assert r.endswith(":false"), r
    # Same two assertions, but for the event-* ANNOTATIONS (labels), which a
    # regression in applyEvents()'s annotation branch (or the theme recolor)
    # could silently leave stuck at their initial visible=False.
    assert len(data["onAnn"]) == 4, data
    assert all(r.endswith(":true") for r in data["onAnn"]), data["onAnn"]
    assert any(r.startswith("overlay:") for r in data["onAnn"]), data["onAnn"]
    assert any(r.startswith("side:") for r in data["onAnn"]), data["onAnn"]
    assert len(data["offAnn"]) == 4, data
    for r in data["offAnn"]:
        assert r.endswith(":false"), r
