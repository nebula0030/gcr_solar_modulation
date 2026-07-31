from __future__ import annotations

import numpy as np
import pytest

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
    for kind, fig, stacked in (
        ("overlay", build_overlay([DetectorSeries("Muon rate", rs, corr)], aligned, meta, rs.bin_mid_utc), False),
        ("side", build_side_by_side([DetectorSeries("Muon rate", rs, corr)], aligned, meta, rs.bin_mid_utc), True),
    ):
        out = tmp_path / (kind + ".html")
        write_html(fig, str(out), stacked=stacked)
        content = out.read_text()
        n_boxes = content.count('type="checkbox"')
        assert n_boxes == len(fig.data), (kind, n_boxes, len(fig.data))
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
    rate_traces = [t for t in fig.data if (t.name or "").startswith("Det")]
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
    assert ov_cfg["stacked"] is False
    assert ov_cfg["nDetectors"] == 1
    assert ov_cfg["hasSecondaryAxis"] is True
    assert ov_cfg["rowOfTrace"] == [1] * len(ov.data)

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
    # One checkbox per series (3 traces here: 1 detector + 2 external), not two.
    assert content.count('type="checkbox"') == len(ov.data)
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
