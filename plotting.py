"""Interactive Plotly figures comparing muon rate to solar activity.

Two views are always produced:

* **Overlay** -- one shared time axis. Because the series span wildly
  different scales (muon rate ~0.5 Hz, neutron monitor ~99 counts/s, X-ray
  ~1e-6 W/m2, Kp 0-9, sunspot ~30-90), the two physically comparable
  series -- corrected muon rate and neutron monitor, which both trace
  cosmic-ray flux -- are shown as percent deviation from their run mean so a
  Forbush decrease appears at a comparable size in both. Everything else goes
  on a secondary axis.
* **Side-by-side** -- stacked panels in native units with linked x-axes, so
  zooming one panel zooms all and the timelines stay aligned.

Only the corrected rate is plotted; the raw rate is deliberately omitted.

Colours follow the dataviz skill's validated categorical palette (fixed hue
order, never cycled per-figure): the muon rate anchors on slot 1 (blue) and
every external series after it draws from slots 2-8 in order, so a given
series keeps the same colour across both figures.
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import anomaly
from align import AlignedSeries
from correction import CorrectionResult
from external_sources import SolarEvent
from rate import RateSeries

#: Sources shown as percent deviation in overlay mode (cosmic-ray flux proxies).
MODULATION_SOURCES: Tuple[str, ...] = ("NMDB",)

#: dataviz categorical palette, slots 2-8 (light mode) -- slot 1 (blue) is
#: reserved for the muon rate so it never collides with the first external
#: series. Fixed order, assigned in sequence, never cycled arbitrarily.
_SERIES_COLORS = (
    "#eb6834",  # slot 2: orange
    "#1baf7a",  # slot 3: aqua
    "#eda100",  # slot 4: yellow
    "#e87ba4",  # slot 5: magenta
    "#008300",  # slot 6: green
    "#4a3aa7",  # slot 7: violet
    "#e34948",  # slot 8: red
)

#: dataviz categorical palette, slot 1 (blue) -- the muon rate's fixed colour.
_MUON_COLOR = "#2a78d6"

#: Flagged-outlier marker colour (theme-adjusted in a later task).
_ALERT_COLOR = "#c0392b"

#: Fixed colour order: detector 0 anchors on blue (today's muon-rate colour),
#: then every subsequent detector and every external series draw from the
#: remaining slots in order -- never cycled arbitrarily.
_ALL_SLOTS = (_MUON_COLOR,) + _SERIES_COLORS  # blue, then slots 2-8


def _detector_color(index: int) -> str:
    return _ALL_SLOTS[index % len(_ALL_SLOTS)]


def _external_color(index: int, n_detectors: int) -> str:
    return _ALL_SLOTS[(n_detectors + index) % len(_ALL_SLOTS)]

#: dataviz chart chrome tokens (light surface).
_GRIDLINE_COLOR = "#e1e0d9"
_MUTED_INK = "#898781"
_SECONDARY_INK = "#52514e"

#: Light -> dark step for every palette slot. Dark mode is *selected* from the
#: same hues re-stepped for the dark surface, not an automatic flip.
_LIGHT_TO_DARK = {
    "#2a78d6": "#3987e5",  # slot 1 blue (muon rate)
    "#eb6834": "#d95926",  # slot 2 orange
    "#1baf7a": "#199e70",  # slot 3 aqua
    "#eda100": "#c98500",  # slot 4 yellow
    "#e87ba4": "#d55181",  # slot 5 magenta
    "#008300": "#008300",  # slot 6 green
    "#4a3aa7": "#9085e9",  # slot 7 violet
    "#e34948": "#e66767",  # slot 8 red
}

#: Surface / ink / chrome tokens per mode, from the dataviz palette.
_THEMES = {
    "light": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "secondary": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
        "panel": "#f2f1ec", "border": "#d8d7d0",
    },
    "dark": {
        "surface": "#1a1a19", "text": "#ffffff", "secondary": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
        "panel": "#232322", "border": "#383835",
    },
}

#: matplotlib-style line specs for the muon rate. Dense runs get unreadable
#: when the connecting line dominates, so markers-only options are offered.
_LINE_SPECS = (
    ("o-", "o-  line + markers", "lines+markers", "solid", 8),
    ("-", "-   line only", "lines", "solid", 8),
    ("o", "o   markers only", "markers", "solid", 8),
    ("--", "--  dashed line", "lines", "dash", 8),
    (".", ".   fine points", "markers", "solid", 4),
)


@dataclass
class DetectorSeries:
    """One detector's corrected rate, for stacking on a shared axis."""

    name: str
    rs: "RateSeries"
    correction: "CorrectionResult"


@dataclass
class PlotMetadata:
    """Labels and provenance shown on the figures.

    ``header_lines`` carries the run parameters (bin size, input file,
    location, meteorology source, time range, requested sources) that are
    printed as a block at the top of both figures, above the graphs.
    """

    run_name: str
    station_label: str
    correction_label: str
    footer_notes: List[str] = field(default_factory=list)
    header_lines: List[str] = field(default_factory=list)


def _titlecase(text: str) -> str:
    """Title-case a label while preserving all-caps tokens (GOES, NMDB, SSN)."""
    out = []
    for word in text.split(" "):
        if any(c.isupper() for c in word) and word.upper() == word:
            out.append(word)  # already an acronym / all-caps token
        else:
            out.append(word[:1].upper() + word[1:])
    return " ".join(out)


def _hover_text(series: AlignedSeries) -> List[str]:
    """Per-point hover text, marking interpolated values as such."""
    out = []
    for value, is_interp in zip(series.values, series.interpolated):
        if not np.isfinite(value):
            out.append("{0}: no data".format(series.name))
        elif is_interp:
            out.append(
                "{0}: {1:.4g} {2} (interpolated)".format(
                    series.name, value, series.units
                )
            )
        else:
            out.append(
                "{0}: {1:.4g} {2}".format(series.name, value, series.units)
            )
    return out


def _header_text(meta: PlotMetadata, title: str) -> Tuple[str, int]:
    """Build the run's metadata block as HTML lines (title excluded).

    Everything the viewer needs to read the plot -- the run's bin size and
    input parameters, the chosen station, the correction, and provenance
    notes -- rendered as an HTML block that ``write_html`` places above the
    graphs. Returns ``(meta_html, line_count)``.
    """
    lines = ["Run: {0}".format(meta.run_name)]
    lines.extend(meta.header_lines)
    lines.append("Station: {0}".format(meta.station_label))
    lines.append("Correction: {0}".format(meta.correction_label))
    for note in meta.footer_notes:
        lines.append(
            '<span style="color:{0}">{1}</span>'.format(_MUTED_INK, note)
        )
    return "<br>".join(lines), len(lines)


def _apply_common_layout(
    fig: go.Figure, meta: PlotMetadata, title: str, plot_height: int,
    top_margin: int = 60, bottom_margin: int = 90,
) -> None:
    """Size the figure and stash the header text for the HTML page.

    The metadata header is NOT baked into the figure -- ``write_html`` renders
    it as an HTML block above the plot, so the whole plotting area goes to the
    graphs instead of a tall top margin. The header text (and its line count)
    is stashed in ``layout.meta`` for ``write_html`` to read. ``top_margin``
    only needs to clear the first subplot title.
    """
    meta_html, _ = _header_text(meta, title)
    fig.update_layout(
        hovermode="x unified",
        template="plotly_white",
        height=top_margin + plot_height + bottom_margin,
        margin=dict(l=70, r=70, t=top_margin, b=bottom_margin),
        meta=dict(header_title=title, header_meta=meta_html),
    )


_GAP_FILL_LIGHT = "rgba(137,135,129,0.12)"  # muted ink at low alpha


def _iso(dt64: np.datetime64) -> str:
    """Render a datetime64 scalar as an ISO-8601 string for Plotly.

    Plotly's JSON encoder serializes a datetime64 *array* (as used for trace
    ``x`` data) to ISO-8601 strings correctly, but a bare datetime64 *scalar*
    (as passed to ``add_vrect``/``add_annotation``) goes through ``.item()``,
    which for nanosecond precision returns a plain integer (ns since epoch)
    that lands off the date axis. Downcasting to microsecond precision first
    forces ``str()`` to produce a clean ISO string instead.
    """
    return str(np.datetime64(dt64, "us"))


def _add_gap_bands(fig: go.Figure, gaps, per_row: int = 1) -> None:
    """Shade each gap region with a faint band across the plotting area.

    On the overlay (per_row=1) one band spans the y-axis. On the stacked
    side-by-side the band is added to every panel via ``add_vrect`` (which
    spans all rows by default). Empty/None gaps -> nothing drawn.
    """
    if not gaps:
        return
    for start, end in gaps:
        fig.add_vrect(
            x0=_iso(start), x1=_iso(end),
            fillcolor=_GAP_FILL_LIGHT, line_width=0, layer="below",
        )
    # Label the first gap as "no data".
    first_start, first_end = gaps[0]
    mid = first_start + (first_end - first_start) / 2
    fig.add_annotation(
        x=_iso(mid), y=1.0, yref="paper", yanchor="bottom",
        text="no data", showarrow=False,
        font=dict(size=10, color=_MUTED_INK),
        name="gap-label",
    )


#: Event-mark ink per kind. Light is drawn into the figure by ``_add_event_marks``;
#: the combined page re-selects the dark step client-side on a theme toggle.
_EVENT_COLORS = {"cme": "#7b3fbf", "flare": "#d98324"}
_EVENT_COLORS_DARK = {"cme": "#b085e0", "flare": "#e0a35a"}


def _event_significance(ev: SolarEvent) -> float:
    """Rank used to pick which event in a crowded cluster gets the label.

    X-class flares sort above CMEs, and stronger flares above weaker ones
    (X8.1 > X1.0), so a major flare is never hidden behind a lesser event
    that merely happened to come first.
    """
    if ev.kind == "flare" and ev.label[:1].upper() == "X":
        try:
            return 100.0 + float(ev.label[1:])
        except ValueError:
            return 100.0
    return 0.0


def _add_event_marks(fig: go.Figure, events: Optional[List[SolarEvent]],
                     xref: str = "x") -> None:
    """Draw a hidden dotted vertical line for every solar event plus a
    decluttered set of top labels.

    Every event gets its own line, but a permanent per-event label does not
    scale: over a long run with many/clustered events (e.g. a multi-flare
    storm) the vertical labels overlap into an unreadable smear. So events
    that fall closer together than ``span/50`` are grouped, and only the most
    significant event in each group is labelled (with ``+N`` when it stands in
    for others). Shapes/annotations are added disabled (``visible=False``);
    the combined page wires the client-side toggle. ``events=None``/empty ->
    nothing drawn.
    """
    if not events:
        return
    evs = sorted(events, key=lambda e: e.utc)

    # A dotted line for every event.
    for ev in evs:
        color = _EVENT_COLORS.get(ev.kind, _MUTED_INK)
        fig.add_shape(type="line", xref=xref, yref="paper",
                      x0=_iso(ev.utc), x1=_iso(ev.utc), y0=0, y1=1,
                      name="event-" + ev.kind, visible=False,
                      line=dict(color=color, width=1, dash="dot"))

    # Group events closer together than span/50 so their labels don't collide.
    span_s = (evs[-1].utc - evs[0].utc) / np.timedelta64(1, "s")
    min_gap_s = span_s / 50.0 if span_s > 0 else 0.0
    clusters: List[List[SolarEvent]] = [[evs[0]]]
    for ev in evs[1:]:
        gap_s = (ev.utc - clusters[-1][-1].utc) / np.timedelta64(1, "s")
        if gap_s <= min_gap_s:
            clusters[-1].append(ev)
        else:
            clusters.append([ev])

    # Label each cluster once, on its most significant member.
    for cluster in clusters:
        rep = max(cluster, key=_event_significance)
        label = rep.label
        if len(cluster) > 1:
            label = "{0} +{1}".format(rep.label, len(cluster) - 1)
        color = _EVENT_COLORS.get(rep.kind, _MUTED_INK)
        fig.add_annotation(xref=xref, yref="paper", x=_iso(rep.utc), y=1.01,
                           text=label, name="event-" + rep.kind, visible=False,
                           showarrow=False, font=dict(color=color, size=10),
                           textangle=-90, yanchor="bottom")


def build_anomaly_payload(detectors: List[DetectorSeries]) -> List[dict]:
    """Per-detector JSON-able payload for client-side anomaly re-thresholding.

    Hz units throughout. ``adj_i = corrected_rate_hz / rate_hz`` (the net
    meteorological correction factor applied to bin i), falling back to 1.0
    where the raw rate is zero or non-finite. Non-finite rates serialize as
    ``None`` so the payload survives ``json.dumps`` untouched.
    """
    out = []
    for det in detectors:
        corrected = np.asarray(det.correction.corrected_rate_hz, dtype=float)
        raw = np.asarray(det.rs.rate_hz, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            adj = np.where((raw != 0) & np.isfinite(raw), corrected / raw, 1.0)
        mu = anomaly.baseline_mean(corrected)
        t = [str(np.datetime64(x, "us")) for x in det.rs.bin_mid_utc]
        rate = [None if not np.isfinite(v) else float(v) for v in corrected]
        good = [bool(np.isfinite(v)) for v in corrected]
        marginal = anomaly.marginal_distribution(
            corrected, det.rs.livetime_s, adj, mu)
        out.append({
            "name": det.name,
            "mu": (None if not np.isfinite(mu) else float(mu)),
            "t": t,
            "counts": [int(c) for c in det.rs.counts],
            "livetime": [float(x) for x in det.rs.livetime_s],
            "adj": [float(x) for x in adj],
            "rate": rate,
            "good": good,
            "marginal": marginal,
        })
    return out


def _anomaly_traces_for_detector(det: DetectorSeries, d_index: int, color: str,
                                 units: str, mu: float) -> List[go.Scatter]:
    """Four tagged hidden Scatter traces (mean, lower, upper, outlier).

    Default two-sided exact-Poisson thresholds at p=0.05. ``units`` selects
    the view: ``"pct"`` renders percent deviation from the run mean (overlay),
    ``"hz"`` renders native Hz (side-by-side).
    """
    x = det.rs.bin_mid_utc
    corrected = np.asarray(det.correction.corrected_rate_hz, dtype=float)
    raw = np.asarray(det.rs.rate_hz, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        adj = np.where((raw != 0) & np.isfinite(raw), corrected / raw, 1.0)
    T = np.asarray(det.rs.livetime_s, dtype=float)
    counts = np.asarray(det.rs.counts)

    def to_view(v):  # Hz value -> view units
        if units == "pct":
            return 100.0 * (v - mu) / mu
        return v

    lower = np.full(corrected.shape, np.nan)
    upper = np.full(corrected.shape, np.nan)
    k_lo = np.full(corrected.shape, -1)
    k_hi = np.full(corrected.shape, np.iinfo(np.int64).max)
    goodmask = np.isfinite(corrected) & np.isfinite(T) & (T > 0)
    for i in np.where(goodmask)[0]:
        lam_i = anomaly.lambda_per_bin(mu, T[i], adj[i])
        if not np.isfinite(lam_i) or lam_i <= 0:
            continue
        klo, khi = anomaly.threshold_counts(float(lam_i), 0.05)
        k_lo[i], k_hi[i] = klo, khi
        lower[i] = to_view(adj[i] * klo / T[i])
        upper[i] = to_view(adj[i] * khi / T[i])
    flags = anomaly.flag_bins(counts, k_lo, k_hi) & goodmask

    mean_y = to_view(mu)
    x0, x1 = x[0], x[-1]

    def trace_meta(role):
        return {"anomaly": role, "det": d_index}

    return [
        go.Scatter(x=[x0, x1], y=[mean_y, mean_y], mode="lines",
                  line=dict(color=color, width=3), visible=False,
                  name="{0} mean".format(det.name), meta=trace_meta("mean"),
                  hoverinfo="skip", showlegend=False),
        go.Scatter(x=x, y=lower, mode="lines",
                  line=dict(color=color, width=1, dash="dash"), visible=False,
                  name="{0} lower".format(det.name), meta=trace_meta("lower"),
                  hoverinfo="skip", showlegend=False),
        go.Scatter(x=x, y=upper, mode="lines",
                  line=dict(color=color, width=1, dash="dash"), visible=False,
                  name="{0} upper".format(det.name), meta=trace_meta("upper"),
                  hoverinfo="skip", showlegend=False),
        go.Scatter(x=[x[i] for i in np.where(flags)[0]],
                  y=[to_view(corrected[i]) for i in np.where(flags)[0]],
                  mode="markers",
                  marker=dict(color=_ALERT_COLOR, size=13, symbol="circle-open",
                              line=dict(width=2)),
                  visible=False, name="{0} flagged".format(det.name),
                  meta=trace_meta("outlier"), showlegend=False,
                  hovertemplate="flagged: %{x}<extra></extra>"),
    ]


#: Poisson-overlay ink for the y-axis marginal (theme-adjusted in a later task).
_MARGINAL_LINE = _SECONDARY_INK


def _detector_adj(det: DetectorSeries) -> np.ndarray:
    """Net meteorological correction factor per bin (corrected/raw, 1.0 fallback)."""
    corrected = np.asarray(det.correction.corrected_rate_hz, dtype=float)
    raw = np.asarray(det.rs.rate_hz, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((raw != 0) & np.isfinite(raw), corrected / raw, 1.0)


def _marginal_traces_for_detector(det: DetectorSeries, d_index: int, color: str,
                                  units: str, mu: float, xaxis_id: str) -> list:
    """Two tagged hidden traces for the y-axis marginal: a rotated observed-rate
    histogram (``go.Bar`` orientation="h") plus an exact-Poisson expected
    overlay (``go.Scatter`` line), both plotted against the muon-rate y-axis on
    a right-margin x-axis. Bucket midpoints are converted to the view units:
    ``"pct"`` -> percent deviation from the run mean (overlay), ``"hz"`` ->
    native Hz (side-by-side)."""
    adj = _detector_adj(det)
    m = anomaly.marginal_distribution(
        det.correction.corrected_rate_hz, det.rs.livetime_s, adj, mu)
    edges = np.asarray(m["edges"], dtype=float)
    observed = m["observed"]
    expected = m["expected"]
    if edges.size >= 2:
        mids = (edges[:-1] + edges[1:]) / 2.0
        mids_view = ((100.0 * (mids - mu) / mu) if units == "pct" else mids).tolist()
    else:
        mids_view = []

    def trace_meta(role):
        return {"anomaly": role, "det": d_index}

    return [
        go.Bar(orientation="h", y=mids_view, x=observed,
               marker=dict(color=color, opacity=0.45),
               xaxis=xaxis_id, yaxis="y", visible=False,
               name="{0} observed".format(det.name), meta=trace_meta("hist"),
               hoverinfo="skip", showlegend=False),
        go.Scatter(mode="lines", y=mids_view, x=expected,
                   line=dict(color=_MARGINAL_LINE, width=2),
                   xaxis=xaxis_id, yaxis="y", visible=False,
                   name="{0} poisson".format(det.name), meta=trace_meta("poisson"),
                   hoverinfo="skip", showlegend=False),
    ]


def build_overlay(
    detectors: List[DetectorSeries],
    aligned: List[AlignedSeries],
    meta: PlotMetadata,
    master_utc: np.ndarray,
    gaps: Optional[List[Tuple[np.datetime64, np.datetime64]]] = None,
    events: Optional[List[SolarEvent]] = None,
) -> go.Figure:
    """One shared time axis; each detector as percent deviation, external on
    the secondary axis (plotted at the shared master grid)."""
    fig = go.Figure()

    for d_index, det in enumerate(detectors):
        color = _detector_color(d_index)
        corrected = det.correction.corrected_rate_hz
        mean_rate = float(np.nanmean(corrected))
        rate_pct = 100.0 * (corrected - mean_rate) / mean_rate
        rate_err_pct = 100.0 * det.correction.corrected_err_hz / mean_rate
        fig.add_trace(go.Scatter(
            x=det.rs.bin_mid_utc, y=rate_pct,
            error_y=dict(type="data", array=rate_err_pct, visible=True,
                         thickness=1),
            name="{0} (corrected)".format(det.name),
            mode="lines+markers",
            line=dict(color=color, width=2), marker=dict(size=8),
            hovertemplate="{0}: %{{y:.2f}}%<extra></extra>".format(det.name),
        ))

    for e_index, series in enumerate(aligned):
        color = _external_color(e_index, len(detectors))
        if series.source in MODULATION_SOURCES:
            fig.add_trace(go.Scatter(
                x=master_utc, y=series.percent_deviation,
                name="{0} (% dev)".format(series.name), mode="lines",
                line=dict(color=color, width=2),
                text=_hover_text(series),
                hovertemplate="%{text}<extra></extra>",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=master_utc, y=series.values,
                name="{0} [{1}]".format(series.name, series.units),
                mode="lines", yaxis="y2",
                line=dict(color=color, width=2, dash="dot"),
                text=_hover_text(series),
                hovertemplate="%{text}<extra></extra>",
            ))

    for d_index, det in enumerate(detectors):
        mu = float(np.nanmean(det.correction.corrected_rate_hz))
        for tr in _anomaly_traces_for_detector(det, d_index,
                                               _detector_color(d_index), "pct", mu):
            fig.add_trace(tr)

    for d_index, det in enumerate(detectors):
        mu = float(np.nanmean(det.correction.corrected_rate_hz))
        for tr in _marginal_traces_for_detector(
                det, d_index, _detector_color(d_index), "pct", mu, "x3"):
            fig.add_trace(tr)

    # The secondary axis carries external series plotted in native units (all
    # non-modulation sources; NMDB is shown as % deviation on the primary
    # axis). Only add it when something actually uses it -- otherwise Plotly
    # draws an empty, auto-ranged axis whose SI-prefixed ticks ("15u" etc.)
    # clutter the plot and sit confusingly next to the y-axis marginal.
    has_secondary = any(s.source not in MODULATION_SOURCES for s in aligned)
    layout = dict(
        xaxis=dict(title="Time (UTC)", gridcolor=_GRIDLINE_COLOR,
                   linecolor=_MUTED_INK, domain=[0.0, 0.82]),
        yaxis=dict(title="Deviation from run mean (%)",
                   gridcolor=_GRIDLINE_COLOR, linecolor=_MUTED_INK),
        xaxis3=dict(domain=[0.85, 1.0], anchor="y", title="bins",
                    showgrid=False, linecolor=_MUTED_INK),
        legend=dict(orientation="h", yanchor="top", y=-0.14, x=0),
        barmode="overlay",
    )
    if has_secondary:
        layout["yaxis2"] = dict(title="Other indices (native units)",
                                overlaying="y", side="right", showgrid=False,
                                linecolor=_MUTED_INK)
    fig.update_layout(**layout)
    _add_gap_bands(fig, gaps)
    _add_event_marks(fig, events)
    _apply_common_layout(fig, meta, "Muon Rate vs. Solar Activity — Overlay",
                         plot_height=460, bottom_margin=120)
    if events is not None:
        # ``update_layout(meta=...)`` replaces the whole dict rather than
        # deep-merging it, so the header_title/header_meta stashed by
        # _apply_common_layout must be carried forward explicitly.
        fig.update_layout(meta=dict(fig.layout.meta or {}, eventsRequested=True))
    return fig


def build_side_by_side(
    detectors: List[DetectorSeries],
    aligned: List[AlignedSeries],
    meta: PlotMetadata,
    master_utc: np.ndarray,
    gaps: Optional[List[Tuple[np.datetime64, np.datetime64]]] = None,
    events: Optional[List[SolarEvent]] = None,
) -> go.Figure:
    """One rate panel (all detectors overlaid) plus one panel per external
    source, sharing a linked time axis."""
    n_rows = 1 + len(aligned)
    titles = ["Muon Rate (Corrected) [Hz]"] + [
        _titlecase("{0} [{1}]".format(s.name, s.units)) for s in aligned
    ]
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        vertical_spacing=min(0.06, 0.6 / max(n_rows, 1)),
                        subplot_titles=titles)

    for d_index, det in enumerate(detectors):
        fig.add_trace(go.Scatter(
            x=det.rs.bin_mid_utc, y=det.correction.corrected_rate_hz,
            error_y=dict(type="data", array=det.correction.corrected_err_hz,
                         visible=True, thickness=1),
            name="{0} (corrected)".format(det.name),
            mode="lines+markers",
            line=dict(color=_detector_color(d_index), width=2),
            marker=dict(size=8),
            hovertemplate="{0}: %{{y:.4f}} Hz<extra></extra>".format(det.name),
        ), row=1, col=1)

    for e_index, series in enumerate(aligned):
        log_scale = "W/m" in series.units
        fig.add_trace(go.Scatter(
            x=master_utc, y=series.values, name=series.name, mode="lines",
            line=dict(color=_external_color(e_index, len(detectors)), width=2),
            text=_hover_text(series),
            hovertemplate="%{text}<extra></extra>",
        ), row=e_index + 2, col=1)
        if log_scale:
            fig.update_yaxes(type="log", row=e_index + 2, col=1)

    for d_index, det in enumerate(detectors):
        mu = float(np.nanmean(det.correction.corrected_rate_hz))
        for tr in _anomaly_traces_for_detector(det, d_index,
                                               _detector_color(d_index), "hz", mu):
            fig.add_trace(tr, row=1, col=1)

    # Marginal traces share row-1's rate y-axis but ride a dedicated right-margin
    # x-axis (added AFTER the bulk update_xaxes below so its independent range /
    # domain survive the matches="x" + domain=[0,0.82] sweep over the time axes).
    marg_ref = "x{0}".format(n_rows + 1)
    for d_index, det in enumerate(detectors):
        mu = float(np.nanmean(det.correction.corrected_rate_hz))
        for tr in _marginal_traces_for_detector(
                det, d_index, _detector_color(d_index), "hz", mu, marg_ref):
            fig.add_trace(tr)

    fig.update_xaxes(title_text="Time (UTC)", row=n_rows, col=1)
    fig.update_xaxes(matches="x", gridcolor=_GRIDLINE_COLOR, linecolor=_MUTED_INK)
    # Shrink every stacked time panel to the same left strip so they stay
    # vertically aligned and zoom-synced with the marginal parked at the right.
    fig.update_xaxes(domain=[0.0, 0.82])
    fig.update_yaxes(gridcolor=_GRIDLINE_COLOR, linecolor=_MUTED_INK)
    fig.update_layout(**{
        "xaxis{0}".format(n_rows + 1): dict(
            domain=[0.85, 1.0], anchor="y", title="bins", showgrid=False,
            linecolor=_MUTED_INK),
    })
    fig.update_layout(showlegend=False, barmode="overlay")
    _add_gap_bands(fig, gaps, per_row=n_rows)
    _add_event_marks(fig, events, xref="x")
    _apply_common_layout(fig, meta,
                         "Muon Rate vs. Solar Activity — Aligned Panels",
                         plot_height=max(240 * n_rows, 480))
    if events is not None:
        fig.update_layout(meta=dict(fig.layout.meta or {}, eventsRequested=True))
    return fig


_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light; }}
  html[data-theme="dark"] {{ color-scheme: dark; }}
  body {{
    margin: 0; padding: 16px;
    font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif;
    background: var(--surface); color: var(--text);
  }}
  html[data-theme="light"] body {{ --surface:#fcfcfb; --text:#0b0b0b;
    --secondary:#52514e; --panel:#f2f1ec; --border:#d8d7d0; }}
  html[data-theme="dark"] body {{ --surface:#1a1a19; --text:#ffffff;
    --secondary:#c3c2b7; --panel:#232322; --border:#383835; }}
  .controls {{
    display: flex; flex-wrap: wrap; gap: 18px; align-items: flex-start;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 12px 14px; margin-bottom: 14px;
  }}
  .controls fieldset {{ border: 0; margin: 0; padding: 0; }}
  .controls legend, .ctl-label {{
    font-size: 12px; font-weight: 600; color: var(--secondary);
    text-transform: uppercase; letter-spacing: .04em;
    padding: 0; margin-bottom: 6px; display: block;
  }}
  .series-list {{ display: flex; flex-wrap: wrap; gap: 4px 16px; }}
  .series-list label {{ display: flex; align-items: center; gap: 6px;
    white-space: nowrap; cursor: pointer; }}
  .swatch {{ width: 11px; height: 11px; border-radius: 2px; flex: none; }}
  select, button {{
    font: inherit; padding: 5px 9px; border-radius: 6px;
    border: 1px solid var(--border); background: var(--surface);
    color: var(--text); cursor: pointer;
  }}
  #viz-plot {{ background: var(--surface); }}
  .run-header {{ margin: 8px 4px 10px; color: var(--secondary); }}
  .run-header > summary {{ cursor: pointer; font-size: 17px; font-weight: 700;
    color: var(--text); margin-bottom: 6px; list-style-position: outside; }}
  .run-header .meta {{ font-size: 12px; line-height: 1.5; }}
</style>
</head>
<body>
<div class="controls">
  <div>
    <span class="ctl-label">Appearance</span>
    <button id="theme-toggle" type="button">Switch to dark mode</button>
  </div>
  <div>
    <span class="ctl-label">Muon rate line spec</span>
    <select id="line-spec">{line_spec_options}</select>
  </div>
  <fieldset>
    <legend>{series_legend}</legend>
    <div class="series-list" id="series-list">{series_checkboxes}</div>
  </fieldset>
</div>
<details class="run-header" open>
  <summary>{header_title}</summary>
  <div class="meta">{header_meta}</div>
</details>
{plot_div}
<script>
(function () {{
  var CFG = {config_json};
  var gd = document.getElementById("viz-plot");
  var root = document.documentElement;

  function axisKey(prefix, row) {{ return row === 1 ? prefix : prefix + row; }}

  /* ---- fit the figure to the viewport ------------------------------- */
  function chromeHeight() {{
    var h = 24;  /* body padding top+bottom */
    ["controls", "run-header"].forEach(function (cls) {{
      var el = document.querySelector("." + cls);
      if (el) {{ h += el.getBoundingClientRect().height; }}
    }});
    return h;
  }}

  function visiblePanelCount() {{
    if (!CFG.stacked) {{ return 1; }}
    var boxes = document.querySelectorAll("#series-list input[type=checkbox]");
    var n = 0;
    boxes.forEach(function (b) {{ if (b.checked) {{ n++; }} }});
    return Math.max(n, 1);
  }}

  /* Height that keeps every visible panel on screen: fill whatever the
     viewport leaves after the controls and header, but never shrink a panel
     below a readable floor (then the page scrolls instead of cropping).
     The overlay is a single chart, so it simply fills the viewport -- which
     is what makes collapsing the header visibly enlarge it. */
  function fitHeight() {{
    var k = visiblePanelCount();
    var avail = window.innerHeight - chromeHeight();
    /* Floor keeps panels readable (stacked) or keeps the overlay tall enough
       that its bottom legend clears the x-axis title (non-stacked). */
    var floorPlot = CFG.stacked ? CFG.minPanelPx * k : CFG.minOverlayPlot;
    var floor = CFG.topMargin + floorPlot + CFG.bottomMargin;
    var ideal = CFG.stacked
      ? CFG.topMargin + CFG.perPanelPx * k + CFG.bottomMargin
      : avail;
    var target = Math.max(Math.min(ideal, avail), floor);
    Plotly.relayout(gd, {{ height: target }});
  }}

  window.addEventListener("resize", fitHeight);
  var hdr = document.querySelector(".run-header");
  if (hdr) {{ hdr.addEventListener("toggle", fitHeight); }}

  /* ---- theme -------------------------------------------------------- */
  function applyTheme(mode) {{
    var t = CFG.themes[mode];
    root.setAttribute("data-theme", mode);
    document.getElementById("theme-toggle").textContent =
      mode === "dark" ? "Switch to light mode" : "Switch to dark mode";

    var lay = {{
      "paper_bgcolor": t.surface, "plot_bgcolor": t.surface,
      "font.color": t.text
    }};
    for (var r = 1; r <= CFG.nRows; r++) {{
      lay[axisKey("xaxis", r) + ".gridcolor"] = t.grid;
      lay[axisKey("xaxis", r) + ".linecolor"] = t.axis;
      lay[axisKey("yaxis", r) + ".gridcolor"] = t.grid;
      lay[axisKey("yaxis", r) + ".linecolor"] = t.axis;
    }}
    if (CFG.hasSecondaryAxis) {{ lay["yaxis2.linecolor"] = t.axis; }}
    CFG.annotationRoles.forEach(function (role, i) {{
      lay["annotations[" + i + "].font.color"] =
        role === "muted" ? t.muted : (role === "header" ? t.secondary : t.text);
    }});
    Plotly.relayout(gd, lay);
    Plotly.restyle(gd, {{
      "line.color": CFG.traceColors[mode],
      "marker.color": CFG.traceColors[mode]
    }});
    document.querySelectorAll(".swatch").forEach(function (sw, i) {{
      sw.style.background = CFG.traceColors[mode][i];
    }});
  }}

  document.getElementById("theme-toggle").addEventListener("click", function () {{
    applyTheme(root.getAttribute("data-theme") === "dark" ? "light" : "dark");
  }});

  /* ---- series checkboxes -------------------------------------------- */
  function applyVisibility() {{
    var boxes = Array.prototype.slice.call(
      document.querySelectorAll("#series-list input[type=checkbox]"));
    var vis = boxes.map(function (b) {{ return b.checked ? true : "legendonly"; }});
    Plotly.restyle(gd, {{ visible: vis }}, vis.map(function (_, i) {{ return i; }}));

    if (!CFG.stacked) {{ fitHeight(); return; }}
    /* Collapse hidden panels so the remaining ones expand to fill. Row 1 is
       the shared rate panel: it stays visible if ANY detector box is
       checked, and collapses only when every detector box is unchecked. */
    var rowsShown = {{}};
    boxes.forEach(function (b, i) {{
      var row = CFG.rowOfTrace[i];
      if (b.checked) {{ rowsShown[row] = true; }}
      else if (!(row in rowsShown)) {{ rowsShown[row] = false; }}
    }});
    var shownRows = [], hiddenRows = [];
    for (var r = 1; r <= CFG.nRows; r++) {{
      if (rowsShown[r]) {{ shownRows.push(r); }} else {{ hiddenRows.push(r); }}
    }}
    if (!shownRows.length) {{ return; }}
    var gap = 0.06, k = shownRows.length;
    var h = (1 - gap * (k - 1)) / k;
    var lay = {{}};
    shownRows.forEach(function (row, i) {{
      var top = 1 - i * (h + gap);
      var isBottom = (i === k - 1);
      lay[axisKey("yaxis", row) + ".domain"] = [Math.max(top - h, 0), top];
      lay[axisKey("yaxis", row) + ".visible"] = true;
      lay[axisKey("xaxis", row) + ".visible"] = true;
      /* Tick labels live on the bottom panel only, so whichever panel ends
         up last must take over the time axis when others are deselected. */
      lay[axisKey("xaxis", row) + ".showticklabels"] = isBottom;
      lay[axisKey("xaxis", row) + ".title.text"] = isBottom ? "Time (UTC)" : "";
      lay["annotations[" + (row - 1) + "].y"] = Math.min(top + 0.012, 1);
      lay["annotations[" + (row - 1) + "].visible"] = true;
    }});
    hiddenRows.forEach(function (row) {{
      lay[axisKey("yaxis", row) + ".visible"] = false;
      lay[axisKey("xaxis", row) + ".visible"] = false;
      lay["annotations[" + (row - 1) + "].visible"] = false;
    }});
    Plotly.relayout(gd, lay).then(fitHeight);
  }}

  document.getElementById("series-list")
    .addEventListener("change", applyVisibility);

  /* size to the viewport once Plotly has laid out the initial figure */
  fitHeight();

  /* ---- muon-rate line spec ------------------------------------------ */
  document.getElementById("line-spec").addEventListener("change", function (e) {{
    var s = CFG.lineSpecs[e.target.value];
    Plotly.restyle(gd,
      {{ mode: s.mode, "line.dash": s.dash, "marker.size": s.size }},
      [CFG.muonTrace]);
  }});
}})();
</script>
</body>
</html>
"""


_COMBINED_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light; }
  html[data-theme="dark"] { color-scheme: dark; }
  body { margin: 0; padding: 16px;
    font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif;
    background: var(--surface); color: var(--text); }
  html[data-theme="light"] body { --surface:#fcfcfb; --text:#0b0b0b;
    --secondary:#52514e; --panel:#f2f1ec; --border:#d8d7d0; --accent:#2a78d6; }
  html[data-theme="dark"] body { --surface:#1a1a19; --text:#ffffff;
    --secondary:#c3c2b7; --panel:#232322; --border:#383835; --accent:#3987e5; }
  .controls { display:flex; flex-wrap:wrap; gap:18px; align-items:flex-start;
    background:var(--panel); border:1px solid var(--border); border-radius:8px;
    padding:12px 14px; margin-bottom:14px; }
  .controls fieldset { border:0; margin:0; padding:0; }
  .ctl-label, .controls legend { font-size:12px; font-weight:600;
    color:var(--secondary); text-transform:uppercase; letter-spacing:.04em;
    margin-bottom:6px; display:block; }
  .series-list { display:flex; flex-wrap:wrap; gap:4px 16px; }
  .series-list label { display:flex; align-items:center; gap:6px;
    white-space:nowrap; cursor:pointer; }
  .swatch { width:11px; height:11px; border-radius:2px; flex:none; }
  select, button, input[type=number] { font:inherit; padding:5px 9px;
    border-radius:6px; border:1px solid var(--border);
    background:var(--surface); color:var(--text); cursor:pointer; }
  input[type=number]:disabled { opacity:.5; cursor:not-allowed; }
  .muted { color:var(--secondary); font-size:12px; }
  .anomaly-ctl label { display:inline-flex; align-items:center; gap:6px;
    white-space:nowrap; cursor:pointer; }
  .events-ctl label { display:inline-flex; align-items:center; gap:6px;
    white-space:nowrap; cursor:pointer; }
  #events-note { margin-left:12px; }
  #flagged-list { margin-top:6px; max-width:560px; line-height:1.6; }
  .run-header { margin:8px 4px 10px; color:var(--secondary); }
  .run-header > summary { cursor:pointer; font-size:17px; font-weight:700;
    color:var(--text); margin-bottom:6px; }
  .run-header .meta { font-size:12px; line-height:1.5; }
  .tabbar { display:flex; gap:6px; margin:6px 4px 4px; border-bottom:1px solid var(--border); }
  .tabbar button { border:1px solid var(--border); border-bottom:none;
    border-radius:6px 6px 0 0; background:var(--panel); padding:7px 16px; }
  .tabbar button.active { background:var(--surface); color:var(--accent);
    font-weight:600; }
  .viz { background:var(--surface); }
</style>
</head>
<body>
<div class="controls">
  <div><span class="ctl-label">Appearance</span>
    <button id="theme-toggle" type="button">Switch to dark mode</button></div>
  <div><span class="ctl-label">Muon rate line spec</span>
    <select id="line-spec">__LINE_SPEC_OPTIONS__</select></div>
  <fieldset><legend>Series</legend>
    <div class="series-list" id="series-list">__SERIES_CHECKBOXES__</div></fieldset>
  <div class="anomaly-ctl"><span class="ctl-label">Anomaly detection</span>
    <label><input type="checkbox" id="anomaly-toggle"> Flag outliers (Poisson)</label>
    <label style="margin-left:12px">p =
      <input type="number" id="anomaly-p" value="0.05" min="0.0001" max="0.9999"
        step="0.01" style="width:5em" disabled></label>
    <span id="anomaly-readout" class="muted" style="margin-left:12px"></span>
    <div id="flagged-list" class="muted"></div></div>
  __EVENTS_CONTROL__
</div>
<details class="run-header" open>
  <summary>__HEADER_TITLE__</summary>
  <div class="meta">__HEADER_META__</div>
</details>
<div class="tabbar">
  <button id="tab-overlay" class="active" type="button">Overlay</button>
  <button id="tab-side" type="button">Side-by-side</button>
</div>
<div id="viz-overlay" class="viz">__PLOT_OVERLAY__</div>
<div id="viz-side" class="viz" style="display:none">__PLOT_SIDE__</div>
__NUMERICS_JS__
<script>
__COMBINED_JS__
</script>
</body>
</html>
"""

_COMBINED_JS = """
(function () {
  var CFG = __CONFIG__;   /* {views:{overlay,side}, themes, lineSpecs} */
  var GD = { overlay: document.getElementById("viz-overlay").querySelector(".plotly-graph-div"),
             side: document.getElementById("viz-side").querySelector(".plotly-graph-div") };
  var VIEWS = ["overlay", "side"];
  var active = "overlay";
  var mode = "light";
  var root = document.documentElement;

  function axisKey(prefix, row) { return row === 1 ? prefix : prefix + row; }

  function checkboxes() {
    return Array.prototype.slice.call(
      document.querySelectorAll("#series-list input[type=checkbox]"));
  }

  /* rows (panels) currently shown in the stacked side view: row is shown if
     ANY checkbox mapping to it is checked. */
  function shownRows(cfg, boxes) {
    var seen = {};
    boxes.forEach(function (b, i) {
      var row = cfg.rowOfTrace[i];
      if (b.checked) { seen[row] = true; }
      else if (!(row in seen)) { seen[row] = false; }
    });
    var rows = [];
    for (var r = 1; r <= cfg.nRows; r++) { if (seen[r]) { rows.push(r); } }
    return rows;
  }

  /* ---- height ------------------------------------------------------- */
  function chromeHeight() {
    var h = 24;
    ["controls", "run-header", "tabbar"].forEach(function (cls) {
      var el = document.querySelector("." + cls);
      if (el) { h += el.getBoundingClientRect().height; }
    });
    return h;
  }

  function fitActive() {
    var cfg = CFG.views[active], gd = GD[active];
    var avail = window.innerHeight - chromeHeight();
    var k = cfg.stacked ? Math.max(shownRows(cfg, checkboxes()).length, 1) : 1;
    var floorPlot = cfg.stacked ? cfg.minPanelPx * k : cfg.minOverlayPlot;
    var floor = cfg.topMargin + floorPlot + cfg.bottomMargin;
    var ideal = cfg.stacked
      ? cfg.topMargin + cfg.perPanelPx * k + cfg.bottomMargin
      : avail;
    var target = Math.max(Math.min(ideal, avail), floor);
    Plotly.relayout(gd, { height: target });
  }

  window.addEventListener("resize", fitActive);
  var hdr = document.querySelector(".run-header");
  if (hdr) { hdr.addEventListener("toggle", fitActive); }

  /* ---- theme (both views) ------------------------------------------- */
  function applyThemeToView(view, t) {
    var cfg = CFG.views[view], gd = GD[view];
    var lay = { "paper_bgcolor": t.surface, "plot_bgcolor": t.surface,
                "font.color": t.text };
    for (var r = 1; r <= cfg.nRows; r++) {
      lay[axisKey("xaxis", r) + ".gridcolor"] = t.grid;
      lay[axisKey("xaxis", r) + ".linecolor"] = t.axis;
      lay[axisKey("yaxis", r) + ".gridcolor"] = t.grid;
      lay[axisKey("yaxis", r) + ".linecolor"] = t.axis;
    }
    if (cfg.hasSecondaryAxis) { lay["yaxis2.linecolor"] = t.axis; }
    cfg.annotationRoles.forEach(function (role, i) {
      if (role === "event") { return; }  /* event marks recolored below */
      lay["annotations[" + i + "].font.color"] =
        role === "muted" ? t.muted : (role === "header" ? t.secondary : t.text);
    });
    /* Event marks (Task 5): recolor per kind for the current theme WITHOUT
       touching their visibility -- only the checkbox drives that. */
    var ec = (CFG.eventColors || {})[mode] || {};
    (cfg.eventShapes || []).forEach(function (i, k) {
      var c = ec[cfg.eventShapeKinds[k]];
      if (c) { lay["shapes[" + i + "].line.color"] = c; }
    });
    (cfg.eventAnnotations || []).forEach(function (j, k) {
      var c = ec[cfg.eventAnnotationKinds[k]];
      if (c) { lay["annotations[" + j + "].font.color"] = c; }
    });
    Plotly.relayout(gd, lay);
    Plotly.restyle(gd, { "line.color": cfg.traceColors[mode],
                         "marker.color": cfg.traceColors[mode] });
  }

  function applyTheme(next) {
    mode = next;
    var t = CFG.themes[mode];
    root.setAttribute("data-theme", mode);
    document.getElementById("theme-toggle").textContent =
      mode === "dark" ? "Switch to light mode" : "Switch to dark mode";
    VIEWS.forEach(function (v) { applyThemeToView(v, t); });
    var colors = CFG.views.overlay.traceColors[mode];
    document.querySelectorAll(".swatch").forEach(function (sw, i) {
      sw.style.background = colors[i];
    });
  }

  document.getElementById("theme-toggle").addEventListener("click", function () {
    applyTheme(mode === "dark" ? "light" : "dark");
    applyAnomaly();  /* re-apply alert/ink colours the theme restyle overwrote */
  });

  /* ---- series checkboxes (both views) ------------------------------- */
  function collapseStacked(cfg, gd, boxes) {
    var rows = shownRows(cfg, boxes);
    if (!rows.length) { return Promise.resolve(); }
    var gap = 0.06, k = rows.length, h = (1 - gap * (k - 1)) / k, lay = {};
    var hidden = [];
    for (var r = 1; r <= cfg.nRows; r++) {
      if (rows.indexOf(r) === -1) { hidden.push(r); }
    }
    rows.forEach(function (row, i) {
      var top = 1 - i * (h + gap), isBottom = (i === k - 1);
      lay[axisKey("yaxis", row) + ".domain"] = [Math.max(top - h, 0), top];
      lay[axisKey("yaxis", row) + ".visible"] = true;
      lay[axisKey("xaxis", row) + ".visible"] = true;
      lay[axisKey("xaxis", row) + ".showticklabels"] = isBottom;
      lay[axisKey("xaxis", row) + ".title.text"] = isBottom ? "Time (UTC)" : "";
      lay["annotations[" + (row - 1) + "].y"] = Math.min(top + 0.012, 1);
      lay["annotations[" + (row - 1) + "].visible"] = true;
    });
    hidden.forEach(function (row) {
      lay[axisKey("yaxis", row) + ".visible"] = false;
      lay[axisKey("xaxis", row) + ".visible"] = false;
      lay["annotations[" + (row - 1) + "].visible"] = false;
    });
    return Plotly.relayout(gd, lay);
  }

  function applyVisibility() {
    var boxes = checkboxes();
    var vis = boxes.map(function (b) { return b.checked ? true : "legendonly"; });
    var idx = vis.map(function (_, i) { return i; });
    Plotly.restyle(GD.overlay, { visible: vis }, idx);
    Plotly.restyle(GD.side, { visible: vis }, idx);
    collapseStacked(CFG.views.side, GD.side, boxes).then(fitActive);
  }

  document.getElementById("series-list")
    .addEventListener("change", function () { applyVisibility(); applyAnomaly(); });

  /* ---- muon-rate line spec (detector traces, both views) ------------ */
  document.getElementById("line-spec").addEventListener("change", function (e) {
    var s = CFG.lineSpecs[e.target.value];
    VIEWS.forEach(function (v) {
      var nd = CFG.views[v].nDetectors, idx = [];
      for (var i = 0; i < nd; i++) { idx.push(i); }
      Plotly.restyle(GD[v],
        { mode: s.mode, "line.dash": s.dash, "marker.size": s.size }, idx);
    });
  });

  /* ---- tabs --------------------------------------------------------- */
  function activate(view) {
    active = view;
    VIEWS.forEach(function (v) {
      document.getElementById("viz-" + v).style.display = (v === view) ? "" : "none";
      document.getElementById("tab-" + v).classList.toggle("active", v === view);
    });
    /* Plotly bakes a fallback fixed width (commonly 700px) into a figure
       that was newPlot'd while its container had display:none (getBoundingClientRect
       reports 0 there), and Plotly.Plots.resize alone does not override an
       explicit layout.width once set. Re-enabling autosize forces Plotly to
       re-measure the now-visible container before resize/fit run. */
    Plotly.relayout(GD[view], { autosize: true });
    Plotly.Plots.resize(GD[view]);
    fitActive();
    applyAnomaly();  /* keep the newly-shown view's anomaly overlay correct */
    applyEvents();   /* freshly-shown tab reflects the current toggle state */
  }
  document.getElementById("tab-overlay").addEventListener("click",
    function () { activate("overlay"); });
  document.getElementById("tab-side").addEventListener("click",
    function () { activate("side"); });

  /* ---- solar events (both views) ------------------------------------ */
  /* Flip every event-* shape/annotation to the checkbox state in both views
     via relayout. Guarded so a view with no event marks is a no-op, and so
     the whole feature is inert when the checkbox is absent. */
  function applyEvents() {
    var box = document.getElementById("events-toggle");
    if (!box) { return; }
    var on = box.checked;
    VIEWS.forEach(function (v) {
      var gd = GD[v], vw = CFG.views[v], lay = {};
      (vw.eventShapes || []).forEach(function (i) {
        lay["shapes[" + i + "].visible"] = on;
      });
      (vw.eventAnnotations || []).forEach(function (j) {
        lay["annotations[" + j + "].visible"] = on;
      });
      if (Object.keys(lay).length) { Plotly.relayout(gd, lay); }
    });
  }

  var _evBox = document.getElementById("events-toggle");
  if (_evBox) { _evBox.addEventListener("change", applyEvents); }

  /* ---- anomaly detection (exact-Poisson, both views) ---------------- */
  function anomActive() {
    return document.getElementById("anomaly-toggle").checked;
  }
  function anomP() {
    return Math.min(0.9999, Math.max(1e-4,
      parseFloat(document.getElementById("anomaly-p").value) || 0.05));
  }
  function toView(v, mu, units) {
    return units === "pct" ? 100 * (v - mu) / mu : v;
  }

  /* Per-detector exact-Poisson thresholds + flagged bin indices. kLo/kHi
     hold null for bins that are skipped (bad data / non-finite lambda) so
     the caller can leave those threshold-line points blank. */
  function computeDetector(d, p) {
    var kLo = [], kHi = [], flag = [];
    for (var i = 0; i < d.counts.length; i++) {
      if (!d.good[i]) { kLo.push(null); kHi.push(null); continue; }
      var lam = d.mu * d.livetime[i] / d.adj[i];
      if (!isFinite(lam) || lam <= 0) { kLo.push(null); kHi.push(null); continue; }
      var t = window.__anom.poissonThresholds(lam, p);
      kLo.push(t.kLo); kHi.push(t.kHi);
      if (d.counts[i] <= t.kLo || d.counts[i] >= t.kHi) { flag.push(i); }
    }
    return { kLo: kLo, kHi: kHi, flag: flag };
  }

  function applyAnomaly() {
    var on = anomActive(), p = anomP();
    document.getElementById("anomaly-p").disabled = !on;
    document.getElementById("anomaly-readout").textContent = on
      ? ("p = " + p + " \\u2192 " + (100 * p / 2).toFixed(2) + "% per tail") : "";

    var alertCol = (mode === "dark") ? "#e66767" : "#c0392b";
    var inkCol = (mode === "dark") ? "#c3c2b7" : "#52514e";
    var boxes = checkboxes();
    var dets = CFG.detectorsAnomaly || [];
    var comp = dets.map(function (d) { return computeDetector(d, p); });

    VIEWS.forEach(function (vn) {
      var gd = GD[vn], view = CFG.views[vn], units = view.units;
      var aByDet = view.anomalyByDetector || [];
      var mByDet = view.marginalByDetector || [];
      dets.forEach(function (d, di) {
        var idx = aByDet[di], midx = mByDet[di];
        if (!idx || !midx) { return; }
        var aidx = [idx.mean, idx.lower, idx.upper, idx.outlier,
                    midx.hist, midx.poisson];
        var vis = on && boxes[di] && boxes[di].checked;
        if (!vis) { Plotly.restyle(gd, { visible: false }, aidx); return; }

        var c = comp[di], mu = d.mu, lower = [], upper = [];
        for (var i = 0; i < d.livetime.length; i++) {
          if (c.kLo[i] === null) { lower.push(null); upper.push(null); }
          else {
            lower.push(toView(d.adj[i] * c.kLo[i] / d.livetime[i], mu, units));
            upper.push(toView(d.adj[i] * c.kHi[i] / d.livetime[i], mu, units));
          }
        }
        Plotly.restyle(gd, { y: [lower] }, [idx.lower]);
        Plotly.restyle(gd, { y: [upper] }, [idx.upper]);
        Plotly.restyle(gd, {
          x: [c.flag.map(function (i) { return d.t[i]; })],
          y: [c.flag.map(function (i) { return toView(d.rate[i], mu, units); })]
        }, [idx.outlier]);

        var col = view.traceColors[mode][di];
        Plotly.restyle(gd, { "line.color": col },
          [idx.mean, idx.lower, idx.upper]);
        Plotly.restyle(gd,
          { "marker.color": alertCol, "marker.line.color": alertCol },
          [idx.outlier]);
        Plotly.restyle(gd, { "marker.color": col }, [midx.hist]);
        Plotly.restyle(gd, { "line.color": inkCol }, [midx.poisson]);
        Plotly.restyle(gd, { visible: true }, aidx);
      });
    });

    function escHtml(s) {
      return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }
    function fmtT(s) { return String(s).slice(0, 16).replace("T", " ") + " UTC"; }
    var listHtml = "";
    if (on) {
      dets.forEach(function (d, di) {
        var c = comp[di];
        listHtml += "<b>" + escHtml(d.name) + "</b>: " + c.flag.length
          + " flagged";
        if (c.flag.length) {
          listHtml += " \\u2014 " + c.flag.slice(0, 20).map(function (i) {
            return fmtT(d.t[i]); }).join(", ");
          if (c.flag.length > 20) { listHtml += ", \\u2026"; }
        }
        listHtml += "<br>";
      });
    }
    document.getElementById("flagged-list").innerHTML = listHtml;
  }

  document.getElementById("anomaly-toggle")
    .addEventListener("change", applyAnomaly);
  var _anomTimer;
  document.getElementById("anomaly-p").addEventListener("input", function () {
    clearTimeout(_anomTimer);
    _anomTimer = setTimeout(applyAnomaly, 150);
  });

  /* initial sizing of the active (overlay) view */
  fitActive();
  applyAnomaly();  /* set control state + keep hidden traces hidden on load */
  applyEvents();   /* reflect the (default-off) checkbox on load */
})();
"""


def _is_anomaly(t) -> bool:
    """True for the hidden per-detector anomaly traces (mean/lower/upper/flagged)
    tagged with meta={"anomaly": role, "det": idx}, so callers can exclude them
    from anything keyed to the primary (detector+external) trace indices."""
    return isinstance(t.meta, dict) and "anomaly" in t.meta


def _trace_colors(fig: go.Figure) -> Tuple[List[str], List[str]]:
    """Light and dark colour for every trace, in trace order."""
    light, dark = [], []
    for trace in fig.data:
        color = None
        if getattr(trace, "line", None) is not None:
            color = trace.line.color
        if color is None:
            color = _MUON_COLOR
        light.append(color)
        dark.append(_LIGHT_TO_DARK.get(color, color))
    return light, dark


def _figure_config(fig: go.Figure, stacked: bool) -> dict:
    """Per-figure config consumed by the page JS (colours, row map, margins,
    axis/annotation metadata). Shared by write_html and write_combined_html."""
    light, dark = _trace_colors(fig)

    n_primary = sum(1 for t in fig.data if not _is_anomaly(t))
    n_detectors = sum(1 for t in fig.data
                      if (t.name or "").endswith("(corrected)"))
    if stacked:
        row_of_trace = [1] * n_detectors + list(
            range(2, n_primary - n_detectors + 2))
    else:
        row_of_trace = [1] * n_primary
    n_rows = (1 + (n_primary - n_detectors)) if stacked else 1
    annotation_roles = [
        "event" if (ann.name or "").startswith("event-")
        else "muted" if (ann.name == "gap-label" or ann.text == "no data")
        else "subplot"
        for ann in fig.layout.annotations
    ]

    # Event marks (Task 4): hidden dotted line + top label per solar event,
    # tagged name="event-<kind>". Collect their layout indices (and kinds, so
    # the page can theme-recolor each mark by kind) for the toggle/theme JS.
    event_shapes, event_shape_kinds = [], []
    for i, s in enumerate(fig.layout.shapes):
        if (s.name or "").startswith("event-"):
            event_shapes.append(i)
            event_shape_kinds.append(s.name[len("event-"):])
    event_annotations, event_annotation_kinds = [], []
    for j, ann in enumerate(fig.layout.annotations):
        if (ann.name or "").startswith("event-"):
            event_annotations.append(j)
            event_annotation_kinds.append(ann.name[len("event-"):])

    units = "hz" if stacked else "pct"
    anomaly_by_det = {}
    marginal_by_det = {}
    for idx, t in enumerate(fig.data):
        m = t.meta if isinstance(t.meta, dict) else {}
        role = m.get("anomaly")
        if role in ("mean", "lower", "upper", "outlier"):
            anomaly_by_det.setdefault(m["det"], {})[role] = idx
        elif role in ("hist", "poisson"):
            marginal_by_det.setdefault(m["det"], {})[role] = idx

    return {
        "traceColors": {"light": light, "dark": dark},
        "stacked": stacked,
        "nRows": n_rows,
        "nDetectors": n_detectors,
        "topMargin": int(fig.layout.margin.t or 90),
        "bottomMargin": int(fig.layout.margin.b or 90),
        "perPanelPx": 240,
        "minPanelPx": 150,
        "minOverlayPlot": 340,
        "rowOfTrace": row_of_trace,
        "muonTrace": 0,
        "hasSecondaryAxis":
            getattr(getattr(fig.layout, "yaxis2", None), "overlaying", None) == "y",
        "annotationRoles": annotation_roles,
        "units": units,
        "anomalyByDetector": [anomaly_by_det[d] for d in sorted(anomaly_by_det)],
        "marginalByDetector": [marginal_by_det[d] for d in sorted(marginal_by_det)],
        "eventShapes": event_shapes,
        "eventShapeKinds": event_shape_kinds,
        "eventAnnotations": event_annotations,
        "eventAnnotationKinds": event_annotation_kinds,
        "eventShapeCount": len(event_shapes),
    }


def write_html(fig: go.Figure, path: str, stacked: bool = False) -> None:
    """Write a self-contained interactive HTML page.

    The page wraps the Plotly figure in a control bar offering a light/dark
    theme toggle, a per-series checkbox list, and a matplotlib-style line-spec
    selector for the muon rate (dense runs are easier to read as markers only,
    where the connecting line would otherwise bury the trend).

    ``stacked`` marks the side-by-side figure, whose checkboxes additionally
    collapse a deselected panel so the remaining panels expand to fill.
    """
    names = [t.name or "Series {0}".format(i + 1)
             for i, t in enumerate(fig.data) if not _is_anomaly(t)]
    cfg = _figure_config(fig, stacked)
    light = cfg["traceColors"]["light"]

    fig_meta = fig.layout.meta or {}
    header_title = fig_meta.get("header_title", "Muon Rate vs. Solar Activity")
    header_meta = fig_meta.get("header_meta", "")

    config = dict(cfg)
    config["themes"] = _THEMES
    config["lineSpecs"] = {
        key: {"mode": mode, "dash": dash, "size": size}
        for key, _label, mode, dash, size in _LINE_SPECS
    }

    checkboxes = "".join(
        '<label><input type="checkbox" checked data-i="{i}">'
        '<span class="swatch" style="background:{color}"></span>{name}</label>'.format(
            i=i, color=light[i], name=html.escape(names[i])
        )
        for i in range(len(names))
    )
    options = "".join(
        '<option value="{key}"{sel}>{label}</option>'.format(
            key=html.escape(key), label=html.escape(label),
            sel=" selected" if key == "o-" else "",
        )
        for key, label, _mode, _dash, _size in _LINE_SPECS
    )

    plot_div = fig.to_html(
        full_html=False, include_plotlyjs=True, div_id="viz-plot",
        config={"scrollZoom": True, "displaylogo": False},
    )

    page = _PAGE_TEMPLATE.format(
        title=html.escape(header_title),
        header_title=html.escape(header_title),
        header_meta=header_meta,
        line_spec_options=options,
        series_legend="Panels" if stacked else "Lines",
        series_checkboxes=checkboxes,
        plot_div=plot_div,
        config_json=json.dumps(config),
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(page)


def _anomaly_js_numerics():
    """Self-contained JS exact-Poisson numerics, mirroring anomaly.py.

    Exposes ``window.__anom.poissonCdf(k, lam)`` and
    ``poissonThresholds(lam, p) -> {kLo, kHi}`` so the interactive anomaly
    control can recompute exact-Poisson thresholds in the browser as ``p``
    changes. ``gser``/``gcf`` use the same adaptive iteration cap and
    non-convergence guard as anomaly._gser/_gcf, so large lambda (e.g. daily
    binning, lambda ~ 43000) converges instead of silently under-summing.
    """
    return r"""
<script>
(function(){
  var EPS=3e-14, FPMIN=1e-300;
  function lgamma(x){
    var c=[76.18009172947146,-86.50532032941677,24.01409824083091,
      -1.231739572450155,0.1208650973866179e-2,-0.5395239384953e-5];
    var y=x, tmp=x+5.5; tmp-=(x+0.5)*Math.log(tmp); var ser=1.000000000190015;
    for(var j=0;j<6;j++){y+=1; ser+=c[j]/y;}
    return -tmp+Math.log(2.5066282746310005*ser/x);
  }
  function gser(a,x){ if(x<=0)return 0; var ap=a,sum=1/a,del=sum;
    var itmax=Math.max(1000, Math.floor(4*(a+x))), ok=false;
    for(var n=0;n<itmax;n++){ap+=1; del*=x/ap; sum+=del;
      if(Math.abs(del)<Math.abs(sum)*EPS){ok=true;break;}}
    if(!ok) throw new Error("gser failed to converge");
    return sum*Math.exp(-x+a*Math.log(x)-lgamma(a)); }
  function gcf(a,x){ var b=x+1-a,c=1/FPMIN,d=1/b,h=d;
    var itmax=Math.max(1000, Math.floor(4*(a+x))), ok=false;
    for(var i=1;i<itmax;i++){var an=-i*(i-a); b+=2; d=an*d+b;
      if(Math.abs(d)<FPMIN)d=FPMIN; c=b+an/c; if(Math.abs(c)<FPMIN)c=FPMIN;
      d=1/d; var del=d*c; h*=del; if(Math.abs(del-1)<EPS){ok=true;break;}}
    if(!ok) throw new Error("gcf failed to converge");
    return Math.exp(-x+a*Math.log(x)-lgamma(a))*h; }
  function gammq(a,x){ if(x<a+1) return 1-gser(a,x); return gcf(a,x); }
  function poissonCdf(k,lam){ if(k<0)return 0; if(lam<=0)return 1;
    return gammq(k+1,lam); }
  function poissonThresholds(lam,p){
    var half=p/2, spread=Math.floor(10*Math.sqrt(lam))+10;
    var hiB=Math.floor(lam)+spread, loB=Math.max(0,Math.floor(lam)-spread);
    var kLo, lo, hi, mid;
    if(poissonCdf(0,lam)>half){kLo=-1;}
    else{lo=0; hi=hiB; while(lo<hi){mid=(lo+hi+1)>>1;
      if(poissonCdf(mid,lam)<=half)lo=mid; else hi=mid-1;} kLo=lo;}
    lo=loB; hi=hiB; while(lo<hi){mid=(lo+hi)>>1;
      if(1-poissonCdf(mid-1,lam)<=half)hi=mid; else lo=mid+1;}
    return {kLo:kLo, kHi:lo};
  }
  window.__anom={poissonCdf:poissonCdf, poissonThresholds:poissonThresholds};
})();
</script>"""


def write_combined_html(
    overlay_fig: go.Figure, side_fig: go.Figure, path: str,
    anomaly: Optional[List[dict]] = None,
) -> None:
    """Write one self-contained page hosting both figures in tabs.

    Plotly's JS is inlined once (with the overlay); the side-by-side figure
    reuses it. A shared control bar drives both figures: a series checkbox
    toggles that series in both views (and collapses the side-by-side panel),
    the theme toggle recolours both, and the line-spec restyles the detector
    traces in both. Switching tabs resizes the newly-shown figure.

    ``anomaly`` is the ``build_anomaly_payload`` output (one dict per
    detector); it is embedded verbatim as ``CFG.detectorsAnomaly`` for the
    page JS to consume (client-side re-thresholding is wired in a later
    task). ``None`` embeds as an empty list.
    """
    names = [t.name or "Series {0}".format(i + 1)
             for i, t in enumerate(overlay_fig.data) if not _is_anomaly(t)]
    ov_cfg = _figure_config(overlay_fig, stacked=False)
    sb_cfg = _figure_config(side_fig, stacked=True)
    light = ov_cfg["traceColors"]["light"]

    fig_meta = overlay_fig.layout.meta or {}
    header_title = fig_meta.get("header_title", "Muon Rate vs. Solar Activity")
    header_meta = fig_meta.get("header_meta", "")

    # Solar events (Task 5): the checkbox appears when the overlay figure
    # carries event-* marks, or when events were requested but the window
    # held none (meta["eventsRequested"], set by the caller) so the
    # empty-window note can explain the absence. Only overlay_fig is checked:
    # both figures are built from the same single call site and always
    # receive identical events, so overlay is representative of side_fig too.
    has_events = any((s.name or "").startswith("event-")
                     for s in overlay_fig.layout.shapes)
    events_requested = bool(
        overlay_fig.layout.meta
        and overlay_fig.layout.meta.get("eventsRequested"))

    config = {
        "themes": _THEMES,
        "lineSpecs": {
            key: {"mode": mode, "dash": dash, "size": size}
            for key, _label, mode, dash, size in _LINE_SPECS
        },
        "views": {"overlay": ov_cfg, "side": sb_cfg},
        "detectorsAnomaly": anomaly or [],
        "hasEvents": has_events,
        "eventColors": {"light": _EVENT_COLORS, "dark": _EVENT_COLORS_DARK},
    }

    if has_events or events_requested:
        note = ("no CME/X-flare events in this window"
                if (events_requested and not has_events) else "")
        events_control = (
            '<div class="events-ctl"><span class="ctl-label">Solar events</span>'
            '<label><input type="checkbox" id="events-toggle"> '
            'Show events (CME / X-flare)</label>'
            '<span id="events-note" class="muted">{note}</span></div>'.format(
                note=html.escape(note))
        )
    else:
        events_control = ""

    checkboxes = "".join(
        '<label><input type="checkbox" checked data-i="{i}">'
        '<span class="swatch" style="background:{color}"></span>{name}</label>'.format(
            i=i, color=light[i], name=html.escape(names[i])
        )
        for i in range(len(names))
    )
    options = "".join(
        '<option value="{key}"{sel}>{label}</option>'.format(
            key=html.escape(key), label=html.escape(label),
            sel=" selected" if key == "o-" else "",
        )
        for key, label, _mode, _dash, _size in _LINE_SPECS
    )

    plot_overlay = overlay_fig.to_html(
        full_html=False, include_plotlyjs=True, div_id="plot-overlay",
        config={"scrollZoom": True, "displaylogo": False})
    plot_side = side_fig.to_html(
        full_html=False, include_plotlyjs=False, div_id="plot-side",
        config={"scrollZoom": True, "displaylogo": False})

    js = _COMBINED_JS.replace("__CONFIG__", json.dumps(config))
    page = (_COMBINED_TEMPLATE
            .replace("__TITLE__", html.escape(header_title))
            .replace("__HEADER_TITLE__", html.escape(header_title))
            .replace("__HEADER_META__", header_meta)
            .replace("__LINE_SPEC_OPTIONS__", options)
            .replace("__SERIES_CHECKBOXES__", checkboxes)
            .replace("__EVENTS_CONTROL__", events_control)
            .replace("__PLOT_OVERLAY__", plot_overlay)
            .replace("__PLOT_SIDE__", plot_side)
            .replace("__NUMERICS_JS__", _anomaly_js_numerics())
            .replace("__COMBINED_JS__", js))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(page)
