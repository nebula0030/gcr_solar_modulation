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

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from align import AlignedSeries
from correction import CorrectionResult
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

#: dataviz chart chrome tokens (light surface).
_GRIDLINE_COLOR = "#e1e0d9"
_MUTED_INK = "#898781"
_SECONDARY_INK = "#52514e"


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
    """Build the top-of-figure header block and count its lines.

    Everything the viewer needs to read the plot -- the title, the run's bin
    size and input parameters, the chosen station, the correction, and
    provenance notes -- is placed above the graphs in the top margin, so no
    text ever overlaps the plotting area.
    """
    lines = ['<span style="font-size:17px"><b>{0}</b></span>'.format(title)]
    lines.append("Run: {0}".format(meta.run_name))
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
    header_yshift: int = 8, bottom_margin: int = 90,
) -> None:
    """Attach the top header block and size the figure to hold it.

    The header lives entirely in the top margin, so the figure's total height
    is the header margin plus ``plot_height`` (the room the graphs themselves
    need) plus the bottom margin. This keeps the plotting area a fixed, legible
    size no matter how many header lines the run produces. ``header_yshift``
    lifts the block above any subplot titles at the top of the plotting area
    (needed for the stacked side-by-side figure).
    """
    text, n_lines = _header_text(meta, title)
    top_margin = 40 + 20 * n_lines + header_yshift
    fig.update_layout(
        hovermode="x unified",
        template="plotly_white",
        height=top_margin + plot_height + bottom_margin,
        margin=dict(l=70, r=70, t=top_margin, b=bottom_margin),
    )
    fig.add_annotation(
        text=text,
        xref="paper", yref="paper", x=0, y=1.0,
        xanchor="left", yanchor="bottom", yshift=header_yshift,
        showarrow=False, align="left",
        font=dict(size=12, color=_SECONDARY_INK),
    )


def build_overlay(
    rs: RateSeries,
    correction: CorrectionResult,
    aligned: List[AlignedSeries],
    meta: PlotMetadata,
) -> go.Figure:
    """One shared time axis; comparable series as percent deviation."""
    fig = go.Figure()

    mean_rate = float(np.nanmean(correction.corrected_rate_hz))
    rate_pct = 100.0 * (correction.corrected_rate_hz - mean_rate) / mean_rate
    rate_err_pct = 100.0 * correction.corrected_err_hz / mean_rate

    fig.add_trace(
        go.Scatter(
            x=rs.bin_mid_utc,
            y=rate_pct,
            error_y=dict(type="data", array=rate_err_pct, visible=True,
                         thickness=1),
            name="Muon rate (corrected)",
            mode="lines+markers",
            line=dict(color=_MUON_COLOR, width=2),
            marker=dict(size=8),
            hovertemplate="Muon rate: %{y:.2f}%<extra></extra>",
        )
    )

    color_index = 0
    for series in aligned:
        color = _SERIES_COLORS[color_index % len(_SERIES_COLORS)]
        color_index += 1
        if series.source in MODULATION_SOURCES:
            fig.add_trace(
                go.Scatter(
                    x=rs.bin_mid_utc,
                    y=series.percent_deviation,
                    name="{0} (% dev)".format(series.name),
                    mode="lines",
                    line=dict(color=color, width=2),
                    text=_hover_text(series),
                    hovertemplate="%{text}<extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=rs.bin_mid_utc,
                    y=series.values,
                    name="{0} [{1}]".format(series.name, series.units),
                    mode="lines",
                    yaxis="y2",
                    line=dict(color=color, width=2, dash="dot"),
                    text=_hover_text(series),
                    hovertemplate="%{text}<extra></extra>",
                )
            )

    fig.update_layout(
        xaxis=dict(
            title="Time (UTC)",
            gridcolor=_GRIDLINE_COLOR, linecolor=_MUTED_INK,
        ),
        yaxis=dict(
            title="Deviation from run mean (%)",
            gridcolor=_GRIDLINE_COLOR, linecolor=_MUTED_INK,
        ),
        yaxis2=dict(title="Other indices (native units)", overlaying="y",
                    side="right", showgrid=False, linecolor=_MUTED_INK),
        legend=dict(orientation="h", yanchor="top", y=-0.14, x=0),
    )
    _apply_common_layout(
        fig, meta, "Muon Rate vs. Solar Activity — Overlay",
        plot_height=460, bottom_margin=120,
    )
    return fig


def build_side_by_side(
    rs: RateSeries,
    correction: CorrectionResult,
    aligned: List[AlignedSeries],
    meta: PlotMetadata,
) -> go.Figure:
    """Stacked panels in native units with linked x-axes."""
    n_rows = 1 + len(aligned)
    titles = ["Muon Rate (Corrected) [Hz]"] + [
        _titlecase("{0} [{1}]".format(s.name, s.units)) for s in aligned
    ]
    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        vertical_spacing=min(0.06, 0.6 / max(n_rows, 1)),
        subplot_titles=titles,
    )

    fig.add_trace(
        go.Scatter(
            x=rs.bin_mid_utc,
            y=correction.corrected_rate_hz,
            error_y=dict(type="data", array=correction.corrected_err_hz,
                         visible=True, thickness=1),
            name="Muon rate (corrected)",
            mode="lines+markers",
            line=dict(color=_MUON_COLOR, width=2),
            marker=dict(size=8),
            hovertemplate="%{y:.4f} Hz<extra></extra>",
        ),
        row=1, col=1,
    )

    for index, series in enumerate(aligned):
        color = _SERIES_COLORS[index % len(_SERIES_COLORS)]
        log_scale = "W/m" in series.units
        fig.add_trace(
            go.Scatter(
                x=rs.bin_mid_utc,
                y=series.values,
                name=series.name,
                mode="lines",
                line=dict(color=color, width=2),
                text=_hover_text(series),
                hovertemplate="%{text}<extra></extra>",
            ),
            row=index + 2, col=1,
        )
        if log_scale:
            fig.update_yaxes(type="log", row=index + 2, col=1)

    fig.update_xaxes(title_text="Time (UTC)", row=n_rows, col=1)
    fig.update_xaxes(matches="x", gridcolor=_GRIDLINE_COLOR, linecolor=_MUTED_INK)
    fig.update_yaxes(gridcolor=_GRIDLINE_COLOR, linecolor=_MUTED_INK)
    fig.update_layout(showlegend=False)
    # Lift the header clear of the first panel's subplot title, and give each
    # stacked panel a fixed slice of height.
    _apply_common_layout(
        fig, meta, "Muon Rate vs. Solar Activity — Aligned Panels",
        plot_height=max(240 * n_rows, 480), header_yshift=30,
    )
    # make_subplots with shared_xaxes should set `matches`; the explicit
    # update_xaxes(matches="x") above guarantees it regardless of Plotly
    # version behaviour.
    return fig


def write_html(fig: go.Figure, path: str) -> None:
    """Write a self-contained interactive HTML file."""
    fig.write_html(
        path,
        include_plotlyjs=True,
        full_html=True,
        config={"scrollZoom": True, "displaylogo": False},
    )
