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
{plot_div}
<script>
(function () {{
  var CFG = {config_json};
  var gd = document.getElementById("viz-plot");
  var root = document.documentElement;

  function axisKey(prefix, row) {{ return row === 1 ? prefix : prefix + row; }}

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
        role === "header" ? t.secondary : t.text;
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
    Plotly.restyle(gd, {{ visible: vis }});

    if (!CFG.stacked) {{ return; }}
    /* Collapse hidden panels so the remaining ones expand to fill. */
    var shownRows = [], hiddenRows = [];
    boxes.forEach(function (b, i) {{
      (b.checked ? shownRows : hiddenRows).push(CFG.rowOfTrace[i]);
    }});
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
    Plotly.relayout(gd, lay);
  }}

  document.getElementById("series-list")
    .addEventListener("change", applyVisibility);

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


def write_html(fig: go.Figure, path: str, stacked: bool = False) -> None:
    """Write a self-contained interactive HTML page.

    The page wraps the Plotly figure in a control bar offering a light/dark
    theme toggle, a per-series checkbox list, and a matplotlib-style line-spec
    selector for the muon rate (dense runs are easier to read as markers only,
    where the connecting line would otherwise bury the trend).

    ``stacked`` marks the side-by-side figure, whose checkboxes additionally
    collapse a deselected panel so the remaining panels expand to fill.
    """
    names = [t.name or "Series {0}".format(i + 1) for i, t in enumerate(fig.data)]
    light, dark = _trace_colors(fig)

    n_rows = len(fig.data) if stacked else 1
    annotation_roles = ["subplot"] * (len(fig.layout.annotations) - 1) + ["header"]

    config = {
        "themes": _THEMES,
        "traceColors": {"light": light, "dark": dark},
        "stacked": stacked,
        "nRows": n_rows,
        "rowOfTrace": list(range(1, len(fig.data) + 1)) if stacked else [1] * len(fig.data),
        "muonTrace": 0,
        "hasSecondaryAxis": not stacked,
        "annotationRoles": annotation_roles,
        "lineSpecs": {
            key: {"mode": mode, "dash": dash, "size": size}
            for key, _label, mode, dash, size in _LINE_SPECS
        },
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
        title=html.escape("Muon Rate vs. Solar Activity"),
        line_spec_options=options,
        series_legend="Panels" if stacked else "Lines",
        series_checkboxes=checkboxes,
        plot_div=plot_div,
        config_json=json.dumps(config),
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(page)
