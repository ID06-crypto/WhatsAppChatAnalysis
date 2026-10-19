"""Plotly figures built from Chat.timeseries / Chat.aggregate output."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from whatsapp_analysis import METRICS

# Okabe-Ito: distinguishable for color-blind viewers; people keep their color across charts.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#999999"]


def person_colors(people: list[str]) -> dict[str, str]:
    return {p: PALETTE[i % len(PALETTE)] for i, p in enumerate(people)}


def _base(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title, template="plotly_white", hovermode="closest", height=480,
        margin=dict(l=40, r=20, t=60, b=40), legend=dict(orientation="h", y=1.1, x=0),
    )
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def timeseries_figure(ts: pd.DataFrame, metric: str, freq: str, colors: dict[str, str],
                      connect: bool = True, fill: bool = True, smooth: int = 0) -> go.Figure:
    """Per-person curves. `fill` shades under each curve; `smooth` > 1 also rounds the line.

    Smoothing itself (a rolling average) is done by the caller on `ts`; here it only
    switches the line to a spline. Very large series use WebGL, which has no spline or
    shading-friendly rendering, so they stay straight-edged.
    """
    big = ts.count().sum() > 20_000
    Trace = go.Scattergl if big else go.Scatter
    fig = go.Figure()
    for person in ts.columns:
        s = ts[person].dropna()
        color = colors.get(person)
        line = dict(width=2, color=color)
        if smooth > 1 and not big:
            line.update(shape="spline", smoothing=1.0)
        fig.add_trace(Trace(
            x=s.index, y=s.values, name=person, mode="lines+markers" if connect else "lines",
            marker=dict(size=5, color=color), line=line,
            fill="tozeroy" if fill else None, fillcolor=_rgba(color, 0.18) if fill and color else None,
            hovertemplate=f"<b>{person}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>",
        ))
    suffix = f" ({smooth}-period average)" if smooth > 1 else ""
    fig = _base(fig, f"{METRICS[metric].label} per {freq}{suffix}")
    fig.update_xaxes(title="Time", rangeslider_visible=True)
    fig.update_yaxes(title=METRICS[metric].label, rangemode="tozero" if metric != "sentiment" else "normal")
    if metric == "sentiment":
        fig.update_yaxes(range=[-1, 1], zeroline=True)
    return fig


def aggregate_figure(agg: pd.DataFrame, colors: dict[str, str], normalize: bool) -> go.Figure:
    """Grouped bars: x = metric, one bar per person."""
    fig = go.Figure()
    labels = [METRICS[m].label for m in agg.columns]
    for person in agg.index:
        fig.add_trace(go.Bar(
            x=labels, y=agg.loc[person].values, name=person, marker_color=colors.get(person),
            text=agg.loc[person].values, texttemplate="%{text:.1f}%" if normalize else "%{text:,.0f}",
            hovertemplate=f"<b>{person}</b><br>%{{x}}: %{{y:,.1f}}<extra></extra>",
        ))
    fig.update_layout(barmode="group")
    fig = _base(fig, "Share of each person's texts" if normalize else "Totals per person")
    fig.update_yaxes(title="% of their texts" if normalize else "Texts")
    return fig


def sentiment_bar_figure(agg: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=list(agg.index), y=agg["sentiment"].values, marker_color=[colors.get(p) for p in agg.index],
        text=agg["sentiment"].values, texttemplate="%{text:.2f}",
        hovertemplate="<b>%{x}</b><br>avg sentiment %{y:.3f}<extra></extra>",
    ))
    fig = _base(fig, "Average sentiment (-1 negative, +1 positive)")
    fig.update_layout(height=340)
    fig.update_yaxes(range=[-1, 1], zeroline=True)
    return fig
