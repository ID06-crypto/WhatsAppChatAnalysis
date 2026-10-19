"""Streamlit UI. Run with:  streamlit run app.py"""
from __future__ import annotations

from datetime import datetime, time

import pandas as pd
import streamlit as st

import charts
from project import Project
from whatsapp_analysis import FREQS, METRICS, Chat

st.set_page_config(page_title="WhatsApp Analysis", page_icon="💬", layout="wide")

PRESET_DAYS = {"All time": None, "Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90, "Last year": 365}
PRESETS = list(PRESET_DAYS) + ["Custom"]
VIEWS = ["Over time", "Compare people", "Saved graphs"]
ss = st.session_state


# --------------------------------------------------------------------------- helpers

@st.cache_resource(show_spinner="Reading chat and scoring sentiment…")
def load_chat(name: str) -> Chat:
    chat = Project(name).chat
    chat.features  # compute once up front (VADER on a long chat takes a few seconds)
    return chat


def resolve_window(w: dict, chat: Chat) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Window dict -> (start, end). A missing end means 'through the latest message'."""
    first, last = chat.bounds
    days = PRESET_DAYS.get(w["preset"])
    if w["preset"] != "Custom":
        return (last - pd.Timedelta(days=days) if days else None), None
    start = pd.Timestamp(w["start"]) if w.get("start") else None
    end = pd.Timestamp(w["end"]) if w.get("end") else None
    return start, end


def window_label(w: dict) -> str:
    if w["preset"] != "Custom":
        return w["preset"]
    return f"{(w.get('start') or 'beginning')[:16]} → {(w.get('end') or 'latest')[:16]}"


def window_controls(chat: Chat) -> dict:
    first, last = chat.bounds
    st.sidebar.subheader("Time window")
    preset = st.sidebar.selectbox("Range", PRESETS, key="preset", label_visibility="collapsed")
    if preset != "Custom":
        return {"preset": preset}

    d0, d1 = first.date(), last.date()
    ss.setdefault("start_d", d0)
    ss.setdefault("start_t", time(0, 0))
    ss.setdefault("end_open", True)
    ss.setdefault("end_d", d1)
    ss.setdefault("end_t", time(23, 59))
    c1, c2 = st.sidebar.columns([3, 2])
    sd = c1.date_input("Start date", min_value=d0, max_value=d1, key="start_d")
    stime = c2.time_input("Start time", key="start_t")
    start = datetime.combine(sd, stime)

    # Leaving the end open means "through the most recent message".
    end_open = st.sidebar.checkbox("End at the latest message", key="end_open")
    end = None
    if not end_open:
        c3, c4 = st.sidebar.columns([3, 2])
        ed = c3.date_input("End date", min_value=d0, max_value=d1, key="end_d")
        etime = c4.time_input("End time", key="end_t")
        end = datetime.combine(ed, etime)
        if end < start:
            st.sidebar.error("End is before start.")
    return {"preset": "Custom", "start": start.isoformat(), "end": end.isoformat() if end else None}


def graph_figure(chat: Chat, recipe: dict, colors: dict[str, str]):
    """Rebuild a figure from a recipe; the same path serves live views and saved graphs."""
    start, end = resolve_window(recipe["window"], chat)
    people = recipe["people"] or None
    if recipe["kind"] == "timeseries":
        ts = chat.timeseries(recipe["metric"], recipe["freq"], people, start, end)
        if ts.empty:
            return None, ts
        return charts.timeseries_figure(ts, recipe["metric"], recipe["freq"], colors, recipe.get("connect", True)), ts
    metrics = recipe["metrics"]
    agg = chat.aggregate(metrics, people, start, end, normalize=recipe["normalize"])
    return agg, agg


def show_recipe(chat: Chat, recipe: dict, colors: dict[str, str], key: str) -> None:
    result, data = graph_figure(chat, recipe, colors)
    if result is None or data.empty:
        st.info("No messages in this window.")
        return
    if recipe["kind"] == "timeseries":
        st.plotly_chart(result, width="stretch", key=f"{key}-ts")
        return
    agg = result
    bars = agg.drop(columns=["sentiment"], errors="ignore")
    if not bars.empty and len(bars.columns):
        st.plotly_chart(charts.aggregate_figure(bars, colors, recipe["normalize"]),
                        width="stretch", key=f"{key}-agg")
    if "sentiment" in agg:
        st.plotly_chart(charts.sentiment_bar_figure(agg, colors), width="stretch", key=f"{key}-sent")


def load_recipe(r: dict) -> None:
    """Button callback: push a saved recipe into the widgets (runs before widgets are drawn)."""
    w = r["window"]
    ss.preset = w["preset"]
    if w["preset"] == "Custom":
        if w.get("start"):
            ss.start_d = datetime.fromisoformat(w["start"]).date()
            ss.start_t = datetime.fromisoformat(w["start"]).time()
        ss.end_open = w.get("end") is None
        if w.get("end"):
            ss.end_d = datetime.fromisoformat(w["end"]).date()
            ss.end_t = datetime.fromisoformat(w["end"]).time()
    ss.people = r["people"]
    if r["kind"] == "timeseries":
        ss.metric, ss.freq, ss.connect = r["metric"], r["freq"], r.get("connect", True)
        ss.view = "Over time"
    else:
        ss.agg_metrics, ss.normalize = r["metrics"], r["normalize"]
        ss.view = "Compare people"


def save_box(project: Project, recipe: dict, key: str) -> None:
    c1, c2 = st.columns([3, 1])
    name = c1.text_input("Graph name", key=f"name-{key}", placeholder="e.g. Weekly texts, last 90 days",
                         label_visibility="collapsed")
    if c2.button("💾 Save graph", key=f"save-{key}", width="stretch"):
        if not name.strip():
            st.warning("Give the graph a name first.")
        else:
            project.save_graph({**recipe, "name": name.strip()})
            st.toast(f"Saved '{name.strip()}'")


# --------------------------------------------------------------------------- pages

def home() -> None:
    st.title("💬 WhatsApp Analysis")
    st.caption("Everything stays on your computer.")
    left, right = st.columns(2, gap="large")

    with left:
        st.subheader("New project")
        with st.form("new"):
            name = st.text_input("Project name", placeholder="e.g. Sam")
            file = st.file_uploader("Exported chat (.txt)", type=["txt"],
                                    help="In WhatsApp: open the chat → ⋮ → More → Export chat → Without media")
            order = st.radio("Date format in the file", ["Detect automatically", "Day / Month", "Month / Day"],
                             horizontal=True, help="Only matters if every date is ambiguous, like 3/4/24.")
            go = st.form_submit_button("Import", type="primary")
        if go:
            if not name.strip() or file is None:
                st.warning("Enter a name and choose a file.")
            else:
                try:
                    dayfirst = {"Detect automatically": None, "Day / Month": True, "Month / Day": False}[order]
                    Project.create(name, file.getvalue(), dayfirst)
                    open_project(name.strip())
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    with right:
        st.subheader("Open a project")
        names = Project.list_all()
        if not names:
            st.write("No projects yet.")
        for n in names:
            c1, c2 = st.columns([4, 1])
            if c1.button(n, key=f"open-{n}", width="stretch"):
                open_project(n)
                st.rerun()
            if c2.button("🗑", key=f"del-{n}", help="Delete project"):
                Project(n).delete()
                load_chat.clear()
                st.rerun()


def open_project(name: str) -> None:
    for k in list(ss.keys()):  # fresh widget state for each project
        del ss[k]
    ss.project = name


def workspace(name: str) -> None:
    project = Project(name)
    chat = load_chat(name)
    first, last = chat.bounds
    colors = charts.person_colors(chat.people)

    st.sidebar.title(f"💬 {name}")
    if st.sidebar.button("← All projects"):
        for k in list(ss.keys()):
            del ss[k]
        st.rerun()
    st.sidebar.caption(f"{len(chat.messages):,} messages · {first:%b %d, %Y} → {last:%b %d, %Y}")
    st.sidebar.divider()
    ss.setdefault("people", chat.people)
    people = st.sidebar.multiselect("People", chat.people, key="people")
    window = window_controls(chat)
    start, end = resolve_window(window, chat)
    n = len(chat._select(people or None, start, end))
    st.sidebar.caption(f"{n:,} messages in view")

    view = st.radio("View", VIEWS, horizontal=True, key="view", label_visibility="collapsed")

    if view == "Over time":
        c1, c2, c3 = st.columns([3, 2, 2])
        metric = c1.selectbox("Metric", list(METRICS), format_func=lambda m: METRICS[m].label, key="metric")
        ss.setdefault("freq", "week")
        ss.setdefault("connect", True)
        freq = c2.selectbox("Group by", list(FREQS), key="freq")
        connect = c3.toggle("Connect points", key="connect")
        recipe = {"kind": "timeseries", "metric": metric, "freq": freq, "people": people,
                  "window": window, "connect": connect}
        show_recipe(chat, recipe, colors, "live")
        save_box(project, recipe, "ts")

    elif view == "Compare people":
        c1, c2 = st.columns([4, 1])
        ss.setdefault("agg_metrics", list(METRICS)[:5])
        metrics = c1.multiselect("Metrics", list(METRICS),
                                 format_func=lambda m: METRICS[m].label, key="agg_metrics")
        ss.setdefault("normalize", True)
        normalize = c2.toggle("As % of texts", key="normalize",
                              help="Off = raw totals, which mostly show who texts more.")
        if not metrics:
            st.info("Pick at least one metric.")
        else:
            recipe = {"kind": "aggregate", "metrics": metrics, "normalize": normalize,
                      "people": people, "window": window}
            show_recipe(chat, recipe, colors, "live")
            save_box(project, recipe, "agg")

    else:
        graphs = project.graphs
        if not graphs:
            st.info("Nothing saved yet. Make a graph in the other views and press Save.")
        for i, g in enumerate(graphs):
            with st.container(border=True):
                h1, h2, h3 = st.columns([6, 1.3, 0.8])
                h1.markdown(f"**{g['name']}**  \n:gray[{g['kind']} · {window_label(g['window'])}]")
                h2.button("Edit", key=f"edit-{i}", on_click=load_recipe, args=(g,))
                if h3.button("🗑", key=f"rm-{i}"):
                    project.delete_graph(g["name"])
                    st.rerun()
                show_recipe(chat, g, colors, f"saved-{i}")


if ss.get("project"):
    try:
        workspace(ss.project)
    except FileNotFoundError:
        ss.pop("project")
        st.rerun()
else:
    home()
