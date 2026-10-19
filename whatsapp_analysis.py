"""WhatsApp chat analysis: parse an exported .txt chat and compute/plot per-person stats.

Typical use:
    chat = parse_chat("chat.txt")
    ts = chat.timeseries("questions", freq="week")   # DataFrame: index=time bucket, columns=people
    agg = chat.aggregate(normalize=True)             # DataFrame: index=person, columns=metrics
    plot_timeseries(chat, "sentiment", freq="month")
    plot_aggregate(chat)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- parsing

# Invisible characters WhatsApp sprinkles into exports (LRM/RLM, BOM, narrow no-break space).
_INVISIBLE = dict.fromkeys(map(ord, "\u200E\u200F\u202A\u202B\u202C﻿"))

# Matches the start of a message in both iOS ("[1/31/24, 9:41:02 PM] Name: hi")
# and Android ("1/31/24, 9:41 PM - Name: hi") exports.
_LINE_START = re.compile(
    r"""^\[?
        (?P<date>\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4}),?\s+
        (?P<time>\d{1,2}[:.]\d{2}(?:[:.]\d{2})?)\s*
        (?P<ampm>[AaPp]\.?\s?[Mm]\.?)?
        \]?\s*(?:-\s*)?
        (?P<rest>.*)$""",
    re.VERBOSE,
)

_MEDIA_MARKERS = (
    "<media omitted>", "image omitted", "video omitted", "audio omitted", "sticker omitted",
    "gif omitted", "document omitted", "contact card omitted", "<attached:",
)
_DELETED_MARKERS = ("this message was deleted", "you deleted this message")

_URL = re.compile(r"https?://\S+|www\.\S+")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U0001F000-\U0001F2FF☀-➿⭐⭕⬆⬇⤴⤵"
    "⌚⌛⏩-⏳⏸-⏺❤‼⁉™ℹ©®]"
)


def _detect_order(dates: list[str], dayfirst: bool | None) -> str:
    """Return 'ymd', 'dmy' or 'mdy'. Exports are ambiguous (3/4/24), so infer from the data."""
    if dates and len(re.split(r"[/.\-]", dates[0])[0]) == 4:
        return "ymd"
    if dayfirst is not None:
        return "dmy" if dayfirst else "mdy"
    first_gt12 = second_gt12 = False
    for d in dates:
        a, b, _ = re.split(r"[/.\-]", d)
        first_gt12 |= int(a) > 12
        second_gt12 |= int(b) > 12
        if first_gt12 and second_gt12:
            break
    if first_gt12:
        return "dmy"
    if second_gt12:
        return "mdy"
    return "mdy"  # truly ambiguous: assume US; caller can override with dayfirst=True


def _to_datetime(date: str, time: str, ampm: str | None, order: str) -> datetime:
    p = [int(x) for x in re.split(r"[/.\-]", date)]
    if order == "ymd":
        y, m, d = p
    elif order == "dmy":
        d, m, y = p
    else:
        m, d, y = p
    if y < 100:
        y += 2000
    t = [int(x) for x in re.split(r"[:.]", time)]
    h, mi, s = t[0], t[1], (t[2] if len(t) > 2 else 0)
    if ampm:
        pm = ampm.lower().startswith("p")
        h = h % 12 + (12 if pm else 0)
    return datetime(y, m, d, h, mi, s)


def _split_sender(rest: str) -> tuple[str | None, str]:
    """'Name: text' -> (Name, text); system lines (no sender) -> (None, rest)."""
    name, sep, text = rest.partition(": ")
    if sep and 0 < len(name) <= 60 and "\n" not in name:
        return name.strip(), text
    if rest.endswith(":"):  # sender with an empty message
        return rest[:-1].strip(), ""
    return None, rest


def parse_chat(path: str | Path, dayfirst: bool | None = None, encoding: str = "utf-8") -> "Chat":
    """Parse a WhatsApp 'Export chat' .txt file into a Chat.

    dayfirst: force DD/MM (True) or MM/DD (False). Default None infers it from the file.
    """
    raw = Path(path).read_text(encoding=encoding, errors="replace")
    raw = raw.replace(" ", " ").replace(" ", " ").translate(_INVISIBLE)

    records: list[list] = []  # [date, time, ampm, rest]
    for line in raw.splitlines():
        m = _LINE_START.match(line)
        if m:
            records.append([m["date"], m["time"], m["ampm"], m["rest"]])
        elif records:  # continuation of a multi-line message
            records[-1][3] += "\n" + line
    if not records:
        raise ValueError(f"No WhatsApp messages found in {path}. Is this an exported chat .txt?")

    order = _detect_order([r[0] for r in records], dayfirst)
    rows = []
    for date, time, ampm, rest in records:
        try:
            ts = _to_datetime(date, time, ampm, order)
        except ValueError:  # e.g. forced dayfirst on a month-first file
            raise ValueError(f"Could not read date '{date}' as {order}; try dayfirst={not dayfirst}") from None
        sender, text = _split_sender(rest)
        low = text.strip().lower()
        rows.append({
            "timestamp": ts,
            "sender": sender,
            "text": text.strip(),
            "is_system": sender is None,
            "is_media": sender is not None and any(low.startswith(k) or low.endswith(k) for k in _MEDIA_MARKERS),
            "is_deleted": sender is not None and low in _DELETED_MARKERS,
        })
    df = pd.DataFrame(rows).sort_values("timestamp", kind="stable").reset_index(drop=True)
    return Chat(df, source=str(path))


# --------------------------------------------------------------------------- features

def count_questions(text: str) -> int:
    return int("?" in text)


def count_exclamations(text: str) -> int:
    return int("!" in text)


def count_caps_words(text: str) -> int:
    """Number of ALL-CAPS words of 2+ letters (URLs ignored; single 'I'/'A' don't count)."""
    return sum(1 for w in _WORD.findall(_URL.sub(" ", text)) if len(w) >= 2 and w.isupper())


def has_caps_word(text: str) -> int:
    return int(count_caps_words(text) > 0)


def has_emoji(text: str) -> int:
    return int(bool(_EMOJI.search(text)))


_vader = None


def sentiment_score(text: str) -> float:
    """VADER compound score in [-1, 1] (negative -> positive)."""
    global _vader
    if _vader is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError as e:
            raise ImportError("Sentiment needs vaderSentiment: pip install vaderSentiment") from e
        _vader = SentimentIntensityAnalyzer()
    return _vader.polarity_scores(text)["compound"]


@dataclass(frozen=True)
class Metric:
    label: str
    how: str  # "sum" for per-message flags/counts, "mean" for scores


# Each metric is a message-level column; "texts with X" metrics are 0/1 flags per message.
METRICS: dict[str, Metric] = {
    "messages": Metric("Texts", "sum"),
    "questions": Metric("Texts with a '?'", "sum"),
    "exclamations": Metric("Texts with a '!'", "sum"),
    "caps": Metric("Texts with ALL-CAPS words", "sum"),
    "emojis": Metric("Texts with emojis", "sum"),
    "sentiment": Metric("Sentiment (VADER, -1 to 1)", "mean"),
}

MAX_FILLED_BUCKETS = 100_000
FREQS = {"minute": "min", "hour": "h", "day": "D", "week": "W-MON", "month": "MS", "year": "YS"}


# --------------------------------------------------------------------------- Chat object

@dataclass
class Chat:
    """Parsed conversation. `df` has one row per message (including system lines)."""
    df: pd.DataFrame
    source: str = ""
    _features: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def people(self) -> list[str]:
        return list(self.messages["sender"].value_counts().index)

    @property
    def messages(self) -> pd.DataFrame:
        """Real messages only (system notices like 'X joined' removed)."""
        return self.df[~self.df["is_system"]]

    @property
    def features(self) -> pd.DataFrame:
        """Messages plus one numeric column per metric. Computed once, then cached."""
        if self._features is None:
            f = self.messages.copy()
            text = f["text"].where(~(f["is_media"] | f["is_deleted"]), "")
            f["messages"] = 1
            f["questions"] = text.map(count_questions)
            f["exclamations"] = text.map(count_exclamations)
            f["caps"] = text.map(has_caps_word)
            f["emojis"] = text.map(has_emoji)
            f["sentiment"] = [sentiment_score(t) if t else float("nan") for t in text]
            self._features = f
        return self._features

    def _select(self, people: list[str] | None, start, end) -> pd.DataFrame:
        f = self.features
        if people is not None:
            f = f[f["sender"].isin(people)]
        if start is not None:
            f = f[f["timestamp"] >= pd.Timestamp(start)]
        if end is not None:
            f = f[f["timestamp"] <= pd.Timestamp(end)]
        return f

    def timeseries(self, metric: str, freq: str = "day", people: list[str] | None = None,
                   start=None, end=None, fill_gaps: bool = True) -> pd.DataFrame:
        """Metric over time, one column per person.

        freq: minute | hour | day | week | month | year.
        fill_gaps: buckets with no messages become 0 for counts (sentiment stays NaN).
        """
        if metric not in METRICS:
            raise ValueError(f"Unknown metric '{metric}'. Choose from {list(METRICS)}")
        if freq not in FREQS:
            raise ValueError(f"Unknown freq '{freq}'. Choose from {list(FREQS)}")
        f = self._select(people, start, end)
        if f.empty:
            return pd.DataFrame()
        grouped = f.groupby([pd.Grouper(key="timestamp", freq=FREQS[freq]), "sender"])[metric]
        out = (grouped.mean() if METRICS[metric].how == "mean" else grouped.sum()).unstack("sender")
        if fill_gaps:
            full = pd.date_range(out.index.min(), out.index.max(), freq=FREQS[freq])
            if len(full) <= MAX_FILLED_BUCKETS:  # e.g. minute buckets over years: skip, too many
                out = out.reindex(full)
                if METRICS[metric].how == "sum":
                    out = out.fillna(0)
        out.index.name = "time"
        return out[[p for p in (people or self.people) if p in out.columns]]

    def aggregate(self, metrics: list[str] | None = None, people: list[str] | None = None,
                  start=None, end=None, normalize: bool = False) -> pd.DataFrame:
        """One row per person, one column per metric.

        normalize=True turns count metrics into % of that person's texts, so a chatty
        person doesn't dominate every bar. Sentiment is always the mean.
        """
        metrics = metrics or list(METRICS)
        f = self._select(people, start, end)
        g = f.groupby("sender")
        out = pd.DataFrame({m: (g[m].mean() if METRICS[m].how == "mean" else g[m].sum()) for m in metrics})
        if normalize:
            n = g["messages"].sum()
            for m in metrics:
                if METRICS[m].how == "sum" and m != "messages":
                    out[m] = 100 * out[m] / n
        return out.loc[[p for p in self.people if p in out.index]]

    @property
    def bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Timestamps of the first and most recent message."""
        t = self.messages["timestamp"]
        return t.min(), t.max()

    def summary(self) -> dict:
        m = self.messages
        return {
            "source": self.source,
            "people": self.people,
            "messages": len(m),
            "start": m["timestamp"].min(),
            "end": m["timestamp"].max(),
            "media": int(m["is_media"].sum()),
        }


# --------------------------------------------------------------------------- plotting

def _hover(fig, ax, artists_data):
    """Tooltip on hover: artists_data = [(PathCollection, label, x_values, y_values)]."""
    import matplotlib.dates as mdates

    note = ax.annotate("", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
                       bbox=dict(boxstyle="round", fc="white", ec="0.6"), visible=False)

    def on_move(event):
        if event.inaxes is not ax:
            return
        for art, label, xs, ys in artists_data:
            hit, idx = art.contains(event)
            if hit:
                i = idx["ind"][0]
                note.xy = (mdates.date2num(xs[i]), ys[i])
                note.set_text(f"{label}\n{xs[i]:%Y-%m-%d %H:%M}\n{ys[i]:.2f}".replace(".00", ""))
                note.set_visible(True)
                fig.canvas.draw_idle()
                return
        if note.get_visible():
            note.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)


def plot_timeseries(chat: Chat, metric: str, freq: str = "week", people: list[str] | None = None,
                    start=None, end=None, connect: bool = True, save_path: str | None = None,
                    show: bool = True):
    """Scatter of a metric over time, one color per person. Hover points for values."""
    import matplotlib.pyplot as plt

    ts = chat.timeseries(metric, freq, people, start, end)
    fig, ax = plt.subplots(figsize=(11, 5))
    hover = []
    for person in ts.columns:
        s = ts[person].dropna()
        if connect:
            ax.plot(s.index, s.values, alpha=0.35, lw=1)
        sc = ax.scatter(s.index, s.values, s=18, label=person)
        hover.append((sc, person, list(s.index), list(s.values)))
    ax.set(xlabel="Time", ylabel=f"{METRICS[metric].label} per {freq}",
           title=f"{METRICS[metric].label} per {freq}")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    _hover(fig, ax, hover)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


def plot_aggregate(chat: Chat, metrics: list[str] | None = None, people: list[str] | None = None,
                   start=None, end=None, normalize: bool = True, save_path: str | None = None,
                   show: bool = True):
    """Grouped bar chart: one group per metric, one bar per person.

    Sentiment lives on a different scale (-1..1), so it's left out of the bars when
    other metrics are shown; plot it alone or via plot_timeseries.
    """
    import matplotlib.pyplot as plt

    metrics = [m for m in (metrics or list(METRICS)) if m != "sentiment" or metrics == ["sentiment"]]
    if normalize and metrics != ["sentiment"]:
        metrics = [m for m in metrics if m != "messages"] or metrics
    agg = chat.aggregate(metrics, people, start, end, normalize=normalize)
    ax = agg.T.rename(index={m: METRICS[m].label for m in metrics}).plot.bar(figsize=(11, 5), width=0.8)
    ax.set(ylabel="% of that person's texts" if normalize else "Total",
           title="Per-person totals" if not normalize else "Per-person rates")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.0f" if not normalize else "%.1f", fontsize=8)
    fig = ax.get_figure()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    import sys

    chat = parse_chat(sys.argv[1])
    print(chat.summary())
    print(chat.aggregate(normalize=True).round(2))
    plot_timeseries(chat, "messages", freq="week")
