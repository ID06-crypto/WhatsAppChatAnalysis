# WhatsApp Chat Analysis

A local web app for exploring an exported WhatsApp conversation: who texts more, who asks more questions, how the mood of the chat changes over time, and more. Everything runs on your own computer. Your chats are never uploaded anywhere.

## What it does

Import a WhatsApp chat export (`.txt`) as a project, then build interactive graphs from it.

**Metrics** (each counts *texts containing* the thing, so "wow!!!" counts once):

| Metric | Meaning |
| --- | --- |
| Texts | Every message sent |
| Texts with a `?` | Messages containing a question mark |
| Texts with a `!` | Messages containing an exclamation point |
| Texts with ALL-CAPS words | Messages with a word of 2+ capital letters (a lone "I" doesn't count) |
| Texts with emojis | Messages containing at least one emoji |
| Sentiment | Average [VADER](https://github.com/cjhutto/vaderSentiment) score from -1 (negative) to +1 (positive) |

**Views**

- **Over time**: any metric, per person, as an interactive scatter. Group by minute, hour, day, week, month or year. Hover for values, drag to zoom, use the slider to move through time. Curves are shaded underneath so people are easy to tell apart, and a **Smoothing** slider averages neighboring buckets for a rounder, less jagged line (0 = raw data).
- **Compare people**: grouped bars of the chosen metrics per person. Show them as a percentage of each person's texts (fair comparison) or as raw totals. Sentiment gets its own bar chart.
- **Saved graphs**: save any graph by name and come back to it later. Saved graphs store their settings, not an image, and can be re-opened in the editor with **Edit**.

**Time window**: restrict everything to All time, the last 7 / 30 / 90 days, the last year, or a custom range. For a custom range, set just a start and the window runs through the most recent message, or untick "End at the latest message" to set an end date and time too. Presets count back from the chat's last message, not today's date.

Media messages, deleted messages and WhatsApp system notices ("X added Y") are not counted in the text metrics.

## Setup

Requires Python 3.9+.

```
pip install -r requirements.txt
```

## Usage

### 1. Export a chat from WhatsApp

Open the chat → tap the contact or group name → **Export chat** (Android: ⋮ → More → Export chat) → **Without media**. Get the `.txt` file onto your computer. On iPhone the export may arrive as a `.zip`; unzip it and use the `.txt` inside.

Both iOS and Android export formats are supported, including multi-line messages.

### 2. Start the app

```
streamlit run app.py
```

It opens in your browser (usually `localhost:8501`). Press Ctrl+C in the terminal to stop it. If Streamlit asks for an email on first run, you can leave it blank and press Enter.

### 3. Create a project

On the home screen, enter a name, upload the `.txt`, and click **Import**. Leave the date format on "Detect automatically" unless dates come out with days and months swapped (this can only happen when every date in the chat is ambiguous, like 3/4/24). In that case delete the project and re-import with the right order.

### 4. Explore

Use the sidebar to choose people and a time window, then pick a view. Type a name under a graph and click **Save graph** to keep it.

Projects live in a `projects/` folder next to the code, one folder each, containing the original export and a `graphs.json` of saved graphs. Delete a project from the home screen with 🗑.

## Using it as a library

The analysis code works without the UI:

```python
from whatsapp_analysis import parse_chat

chat = parse_chat("chat.txt")                      # dayfirst=True/False to force date order
chat.people                                        # ['Alice', 'Bob']
chat.timeseries("questions", freq="week")          # DataFrame: time buckets x people
chat.timeseries("emojis", "day", start="2024-01-01")   # no end = through the latest message
chat.aggregate(normalize=True)                     # DataFrame: people x metrics
```

Metric names: `messages`, `questions`, `exclamations`, `caps`, `emojis`, `sentiment`. Frequencies: `minute`, `hour`, `day`, `week`, `month`, `year`.

## Project layout

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI |
| `charts.py` | Plotly figures |
| `project.py` | Project folders and saved graphs |
| `whatsapp_analysis.py` | Chat parsing, metrics, time series and aggregation |

## Notes and limits

- Sentiment is VADER, which is tuned for English. Other languages will score close to neutral.
- Very fine grouping over a long window (e.g. minutes across years) produces a lot of points and will be slow. Narrow the window or group by hour or day.
- Emoji detection uses Unicode ranges and may miss a few rare symbols.
