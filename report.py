"""Builds a self-contained HTML session report from a CSV log.

No external chart library is used -- the engagement-over-time chart is drawn
as inline SVG built directly from the logged points, so the report has zero
runtime dependencies and opens standalone in any browser.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

import config

CHART_WIDTH = 900
CHART_HEIGHT = 260
CHART_PAD_L = 45
CHART_PAD_R = 15
CHART_PAD_T = 15
CHART_PAD_B = 30


@dataclass
class DipWindow:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def find_dip_windows(
    df: pd.DataFrame, threshold_pct: float, min_seconds: float
) -> list[DipWindow]:
    below = df["engagement_pct"] < threshold_pct
    windows: list[DipWindow] = []
    start_idx = None

    for i, is_below in enumerate(below):
        if is_below and start_idx is None:
            start_idx = i
        elif not is_below and start_idx is not None:
            start_s = df["elapsed_s"].iloc[start_idx]
            end_s = df["elapsed_s"].iloc[i - 1]
            if end_s - start_s >= min_seconds:
                windows.append(DipWindow(start_s, end_s))
            start_idx = None

    if start_idx is not None:
        start_s = df["elapsed_s"].iloc[start_idx]
        end_s = df["elapsed_s"].iloc[len(df) - 1]
        if end_s - start_s >= min_seconds:
            windows.append(DipWindow(start_s, end_s))

    return windows


def _x(elapsed: float, max_elapsed: float) -> float:
    span = max(max_elapsed, 1e-6)
    usable = CHART_WIDTH - CHART_PAD_L - CHART_PAD_R
    return CHART_PAD_L + (elapsed / span) * usable


def _y(pct: float) -> float:
    usable = CHART_HEIGHT - CHART_PAD_T - CHART_PAD_B
    return CHART_PAD_T + usable * (1 - pct / 100.0)


def build_chart_svg(df: pd.DataFrame, dips: list[DipWindow]) -> str:
    max_elapsed = float(df["elapsed_s"].max())
    points = " ".join(
        f"{_x(row.elapsed_s, max_elapsed):.1f},{_y(row.engagement_pct):.1f}"
        for row in df.itertuples()
    )

    grid_lines = []
    labels = []
    for pct in (0, 25, 50, 75, 100):
        y = _y(pct)
        grid_lines.append(
            f'<line x1="{CHART_PAD_L}" y1="{y:.1f}" x2="{CHART_WIDTH - CHART_PAD_R}" '
            f'y2="{y:.1f}" stroke="#3a3f4b" stroke-width="1" />'
        )
        labels.append(
            f'<text x="{CHART_PAD_L - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#9aa4b2">{pct}%</text>'
        )

    dip_rects = []
    for dip in dips:
        x1 = _x(dip.start_s, max_elapsed)
        x2 = _x(dip.end_s, max_elapsed)
        dip_rects.append(
            f'<rect x="{x1:.1f}" y="{CHART_PAD_T}" width="{max(x2 - x1, 1):.1f}" '
            f'height="{CHART_HEIGHT - CHART_PAD_T - CHART_PAD_B}" '
            f'fill="#e5484d" opacity="0.15" />'
        )

    time_labels = []
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        elapsed = max_elapsed * frac
        x = _x(elapsed, max_elapsed)
        time_labels.append(
            f'<text x="{x:.1f}" y="{CHART_HEIGHT - 8}" text-anchor="middle" '
            f'font-size="11" fill="#9aa4b2">{elapsed / 60:.1f}m</text>'
        )

    return f"""
<svg viewBox="0 0 {CHART_WIDTH} {CHART_HEIGHT}" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;height:auto;background:#1b1e26;border-radius:8px;">
  {''.join(dip_rects)}
  {''.join(grid_lines)}
  {''.join(labels)}
  {''.join(time_labels)}
  <polyline points="{points}" fill="none" stroke="#4cc38a" stroke-width="2" />
</svg>
""".strip()


def build_report_html(csv_path: Path, cfg: config.Config) -> str:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No data in {csv_path}")

    dips = find_dip_windows(df, cfg.dip_threshold_pct, cfg.dip_min_seconds)
    chart = build_chart_svg(df, dips)

    duration_s = float(df["elapsed_s"].max())
    mean_pct = float(df["engagement_pct"].mean())
    min_pct = float(df["engagement_pct"].min())
    peak_pct = float(df["engagement_pct"].max())
    mean_faces = float(df["faces_detected"].mean())

    dip_rows = "".join(
        f"<tr><td>{d.start_s / 60:.1f}m</td><td>{d.end_s / 60:.1f}m</td>"
        f"<td>{d.duration_s:.0f}s</td></tr>"
        for d in dips
    ) or '<tr><td colspan="3">No dips below threshold</td></tr>'

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Session Report - {csv_path.stem}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#12141a;
         color:#e6e9ef; max-width:960px; margin:32px auto; padding:0 16px; }}
  h1 {{ font-size:22px; margin-bottom:4px; }}
  .meta {{ color:#9aa4b2; font-size:13px; margin-bottom:24px; }}
  .stats {{ display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin-bottom:24px; }}
  .stat {{ background:#1b1e26; border-radius:8px; padding:14px; }}
  .stat .label {{ font-size:12px; color:#9aa4b2; }}
  .stat .value {{ font-size:24px; font-weight:600; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:8px; }}
  th, td {{ text-align:left; padding:6px 10px; border-bottom:1px solid #2a2e38; font-size:13px; }}
  th {{ color:#9aa4b2; font-weight:500; }}
  section {{ margin-bottom:28px; }}
</style>
</head>
<body>
  <h1>Live Audience Attention Report</h1>
  <div class="meta">Source log: {csv_path.name} &middot; Generated {generated}</div>

  <div class="stats">
    <div class="stat"><div class="label">Duration</div><div class="value">{duration_s / 60:.1f}m</div></div>
    <div class="stat"><div class="label">Mean Engagement</div><div class="value">{mean_pct:.0f}%</div></div>
    <div class="stat"><div class="label">Peak / Min</div><div class="value">{peak_pct:.0f}% / {min_pct:.0f}%</div></div>
    <div class="stat"><div class="label">Avg Faces Tracked</div><div class="value">{mean_faces:.1f}</div></div>
  </div>

  <section>
    <h2>Engagement over time</h2>
    {chart}
  </section>

  <section>
    <h2>Attention dips (below {cfg.dip_threshold_pct:.0f}% for &ge; {cfg.dip_min_seconds}s)</h2>
    <table>
      <thead><tr><th>Start</th><th>End</th><th>Duration</th></tr></thead>
      <tbody>{dip_rows}</tbody>
    </table>
  </section>
</body>
</html>
"""


def generate_report(csv_path: Path, cfg: config.Config | None = None) -> Path:
    cfg = cfg or config.Config()
    config.ensure_dirs()
    html = build_report_html(csv_path, cfg)
    out_path = config.REPORTS_DIR / f"{csv_path.stem}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: py report.py <path-to-session-csv>")
        raise SystemExit(1)
    out = generate_report(Path(sys.argv[1]))
    print(f"Report written to {out}")
