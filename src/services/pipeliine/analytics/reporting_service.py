from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import html as _html_mod  # for HTML entity unescape
import pytz  # for timezone-aware formatting
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
    Table, TableStyle, Preformatted
)

from src.config.env_loader import SETTINGS
from src.utils.data_io_utils import save_report_pdf


# =========================
# Public primitives
# =========================

@dataclass
class Section:
    title: str
    html: str  # HTML fragment to embed (table/img/p/headers/etc.)


def _format_human_ts(dt: Optional[datetime] = None, tz_name: str = "America/Los_Angeles") -> str:
    """
    Return timestamp like 'Nov 09, 2025 07:02 am' localized to tz_name if possible.
    Falls back to local time if pytz unavailable.
    """
    dt = dt or datetime.now()
    if pytz is not None:
        try:
            tz = pytz.timezone(tz_name)
            dt = dt.astimezone(tz)
        except Exception:
            pass
    s = dt.strftime("%b %d, %Y %I:%M %p")
    # lower-case AM/PM to 'am'/'pm'
    return s[:-2] + s[-2:].lower()


def _html_escape(s: str) -> str:
    if _html_mod:
        return _html_mod.escape(s)
    # minimal fallback
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _html_unescape(s: str) -> str:
    if _html_mod:
        return _html_mod.unescape(s)
    return s


def _html_table_from_mapping(d: Dict[str, object], table_class: str = "tbl") -> str:
    """Render a dict as a 2-column HTML table (Key / Value)."""
    rows = []
    for k, v in d.items():
        if isinstance(v, (dict, list)):
            # pretty-print nested structure, but escaped
            v_str = _html_escape(json.dumps(v, indent=2))
            v_html = f"<pre class='pre'>{v_str}</pre>"
        else:
            v_html = _html_escape(str(v))
        rows.append(f"<tr><th>{_html_escape(str(k))}</th><td>{v_html}</td></tr>")
    return f"<table class='{table_class}'><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _html_table_from_records(recs: List[Dict[str, object]], table_class: str = "tbl") -> str:
    """Render a list[dict] as an HTML table."""
    if not recs:
        return "<div class='muted'>No data</div>"
    # union of keys preserves order by first row, then add unseen keys
    keys = list(recs[0].keys())
    for r in recs[1:]:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    head = "".join(f"<th>{_html_escape(str(k))}</th>" for k in keys)
    body_rows = []
    for r in recs:
        tds = []
        for k in keys:
            val = r.get(k, "")
            if isinstance(val, float):
                # simple numeric formatting
                v = f"{val:,.4f}"
            elif isinstance(val, (dict, list)):
                v = f"<pre class='pre'>{_html_escape(json.dumps(val, indent=2))}</pre>"
            else:
                v = _html_escape(str(val))
            tds.append(f"<td>{v}</td>")
        body_rows.append(f"<tr>{''.join(tds)}</tr>")
    return f"<table class='{table_class}'><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def build_html_report(title: str, meta: Dict, sections: List[Section]) -> str:
    """
    Build a complete HTML report page.

    - Renders a formatted 'Generated' timestamp in America/Los_Angeles as 'MMM dd, yyyy hh:mm am/pm'
    - Renders meta as a 2-column table (Field/Value) instead of raw JSON
    - Includes simple table/typography CSS so tables look good in both browser & PDF
    """
    gen_ts = _format_human_ts()
    meta_tbl = _html_table_from_mapping(meta)

    head = f"""<!doctype html>
    <html>
    <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>{_html_escape(title)}</title>
    <style>
      body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; color:#222; line-height:1.45; }}
      h1 {{ margin: 0 0 8px 0; font-size: 22px; }}
      h2 {{ margin: 20px 0 8px 0; font-size: 18px; border-bottom: 1px solid #e5e5e5; padding-bottom: 4px; }}
      h3 {{ margin: 14px 0 6px 0; font-size: 16px; }}
      .muted {{ color: #777; font-style: italic; }}
      .meta {{ margin-top: 6px; margin-bottom: 18px; }}
      .pre {{ white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, "Courier New", monospace; font-size: 12px; background:#fafafa; padding:8px; border:1px solid #eee; border-radius:6px; }}
      table.tbl {{ border-collapse: collapse; width: 100%; margin: 8px 0 16px 0; font-size: 13px; }}
      table.tbl th, table.tbl td {{ border: 1px solid #e0e0e0; padding: 6px 8px; vertical-align: top; text-align: left; }}
      table.tbl thead th {{ background: #f5f7fa; font-weight: 600; }}
      table.tbl tbody tr:nth-child(even) {{ background: #fbfcff; }}
    </style>
    </head>
    <body>
    <h1>{_html_escape(title)}</h1>
    <div class="meta"><strong>Generated:</strong> {gen_ts}</div>
    {meta_tbl}
    <hr/>
    """

    parts = []
    for s in (sections or []):
        if isinstance(s, Section):
            parts.append(f"<h2>{_html_escape(s.title)}</h2>\n{s.html or ''}")
        else:
            # fallback: treat as raw html string
            parts.append(str(s))
    body = "".join(parts)
    return head + body + "</body></html>"


def save_html_report(html: str, report_name: str) -> str:
    """
    Save the HTML report under REPORTS_DIR. Returns the path written.
    """
    reports_dir = Path(SETTINGS.REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"{report_name}.html"
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


# =========================
# PDF BUILDING (ReportLab)
# =========================

# Styles
_STYLES = getSampleStyleSheet()
_STYLES.add(ParagraphStyle(name="H1", parent=_STYLES["Heading1"], spaceAfter=8))
_STYLES.add(ParagraphStyle(name="H2", parent=_STYLES["Heading2"], spaceAfter=6))
_STYLES.add(ParagraphStyle(name="H3", parent=_STYLES["Heading3"], spaceAfter=6))
_STYLES.add(ParagraphStyle(name="Body", parent=_STYLES["BodyText"], leading=14))
if "Code" in _STYLES.byName:
    _STYLES.add(ParagraphStyle(name="Mono", parent=_STYLES["Code"], fontName="Courier", leading=12))
else:
    _STYLES.add(ParagraphStyle(name="Mono", parent=_STYLES["BodyText"], fontName="Courier", leading=12))

# Minimal HTML token regex
_RE_H1 = re.compile(r"<h1>(.*?)</h1>", re.I | re.S)
_RE_H2 = re.compile(r"<h2>(.*?)</h2>", re.I | re.S)
_RE_H3 = re.compile(r"<h3>(.*?)</h3>", re.I | re.S)
_RE_P = re.compile(r"<p>(.*?)</p>", re.I | re.S)
_RE_BR = re.compile(r"<br\s*/?>", re.I)
_RE_PRE = re.compile(r"<pre>(.*?)</pre>", re.I | re.S)
_RE_CODE = re.compile(r"<code>(.*?)</code>", re.I | re.S)
_RE_IMG = re.compile(r"<img[^>]*src=['\"](data:image/[^;]+;base64,([^'\"]+))['\"][^>]*>", re.I | re.S)
# Simple table parsing
_RE_TABLE = re.compile(r"<table.*?>.*?</table>", re.I | re.S)
_RE_TR = re.compile(r"<tr.*?>(.*?)</tr>", re.I | re.S)
_RE_TH = re.compile(r"<th.*?>(.*?)</th>", re.I | re.S)
_RE_TD = re.compile(r"<td.*?>(.*?)</td>", re.I | re.S)
_RE_BODY = re.compile(r"<body[^>]*>(.*?)</body>", re.I | re.S)
_RE_STYLE = re.compile(r"<style.*?>.*?</style>", re.I | re.S)
_RE_HEAD = re.compile(r"<head.*?>.*?</head>", re.I | re.S)

# Placeholder parser (robust)
_PLACEHOLDER_RE = re.compile(r"^__IMG_PLACEHOLDER_(\d+)__$")


def _normalize_breaks(text: str) -> str:
    return _RE_BR.sub("<br/>", (text or "").strip())


def _placeholder_index(seg: str) -> Optional[int]:
    m = _PLACEHOLDER_RE.match(seg)
    return int(m.group(1)) if m else None


def _data_uri_to_image_flowable(data_b64: str, max_width: float) -> Image:
    img_bytes = base64.b64decode(data_b64)
    ir = ImageReader(BytesIO(img_bytes))
    iw, ih = ir.getSize()
    scale = min(max_width / float(iw), 1.0)
    return Image(BytesIO(img_bytes), width=iw * scale, height=ih * scale)


def _parse_table_to_flowable(tbl_html: str) -> Optional[Table]:
    """
    Parse a simple HTML table into a ReportLab Table.
    Handles <thead>, <tbody>, <tr>, and rows that contain both <th> and <td>.
    """
    try:
        rows = _RE_TR.findall(tbl_html)
        if not rows:
            return None

        data: List[List[str]] = []
        header_done = False

        for row_html in rows:
            ths = _RE_TH.findall(row_html)
            tds = _RE_TD.findall(row_html)

            # Header row (pure THs)
            if ths and not tds and not header_done:
                header_done = True
                data.append([_html_unescape(re.sub("<.*?>", "", c)).strip() for c in ths])
                continue

            # Data row with TH in first col and TDs after (e.g., meta tables)
            if ths and tds:
                row_cells = [ths[0]] + tds
                data.append([_html_unescape(re.sub("<.*?>", "", c)).strip() for c in row_cells])
                continue

            # Pure TD row
            if tds:
                data.append([_html_unescape(re.sub("<.*?>", "", c)).strip() for c in tds])

        if not data:
            return None

        tbl = Table(data, repeatRows=1, splitByRow=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ALIGN", (0, 0), (-1, 0), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        return tbl
    except Exception:
        return None


def _json_to_flowables(text: str) -> List:
    """
    If `text` is JSON for performance payload, render as tables:
      - per_model_metrics: rows with model/RMSE/MAE/R2 (and bp if present)
      - ensemble_avg_metrics / ensemble_wgt_metrics: small key/value tables
    Otherwise, return a monospaced block.
    """
    try:
        payload = json.loads(text)
    except Exception:
        # not JSON → fallback to monospaced
        return [Preformatted(text.strip(), _STYLES["Mono"]), Spacer(1, 6)]

    flws: List = []

    # per_model_metrics table
    pmm = payload.get("per_model_metrics")
    if isinstance(pmm, list) and pmm:
        # columns (include bp if present)
        cols = ["model", "RMSE", "MAE", "R2"]
        if any("bp" in r for r in pmm):
            cols.append("bp")
        data = [cols]
        for r in pmm:
            row = []
            for c in cols:
                v = r.get(c, "")
                if isinstance(v, float):
                    v = f"{v:,.4f}"
                row.append(str(v))
            data.append(row)
        tbl = Table(data, repeatRows=1, splitByRow=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ALIGN", (0, 0), (-1, 0), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        flws += [Paragraph("Per-model metrics", _STYLES["H3"]), tbl, Spacer(1, 8)]

    # ensemble avg / weighted tables
    for key, title in [
        ("ensemble_avg_metrics", "Ensemble (average)"),
        ("ensemble_wgt_metrics", "Ensemble (weighted by 1/RMSE)"),
    ]:
        m = payload.get(key)
        if isinstance(m, dict) and m:
            data = [["Metric", "Value"]] + [[k, f"{v:,.4f}" if isinstance(v, float) else str(v)] for k, v in m.items()]
            tbl = Table(data, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]))
            flws += [Paragraph(title, _STYLES["H3"]), tbl, Spacer(1, 8)]

    if not flws:
        # JSON but not the structure we know → pretty print
        return [Preformatted(json.dumps(payload, indent=2), _STYLES["Mono"]), Spacer(1, 6)]
    return flws


def _html_to_story(html: str, page_width: float) -> List:
    """
    Convert a small subset of HTML to ReportLab flowables and
    force the 'Dataset Preview' section to the very end of the PDF.
    Supports: h1/h2/h3, p, br, pre/code (with JSON→tables), img (base64), table/th/td.
    """
    story: List = []
    preview_story: List = []  # captured 'Dataset Preview' section goes here
    if not html:
        return story

    # Keep only body and strip head/style so CSS doesn’t print
    m = _RE_BODY.search(html)
    html = m.group(1) if m else html
    html = _RE_STYLE.sub("", html)
    html = _RE_HEAD.sub("", html)

    # Extract base64 images and replace with placeholders
    imgs: List[str] = []

    def _img_repl(m: re.Match) -> str:
        imgs.append(m.group(2))
        return f"__IMG_PLACEHOLDER_{len(imgs) - 1}__"

    html = _RE_IMG.sub(_img_repl, html)

    # Tokenize by known block tags (include <table>)
    token_pat = re.compile(
        r"(<h1>.*?</h1>|<h2>.*?</h2>|<h3>.*?</h3>|<pre>.*?</pre>|<code>.*?</code>|<p>.*?</p>|<table.*?>.*?</table>)",
        re.I | re.S,
    )
    parts = [p for p in token_pat.split(html) if p and p.strip()]

    # Helper to render one "chunk" into a target list (story or preview_story)
    def _render_chunk(chunk: str, target: List):
        # Headings
        if _RE_H1.match(chunk):
            txt = _RE_H1.findall(chunk)[0]
            target.append(Paragraph(_normalize_breaks(txt), _STYLES["H1"]))
            target.append(Spacer(1, 6))
            return True
        if _RE_H2.match(chunk):
            txt = _RE_H2.findall(chunk)[0]
            target.append(Paragraph(_normalize_breaks(txt), _STYLES["H2"]))
            target.append(Spacer(1, 6))
            return True
        if _RE_H3.match(chunk):
            txt = _RE_H3.findall(chunk)[0]
            target.append(Paragraph(_normalize_breaks(txt), _STYLES["H3"]))
            target.append(Spacer(1, 6))
            return True

        # Pre/Code
        if _RE_PRE.match(chunk):
            txt = _RE_PRE.findall(chunk)[0]
            target += _json_to_flowables(_html_unescape(txt))
            return True
        if _RE_CODE.match(chunk):
            txt = _RE_CODE.findall(chunk)[0]
            txt = _RE_BR.sub("\n", txt)
            target.append(Preformatted(txt.strip(), _STYLES["Mono"]))
            target.append(Spacer(1, 6))
            return True

        # Table
        if _RE_TABLE.match(chunk):
            tbl = _parse_table_to_flowable(chunk)
            if tbl is not None:
                target.append(tbl)
                target.append(Spacer(1, 8))
                return True
            # fall through to text if parsing failed

        # Paragraphs: may contain image placeholders
        if _RE_P.match(chunk):
            body = _RE_P.findall(chunk)[0]
            segments = re.split(r"(__IMG_PLACEHOLDER_\d+__)", body)
            for seg in segments:
                if seg.startswith("__IMG_PLACEHOLDER_"):
                    idx = _placeholder_index(seg)
                    if idx is None or idx < 0 or idx >= len(imgs):
                        continue
                    target.append(_data_uri_to_image_flowable(imgs[idx], page_width))
                    target.append(Spacer(1, 6))
                else:
                    txt = _normalize_breaks(seg)
                    if txt.strip():
                        target.append(Paragraph(txt, _STYLES["Body"]))
                        target.append(Spacer(1, 4))
            return True

        # Fallback: any stray placeholders or text
        segments = re.split(r"(__IMG_PLACEHOLDER_\d+__)", chunk)
        for seg in segments:
            if seg.startswith("__IMG_PLACEHOLDER_"):
                idx = _placeholder_index(seg)
                if idx is None or idx < 0 or idx >= len(imgs):
                    continue
                target.append(_data_uri_to_image_flowable(imgs[idx], page_width))
                target.append(Spacer(1, 6))
            else:
                txt = _normalize_breaks(seg)
                if txt.strip():
                    target.append(Paragraph(txt, _STYLES["Body"]))
                    target.append(Spacer(1, 4))
        return True

    # Walk parts; when we hit an H2 named 'Dataset Preview', start capturing into preview_story
    i = 0
    while i < len(parts):
        chunk = parts[i]
        # Detect an <h2> and read its text
        if _RE_H2.match(chunk):
            h2_txt = _RE_H2.findall(chunk)[0].strip()
            is_preview_h2 = ("dataset preview" in h2_txt.lower())
            if is_preview_h2:
                # Start capturing this H2 and everything until the next H2 into preview_story
                _render_chunk(chunk, preview_story)  # the H2 itself
                i += 1
                # capture following chunks until next H2 or end
                while i < len(parts) and not _RE_H2.match(parts[i]):
                    _render_chunk(parts[i], preview_story)
                    i += 1
                # do NOT render this section now; it will be appended at the end with a PageBreak
                continue
            else:
                _render_chunk(chunk, story)
                i += 1
                continue

        # Not an H2 — render to main story unless we’re already capturing (handled above)
        _render_chunk(chunk, story)
        i += 1

    # If we captured a preview section, push it to the end with a page break
    if preview_story:
        story.append(PageBreak())
        story.extend(preview_story)

    return story


def df_to_table(df, max_rows: int = 25):
    """Convert a DataFrame head to a compact ReportLab Table flowable list (for PDF)."""
    try:
        import pandas as pd  # local import
    except Exception:
        return []

    if df is None:
        return []
    d = df.head(max_rows)
    if d.empty:
        return []

    data = [list(map(str, d.columns))] + d.astype(str).values.tolist()
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (0, 0), (-1, 0), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    return [tbl, Spacer(1, 8)]


from io import BytesIO

...


def build_pdf_from_html(
        html_report: str,
        report_name: str,
        base_dir: str,
        df_preview=None,  # optional DataFrame to include as a page
) -> str:
    """
    Convert a **subset** of HTML (incl. tables & images) to a paginated PDF using ReportLab,
    then persist it via data_io_utils.save_report_pdf, abstracting over LOCAL vs S3.

    Args:
      html_report: HTML string you also render in Streamlit.
      report_name: file name without extension.
      base_dir: logical subdir / run_id where the report should be stored.
      df_preview: optional DataFrame for an extra 'Dataset Preview' page.

    Returns:
      LOCAL → filesystem path as str
      S3    → s3://bucket/key URI
    """
    # Build PDF into memory first
    buf = BytesIO()

    left, right, top, bottom = (36, 36, 36, 36)
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=left, rightMargin=right, topMargin=top, bottomMargin=bottom,
    )

    story: List = []
    page_width = doc.width

    # Convert HTML (head/sections) to flowables
    story += _html_to_story(html_report, page_width)

    # Optional DataFrame preview page rendered as a proper table
    if df_preview is not None:
        story.append(PageBreak())
        story.append(Paragraph("Dataset Preview (first rows)", _STYLES["H2"]))
        story += df_to_table(df_preview, max_rows=25)

    def _numbered(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(doc_.pagesize[0] - right, 20, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_numbered, onLaterPages=_numbered)

    pdf_bytes = buf.getvalue()
    buf.close()

    # Persist via data_io_utils (LOCAL or S3)
    return save_report_pdf(base_dir=base_dir, name=report_name, pdf_bytes=pdf_bytes)
