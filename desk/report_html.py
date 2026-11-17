"""HTML rendering of the daily report.

Email clients are a hostile target: no external stylesheets, no flexbox, no web fonts, and several
force their own dark mode. So everything here is table layout with inline styles and explicit
backgrounds on every cell. Same ctx dict as the plain-text build_report, so the two cannot drift.
"""
from __future__ import annotations

import re
from html import escape

from desk.report import snapshot_line, trim_text

INK, MUTED, LINE, BG, PANEL = "#16181d", "#6b7280", "#e6e8eb", "#ffffff", "#f7f8fa"
POS, NEG = "#0f7a3d", "#b3261e"
LINK = "#1d4ed8"
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def _pct(x, signed=True) -> str:
    return "n/a" if x is None else (f"{x:+.2%}" if signed else f"{x:.2%}")


def _colour(x) -> str:
    return MUTED if x is None else (POS if x > 0 else NEG if x < 0 else INK)


def _num(text, x=None, bold=False) -> str:
    """Right-aligned monospace cell, coloured by sign when a value is given."""
    c = _colour(x) if x is not None else INK
    w = "600" if bold else "400"
    return (f'<td style="padding:7px 10px;border-bottom:1px solid {LINE};font-family:{MONO};font-size:13px;'
            f'text-align:right;color:{c};font-weight:{w};white-space:nowrap;background:{BG};">{escape(str(text))}</td>')


def _cell(text, muted=False, width="") -> str:
    c = MUTED if muted else INK
    w = f"width:{width};" if width else ""
    return (f'<td style="padding:7px 10px;border-bottom:1px solid {LINE};font-family:{FONT};font-size:13px;'
            f'color:{c};{w}background:{BG};">{text}</td>')


def _section(title: str, inner: str) -> str:
    return (f'<tr><td style="padding:26px 0 8px 0;background:{BG};"><div style="font-family:{FONT};font-size:11px;'
            f'letter-spacing:.09em;text-transform:uppercase;color:{MUTED};font-weight:600;">{escape(title)}</div>'
            f'</td></tr><tr><td style="background:{BG};">{inner}</td></tr>')


def _table(rows: str) -> str:
    return (f'<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" '
            f'style="border-collapse:collapse;width:100%;background:{BG};">{rows}</table>')


def _note_row(text: str, cols: int = 4) -> str:
    """Analyst prose gets the full width. Inside a narrow cell it wraps to a dozen lines on a phone."""
    if not text:
        return ""
    return (f'<tr><td colspan="{cols}" style="padding:0 10px 9px 10px;border-bottom:1px solid {LINE};'
            f'background:{BG};font-family:{FONT};font-size:12px;line-height:1.5;color:{MUTED};">'
            f'{escape(text)}</td></tr>')


def _ticker(symbol: str, links: dict | None) -> str:
    """A ticker, linked to its Google Finance page when its exchange is known."""
    url = (links or {}).get(symbol)
    if not url:
        return f"<b>{escape(symbol)}</b>"
    return (f'<a href="{escape(url, quote=True)}" style="color:{LINK};text-decoration:none;font-weight:700;">'
            f'{escape(symbol)}</a>')


def _link(text: str, url) -> str:
    """Escaped text, linked only to a plain http(s) address. The URL comes from the news feed, so anything
    else (javascript:, data:, a malformed value) is left as unlinked text rather than trusted."""
    u = str(url or "").strip()
    if not re.match(r"^https?://[^\s\"'<>]+$", u):
        return escape(text)
    return (f'<a href="{escape(u, quote=True)}" style="color:{LINK};text-decoration:underline;">'
            f'{escape(text)}</a>')


def _empty(text: str) -> str:
    return (f'<div style="font-family:{FONT};font-size:13px;color:{MUTED};padding:4px 0;">{escape(text)}</div>')


def _head_row(cells: list, left: int = 1, html: dict | None = None) -> str:
    """`left` = how many leading columns are left-aligned, to match their cells. `html` = {index: markup}
    for a header that is a link rather than plain text."""
    out = ""
    for i, c in enumerate(cells):
        align = "left" if i < left else "right"
        out += (f'<th style="padding:0 10px 6px 10px;border-bottom:1px solid {LINE};font-family:{FONT};font-size:10px;'
                f'letter-spacing:.06em;text-transform:uppercase;color:{MUTED};text-align:{align};font-weight:600;'
                f'background:{BG};">{(html or {}).get(i) or escape(c)}</th>')
    return f"<tr>{out}</tr>"


def build_html(ctx: dict) -> str:
    eq, day_pct, day_pl = ctx["equity"], ctx["day_pct"], ctx["day_pl"]
    links = ctx.get("quote_links") or {}
    parts = []

    # Headline: equity, today's move, cash.
    halt = ctx.get("halted")
    banner = ""
    if halt:
        banner = (f'<tr><td style="padding:12px 14px;background:#fdecea;border:1px solid #f3c2bd;border-radius:6px;'
                  f'font-family:{FONT};font-size:13px;color:{NEG};font-weight:600;">HALTED &mdash; '
                  f'{escape(str(halt))}. No trading until restarted.</td></tr>')
    parts.append(
        f'<tr><td style="padding:0 0 4px 0;background:{BG};">'
        f'<div style="font-family:{FONT};font-size:11px;letter-spacing:.09em;text-transform:uppercase;'
        f'color:{MUTED};font-weight:600;">TeamRobo daily &middot; {ctx["day"].isoformat()}</div>'
        f'<div style="font-family:{MONO};font-size:30px;font-weight:600;color:{INK};padding:8px 0 2px 0;">'
        f'${eq:,.2f}</div>'
        f'<div style="font-family:{FONT};font-size:14px;color:{_colour(day_pct)};font-weight:600;">'
        f'{_pct(day_pct)} today <span style="color:{MUTED};font-weight:400;">'
        f'({day_pl:+,.2f}) &middot; cash ${ctx["cash"]:,.0f} ({ctx["cash"] / eq:.0%}) &middot; '
        f'{len(ctx["positions"])} holdings</span></div></td></tr>')
    snap = snapshot_line(ctx.get("now"))
    if snap:
        parts.append(f'<tr><td style="padding:6px 0 0 0;background:{BG};font-family:{FONT};font-size:11px;'
                     f'line-height:1.5;color:{MUTED};">{escape(snap)}</td></tr>')
    if banner:
        parts.append(f'<tr><td style="padding:14px 0 0 0;background:{BG};">{_table(banner)}</td></tr>')

    # Things only the owner can do (the desk will not trim a held position): before everything else.
    if ctx.get("manual"):
        items = "".join(f'<li style="font-family:{FONT};font-size:13px;color:{INK};padding:3px 0;">'
                        f'{escape(trim_text(m, max_sentences=2, max_chars=300))}</li>' for m in ctx["manual"])
        parts.append(_section("Needs your action", f'<ul style="margin:4px 0 0 0;padding-left:20px;">{items}</ul>'))

    # Performance against SPY - the headline comparison, so it sits first.
    perf = ctx.get("perf") or []
    if perf:
        spy = links.get("SPY")
        rows = _head_row(["Period", "Desk", "SPY", "Difference"], html={2: (
            f'<a href="{escape(spy, quote=True)}" style="color:{MUTED};text-decoration:underline;">SPY</a>'
            if spy else None)})
        for r in perf:
            rows += ("<tr>" + _cell(f'<span style="white-space:nowrap;">{escape(r["period"])}</span>'
                                    f'<div style="color:{MUTED};font-size:11px;white-space:nowrap;padding-top:1px;">'
                                    f'from {r["from"].isoformat()}</div>')
                     + _num(_pct(r["desk_pct"]), r["desk_pct"]) + _num(_pct(r["spy_pct"]), r["spy_pct"])
                     + _num(_pct(r["diff_pct"]), r["diff_pct"], bold=True) + "</tr>")
        parts.append(_section("Performance vs SPY", _table(rows)))
    else:
        parts.append(_section("Performance vs SPY", _empty("not enough history yet")))

    # Live against backtest: volatility and drawdown are what a backtest actually predicts (Wiecki et al. 2016).
    ev = ctx.get("evidence")
    if ev and ev.get("days"):
        ref = ev.get("reference")
        if ref:
            def chip(status):
                bad = status.startswith(("ABOVE", "BEYOND"))
                warn = status.startswith(("below", "in the backtest's worst"))
                bg, fg = ("#fdecea", NEG) if bad else ("#fff4e5", "#9a6700") if warn else (PANEL, MUTED)
                return (f'<span style="background:{bg};color:{fg};font-size:10px;font-weight:600;padding:2px 6px;'
                        f'border-radius:3px;white-space:nowrap;">{escape(status)}</span>')
            rows = _head_row(["Measure", "Live", "Backtest", "Status"])
            rows += ("<tr>" + _cell("20-day volatility")
                     + _num(f"{ev['vol']:.1%}" if ev["vol"] is not None else "n/a")
                     + _num(f"{ref['vol_p05']:.0%}-{ref['vol_p95']:.0%}") + _cell(chip(ev["vol_status"])) + "</tr>")
            rows += ("<tr>" + _cell("Drawdown from peak") + _num(f"{ev['dd_now']:.1%}", ev["dd_now"] or None)
                     + _num(f"worst {ref['max_dd']:.1%}") + _cell(chip(ev["dd_status"])) + "</tr>")
            note = (f"Worst since trading began: {ev['dd_max']:.1%} over {ev['days']} trading days. Backtest: current "
                    f"config, {ref['window']}, median 20-day volatility {ref['vol_p50']:.0%}.")
            if ev.get("stale_reference"):
                note += " The reference is from an older config; rerun the backtest without overrides to refresh it."
        else:
            rows, note = "", f"{ev['vol_status']}: run backtest.py once without overrides to create it."
        inner = (_table(rows) if rows else "") + (
            f'<div style="font-family:{FONT};font-size:12px;color:{MUTED};padding:8px 10px 0 10px;">{escape(note)}</div>')
        if ctx.get("research"):
            inner += (f'<div style="font-family:{FONT};font-size:12px;color:{INK};padding:6px 10px 0 10px;">'
                      f'<b>Research:</b> {escape(ctx["research"])}</div>')
        parts.append(_section("Live vs backtest", inner))

    # Results today: exits with P&L, then fills.
    closed, filled = ctx.get("closed_today") or [], [p for p in ctx["positions"] if p.get("filled_today")]
    plain = ctx.get("moves_plain") or {}
    if closed or filled:
        rows = _head_row(["", "Symbol", "Price", "Result"], left=2)
        for c in closed:
            rows += ("<tr>" + _cell(f'<span style="color:{NEG};font-weight:600;">SOLD</span>', width="62px")
                     + _cell(f'<span style="white-space:nowrap;">{_ticker(c["symbol"], links)} &times;{c["qty"]:.0f}</span>')
                     + _num(f'{c["exit_price"]:,.2f}')
                     + _num(f'{c["pnl_pct"]:+.2%} ({c["pnl"]:+,.2f})', c["pnl_pct"], bold=True) + "</tr>"
                     + _note_row(plain.get(f'sold:{c["symbol"]}', "")))
        for p in filled:
            rows += ("<tr>" + _cell(f'<span style="color:{POS};font-weight:600;">BOUGHT</span>', width="62px")
                     + _cell(f'<span style="white-space:nowrap;">{_ticker(p["symbol"], links)} &times;{p["qty"]:.0f}</span>')
                     + _num(f'{p["entry_price"]:,.2f}') + _num("—") + "</tr>"
                     + _note_row(plain.get(f'bought:{p["symbol"]}', "")))
        parts.append(_section("Results today", _table(rows)))
    else:
        parts.append(_section("Results today", _empty("nothing bought or sold")))
    parts.append(f'<tr><td style="padding:6px 10px;background:{BG};font-family:{FONT};font-size:12px;color:{MUTED};">'
                 f'Unrealised across {len(ctx["positions"])} holdings at the close: '
                 f'<span style="color:{_colour(ctx.get("unrealized", 0))};font-weight:600;font-family:{MONO};">'
                 f'{ctx.get("unrealized", 0):+,.2f}</span></td></tr>')

    # Every holding at the official close, largest first.
    hold = ctx.get("holdings")
    if hold and hold["rows"]:
        rows = _head_row(["Symbol", "Close", "Value", "Day", "Since entry"])
        for r in hold["rows"]:
            sub = f'&times;{r["qty"]:.0f} &middot; {r["weight"]:.1%}' + (f' &middot; stop {r["stop"]:,.2f}' if r.get("stop") else "")
            if r["live_mark"]:
                sub += " &middot; live mark"
            rows += ("<tr>" + _cell(f'<span style="white-space:nowrap;">{_ticker(r["symbol"], links)}</span>'
                                    f'<div style="color:{MUTED};font-size:11px;white-space:nowrap;padding-top:2px;">{sub}</div>')
                     + _num(f'{r["close"]:,.2f}') + _num(f'{r["value"]:,.2f}')
                     + _num(_pct(r["day_pct"]), r["day_pct"])
                     + _num(f'{_pct(r["pnl_pct"])} ({r["pnl"]:+,.2f})', r["pnl_pct"], bold=True) + "</tr>")
        rows += ("<tr>" + _cell(f'<b>Total</b> <span style="color:{MUTED};font-size:11px;">{len(hold["rows"])} holdings</span>')
                 + _num("") + _num(f'{hold["invested"]:,.2f}', bold=True) + _num("")
                 + _num(f'{hold["pnl"]:+,.2f}', hold["pnl"], bold=True) + "</tr>")
        inner = _table(rows) + (
            f'<div style="font-family:{FONT};font-size:12px;color:{MUTED};padding:8px 10px 0 10px;">'
            f'Invested ${hold["invested"]:,.2f} ({hold["invested"] / hold["equity"]:.0%}) + cash ${hold["cash"]:,.2f} '
            f'= ${hold["equity"]:,.2f} at the close.</div>')
        title = f'Current holdings · as of close {hold["asof"]}' if hold.get("asof") else "Current holdings"
        parts.append(_section(title, inner))

    # Orders resting for the next open.
    staged = ctx.get("staged") or []
    if staged:
        rows = _head_row(["", "Symbol", "Limit", "Stop"], left=2)
        for s in staged:
            buy = s["side"] == "buy"
            tag = (f'<span style="color:{POS if buy else NEG};font-weight:600;">{"BUY" if buy else "SELL"}</span>')
            note = trim_text(str(s.get("note", "")), max_chars=200)
            rows += ("<tr>" + _cell(tag, width="62px")
                     + _cell(f'<span style="white-space:nowrap;">{_ticker(s["symbol"], links)} &times;{s["qty"]:.0f}</span>')
                     + _num(f'{s["limit"]:,.2f}' if s["limit"] else "market")
                     + _num(f'{s["stop"]:,.2f}' if s["stop"] else "—") + "</tr>"
                     + _note_row(note))
        parts.append(_section("Working for the next open", _table(rows)))
    else:
        parts.append(_section("Working for the next open", _empty("no orders resting")))

    # Movers.
    mv = ctx.get("movers") or {"notable": [], "rest": 0}
    if mv["notable"]:
        rows = _head_row(["Symbol", "Today", "Since entry", "Price"])
        for p in mv["notable"]:
            stop = (f'<div style="color:{MUTED};font-size:11px;padding-top:2px;">stop {p["stop"]:,.2f}</div>'
                    if p.get("stop") else "")
            rows += ("<tr>" + _cell(f'{_ticker(p["symbol"], links)}{stop}')
                     + _num(_pct(p["day_pct"]), p["day_pct"], bold=True)
                     + _num(_pct(p["total_pct"]), p["total_pct"])
                     + _num(f'{p.get("price", 0):,.2f}') + "</tr>")
        inner = _table(rows)
        if mv["rest"]:
            inner += (f'<div style="font-family:{FONT};font-size:12px;color:{MUTED};padding:8px 10px 0 10px;">'
                      f'{mv["rest"]} other holding{"s" if mv["rest"] != 1 else ""} moved less than 3%.</div>')
        parts.append(_section("Notable moves", inner))

    # News.
    news = ctx.get("news") or []
    if news:
        rows = ""
        for n in news:
            flags = ""
            if n["flags"]:
                flags = (f'<span style="background:#fdecea;color:{NEG};font-size:10px;font-weight:600;'
                         f'padding:2px 6px;border-radius:3px;margin-left:6px;white-space:nowrap;">'
                         f'{escape(", ".join(n["flags"]))}</span>')
            rows += ("<tr>" + _cell(_ticker(n["symbol"], links), width="64px")
                     + _cell(f'{_link(n["headline"][:140], n.get("url"))}{flags}')
                     + _num(f'{n["age_h"]:.0f}h') + "</tr>")
        parts.append(_section("Notable news", _table(rows)))
    else:
        parts.append(_section("Notable news", _empty("no fresh coverage on held names")))

    # Health of the agents themselves, from monitor.py. Missing is shown as missing, never as healthy.
    h = ctx.get("health")
    if h:
        tone = {"healthy": POS, "degraded": "#9a6700", "incident": NEG}.get(h["status"], MUTED)
        rows = (f'<tr><td colspan="2" style="padding:6px 10px;background:{BG};font-family:{FONT};font-size:13px;">'
                f'<span style="color:{tone};font-weight:700;text-transform:uppercase;">{escape(h["status"])}</span>'
                f'<span style="color:{MUTED};"> &middot; {escape(trim_text(h.get("summary", ""), max_sentences=2, max_chars=300))}'
                f'</span></td></tr>')
        if h.get("findings") and h.get("analysis") == "deterministic" and h.get("reviewed_on"):
            rows += ("<tr>" + _cell("", width="52px") + _cell(f'Open findings as of the {escape(h["reviewed_on"])} model '
                                                              f'review (not re-checked today):', muted=True) + "</tr>")
        for f in h.get("findings", [])[:8]:
            sev = f["severity"]
            rows += ("<tr>" + _cell(f'<span style="color:{NEG if sev in ("SEV1", "SEV2") else MUTED};font-weight:600;'
                                    f'white-space:nowrap;">{escape(sev)}</span>', width="52px")
                     + _cell(f'<b>{escape(f["id"])}</b> {escape(f["title"])}'
                             + (f' <span style="color:{MUTED};">({escape(f["trend"])})</span>' if f.get("trend") else "")
                             + (f' <span style="color:{POS};">&middot; owner: {escape(f["resolved"])}, awaiting the next '
                                f'model review</span>' if f.get("resolved") else ""))
                     + "</tr>")
        if h.get("parked"):
            rows += ("<tr>" + _cell("", width="52px")
                     + _cell(f'+ {h["parked"]} parked (no new evidence for 3+ reviews; listed in the full report)', muted=True)
                     + "</tr>")
        for r in h.get("recommendations", [])[:5]:
            rows += ("<tr>" + _cell(f'<span style="color:{MUTED};font-weight:600;">{escape(r["priority"])}</span>', width="52px")
                     + _cell(f'{escape(r["id"])} {escape(r["title"])}'
                             + (f' &middot; owner: {escape(r["resolved"])}' if r.get("resolved") else ""), muted=True)
                     + "</tr>")
        inner = _table(rows)
        if h.get("report"):
            inner += (f'<div style="font-family:{FONT};font-size:11px;color:{MUTED};padding:6px 10px 0 10px;">'
                      f'Full report: {escape(h["report"])}</div>')
        parts.append(_section("System health", inner))

    # Changes to the system itself.
    ch = ctx.get("changes") or []
    if ch:
        items = "".join(
            f'<li style="font-family:{MONO};font-size:12px;color:{INK};padding:3px 0;">{escape(c)}</li>' for c in ch)
        inner = f'<ul style="margin:4px 0 0 0;padding-left:20px;">{items}</ul>'
    else:
        inner = _empty("no change to config, models or code since the last report")
    parts.append(_section("Changes to models, agents and approach", inner))

    # Run notes, de-emphasised.
    notes = ctx.get("notes") or []
    if notes:
        items = "".join(
            f'<li style="font-family:{FONT};font-size:12px;color:{MUTED};padding:2px 0;">{escape(n)}</li>'
            for n in notes[:12])
        parts.append(_section("From today's run", f'<ul style="margin:4px 0 0 0;padding-left:20px;">{items}</ul>'))

    body = "".join(parts)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light only"><meta name="supported-color-schemes" content="light only">
<title>TeamRobo daily {ctx['day'].isoformat()}</title></head>
<body style="margin:0;padding:0;background:{PANEL};">
<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="background:{PANEL};">
<tr><td align="center" style="padding:22px 12px;">
<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="640"
 style="max-width:640px;width:100%;background:{BG};border:1px solid {LINE};border-radius:10px;padding:26px 24px 30px 24px;">
{body}
<tr><td style="padding:26px 0 0 0;border-top:1px solid {LINE};background:{BG};">
<div style="font-family:{FONT};font-size:11px;color:{MUTED};line-height:1.55;">
Paper trading on an Alpaca practice account &mdash; no real money. Not investment advice; the strategy is
unproven. Generated automatically at 14:00 by report_daily.py.</div></td></tr>
</table></td></tr></table></body></html>"""
