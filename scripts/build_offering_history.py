#!/usr/bin/env python
"""Point-in-time history of the filings behind the live dilution block, for the backtest.

The live desk hard-blocks a candidate with an S-1/S-1/A or an 8-K item 3.02 (unregistered equity sale) filed in
the last 90 days (desk/edgar.py offering_filings). It was the most frequent block live (86 brief notes by 10-16:
SNOW, MRVL, DT and others) and was never backtested (engineering review 2026-10-17, item 6).

Each point-in-time member (state/history/pit/members.csv) is mapped to its SEC CIK from the archived
company_tickers.json snapshots the universe build saved (state/history/pit/sec/), using the snapshot nearest its
membership, so delisted names are covered too. Its full EDGAR filing history (the submissions file plus the older
pages it links) is read, and every qualifying filing dated inside the name's membership (plus 90 days before it)
is written to state/history/offerings.csv: date, symbol, cik, form, items.

  python scripts/build_offering_history.py        ~2,600 CIKs at the SEC's 10 requests/second; cached, re-runnable
"""
from __future__ import annotations
import csv, json, logging, sys
from datetime import date, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from desk.config import load_config
from desk.edgar import DILUTION_FORMS, OFFERING_8K_ITEMS, Edgar

log = logging.getLogger("offerings")
WINDOW = timedelta(days=90)


def snapshot_ciks(folder: Path) -> list:
    """[(snapshot date, {ticker: cik})] oldest first."""
    out = []
    for p in sorted(folder.glob("*.json")):
        raw = json.loads(p.read_text())
        rows = raw.values() if isinstance(raw, dict) else raw
        out.append((date(int(p.stem[:4]), int(p.stem[4:6]), int(p.stem[6:8])),
                    {str(v["ticker"]).upper(): int(v["cik_str"]) for v in rows if isinstance(v, dict) and v.get("ticker")}))
    return out


def cik_for(sym: str, first: date, last: date, snaps: list) -> int | None:
    """The CIK the ticker had during its membership: the latest snapshot inside it, else the nearest outside it.
    Tickers are reused (a delisted name's symbol later goes to another company), so the date matters."""
    key = sym.upper().replace(".", "-")
    inside = [m[key] for d, m in snaps if first <= d <= last + timedelta(days=30) and key in m]
    if inside:
        return inside[-1]
    near = sorted(((min(abs((d - first).days), abs((d - last).days)), m[key]) for d, m in snaps if key in m))
    return near[0][1] if near else None


def all_filings(ed: Edgar, cik: int) -> list:
    """Every filing (form, date, items): the 'recent' block and the older pages it lists. Cached for 30 days."""
    raw = json.loads(ed._get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", f"sub_{cik}.json", ttl_hours=24 * 30))
    blocks = [raw["filings"]["recent"]]
    for f in raw["filings"].get("files", []):
        if f.get("filingTo", "9999") >= "2017-09-01":
            blocks.append(json.loads(ed._get(f"https://data.sec.gov/submissions/{f['name']}", f["name"], ttl_hours=24 * 30)))
    out = []
    for b in blocks:
        items = b.get("items") or [""] * len(b["form"])
        out += [(b["form"][i], b["filingDate"][i], items[i] or "") for i in range(len(b["form"]))]
    return out


def qualifies(form: str, items: str) -> bool:
    return form in DILUTION_FORMS or (form in ("8-K", "8-K/A") and bool(OFFERING_8K_ITEMS & {i.strip() for i in items.split(",")}))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = load_config()
    root = Path(cfg["_root"]); pit = root / "state" / "history" / "pit"
    members = pd.read_csv(pit / "members.csv")
    snaps = snapshot_ciks(pit / "sec")
    ed = Edgar(pit / "edgar")
    rows, unmapped, failed = [], [], []
    for k, m in enumerate(members.itertuples(), 1):
        first, last = date.fromisoformat(m.first_week), date.fromisoformat(m.last_week) + timedelta(days=7)
        cik = cik_for(m.symbol, first, last, snaps)
        if cik is None:
            unmapped.append(m.symbol)
            continue
        try:
            fl = all_filings(ed, cik)
        except Exception as e:
            failed.append(f"{m.symbol}:{type(e).__name__}")
            continue
        for form, d, items in fl:
            fd = date.fromisoformat(d)
            if first - WINDOW <= fd <= last and qualifies(form, items):
                rows.append({"date": d, "symbol": m.symbol, "cik": cik, "form": form, "items": items})
        if k % 200 == 0:
            log.info("%d/%d members, %d filings so far", k, len(members), len(rows))
    out = root / "state" / "history" / "offerings.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "symbol", "cik", "form", "items"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["date"], r["symbol"])))
    meta = {"members": len(members), "mapped": len(members) - len(unmapped), "unmapped": len(unmapped),
            "failed": len(failed), "filings": len(rows), "unmapped_examples": unmapped[:30], "failed_examples": failed[:20]}
    (root / "state" / "history" / "offerings_build.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if not k.endswith("examples")}))


if __name__ == "__main__":
    main()
