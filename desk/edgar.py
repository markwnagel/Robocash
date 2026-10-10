from __future__ import annotations

import gzip
import json
import logging
import os
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path

log = logging.getLogger("desk.edgar")

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
# Unambiguous dilution: 8-K 3.02 is an unregistered equity sale, S-1 for an already-trading company is
# typically a resale/dilution registration. These hard-block a candidate.
DILUTION_FORMS = {"S-1", "S-1/A"}
OFFERING_8K_ITEMS = {"3.02"}
# Ambiguous: a shelf takedown may be debt or equity, and a well-known seasoned issuer files S-3ASR as
# routine housekeeping. Form type alone cannot tell them apart, so these are flagged for the analyst.
SHELF_FORMS = {"424B1", "424B2", "424B3", "424B4", "424B5", "424B7", "S-3", "S-3/A", "S-3ASR", "F-1", "F-3", "F-3ASR"}


class Edgar:
    """Polite EDGAR client: SEC requires a contact User-Agent and <= 10 requests/second."""

    def __init__(self, cache_dir: Path, user_agent: str | None = None):
        ua = user_agent or os.environ.get("SEC_USER_AGENT", "")
        if "@" not in ua:
            raise RuntimeError("SEC_USER_AGENT must be set to 'Your Name your@email' (SEC fair-access policy)")
        self.ua = ua
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._last = 0.0
        self._tickers = None

    def _get(self, url: str, cache_key: str, ttl_hours: float) -> bytes:
        p = self.cache / cache_key
        if p.exists() and (time.time() - p.stat().st_mtime) < ttl_hours * 3600:
            return p.read_bytes()
        wait = 0.11 - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": self.ua, "Accept-Encoding": "gzip"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                data = gzip.decompress(data)
        self._last = time.time()
        p.write_bytes(data)
        return data

    def cik(self, symbol: str) -> int | None:
        if self._tickers is None:
            raw = json.loads(self._get(TICKERS_URL, "company_tickers.json", ttl_hours=24 * 7))
            self._tickers = {v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()}
        return self._tickers.get(symbol.upper().replace(".", "-"))

    def filings(self, cik: int, since: date) -> list:
        raw = json.loads(self._get(SUBMISSIONS_URL.format(cik=cik), f"sub_{cik}.json", ttl_hours=12))
        rec = raw["filings"]["recent"]
        items = rec.get("items") or [""] * len(rec["form"])
        out = []
        for i, form in enumerate(rec["form"]):
            fd = date.fromisoformat(rec["filingDate"][i])
            if fd < since:
                continue
            out.append({"form": form, "date": fd, "accession": rec["accessionNumber"][i],
                        "items": items[i] or "", "doc": rec["primaryDocument"][i]})
        return out

    def form4_activity(self, cik: int, filings: list, since: date) -> dict:
        tot = {"buys": 0.0, "sells": 0.0, "net_usd": 0.0, "n": 0, "transactions": []}
        seen = set()
        for f in filings:
            if f["form"] not in ("4", "4/A") or f["date"] < since:
                continue
            acc = f["accession"].replace("-", "")
            doc = f["doc"].split("/", 1)[1] if f["doc"].startswith("xsl") else f["doc"]
            try:
                xml = self._get(ARCHIVE_URL.format(cik=cik, acc=acc, doc=doc), f"f4_{acc}.xml", ttl_hours=24 * 365)
                parsed = parse_form4(xml)
            except Exception as e:  # one bad filing must not sink the fact sheet
                log.warning("form4 %s: %s", acc, e)
                continue
            for t in parsed["transactions"]:
                # A 4/A restates the original filing's rows, and joint filers report one trade twice.
                # Identical rows are one economic trade. (A 4/A that corrects the numbers cannot be matched
                # to its original from the index, so that rarer case can still count twice.)
                row = (t.get("date"), t["code"], t.get("shares"), t.get("price"))
                if row in seen:
                    continue
                seen.add(row)
                tot["transactions"].append({"filed": f["date"].isoformat(), **t})
        tx = tot["transactions"]
        tot["buys"] = sum(t["usd"] for t in tx if t["code"] == "P")
        tot["sells"] = sum(t["usd"] for t in tx if t["code"] == "S")
        tot["n"] = len(tx)
        tot["net_usd"] = tot["buys"] - tot["sells"]
        return tot


def parse_form4(xml_bytes: bytes) -> dict:
    root = ET.fromstring(xml_bytes)
    # The 10b5-1 checkbox (Forms filed from April 2023): trades pre-scheduled under a plan say little about
    # what an insider thinks now, and a large holder's plan sales can swamp every other row.
    plan = (root.findtext("aff10b5One") or "").strip().lower() in ("1", "true")
    # A 10% holder that is not an officer - usually a fund or sponsor selling down a stake, often a director
    # only "by deputization" - is not insider sentiment: Silver Lake's exit from DELL read as -$879M of
    # "insider selling" in 30 days against ~$60M by its executives. A founder-CEO stays an insider.
    flags = lambda r, k: (r.findtext(k) or "").strip().lower() in ("1", "true")
    rels = [r for r in root.iter("reportingOwnerRelationship")]
    holder = any(flags(r, "isTenPercentOwner") for r in rels) and not any(flags(r, "isOfficer") for r in rels)
    buys = sells = 0.0
    n = 0
    txs = []
    for tx in root.iter("nonDerivativeTransaction"):
        code = (tx.findtext("transactionCoding/transactionCode") or "").strip()
        if code not in ("P", "S"):
            continue
        shares = float(tx.findtext("transactionAmounts/transactionShares/value") or 0)
        price = float(tx.findtext("transactionAmounts/transactionPricePerShare/value") or 0)
        usd = shares * price
        if code == "P":
            buys += usd
        else:
            sells += usd
        n += 1
        txs.append({"code": code, "usd": round(usd, 2), "price": price, "shares": shares,
                    "date": tx.findtext("transactionDate/value"), "plan": plan, "holder": holder})
    return {"buys": buys, "sells": sells, "net_usd": buys - sells, "n": n, "transactions": txs}


def offering_filings(filings: list) -> list:
    """Hard-block tier: filings that unambiguously mean equity dilution."""
    hits = []
    for f in filings:
        if f["form"] in DILUTION_FORMS:
            hits.append(f"{f['form']} {f['date'].isoformat()}")
        elif f["form"] in ("8-K", "8-K/A") and OFFERING_8K_ITEMS & set(i.strip() for i in f["items"].split(",")):
            hits.append(f"8-K item 3.02 {f['date'].isoformat()}")
    return hits


def shelf_filings(filings: list) -> list:
    """Analyst-flag tier: shelf registrations and takedowns, which may be debt or equity."""
    return [f"{f['form']} {f['date'].isoformat()}" for f in filings if f["form"] in SHELF_FORMS]


REGISTRATION_FORMS = {"S-3", "S-3/A", "S-3ASR", "F-3", "F-3ASR"}
SUPPLEMENT_FORMS = {"424B2", "424B3", "424B5", "424B7"}


def standing_shelf(filings: list, today: date) -> dict:
    """An issuer's standing ability to sell shares, which a 90-day window misses: a shelf registration lasts
    three years, and an at-the-market program sells under a prospectus supplement filed once, with no further
    filing per sale. ASST on 2026-10-09 had an S-3ASR (2025-09-15) and 424B5s (2026-06-05, 06-08), and the
    fact sheet said no shelf - the red team found the ATM on the web instead (monitor F-31)."""
    regs = sorted((f for f in filings if f["form"] in REGISTRATION_FORMS and (today - f["date"]).days <= 3 * 365),
                  key=lambda f: f["date"], reverse=True)
    sups = sorted((f for f in filings if f["form"] in SUPPLEMENT_FORMS and (today - f["date"]).days <= 365),
                  key=lambda f: f["date"], reverse=True)
    return {"shelf_registration": f"{regs[0]['form']} {regs[0]['date'].isoformat()}" if regs else None,
            "prospectus_supplements_12m": [f"{f['form']} {f['date'].isoformat()}" for f in sups[:5]],
            "prospectus_supplements_12m_n": len(sups)}


def recent_8k_items(filings: list) -> list:
    return [f"{f['date'].isoformat()}: {f['items']}" for f in filings if f["form"] in ("8-K", "8-K/A") and f["items"]]
