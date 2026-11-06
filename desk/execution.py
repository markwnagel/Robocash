from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, OrderStatus, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (GetOrdersRequest, LimitOrderRequest, MarketOrderRequest, StopLossRequest, StopOrderRequest,
                                     TakeProfitRequest)

from . import telemetry
from .state import LEGACY_RELEASE, record_trade

log = logging.getLogger("desk.exec")
DEAD_ORDER_STATUSES = {"expired", "canceled", "rejected", "replaced", "done_for_day", "stopped", "suspended"}
LOOKBACK_DAYS = 90  # order-history window when looking for a position's live exit legs


def raw_positions(trading: TradingClient) -> list:
    """Open positions as plain JSON, not get_all_positions(). The SDK checks each position's exchange against
    a fixed list, so one holding on an exchange it does not know (TXSE, 2026-10-19 - monitor F-39) would fail
    the whole call, and with it the stop checks and the daily report. Nothing here reads the exchange."""
    return trading.get("/positions") or []


class Broker:
    def __init__(self, trading: TradingClient, dry_run: bool = False):
        self.t = trading
        self.dry_run = dry_run

    def account(self) -> dict:
        a = self.t.get_account()
        return {
            "equity": float(a.equity),
            "last_equity": float(a.last_equity),
            "cash": float(a.cash),
            "non_marginable_buying_power": float(a.non_marginable_buying_power or a.cash),
            "daytrade_count": int(a.daytrade_count or 0),
            "pattern_day_trader": bool(a.pattern_day_trader),
            "trading_blocked": bool(a.trading_blocked or a.account_blocked),
            # Which Alpaca account the keys reach: state.json belongs to one (scripts/switch_account.py).
            "account_number": str(getattr(a, "account_number", "") or ""),
        }

    def positions(self) -> dict:
        out = {}
        for p in raw_positions(self.t):
            out[p["symbol"]] = {
                "qty": float(p["qty"]),
                "avg_entry_price": float(p["avg_entry_price"]),
                "current_price": float(p.get("current_price") or 0),
                "market_value": float(p.get("market_value") or 0),
                "unrealized_plpc": float(p.get("unrealized_plpc") or 0),
            }
        return out

    def is_trading_day(self, day: date) -> bool:
        from alpaca.trading.requests import GetCalendarRequest
        cal = self.t.get_calendar(GetCalendarRequest(start=day, end=day))
        return any(c.date == day for c in cal)

    def nth_trading_day_after(self, day: date, n: int) -> date:
        """The n-th session after `day` on the exchange calendar (holidays skipped)."""
        from datetime import timedelta
        from alpaca.trading.requests import GetCalendarRequest
        cal = self.t.get_calendar(GetCalendarRequest(start=day + timedelta(days=1),
                                                    end=day + timedelta(days=2 * n + 14)))
        days = sorted(c.date for c in cal if c.date > day)
        if len(days) < n:
            raise ValueError(f"calendar returned {len(days)} sessions after {day}, need {n}")
        return days[n - 1]

    def market_open(self) -> bool:
        """Whether the regular session is open right now. Unknown reads as open: the callers use it to be
        more careful (day-trade guard, partial-bar warning), never less."""
        try:
            return bool(self.t.get_clock().is_open)
        except Exception as e:
            log.warning("market clock unavailable, assuming the session is open: %s", e)
            return True

    def cancel_symbol_orders(self, symbol: str, side: OrderSide | None = None) -> int:
        orders = self.t.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))
        n = 0
        for o in orders:
            if o.symbol != symbol or (side is not None and o.side != side):
                continue
            if self.dry_run:
                log.info("[dry] cancel %s %s", symbol, o.id)
            else:
                self.t.cancel_order_by_id(o.id)
            n += 1
        if n and not self.dry_run:
            time.sleep(1.5)
        return n

    def close(self, symbol: str) -> None:
        self.cancel_symbol_orders(symbol)
        if self.dry_run:
            log.info("[dry] close %s", symbol)
            return
        self.t.close_position(symbol)

    def liquidate_all(self) -> None:
        if self.dry_run:
            log.info("[dry] LIQUIDATE ALL")
            return
        self.t.cancel_orders()
        time.sleep(1.5)
        self.t.close_all_positions(cancel_orders=True)

    def submit_bracket(self, symbol: str, qty: int, limit: float, stop: float, target: float | None) -> str:
        """Limit entry with a broker-side stop; a take-profit leg too when target is given (bracket),
        otherwise a one-triggers-other order so the stop can be trailed by replacing it."""
        kw = dict(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.DAY,
                  limit_price=round(limit, 2), stop_loss=StopLossRequest(stop_price=round(stop, 2)))
        if target is not None:
            req = LimitOrderRequest(order_class=OrderClass.BRACKET, take_profit=TakeProfitRequest(limit_price=round(target, 2)), **kw)
        else:
            req = LimitOrderRequest(order_class=OrderClass.OTO, **kw)
        if self.dry_run:
            log.info("[dry] BUY %s x%d lim %.2f stop %.2f tgt %s", symbol, qty, limit, stop, f"{target:.2f}" if target else "trail")
            return "dry-run"
        o = self.t.submit_order(req)
        return str(o.id)

    def live_exit_orders(self, symbol: str) -> dict:
        """{'stop', 'target', 'close'}: the sell orders that can still fire (order or None each). 'close' is a
        working market sell - a close sent earlier that has not filled yet.
        Status OPEN does NOT include 'held', which is where a resting stop leg sits, so query ALL."""
        try:
            orders = self.t.get_orders(GetOrdersRequest(
                status=QueryOrderStatus.ALL, symbols=[symbol], nested=False, limit=200,
                after=datetime.now() - timedelta(days=LOOKBACK_DAYS)))
        except Exception as e:
            log.warning("live_exit_orders %s: %s", symbol, e)
            return {"stop": None, "target": None, "close": None, "error": True}
        live = [o for o in orders if o.symbol == symbol and o.side == OrderSide.SELL
                and o.status.value not in DEAD_ORDER_STATUSES and o.status.value != "filled"]
        return {"stop": next((o for o in live if o.stop_price), None),
                "target": next((o for o in live if o.limit_price and not o.stop_price), None),
                "close": next((o for o in live if not o.limit_price and not o.stop_price), None), "error": False}

    def sell_market(self, symbol: str, qty: int) -> str:
        """A DAY market sell; sent after the close it fills at the next open. Used for trims - the holding's
        stop order has to be cancelled first, since it reserves every share."""
        if self.dry_run:
            log.info("[dry] SELL %s x%d market", symbol, qty)
            return "dry-run"
        return str(self.t.submit_order(MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL,
                                                          time_in_force=TimeInForce.DAY)).id)

    def submit_exit_protection(self, symbol: str, qty: int, stop: float, target: float | None) -> str:
        """A GTC stop (paired with a target as OCO when there is one) that survives the session.
        Bracket legs inherit the entry's DAY tif and die at the close of the day the entry fills."""
        if target is not None:
            req = LimitOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC,
                                    order_class=OrderClass.OCO,
                                    take_profit=TakeProfitRequest(limit_price=round(target, 2)),
                                    stop_loss=StopLossRequest(stop_price=round(stop, 2)))
        else:
            req = StopOrderRequest(symbol=symbol, qty=qty, side=OrderSide.SELL,
                                   time_in_force=TimeInForce.GTC, stop_price=round(stop, 2))
        if self.dry_run:
            log.info("[dry] protect %s x%d stop %.2f tgt %s", symbol, qty, stop, f"{target:.2f}" if target else "-")
            return "dry-run"
        return str(self.t.submit_order(req).id)

    def raise_stop(self, symbol: str, new_stop: float) -> float | None:
        """Ratchet the resting sell-stop up to new_stop. Returns the stop price now protecting the
        position at the broker, or None if nothing is. When the broker is already higher its own level
        is returned, not new_stop - reporting success with the lower number would write a stop into
        state that is below the one actually resting."""
        from alpaca.trading.requests import ReplaceOrderRequest
        new_stop = round(new_stop, 2)
        leg = self.live_exit_orders(symbol)["stop"]
        if leg is None:
            return None
        current = float(leg.stop_price)
        if current >= new_stop:
            return current
        if self.dry_run:
            log.info("[dry] raise stop %s %s -> %.2f", symbol, leg.stop_price, new_stop)
        else:
            self.t.replace_order_by_id(leg.id, ReplaceOrderRequest(stop_price=new_stop))
        return new_stop

    def move_exit_levels(self, symbol: str, stop: float, target: float | None) -> None:
        """Re-price the resting stop (and target) legs, in either direction. raise_stop only ratchets up;
        this is for a leg still resting at the staged level after anchor_to_fill moved the position's."""
        from alpaca.trading.requests import ReplaceOrderRequest
        legs = self.live_exit_orders(symbol)
        if legs.get("error") or legs["stop"] is None:
            raise RuntimeError(f"no readable stop leg for {symbol}")
        if self.dry_run:
            log.info("[dry] move exits %s -> stop %.2f tgt %s", symbol, stop, f"{target:.2f}" if target else "-")
            return
        self.t.replace_order_by_id(legs["stop"].id, ReplaceOrderRequest(stop_price=round(stop, 2)))
        if target is not None and legs["target"] is not None:
            self.t.replace_order_by_id(legs["target"].id, ReplaceOrderRequest(limit_price=round(target, 2)))

    def order_status(self, order_id: str) -> str | None:
        """None when the order cannot be read; the caller then waits two days before giving up on it."""
        try:
            return self.t.get_order_by_id(order_id).status.value
        except Exception as e:
            log.warning("order %s status unavailable: %s", order_id, e)
            return None

    def order_fill(self, order_id: str) -> tuple:
        """(status, filled_avg_price or None)."""
        try:
            o = self.t.get_order_by_id(order_id)
            return o.status.value, (float(o.filled_avg_price) if o.filled_avg_price else None)
        except Exception as e:
            log.warning("order %s fill unavailable: %s", order_id, e)
            return None, None

    def entry_fill(self, order_id: str) -> dict:
        """{status, qty, price, filled_at} for an entry order; empty values when it cannot be read."""
        try:
            o = self.t.get_order_by_id(order_id)
            return {"status": o.status.value, "qty": float(o.filled_qty or 0),
                    "price": float(o.filled_avg_price) if o.filled_avg_price else None, "filled_at": o.filled_at}
        except Exception as e:
            log.warning("order %s fill unavailable: %s", order_id, e)
            return {"status": None, "qty": 0.0, "price": None, "filled_at": None}

    def last_sell_fill(self, symbol: str, since: date) -> float | None:
        orders = self.t.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.CLOSED, symbols=[symbol],
            after=datetime(since.year, since.month, since.day), limit=100,
        ))
        fills = [o for o in orders if o.symbol == symbol and o.side == OrderSide.SELL
                 and o.status == OrderStatus.FILLED and o.filled_avg_price]
        if not fills:
            return None
        fills.sort(key=lambda o: o.filled_at or datetime.min)
        return float(fills[-1].filled_avg_price)


def anchor_to_fill(pos: dict, sym: str, fill: float) -> str | None:
    """Move a new holding's stop and target to the fill, the same distances as staged. The bracket is priced
    from the signal-day close because the order goes in before the fill is known, so an entry that opened low
    carried a nearer stop than stop_atr_multiple x ATR (NTAP/FTNT/OKTA at 74-83%, all stopped within three
    sessions - audit 2026-10-17). The backtest measures from the fill, and point-in-time that beat the signal
    anchor on return in 3/4 starts in both windows. The bracket's own legs are DAY orders that expire at the
    close, so the GTC protection placed this evening is the first to use these levels. Returns a brief note."""
    ref, stop = pos.get("entry_price"), pos.get("stop")
    if pos.get("adopted") or stop is None or not ref:
        return None
    ref, stop = float(ref), float(stop)
    dist = ref - stop
    if dist <= 0 or abs(fill - ref) < 0.005:
        return None
    new_stop = round(fill - dist, 2)
    new_target = round(fill + (float(pos["target"]) - ref), 2) if pos.get("target") else None
    note = (f"{sym} filled at {fill:.2f} vs {ref:.2f} staged: stop {stop:.2f} -> {new_stop:.2f}"
            + (f", target {float(pos['target']):.2f} -> {new_target:.2f}" if new_target is not None else "")
            + " (measured from the fill, as tested)")
    pos.update(stop=new_stop, target=new_target, signal_price=ref, staged_stop=stop, staged_target=pos.get("target"),
               anchored_to_fill=True, reanchor_pending=True)
    return note


def sync_positions(state: dict, broker: Broker, live: dict, today: date) -> list:
    """Reconcile state['positions'] with the broker. Returns human-readable notes."""
    notes = []
    st = state["positions"]
    for sym in list(st.keys()):
        pos = st[sym]
        if sym in live:
            first_fill = not pos.get("filled")
            if first_fill:
                # First sight of the fill. entry_date is when the order was staged (the evening before), so
                # the day-trade guard needs this to know which positions were bought today.
                pos["filled_date"] = today.isoformat()
                staged_qty, filled_qty = float(pos.get("qty") or 0), float(live[sym]["qty"])
                shares = f"{filled_qty:g} of {staged_qty:g} shares filled"
                moved = anchor_to_fill(pos, sym, float(live[sym]["avg_entry_price"]))
                if moved:
                    notes.append(f"{moved}; {shares}")       # price and size, so a partial fill reads as one
                if staged_qty and filled_qty < staged_qty:
                    notes.append(f"{sym} partial fill: {shares}")
            pos["filled"] = True
            pos["entry_price"] = live[sym]["avg_entry_price"]
            pos["qty"] = live[sym]["qty"]
            if first_fill and not broker.dry_run:
                telemetry.order_event("entry_filled", sym, order_id=pos.get("order_id"), staged=pos.get("entry_date"),
                                      qty=pos["qty"], price=pos["entry_price"])
            if pos.get("pending_exit_reason") and broker.live_exit_orders(sym).get("close") is not None:
                # The close sent earlier is still working (a second run the same evening, or a sale queued
                # for the open): the position is exiting. Re-evaluating it would re-place a stop beside the
                # queued sale, and whichever fills second would sell shares the account no longer has.
                notes.append(f"{sym} close still working at the broker ({pos['pending_exit_reason']})")
                continue
            # The position was told to close on an earlier run and is still here, so that close did not
            # complete (unfilled market sell, halt, partial fill, rejection). The flag has to go: while it
            # is set the position is skipped by the exit, trailing-stop AND stop-re-placement loops, so it
            # would sit unprotected and un-exited forever.
            stale = pos.pop("pending_exit_reason", None)
            if stale:
                notes.append(f"{sym} did not close on the earlier instruction ({stale}); "
                             f"re-evaluating its exit and protection today")
            if pos.get("add_order_id"):
                status, px = broker.order_fill(pos["add_order_id"]) if not broker.dry_run else (None, None)
                if px:  # filled or partially filled before expiring: it is an add either way
                    pos["adds"] = int(pos.get("adds") or 0) + 1
                    pos["last_add_price"] = px
                    notes.append(f"{sym} pyramid add filled at ~{px:.2f} (add {pos['adds']}, now {pos['qty']:.0f} sh)")
                if px or status in DEAD_ORDER_STATUSES or status == "filled":
                    pos.pop("add_order_id", None)
                elif status is None and (today - date.fromisoformat(pos.get("add_date", pos["entry_date"]))).days >= 2:
                    pos.pop("add_order_id", None)
            continue
        entry = date.fromisoformat(pos["entry_date"])
        if pos.get("filled"):
            exit_price = None
            if not broker.dry_run:
                try:
                    exit_price = broker.last_sell_fill(sym, entry)
                except Exception as e:
                    log.warning("last sell fill %s unavailable, logging the last mark: %s", sym, e)
            exit_price = exit_price or pos.get("last_price") or pos["entry_price"]
            if not broker.dry_run:
                _log_close(pos, sym, exit_price, today, pos.get("pending_exit_reason", "bracket/stop/target"))
                telemetry.order_event("position_closed", sym, price=exit_price, qty=pos.get("qty"),
                                      reason=pos.get("pending_exit_reason", "bracket/stop/target"))
            notes.append(f"{sym} closed at ~{exit_price:.2f} ({pos.get('pending_exit_reason', 'bracket')})")
            # A sell order left resting on a name no longer held (a stop beside a manual close, the other
            # half of an OCO the broker did not cancel) would open a SHORT position when it fires.
            try:
                n = broker.cancel_symbol_orders(sym, side=OrderSide.SELL)
                if n:
                    notes.append(f"{sym}: {n} leftover sell order(s) cancelled - nothing is held to sell")
            except Exception as e:
                log.warning("leftover orders for %s not cancelled: %s", sym, e)
                notes.append(f"MANUAL ACTION: {sym} is closed but its leftover sell orders could not be checked "
                             f"({type(e).__name__}) - cancel any at the broker, or a stop could open a short")
            del st[sym]
        else:
            fill = broker.entry_fill(pos.get("order_id", "")) if not broker.dry_run else {"status": None}
            status = fill["status"]
            if fill.get("price") and fill.get("qty"):
                # Bought and sold between two runs: the entry filled at the open and the bracket's stop (or
                # target) sold it before this run, so the name was never seen held. It stayed a pending entry
                # with a filled order - never dropped, holding a slot, its loss missing from the trade log
                # (TXG, 2026-10-06: bought 98.18 at 09:51 ET, stopped 85.85 at 10:51 ET).
                since = fill["filled_at"].date() if fill.get("filled_at") else entry
                exit_price = None
                try:
                    exit_price = broker.last_sell_fill(sym, since)
                except Exception as e:
                    log.warning("last sell fill %s unavailable: %s", sym, e)
                if exit_price is None:
                    # Left as a pending entry, untouched, until the sale is readable. Marking it filled now would
                    # send it through protect(), which would place a sell stop for shares the account does not
                    # hold - on a margin account, an order that opens a short.
                    notes.append(f"{sym} entry filled at {fill['price']:.2f} but is not held and its sale is not "
                                 f"readable yet - checked again next run")
                    continue
                pos.update(filled=True, filled_date=today.isoformat(), entry_price=fill["price"], qty=fill["qty"])
                telemetry.order_event("entry_filled", sym, order_id=pos.get("order_id"), staged=pos["entry_date"],
                                      qty=fill["qty"], price=fill["price"])
                reason = "bracket/stop/target (same day)"
                _log_close(pos, sym, exit_price, today, reason)
                telemetry.order_event("position_closed", sym, price=exit_price, qty=fill["qty"], reason=reason)
                notes.append(f"{sym} bought at {fill['price']:.2f} and sold at {exit_price:.2f} before this run "
                             f"({(exit_price / fill['price'] - 1):+.1%}, {fill['qty']:.0f} sh) - recorded as a closed trade")
                try:
                    n = broker.cancel_symbol_orders(sym, side=OrderSide.SELL)
                    if n:
                        notes.append(f"{sym}: {n} leftover sell order(s) cancelled - nothing is held to sell")
                except Exception as e:
                    log.warning("leftover orders for %s not cancelled: %s", sym, e)
                    notes.append(f"MANUAL ACTION: {sym} is closed but its leftover sell orders could not be checked "
                                 f"({type(e).__name__}) - cancel any at the broker, or a stop could open a short")
                del st[sym]
                continue
            # An unfilled DAY order that has expired must free its slot the same evening, not two days later.
            dead = status in DEAD_ORDER_STATUSES or (status is None and (today - entry).days >= 2)
            if dead:
                # Name the order: the same run can stage a new entry for the name, and on 2026-09-29 "MXL entry
                # order never filled" beside "BUY MXL" read as one order that later filled (monitor F-23).
                notes.append(f"{sym} entry order never filled ({status or 'stale'}; staged {pos['entry_date']}); dropped")
                telemetry.order_event("entry_expired", sym, order_id=pos.get("order_id"), staged=pos["entry_date"],
                                      status=status or "stale")
                del st[sym]
    for sym, p in live.items():
        if sym not in st:
            st[sym] = {
                "entry_date": today.isoformat(), "entry_price": p["avg_entry_price"], "qty": p["qty"],
                "stop": None, "target": None, "filled": True, "adopted": True, "thesis": "adopted (manual position)",
            }
            notes.append(f"{sym} adopted from broker (no bracket; time-exit only)")
            telemetry.order_event("position_adopted", sym, qty=p["qty"], price=p["avg_entry_price"])
    for sym, p in live.items():
        st[sym]["last_price"] = p["current_price"]
    return notes


def _log_close(pos: dict, sym: str, exit_price: float, today: date, reason: str) -> None:
    entry_price = float(pos["entry_price"])
    qty = float(pos.get("qty") or 0)
    entry = date.fromisoformat(pos["entry_date"])
    record_trade({
        "symbol": sym, "entry_date": pos["entry_date"], "entry_price": entry_price,
        "exit_date": today.isoformat(), "exit_price": exit_price, "qty": qty,
        "pnl": round((exit_price - entry_price) * qty, 2),
        "pnl_pct": round(exit_price / entry_price - 1, 4) if entry_price else 0,
        "hold_days": (today - entry).days, "exit_reason": reason, "thesis": pos.get("thesis", ""),
        "stop": pos.get("stop"), "target": pos.get("target"),
        # A position opened before release tracking was held across the v1 -> v2 cut: v1's pick, closed under v2.
        "release": pos.get("release") or f"{LEGACY_RELEASE} (carried)",
    })
