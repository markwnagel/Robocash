from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, OrderStatus, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (GetOrdersRequest, LimitOrderRequest, StopLossRequest, StopOrderRequest,
                                     TakeProfitRequest)

from .state import record_trade

log = logging.getLogger("desk.exec")
DEAD_ORDER_STATUSES = {"expired", "canceled", "rejected", "replaced", "done_for_day", "stopped", "suspended"}
LOOKBACK_DAYS = 90  # order-history window when looking for a position's live exit legs


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
        }

    def positions(self) -> dict:
        out = {}
        for p in self.t.get_all_positions():
            out[p.symbol] = {
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price or 0),
                "market_value": float(p.market_value or 0),
                "unrealized_plpc": float(p.unrealized_plpc or 0),
            }
        return out

    def is_trading_day(self, day: date) -> bool:
        from alpaca.trading.requests import GetCalendarRequest
        cal = self.t.get_calendar(GetCalendarRequest(start=day, end=day))
        return any(c.date == day for c in cal)

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


def sync_positions(state: dict, broker: Broker, live: dict, today: date) -> list:
    """Reconcile state['positions'] with the broker. Returns human-readable notes."""
    notes = []
    st = state["positions"]
    for sym in list(st.keys()):
        pos = st[sym]
        if sym in live:
            if not pos.get("filled"):
                # First sight of the fill. entry_date is when the order was staged (the evening before), so
                # the day-trade guard needs this to know which positions were bought today.
                pos["filled_date"] = today.isoformat()
            pos["filled"] = True
            pos["entry_price"] = live[sym]["avg_entry_price"]
            pos["qty"] = live[sym]["qty"]
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
            # An unfilled DAY order that has expired must free its slot the same evening, not two days later.
            status = broker.order_status(pos.get("order_id", "")) if not broker.dry_run else None
            dead = status in DEAD_ORDER_STATUSES or (status is None and (today - entry).days >= 2)
            if dead:
                notes.append(f"{sym} entry order never filled ({status or 'stale'}); dropped")
                del st[sym]
    for sym, p in live.items():
        if sym not in st:
            st[sym] = {
                "entry_date": today.isoformat(), "entry_price": p["avg_entry_price"], "qty": p["qty"],
                "stop": None, "target": None, "filled": True, "adopted": True, "thesis": "adopted (manual position)",
            }
            notes.append(f"{sym} adopted from broker (no bracket; time-exit only)")
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
    })
