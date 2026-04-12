"""Thin Alpaca REST broker client for live/paper execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class AlpacaCredentials:
    """Environment-backed credentials for Alpaca."""

    api_key_id: str
    api_secret_key: str
    base_url: str
    timeout_seconds: float = 20.0

    @staticmethod
    def from_env(*, paper: bool) -> "AlpacaCredentials":
        key = os.getenv("ALPACA_API_KEY_ID", "").strip()
        secret = os.getenv("ALPACA_API_SECRET_KEY", "").strip()
        if not key or not secret:
            raise RuntimeError("Missing ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY environment variables.")
        env_url = os.getenv("ALPACA_BASE_URL", "").strip()
        if env_url:
            base = env_url
        else:
            base = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        return AlpacaCredentials(api_key_id=key, api_secret_key=secret, base_url=base)


@dataclass(frozen=True)
class AlpacaAccount:
    equity: float
    buying_power: float
    pattern_day_trader: bool
    trading_blocked: bool


@dataclass(frozen=True)
class AlpacaPosition:
    """Normalized position snapshot."""

    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float
    current_price: float
    side: str

    @property
    def signed_qty(self) -> float:
        return self.qty if self.side.lower() == "long" else -self.qty


@dataclass(frozen=True)
class ProposedOrder:
    symbol: str
    side: str
    qty: float
    notional: float
    reason: str
    dry_run: bool
    response: dict[str, Any] | None = None


class AlpacaBroker:
    """Minimal Alpaca REST broker wrapper."""

    def __init__(self, creds: AlpacaCredentials) -> None:
        self.creds = creds
        self._session = requests.Session()
        self._session.headers.update(
            {
                "APCA-API-KEY-ID": creds.api_key_id,
                "APCA-API-SECRET-KEY": creds.api_secret_key,
                "Content-Type": "application/json",
            }
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.creds.base_url.rstrip('/')}{path}"
        response = self._session.request(
            method=method.upper(),
            url=url,
            params=params,
            json=payload,
            timeout=self.creds.timeout_seconds,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"Alpaca API error {response.status_code}: {response.text}")
        if not response.text:
            return None
        return response.json()

    def get_account(self) -> AlpacaAccount:
        raw = self._request("GET", "/v2/account")
        return AlpacaAccount(
            equity=float(raw.get("equity", 0.0)),
            buying_power=float(raw.get("buying_power", 0.0)),
            pattern_day_trader=bool(raw.get("pattern_day_trader", False)),
            trading_blocked=bool(raw.get("trading_blocked", False)),
        )

    def get_clock(self) -> dict[str, Any]:
        return self._request("GET", "/v2/clock")

    def get_latest_quotes(self, symbols: list[str] | tuple[str, ...] | set[str]) -> dict[str, float]:
        syms = sorted({str(s).upper() for s in symbols if str(s).strip()})
        if not syms:
            return {}
        out: dict[str, float] = {}
        chunk = 150
        for i in range(0, len(syms), chunk):
            sub = syms[i : i + chunk]
            data = self._request("GET", "/v2/stocks/quotes/latest", params={"symbols": ",".join(sub)})
            quotes = data.get("quotes", {}) if isinstance(data, dict) else {}
            for sym in sub:
                row = quotes.get(sym) or {}
                ask = row.get("ap")
                bid = row.get("bp")
                if ask is not None and float(ask) > 0:
                    out[sym] = float(ask)
                elif bid is not None and float(bid) > 0:
                    out[sym] = float(bid)
        return out

    def get_positions(self) -> dict[str, float]:
        raw = self._request("GET", "/v2/positions")
        out: dict[str, float] = {}
        for row in raw:
            symbol = str(row.get("symbol", "")).upper()
            qty = float(row.get("qty", 0.0))
            side = str(row.get("side", "long")).lower()
            signed = qty if side == "long" else -qty
            if symbol:
                out[symbol] = signed
        return out

    def list_open_orders(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v2/orders", params={"status": "open"})

    def cancel_all_orders(self) -> None:
        self._request("DELETE", "/v2/orders")

    def submit_market_order(
        self,
        *,
        symbol: str,
        qty: float,
        side: str,
        time_in_force: str = "day",
    ) -> dict[str, Any]:
        if qty <= 0:
            raise ValueError("Order qty must be positive.")
        payload = {
            "symbol": str(symbol).upper(),
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": time_in_force,
        }
        return self._request("POST", "/v2/orders", payload=payload)


def submit_target_orders(
    *,
    broker: AlpacaBroker,
    target_weights: dict[str, float],
    equity: float,
    dry_run: bool,
    max_notional_per_order: float = 50_000.0,
    min_notional_to_trade: float = 200.0,
) -> list[ProposedOrder]:
    """Convert target weights into incremental market orders."""
    if equity <= 0:
        raise ValueError("Equity must be positive.")
    positions = broker.get_positions()
    symbols = sorted(set(target_weights.keys()) | set(positions.keys()))
    quotes = broker.get_latest_quotes(symbols)
    orders: list[ProposedOrder] = []

    for symbol in symbols:
        px = float(quotes.get(symbol, 0.0))
        if not np.isfinite(px) or px <= 0:
            continue
        target_w = float(target_weights.get(symbol, 0.0))
        current_qty = float(positions.get(symbol, 0.0))
        target_qty = (target_w * float(equity)) / px
        delta = target_qty - current_qty
        notional = abs(delta) * px
        if notional < float(min_notional_to_trade):
            continue
        capped_delta = float(np.sign(delta) * min(abs(delta), float(max_notional_per_order) / px))
        if abs(capped_delta) < 1e-8:
            continue
        side = "buy" if capped_delta > 0 else "sell"
        qty = float(abs(capped_delta))
        if dry_run:
            orders.append(
                ProposedOrder(
                    symbol=symbol,
                    side=side,
                    qty=qty,
                    notional=qty * px,
                    reason="target_rebalance",
                    dry_run=True,
                    response=None,
                )
            )
            continue
        resp = broker.submit_market_order(symbol=symbol, qty=qty, side=side, time_in_force="day")
        orders.append(
            ProposedOrder(
                symbol=symbol,
                side=side,
                qty=qty,
                notional=qty * px,
                reason="target_rebalance",
                dry_run=False,
                response=resp,
            )
        )
    return orders
