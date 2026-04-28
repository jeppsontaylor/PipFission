"""
Python prototype of the online ridge-regression strategy that we'll port to Rust.
Goal: prove the math is sound on real captured OANDA data BEFORE writing Rust.

Algorithm:
  features per tick (instrument-local):
    f1 = log return over last 10 ticks
    f2 = rolling std over last 60 ticks
    f3 = vol-normalized momentum: (mean(last20) - mean(last60)) / std60
    f4 = spread / mid (bps)
    f5 = log return over last 60 ticks
    bias term

  target: forward log return over next H ticks (default H=30)

  fit: ridge regression with lambda=L on a rolling buffer of the last N samples
       train at t = warmup (1000), then every retrain_every (100) new samples

  signal:
    pred = w · features
    threshold = k * sigma_pred  (sigma_pred = std of predictions in buffer)
    if pred > threshold:  long
    elif pred < -threshold: short
    else: flat
"""

from __future__ import annotations

import json
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---- features ---------------------------------------------------------------

WINDOW_LONG = 60
WINDOW_SHORT = 20
WINDOW_MOMENTUM = 10
HORIZON = 30  # predict forward log-return over next 30 ticks
WARMUP = 1000
RETRAIN_EVERY = 100
BUFFER_SIZE = 4000
RIDGE_LAMBDA = 1.0
SIGNAL_K = 1.0  # signal threshold in units of pred std
TRADE_UNITS = 1000  # base units per fill (fixed sizing)
COMMISSION_BPS = 0.0  # we model spread cost via bid/ask, no extra commission

NUM_FEATURES = 6  # f1..f5 + bias


@dataclass
class TickHistory:
    """Bounded mid-price history; computes features on demand."""

    mids: deque = field(default_factory=lambda: deque(maxlen=WINDOW_LONG + 5))
    bids: deque = field(default_factory=lambda: deque(maxlen=4))
    asks: deque = field(default_factory=lambda: deque(maxlen=4))

    def push(self, bid: float, ask: float) -> None:
        mid = 0.5 * (bid + ask)
        self.bids.append(bid)
        self.asks.append(ask)
        self.mids.append(mid)

    def features(self) -> np.ndarray | None:
        if len(self.mids) < WINDOW_LONG:
            return None
        arr = np.asarray(self.mids, dtype=np.float64)
        last_long = arr[-WINDOW_LONG:]
        last_short = arr[-WINDOW_SHORT:]
        last_mom = arr[-WINDOW_MOMENTUM:]

        # log returns: use last vs reference
        log_mid = np.log(last_long)
        f1 = log_mid[-1] - log_mid[-WINDOW_MOMENTUM]
        f5 = log_mid[-1] - log_mid[0]

        # rolling std of log diffs over the long window
        diffs = np.diff(log_mid)
        f2 = float(np.std(diffs)) if diffs.size else 0.0

        # vol-normalized momentum
        mean_short = float(np.mean(last_short))
        mean_long = float(np.mean(last_long))
        std_long = float(np.std(last_long))
        f3 = (mean_short - mean_long) / std_long if std_long > 0 else 0.0

        # spread in bps (mid-relative)
        bid = self.bids[-1]
        ask = self.asks[-1]
        mid = 0.5 * (bid + ask)
        f4 = (ask - bid) / mid if mid > 0 else 0.0

        # last_mom unused beyond f1 — keep as placeholder for future expansion
        _ = last_mom

        return np.array([f1, f2, f3, f4, f5, 1.0])


# ---- ridge regression -------------------------------------------------------

@dataclass
class RidgeOnline:
    """
    Holds a rolling buffer of (X, y) and refits the closed-form ridge solution
    on demand:  w = (XᵀX + λI)⁻¹ Xᵀy.

    Using the closed form on a buffer of ≤4000 × 6 is dirt-cheap (microseconds);
    no need for incremental matrix updates.
    """

    capacity: int = BUFFER_SIZE
    lam: float = RIDGE_LAMBDA
    X: deque = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    y: deque = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    w: np.ndarray | None = None
    pred_std: float = 0.0
    fit_count: int = 0

    def add(self, x: np.ndarray, y: float) -> None:
        self.X.append(x)
        self.y.append(y)

    def fit(self) -> None:
        if len(self.X) < 50:
            return
        X = np.asarray(self.X)
        y = np.asarray(self.y)
        n_features = X.shape[1]
        # standardize features (zero-mean, unit-std) for ridge stability — bias kept as-is.
        mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        sigma_safe = np.where(sigma > 1e-12, sigma, 1.0)
        X_std = (X - mu) / sigma_safe
        X_std[:, -1] = 1.0  # restore bias column
        A = X_std.T @ X_std + self.lam * np.eye(n_features)
        b = X_std.T @ y
        try:
            w_std = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return
        # back-out coefficients in raw feature space.
        w = np.zeros(n_features)
        w[:-1] = w_std[:-1] / sigma_safe[:-1]
        # bias absorbs the means.
        w[-1] = w_std[-1] - float(np.dot(w_std[:-1], mu[:-1] / sigma_safe[:-1]))
        self.w = w
        # predicted std on training set — used as our adaptive signal threshold.
        preds = X @ w
        self.pred_std = float(np.std(preds))
        self.fit_count += 1

    def predict(self, x: np.ndarray) -> float:
        if self.w is None:
            return 0.0
        return float(np.dot(self.w, x))


# ---- backtest ---------------------------------------------------------------

@dataclass
class PaperAccount:
    """Internal paper account, fills at observed bid/ask, tracks net P/L."""

    cash: float
    positions: dict[str, tuple[float, float]] = field(default_factory=dict)  # inst -> (units, avg_price)
    realized: float = 0.0
    fills: int = 0

    def market_order(self, inst: str, units: float, bid: float, ask: float) -> None:
        if units == 0:
            return
        # buy at ask, sell at bid
        fill_price = ask if units > 0 else bid
        cur_units, cur_avg = self.positions.get(inst, (0.0, 0.0))
        new_units = cur_units + units
        if cur_units == 0:
            new_avg = fill_price
        elif (cur_units > 0) == (units > 0):
            new_avg = (cur_avg * cur_units + fill_price * units) / new_units if new_units != 0 else 0.0
        else:
            # closing or flipping
            close_units = -units if abs(units) <= abs(cur_units) else cur_units
            pnl = (fill_price - cur_avg) * (-close_units)
            self.realized += pnl
            self.cash += pnl
            if abs(units) > abs(cur_units):
                # flipped through zero
                new_avg = fill_price
            else:
                new_avg = cur_avg
        self.positions[inst] = (new_units, new_avg)
        self.fills += 1

    def equity(self, mids: dict[str, float]) -> float:
        eq = self.cash
        for inst, (units, avg) in self.positions.items():
            if units != 0 and inst in mids:
                eq += units * (mids[inst] - avg)
        return eq


def load_pricing(data_dir: Path, instrument: str) -> list[tuple[float, float, float]]:
    """Returns sorted [(ts_seconds, bid, ask), ...] for an instrument."""
    rows: list[tuple[float, float, float]] = []
    for f in sorted(data_dir.glob("**/pricing_1s-*.jsonl")):
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("instrument") != instrument:
                    continue
                if obj.get("status") and obj.get("status") != "tradeable":
                    continue
                try:
                    bid = float(obj["bid"])
                    ask = float(obj["ask"])
                except (KeyError, ValueError):
                    continue
                ts_str = obj.get("ts") or obj.get("source_time")
                if not ts_str:
                    continue
                # crude epoch: parse just the seconds component.
                # Robust parse:
                from datetime import datetime
                try:
                    t = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                except Exception:
                    continue
                rows.append((t, bid, ask))
    rows.sort(key=lambda r: r[0])
    # de-dup adjacent identical timestamps
    deduped: list[tuple[float, float, float]] = []
    last_t = None
    for r in rows:
        if r[0] != last_t:
            deduped.append(r)
            last_t = r[0]
    return deduped


def backtest(rows: list[tuple[float, float, float]], inst: str) -> dict:
    history = TickHistory()
    model = RidgeOnline()
    account = PaperAccount(cash=100_000.0)

    pending: deque[tuple[int, float]] = deque()  # (fire_index, mid_at_creation) — for label collection
    samples_since_retrain = 0
    equity_curve: list[float] = []
    signal_curve: list[int] = []
    pred_curve: list[float] = []
    times: list[float] = []
    fills_log: list[tuple[float, str, float, float]] = []  # (ts, side, units, price)

    last_signal = 0  # -1 short, 0 flat, +1 long
    target_position = 0.0

    for i, (ts, bid, ask) in enumerate(rows):
        history.push(bid, ask)
        feats = history.features()
        if feats is None:
            equity_curve.append(account.equity({inst: 0.5 * (bid + ask)}))
            times.append(ts)
            signal_curve.append(0)
            pred_curve.append(0.0)
            continue
        mid = 0.5 * (bid + ask)

        # process pending labels: any pending entry whose forward window has matured
        while pending and i - pending[0][0] >= HORIZON:
            entry_idx, entry_mid = pending.popleft()
            # forward log return
            y = math.log(mid / entry_mid)
            # the features captured at entry are the i-HORIZON-th feature snapshot —
            # we recompute by walking back; cheaper to remember. We saved entry_feats below.
            x = entry_feats_buffer.popleft()
            model.add(x, y)
            samples_since_retrain += 1

        # capture features for label-future pairing
        entry_feats_buffer.append(feats)
        pending.append((i, mid))

        # train if we have warmup and time to retrain
        if model.fit_count == 0 and len(model.X) >= WARMUP:
            model.fit()
            samples_since_retrain = 0
        elif model.fit_count > 0 and samples_since_retrain >= RETRAIN_EVERY:
            model.fit()
            samples_since_retrain = 0

        pred = model.predict(feats)
        thresh = SIGNAL_K * model.pred_std if model.pred_std > 0 else float("inf")
        if model.fit_count == 0:
            sig = 0
        elif pred > thresh:
            sig = 1
        elif pred < -thresh:
            sig = -1
        else:
            sig = 0

        # rebalance to target if signal changed
        if sig != last_signal:
            new_target = sig * TRADE_UNITS
            cur_units = account.positions.get(inst, (0.0, 0.0))[0]
            delta = new_target - cur_units
            if delta != 0:
                fill_price = ask if delta > 0 else bid
                account.market_order(inst, delta, bid, ask)
                fills_log.append((ts, "BUY" if delta > 0 else "SELL", abs(delta), fill_price))
            last_signal = sig
            target_position = new_target

        equity_curve.append(account.equity({inst: mid}))
        signal_curve.append(sig)
        pred_curve.append(pred)
        times.append(ts)

    # close any open position at end
    cur_units = account.positions.get(inst, (0.0, 0.0))[0]
    if cur_units != 0 and rows:
        bid, ask = rows[-1][1], rows[-1][2]
        account.market_order(inst, -cur_units, bid, ask)
        fills_log.append((rows[-1][0], "CLOSE", abs(cur_units), bid if cur_units > 0 else ask))

    return {
        "equity_curve": equity_curve,
        "signal_curve": signal_curve,
        "pred_curve": pred_curve,
        "times": times,
        "fills": fills_log,
        "model_fits": model.fit_count,
        "final_cash": account.cash,
        "final_realized": account.realized,
        "fill_count": account.fills,
        "samples_used": len(model.X),
        "pred_std": model.pred_std,
    }


# entry_feats_buffer is module-level so backtest() can append/pop from it — keeps
# code closer to how the Rust version will look.
entry_feats_buffer: deque[np.ndarray] = deque()


def main() -> int:
    data_dir = Path("oanda_data")
    if not data_dir.exists():
        print("No oanda_data/ directory.", file=sys.stderr)
        return 1
    inst = "EUR_USD"
    print(f"Loading pricing_1s for {inst}...")
    rows = load_pricing(data_dir, inst)
    print(f"  loaded {len(rows)} ticks")
    if len(rows) < WARMUP + HORIZON + 100:
        print(f"  too few ticks for backtest (need ≥ {WARMUP + HORIZON + 100})", file=sys.stderr)
        # still run, but expect zero fills
    print("Running backtest...")
    res = backtest(rows, inst)

    # summary
    eq = res["equity_curve"]
    if eq:
        peak = max(eq)
        trough = min(eq)
        ret = (eq[-1] / eq[0] - 1.0) * 100 if eq[0] else 0.0
        print(f"  start equity: ${eq[0]:,.2f}")
        print(f"  end equity:   ${eq[-1]:,.2f}  ({ret:+.2f}%)")
        print(f"  peak:         ${peak:,.2f}")
        print(f"  trough:       ${trough:,.2f}")
    print(f"  model fits:   {res['model_fits']}")
    print(f"  buffer size:  {res['samples_used']}")
    print(f"  pred_std:     {res['pred_std']:.6e}")
    print(f"  fills:        {res['fill_count']}")
    if res["fills"]:
        print("  first 5 fills:")
        for f in res["fills"][:5]:
            print(f"    {f}")
        print("  last 5 fills:")
        for f in res["fills"][-5:]:
            print(f"    {f}")

    # write equity curve csv for later plotting
    out = Path("strategy_proto/equity_curve.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        fh.write("ts,equity,signal,pred\n")
        for t, e, s, p in zip(res["times"], res["equity_curve"], res["signal_curve"], res["pred_curve"]):
            fh.write(f"{t},{e},{s},{p}\n")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
