#!/usr/bin/env python3
"""
Feature Engineering & CSV Chunking Pipeline for OANDA Data.

Loads raw JSONL data (pricing, order book, position book),
computes technical indicators and book-derived features,
and outputs sliding-window CSV chunks for ML training.

Each CSV contains ALL features for BOTH instruments (EUR_USD, USD_JPY),
with the final two columns being the PRIMARY instrument's bid and ask.
"""

import json
import os
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA LOADING
# ============================================================

def load_jsonl(file_path):
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def load_all_data(base_dir="oanda_data"):
    """Load ALL raw data from all run directories and hourly files."""
    pricing_rows, ob_rows, pb_rows = [], [], []

    for run_dir in sorted(glob.glob(os.path.join(base_dir, "*"))):
        if not os.path.isdir(run_dir):
            continue

        for f in sorted(glob.glob(os.path.join(run_dir, "pricing_1s*.jsonl"))):
            pricing_rows.extend(load_jsonl(f))

        for f in sorted(glob.glob(os.path.join(run_dir, "v20_orderbook*.jsonl"))):
            ob_rows.extend(load_jsonl(f))

        for f in sorted(glob.glob(os.path.join(run_dir, "v20_positionbook*.jsonl"))):
            pb_rows.extend(load_jsonl(f))

    p_df = pd.DataFrame(pricing_rows) if pricing_rows else pd.DataFrame()
    ob_df = pd.DataFrame(ob_rows) if ob_rows else pd.DataFrame()
    pb_df = pd.DataFrame(pb_rows) if pb_rows else pd.DataFrame()

    print(f"Loaded: {len(p_df)} pricing, {len(ob_df)} orderbook, {len(pb_df)} positionbook rows")
    return p_df, ob_df, pb_df


# ============================================================
# 2. PRICING FEATURES
# ============================================================

def build_pricing_grid(pricing_df, instrument):
    """
    Build a 3s-resampled pricing grid for one instrument.
    Returns DataFrame indexed by datetime with bid, ask, mid, spread columns.
    """
    df = pricing_df[pricing_df['instrument'] == instrument].copy()
    if df.empty:
        return pd.DataFrame()

    df['dt'] = pd.to_datetime(df['ts'])
    if df['dt'].dt.tz is not None:
        df['dt'] = df['dt'].dt.tz_convert('UTC').dt.tz_localize(None)
    df = df.sort_values('dt').set_index('dt')

    for col in ['bid', 'ask']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Resample to 3s, forward-fill
    df = df[['bid', 'ask']].resample('3s').last().ffill()
    df['mid_price'] = (df['bid'] + df['ask']) / 2.0
    df['spread'] = df['ask'] - df['bid']
    df['spread_bps'] = (df['spread'] / df['mid_price']) * 10000.0
    df['log_return'] = np.log(df['mid_price'] / df['mid_price'].shift(1))

    return df


# ============================================================
# 3. TECHNICAL INDICATORS
# ============================================================

def compute_rsi(series, window):
    """Compute RSI over a rolling window."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_technical_indicators(df, prefix):
    """
    Compute technical indicators at multiple timeframes.
    Input df must have: mid_price, log_return, bid, ask columns.
    Returns a new DataFrame with indicator columns (prefixed).
    """
    result = pd.DataFrame(index=df.index)
    mid = df['mid_price']
    log_ret = df['log_return']

    # Timeframes in number of 3s bars
    timeframes = {
        '1m': 20,    # 60s / 3s = 20 bars
        '5m': 100,   # 300s / 3s = 100 bars
        '15m': 300,  # 900s / 3s = 300 bars
    }

    for tf_name, window in timeframes.items():
        # Volatility (rolling std of log returns)
        result[f'{prefix}_volatility_{tf_name}'] = log_ret.rolling(window=window, min_periods=1).std()

        # RSI
        result[f'{prefix}_rsi_{tf_name}'] = compute_rsi(mid, window)

        # SMA
        sma = mid.rolling(window=window, min_periods=1).mean()
        result[f'{prefix}_sma_{tf_name}'] = sma

        # Bollinger Bands
        std = mid.rolling(window=window, min_periods=1).std()
        result[f'{prefix}_bb_upper_{tf_name}'] = sma + 2 * std
        result[f'{prefix}_bb_lower_{tf_name}'] = sma - 2 * std
        result[f'{prefix}_bb_width_{tf_name}'] = (4 * std) / sma.replace(0, np.nan)

        # ATR (using bid as low proxy, ask as high proxy)
        high = df['ask'].rolling(window=window, min_periods=1).max()
        low = df['bid'].rolling(window=window, min_periods=1).min()
        tr = high - low
        atr = tr.rolling(window=window, min_periods=1).mean()
        result[f'{prefix}_atr_{tf_name}'] = atr
        result[f'{prefix}_natr_{tf_name}'] = atr / mid.replace(0, np.nan)

    # MACD (using 3s bars: EMA12=12 bars, EMA26=26 bars, Signal=9 bars)
    # Scale to approximate standard periods
    ema_fast = mid.ewm(span=12 * 20, min_periods=1).mean()  # ~12 min
    ema_slow = mid.ewm(span=26 * 20, min_periods=1).mean()  # ~26 min
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9 * 20, min_periods=1).mean()
    result[f'{prefix}_macd'] = macd_line
    result[f'{prefix}_macd_signal'] = macd_signal
    result[f'{prefix}_macd_hist'] = macd_line - macd_signal

    # Force Index
    volume_proxy = df['spread']  # Use spread as activity proxy (no volume in FX)
    result[f'{prefix}_force_index'] = mid.diff() * volume_proxy
    result[f'{prefix}_force_index_ema13'] = result[f'{prefix}_force_index'].ewm(span=13 * 20, min_periods=1).mean()

    return result


# ============================================================
# 4. ORDER BOOK FEATURES
# ============================================================

def parse_book_snapshots(book_df, instrument):
    """
    Parse order/position book snapshots for an instrument.
    Returns list of (timestamp, buckets_list) tuples.
    """
    df = book_df[book_df['instrument'] == instrument].copy()
    if df.empty:
        return []

    df['dt'] = pd.to_datetime(df['ts'])
    if df['dt'].dt.tz is not None:
        df['dt'] = df['dt'].dt.tz_convert('UTC').dt.tz_localize(None)
    df = df.sort_values('dt')

    snapshots = []
    for _, row in df.iterrows():
        dt = row['dt']
        # Handle both schema variants
        buckets = []
        if 'data' in row.index and isinstance(row.get('data'), dict):
            buckets = row['data'].get('buckets', [])
        elif 'buckets' in row.index and isinstance(row.get('buckets'), list):
            buckets = row['buckets']

        if buckets:
            parsed = []
            for b in buckets:
                try:
                    parsed.append({
                        'price': float(b['price']),
                        'long_pct': float(b['longCountPercent']),
                        'short_pct': float(b['shortCountPercent']),
                    })
                except (KeyError, ValueError):
                    continue
            snapshots.append((dt, parsed))

    return snapshots


def compute_book_features_at_snapshot(buckets, current_price, near_pct=0.001, mid_pct=0.005):
    """
    Compute aggregate features from a single book snapshot relative to current price.
    near_pct: ±0.1% for "near" features
    mid_pct:  ±0.5% for "mid" features
    """
    near_lo = current_price * (1 - near_pct)
    near_hi = current_price * (1 + near_pct)
    mid_lo = current_price * (1 - mid_pct)
    mid_hi = current_price * (1 + mid_pct)

    # Accumulators
    near_long, near_short = 0.0, 0.0
    mid_long, mid_short = 0.0, 0.0
    bid_wall, ask_wall = 0.0, 0.0
    bid_depth, ask_depth = 0.0, 0.0
    total_long, total_short = 0.0, 0.0

    for b in buckets:
        p = b['price']
        lp = b['long_pct']
        sp = b['short_pct']
        total_long += lp
        total_short += sp

        if near_lo <= p <= near_hi:
            near_long += lp
            near_short += sp

        if mid_lo <= p <= mid_hi:
            mid_long += lp
            mid_short += sp

        # Bid side (below price)
        if mid_lo <= p < current_price:
            bid_depth += lp
            bid_wall = max(bid_wall, lp)

        # Ask side (above price)
        if current_price <= p <= mid_hi:
            ask_depth += sp
            ask_wall = max(ask_wall, sp)

    imbalance_near = near_long - near_short
    imbalance_mid = mid_long - mid_short
    depth_ratio = bid_depth / max(ask_depth, 1e-10)
    net_total = total_long - total_short
    ratio_total = total_long / max(total_short, 1e-10)

    return {
        'imbalance_near': imbalance_near,
        'imbalance_mid': imbalance_mid,
        'bid_wall': bid_wall,
        'ask_wall': ask_wall,
        'bid_depth': bid_depth,
        'ask_depth': ask_depth,
        'depth_ratio': depth_ratio,
        'long_total': total_long,
        'short_total': total_short,
        'net_total': net_total,
        'ratio_total': ratio_total,
    }


def build_book_feature_series(snapshots, price_grid, prefix, near_pct=0.001, mid_pct=0.005):
    """
    Build a DataFrame of book features aligned to the price grid.
    Snapshots are forward-filled to match the 3s grid.
    """
    if not snapshots:
        # Return empty columns
        cols = [f'{prefix}_{k}' for k in [
            'imbalance_near', 'imbalance_mid', 'bid_wall', 'ask_wall',
            'bid_depth', 'ask_depth', 'depth_ratio',
            'long_total', 'short_total', 'net_total', 'ratio_total'
        ]]
        return pd.DataFrame(np.nan, index=price_grid.index, columns=cols)

    # For each snapshot, compute features using the nearest price
    feat_records = []
    for dt, buckets in snapshots:
        # Find closest price
        idx = price_grid.index.searchsorted(dt)
        if idx >= len(price_grid):
            idx = len(price_grid) - 1
        current_price = price_grid.iloc[idx]

        feats = compute_book_features_at_snapshot(buckets, current_price, near_pct, mid_pct)
        feats['dt'] = dt
        feat_records.append(feats)

    feat_df = pd.DataFrame(feat_records).set_index('dt')
    feat_df = feat_df.rename(columns={k: f'{prefix}_{k}' for k in feat_df.columns})

    # Reindex to 3s grid and forward-fill
    feat_df = feat_df.reindex(price_grid.index, method='ffill')

    return feat_df


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def build_instrument_features(pricing_df, ob_df, pb_df, instrument):
    """Build ALL features for one instrument. Returns a DataFrame."""
    prefix = instrument.replace('_', '')
    print(f"\n--- Building features for {instrument} ({prefix}) ---")

    # 1. Price Grid
    grid = build_pricing_grid(pricing_df, instrument)
    if grid.empty:
        print(f"  WARNING: No pricing data for {instrument}")
        return pd.DataFrame()
    print(f"  Price grid: {len(grid)} rows, {grid.index.min()} to {grid.index.max()}")

    # 2. Technical Indicators
    tech = compute_technical_indicators(grid, prefix)
    print(f"  Technical indicators: {len(tech.columns)} columns")

    # 3. Order Book Features
    ob_snapshots = parse_book_snapshots(ob_df, instrument)
    print(f"  Order Book snapshots: {len(ob_snapshots)}")
    ob_feats = build_book_feature_series(
        ob_snapshots, grid['mid_price'], f'{prefix}_ob',
        near_pct=0.001, mid_pct=0.005
    )

    # 4. Position Book Features (wider range)
    pb_snapshots = parse_book_snapshots(pb_df, instrument)
    print(f"  Position Book snapshots: {len(pb_snapshots)}")
    pb_feats = build_book_feature_series(
        pb_snapshots, grid['mid_price'], f'{prefix}_pb',
        near_pct=0.01, mid_pct=0.05
    )

    # 5. Rename price columns with prefix
    price_cols = grid.rename(columns={
        'mid_price': f'{prefix}_mid_price',
        'spread': f'{prefix}_spread',
        'spread_bps': f'{prefix}_spread_bps',
        'log_return': f'{prefix}_log_return',
        'bid': f'{prefix}_bid',
        'ask': f'{prefix}_ask',
    })

    # 6. Join all
    result = pd.concat([price_cols, tech, ob_feats, pb_feats], axis=1)
    print(f"  Total columns for {instrument}: {len(result.columns)}")

    return result


def main():
    print("=" * 70)
    print("OANDA Feature Engineering & CSV Chunking Pipeline")
    print("=" * 70)

    # Load data
    p_df, ob_df, pb_df = load_all_data()

    instruments = ['EUR_USD', 'USD_JPY']

    # Build features per instrument
    all_features = {}
    for inst in instruments:
        all_features[inst] = build_instrument_features(p_df, ob_df, pb_df, inst)

    # Align all instruments to the same time index (intersection)
    common_index = None
    for inst in instruments:
        if all_features[inst].empty:
            continue
        if common_index is None:
            common_index = all_features[inst].index
        else:
            common_index = common_index.intersection(all_features[inst].index)

    if common_index is None or len(common_index) == 0:
        print("ERROR: No overlapping data between instruments!")
        return

    print(f"\nCommon index: {len(common_index)} rows, {common_index.min()} to {common_index.max()}")

    # Build combined DataFrame
    # For each primary instrument, create CSVs with ALL features
    # but the LAST two columns are that instrument's bid and ask
    output_dir = Path("data_chunks")
    output_dir.mkdir(exist_ok=True)

    for primary_inst in instruments:
        primary_prefix = primary_inst.replace('_', '')

        # Gather all columns from all instruments
        frames = []
        for inst in instruments:
            df = all_features[inst].loc[common_index]
            frames.append(df)

        combined = pd.concat(frames, axis=1)

        # Remove duplicate columns (if any from index alignment)
        combined = combined.loc[:, ~combined.columns.duplicated()]

        # Add poll_ts as first column
        combined.insert(0, 'poll_ts', combined.index)

        # Ensure bid/ask for the PRIMARY instrument are LAST
        bid_col = f'{primary_prefix}_bid'
        ask_col = f'{primary_prefix}_ask'

        # Move bid and ask to the end
        other_cols = [c for c in combined.columns if c not in [bid_col, ask_col]]
        combined = combined[other_cols + [bid_col, ask_col]]

        # Reset index for clean CSV output
        combined = combined.reset_index(drop=True)

        # Drop initial rows with NaN from indicator warmup (keep rows where RSI is not NaN)
        # Use a generous warmup: 300 bars (15m indicators)
        warmup = 300
        combined = combined.iloc[warmup:].reset_index(drop=True)

        total_rows = len(combined)
        print(f"\n{'='*50}")
        print(f"[{primary_inst}] Total usable rows after warmup: {total_rows}")
        print(f"[{primary_inst}] Total columns: {len(combined.columns)}")
        print(f"[{primary_inst}] Columns: {list(combined.columns)}")

        # Chunking: 1100 rows, slide by 100
        window_size = 1100
        step_size = 100
        chunk_count = 0

        for start in range(0, total_rows - window_size + 1, step_size):
            end = start + window_size
            chunk = combined.iloc[start:end]

            filename = f"{primary_prefix}_{start}_{end}_{window_size}.csv"
            filepath = output_dir / filename
            chunk.to_csv(filepath, index=False)
            chunk_count += 1

        print(f"[{primary_inst}] Generated {chunk_count} chunks in {output_dir}/")

    print(f"\n{'='*70}")
    print("DONE! All CSV chunks saved to data_chunks/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
