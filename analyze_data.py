#!/usr/bin/env python3
import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.ndimage
from pathlib import Path
from datetime import datetime

def load_jsonl(file_path):
    data = []
    if not os.path.exists(file_path):
        return pd.DataFrame()
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(data)

def load_all_data(base_dir="oanda_data"):
    pricing_data, ob_data, pb_data = [], [], []
    for run_dir in glob.glob(os.path.join(base_dir, "*")):
        if not os.path.isdir(run_dir): continue
        
        # Load ALL matching files, not just the first one
        p_files = glob.glob(os.path.join(run_dir, "pricing_1s*.jsonl"))
        for f in p_files: pricing_data.append(load_jsonl(f))
            
        o_files = glob.glob(os.path.join(run_dir, "v20_orderbook*.jsonl"))
        for f in o_files: ob_data.append(load_jsonl(f))
            
        pos_files = glob.glob(os.path.join(run_dir, "v20_positionbook*.jsonl"))
        for f in pos_files: pb_data.append(load_jsonl(f))
    
    p_df = pd.concat(pricing_data, ignore_index=True) if pricing_data else pd.DataFrame()
    o_df = pd.concat(ob_data, ignore_index=True) if ob_data else pd.DataFrame()
    pos_df = pd.concat(pb_data, ignore_index=True) if pb_data else pd.DataFrame()
    return p_df, o_df, pos_df

def prepare_rolling_distribution_matrix(book_df, price_series, bins_per_side=20, range_pct=0.005, label="Unknown"):
    """
    Strict implementation of user request:
    1. Resample to 3s steps.
    2. For each step, define 40 bins centered on current price (20 above, 20 below).
    3. Sum liquidity into these bins.
    4. Return (TimeSteps x 40) matrix.
    """
    if book_df.empty or price_series.empty: 
        print(f"DEBUG [{label}]: Empty DF or Price Series.")
        return None, None

    # Time grid: 3 second steps (Aligned to 3s start for global consistency)
    t_start = price_series.index.min().floor('3s')
    t_end = price_series.index.max().ceil('3s')
    time_grid = pd.date_range(start=t_start, end=t_end, freq='3s')
    
    # Pre-process book
    book_df = book_df.copy()
    book_df['dt'] = pd.to_datetime(book_df['ts'])
    if book_df['dt'].dt.tz is None: book_df['dt'] = book_df['dt'].dt.tz_localize('UTC')
    else: book_df['dt'] = book_df['dt'].dt.tz_convert('UTC')
    book_df = book_df.sort_values('dt')
    book_ts = book_df['dt'].values.copy() # Native datetime64[ns], Copy to allow modification
    
    # Resample price (handle duplicates)
    price_series = price_series[~price_series.index.duplicated(keep='first')]
    price_resampled = price_series.resample('3s').ffill().reindex(time_grid).ffill()
    
    grid_ts = time_grid.values # Native datetime64[ns]
    
    # TIMESTAMP ALIGNMENT FIX
    # Detect if Book TS is "Future Dated" (e.g. +1h skew due to Timezone/Server issues)
    if len(book_ts) > 0 and len(grid_ts) > 0:
        skew = book_ts[0] - grid_ts[0]
        # If Book is more than 30 mins ahead of Grid start
        if skew > np.timedelta64(30, 'm'):
            print(f"DEBUG [{label}]: DETECTED FUTURE SKEW ({skew}). Shifting Book TS back by 1 Hour.")
            book_ts -= np.timedelta64(1, 'h')
        
        # Also check if Book is way behind? (Not an issue here, searchsorted handles it)

    indices = np.searchsorted(book_ts, grid_ts, side='right') - 1
    
    # DEBUG TIMESTAMP ALIGNMENT
    if len(book_ts) > 0:
        print(f"DEBUG [{label}]: Grid Range: {grid_ts[0]} to {grid_ts[-1]}")
        print(f"DEBUG [{label}]: Book Range: {book_ts[0]} to {book_ts[-1]}")
    else:
        print(f"DEBUG [{label}]: Book TS is EMPTY!")
    
    # 20 bins total
    total_bins = bins_per_side * 2
    matrix = np.zeros((total_bins, len(time_grid)))
    
    parsed_cache = {}
    debug_printed = False
    
    for i, t_val in enumerate(grid_ts):
        idx = indices[i]


        if idx < 0: continue
        
        current_price = price_resampled.iloc[i]
        if pd.isna(current_price): continue
        
        # Parse book if needed
        if idx not in parsed_cache:
            row = book_df.iloc[idx]
            
            # Robust Bucket Parsing (Handle Schema Differences)
            buckets = []
            if 'buckets' in row.index and isinstance(row['buckets'], list):
                buckets = row['buckets']
            
            if not buckets and 'data' in row.index:
                d = row['data']
                if isinstance(d, dict):
                    buckets = d.get('buckets', [])
            
            parsed_cache[idx] = [(float(b['price']), float(b['longCountPercent']) - float(b['shortCountPercent'])) for b in buckets]
        
        # Binning
        min_p = current_price * (1 - range_pct)
        max_p = current_price * (1 + range_pct)
        bin_edges = np.linspace(min_p, max_p, total_bins + 1)
        
        for price, net_val in parsed_cache[idx]:
            if price < min_p or price >= max_p: continue
            
            # Find which bin this price falls into
            b_idx = np.searchsorted(bin_edges, price) - 1
            if 0 <= b_idx < total_bins:
                matrix[b_idx, i] += net_val

    return time_grid, matrix

def create_dashboard(instrument, pricing_df, ob_df, pb_df, output_path):
    p_inst = pricing_df[pricing_df['instrument'] == instrument].copy()
    if p_inst.empty: return

    p_inst['dt'] = pd.to_datetime(p_inst['ts'])
    if p_inst['dt'].dt.tz is None: p_inst['dt'] = p_inst['dt'].dt.tz_localize('UTC')
    else: p_inst['dt'] = p_inst['dt'].dt.tz_convert('UTC')
    p_inst = p_inst.sort_values('dt').set_index('dt')
    p_inst['bid'] = pd.to_numeric(p_inst['bid'])
    p_inst['ask'] = pd.to_numeric(p_inst['ask'])
    mid_price = (p_inst['bid'] + p_inst['ask']) / 2.0
    start_time = p_inst.index.min()
    
    # CREATE MATRICES
    # Using strict 10 bins above/below logic
    ob_times, ob_matrix = prepare_rolling_distribution_matrix(ob_df[ob_df['instrument']==instrument], mid_price, range_pct=0.005, label="OrderBook")
    
    # Position Book covers a WIDER range of prices (long held positions).
    # Expand to 50% (0.5) to capture all historical positions.
    pb_times, pb_matrix = prepare_rolling_distribution_matrix(pb_df[pb_df['instrument']==instrument], mid_price, range_pct=0.50, label="PositionBook")
    
    # DATA SUMMARY
    if ob_matrix is not None:
        print(f"\n[{instrument}] Rolling Matrix Summary:")
        # 3s per step
        mins = (ob_matrix.shape[1] * 3) / 60
        print(f"Matrix Shape: {ob_matrix.shape} (40 bins x {ob_matrix.shape[1]} steps = {mins:.1f} mins)")
    
    # ... Normalization func
    def normalize_matrix_rows(mat):
        if mat is None: return None
        norm_mat = np.zeros_like(mat)
        for r in range(mat.shape[0]):
            row = mat[r, :]
            if np.max(np.abs(row)) > 1e-9:
                norm_mat[r, :] = pd.Series(row).rank(pct=True).values
            else:
                norm_mat[r, :] = 0.5 
        return norm_mat

    ob_norm = normalize_matrix_rows(ob_matrix)
    
    # Position Book: DEBUG and Enhance
    if pb_matrix is not None:
        flat_pb = pb_matrix.flatten()
        non_zero_pb = flat_pb[np.abs(flat_pb) > 1e-9]
        print(f"\n[DEBUG] {instrument} PB Matrix Stats:")
        print(f"  Range: {np.min(flat_pb):.6f} to {np.max(flat_pb):.6f}")
        if len(non_zero_pb) > 0:
            print(f"  Non-Zero Percentiles (5%, 50%, 95%): {np.percentile(non_zero_pb, [5, 50, 95])}")
        else:
            print("  Matrix is effectively empty (all zeros).")

        # AUTO-SCALING for RICH VISUAL
        # Use Quantile Scaling to force full color usage
        # We want Negative -> Red, Positive -> Green
        # Map negs to 0.0-0.5, pos to 0.5-1.0
        pb_norm = np.full_like(pb_matrix, 0.5)
        
        # Scale Negatives
        mask_neg = pb_matrix < 0
        if np.any(mask_neg):
            vals = pb_matrix[mask_neg]
            # Rank 0..1 then map to 0.0..0.5 (inverted so most neg is 0.0)
            ranks = pd.Series(vals).rank(pct=True).values
            pb_norm[mask_neg] = 0.5 * ranks # 0.0 (most neg) to 0.5 (least neg)
            
        # Scale Positives
        mask_pos = pb_matrix > 0
        if np.any(mask_pos):
            vals = pb_matrix[mask_pos]
            ranks = pd.Series(vals).rank(pct=True).values
            pb_norm[mask_pos] = 0.5 + (0.5 * ranks) # 0.5 (least pos) to 1.0 (most pos)
    else:
        pb_norm = None

    # ---------------------------------------------------------
    # PLOTTING: Merged Dashboard (Dark Theme)
    # ---------------------------------------------------------
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 18), facecolor='#0e0e0e')
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
    
    # Custom Red-Black-Green Colormap
    colors = [(0.0, '#ff3333'), (0.5, '#000000'), (1.0, '#33ff33')]
    cmap = mcolors.LinearSegmentedColormap.from_list("RBG", colors, N=256)
    
    # Panel 1: Price Action
    ax0 = fig.add_subplot(gs[0])
    ax0.set_facecolor('#0e0e0e')
    
    # Align Candles to Matrix Grid (ob_times)
    # Reindex OHLC to the exact same 3s grid used for the matrix
    if ob_times is not None:
        target_grid = ob_times
    else:
        target_grid = pd.date_range(start=start_time, end=p_inst.index.max(), freq='3s')
        
    ohlc = mid_price.resample('3s').ohlc().reindex(target_grid).ffill()
    ohlc['MA20'] = ohlc['close'].rolling(20, min_periods=1).mean() # 1 min MA approx (20*3s)
    ohlc['UpperBB'] = ohlc['MA20'] + 2*ohlc['close'].rolling(20, min_periods=1).std()
    ohlc['LowerBB'] = ohlc['MA20'] - 2*ohlc['close'].rolling(20, min_periods=1).std()
    
    # X-Axis: Data Point Index
    x_indices = np.arange(len(ohlc))
    
    # Candles
    width = 0.6
    for i, (t, row) in enumerate(ohlc.iterrows()):
        c = '#00e676' if row.close >= row.open else '#ff1744'
        ax0.vlines(x_indices[i], row.low, row.high, color=c, linewidth=1)
        ax0.add_patch(plt.Rectangle((x_indices[i]-width/2, min(row.open, row.close)), width, abs(row.open-row.close), color=c, alpha=0.9))
        
    ax0.plot(x_indices, ohlc['MA20'].values, color='cyan', linewidth=1, label='SMA 20 (1m)', alpha=0.7)
    ax0.fill_between(x_indices, ohlc['LowerBB'].values, ohlc['UpperBB'].values, color='cyan', alpha=0.08)
    
    ax0.set_title(f"{instrument}: Market Action (3s Intervals)", fontsize=16, fontweight='bold', color='white')
    ax0.grid(True, color='#333333', linestyle=':', alpha=0.5)
    
    # SECONDARY X-AXIS: Minutes
    def idx_to_min(x):
        return x * 3.0 / 60.0
    def min_to_idx(m):
        return m * 60.0 / 3.0
    
    secax = ax0.secondary_xaxis('top', functions=(idx_to_min, min_to_idx))
    secax.set_xlabel("Minutes Elapsed", color='white', fontsize=12)
    secax.tick_params(axis='x', colors='white')

    def plot_matrix(ax, matrix, times, title):
        if matrix is None: return
        ax.set_facecolor('#0e0e0e')
        
        # Extent: x=Data Points, y=bins (0-40)
        x_max = len(times)
        extent = [0, x_max, 0, 40]
        
        # Plot normalized matrix
        im = ax.imshow(matrix, aspect='auto', origin='lower', extent=extent, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        
        # Center line
        ax.axhline(20, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
        
        ax.set_title(title, fontsize=14, fontweight='bold', color='white')
        
        # Ticks
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_yticklabels(['-0.5%', '-0.25%', 'Price', '+0.25%', '+0.5%'], color='white')
        ax.tick_params(axis='x', colors='white')
        ax.grid(axis='x', color='#333333', linestyle=':', alpha=0.3)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    plot_matrix(ax1, ob_norm, ob_times, f"{instrument}: Order Book (Normalized)")

    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    plot_matrix(ax2, pb_norm, pb_times, f"{instrument}: Position Book (Normalized)")
    ax2.set_xlabel("Data Points (3s Intervals)", color='white')
    
    plt.suptitle(f"OANDA Deep Distribution: {instrument}", fontsize=20, fontweight='bold', color='white', y=0.98)
    plt.savefig(output_path, dpi=150, facecolor='#0e0e0e')
    plt.close()
    print(f"Generated Merged Dashboard: {output_path}")

def plot_distribution(instrument, pricing_df, ob_df, pb_df, output_path):
    o_inst = ob_df[ob_df['instrument'] == instrument].copy()
    if o_inst.empty: return
    o_latest = o_inst.sort_values('ts').iloc[-1].data
    pos_inst = pb_df[pb_df['instrument'] == instrument].copy()
    p_latest = pos_inst.sort_values('ts').iloc[-1].data if not pos_inst.empty else None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    cp = float(o_latest['price'])
    
    def draw_dist(ax, data, title, xlabel):
        if not data: return
        buckets = data.get('buckets', [])
        p = [float(b['price']) for b in buckets]
        val = [float(b['longCountPercent']) - float(b['shortCountPercent']) for b in buckets]
        ax.barh(p, val, color=['#4caf50' if v > 0 else '#f44336' for v in val], alpha=0.7)
        ax.axhline(cp, color='black', linestyle='--')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.2)

    draw_dist(ax1, o_latest, f"Order Book Distribution: {instrument}", "Net Liquidity %")
    draw_dist(ax2, p_latest, f"Position Book Distribution: {instrument}", "Net Sentiment %")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    p_df, o_df, pos_df = load_all_data()
    if p_df.empty: return
    for inst in p_df['instrument'].unique():
        create_dashboard(inst, p_df, o_df, pos_df, f"dashboard_{inst}.png")
        plot_distribution(inst, p_df, o_df, pos_df, f"distribution_{inst}.png")

if __name__ == "__main__":
    main()
