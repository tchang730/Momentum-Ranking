#    Relative Rotation Graph (RRG)
#    Plots each security's relative strength against a benchmark as a point on two axes:
#    RS-Ratio (x, trend of relative strength) and RS-Momentum (y, speed of change of that trend).
#    Both are centred on 100, which splits the chart into four quadrants:
#      Leading    RS-Ratio > 100, RS-Momentum > 100
#      Weakening  RS-Ratio > 100, RS-Momentum < 100
#      Lagging    RS-Ratio < 100, RS-Momentum < 100
#      Improving  RS-Ratio < 100, RS-Momentum > 100
#    The price files are the "<TICKER>-<folder>.csv" files that run.py writes (timestamp, close).
#    Note: this uses the common public approximation of the RRG calculation (rolling z-score
#    normalisation of relative strength), not the proprietary formula of the method's author.
#
#    Usage:
#      python rrg.py --folder 2026-10-10 --benchmark SPY --out rrg.png
#      python rrg.py --folder 2026-10-10 --benchmark SPY --tickers AAPL MSFT --trail 8

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

QUADRANT_NAMES = {
    (True, True): 'Leading',
    (True, False): 'Weakening',
    (False, False): 'Lagging',
    (False, True): 'Improving',
}

QUADRANT_COLOURS = {
    'Leading': '#d9f2d9',
    'Weakening': '#fff5cc',
    'Lagging': '#f9d9d9',
    'Improving': '#d9e8f9',
}

def load_closes(path):
    # run.py writes no header row: column 0 is a timestamp, column 4 of the Koyfin data is the close
    df = pd.read_csv(path, header=None, names=['timestamp', 'close'])
    if pd.api.types.is_numeric_dtype(df['timestamp']):
        unit = 'ms' if df['timestamp'].max() > 1e11 else 's'
        dates = pd.to_datetime(df['timestamp'], unit=unit)
    else:
        dates = pd.to_datetime(df['timestamp'])
    closes = pd.Series(df['close'].astype(float).values, index=dates)
    closes = closes[~closes.index.duplicated(keep='last')].sort_index()
    return closes

def _normalise(series, window):
    # rolling z-score rescaled to be centred on 100
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return 100 + (series - mean) / std

def compute_rrg(price, benchmark, window=10, momentum_window=1):
    # Relative strength of the security against the benchmark, aligned on common dates
    rs = (price / benchmark).dropna()
    # Trend of relative strength: RS as a percentage of its own moving average, then normalised
    rs_trend = rs / rs.rolling(window).mean() * 100
    rs_ratio = _normalise(rs_trend, window)
    # Speed of change of the RS-Ratio, then normalised the same way
    rs_speed = rs_ratio / rs_ratio.shift(momentum_window) * 100
    rs_momentum = _normalise(rs_speed, window)
    out = pd.DataFrame({'rs_ratio': rs_ratio, 'rs_momentum': rs_momentum}).dropna()
    out['quadrant'] = [quadrant(x, y) for x, y in zip(out['rs_ratio'], out['rs_momentum'])]
    return out

def quadrant(rs_ratio, rs_momentum):
    return QUADRANT_NAMES[(rs_ratio > 100, rs_momentum > 100)]

def plot_rrg(results, out_path, trail=5, title='Relative Rotation Graph'):
    # results: dict of ticker -> DataFrame from compute_rrg
    fig, ax = plt.subplots(figsize=(10, 8))

    xs = np.concatenate([r['rs_ratio'].values for r in results.values()])
    ys = np.concatenate([r['rs_momentum'].values for r in results.values()])
    spread = max(np.abs(xs - 100).max(), np.abs(ys - 100).max()) * 1.15
    spread = max(spread, 1.0)
    lo, hi = 100 - spread, 100 + spread

    ax.add_patch(plt.Rectangle((100, 100), hi - 100, hi - 100, color=QUADRANT_COLOURS['Leading'], zorder=0))
    ax.add_patch(plt.Rectangle((100, lo), hi - 100, 100 - lo, color=QUADRANT_COLOURS['Weakening'], zorder=0))
    ax.add_patch(plt.Rectangle((lo, lo), 100 - lo, 100 - lo, color=QUADRANT_COLOURS['Lagging'], zorder=0))
    ax.add_patch(plt.Rectangle((lo, 100), 100 - lo, hi - 100, color=QUADRANT_COLOURS['Improving'], zorder=0))
    for label, x, y in [('Leading', hi, hi), ('Weakening', hi, lo), ('Lagging', lo, lo), ('Improving', lo, hi)]:
        ax.text(x, y, label, ha='right' if x == hi else 'left', va='top' if y == lo else 'bottom',
                fontsize=12, color='#555555', alpha=0.8)

    ax.axvline(100, color='grey', linewidth=0.8)
    ax.axhline(100, color='grey', linewidth=0.8)

    for ticker, r in results.items():
        tail = r.tail(trail)
        if tail.empty:
            continue
        ax.plot(tail['rs_ratio'], tail['rs_momentum'], color='#444444', linewidth=0.8, alpha=0.6, zorder=2)
        ax.scatter(tail['rs_ratio'].iloc[:-1], tail['rs_momentum'].iloc[:-1], s=12, color='#888888', zorder=2)
        last = tail.iloc[-1]
        ax.scatter([last['rs_ratio']], [last['rs_momentum']], s=60, color='#1f4e79', zorder=3)
        ax.annotate(ticker, (last['rs_ratio'], last['rs_momentum']), xytext=(4, 4),
                    textcoords='offset points', fontsize=9, zorder=4)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('RS-Ratio')
    ax.set_ylabel('RS-Momentum')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def load_folder(folder, benchmark, tickers=None):
    tag = os.path.basename(os.path.normpath(folder))
    bench_path = os.path.join(folder, '%s-%s.csv' % (benchmark, tag))
    if not os.path.exists(bench_path):
        raise FileNotFoundError('Benchmark file not found: %s' % bench_path)
    bench = load_closes(bench_path)

    if tickers is None:
        pattern = os.path.join(folder, '*-%s.csv' % tag)
        tickers = []
        for path in sorted(glob.glob(pattern)):
            name = os.path.basename(path)[:-len('-%s.csv' % tag)]
            if name != benchmark:
                tickers.append(name)

    prices = {}
    for ticker in tickers:
        path = os.path.join(folder, '%s-%s.csv' % (ticker, tag))
        if os.path.exists(path):
            prices[ticker] = load_closes(path)
        else:
            print('Skipping %s: no file at %s' % (ticker, path))
    return prices, bench

def main(argv=None):
    parser = argparse.ArgumentParser(description='Build a Relative Rotation Graph from downloaded price files.')
    parser.add_argument('--folder', required=True, help='dated folder holding the <TICKER>-<date>.csv files')
    parser.add_argument('--benchmark', required=True, help='ticker to measure relative strength against, e.g. SPY')
    parser.add_argument('--tickers', nargs='*', help='tickers to plot (default: every file in the folder)')
    parser.add_argument('--window', type=int, default=10, help='smoothing window in periods (default 10)')
    parser.add_argument('--momentum-window', type=int, default=1, help='periods used for RS-Momentum (default 1)')
    parser.add_argument('--trail', type=int, default=5, help='number of past periods drawn as a trail (default 5)')
    parser.add_argument('--out', default='rrg.png', help='output image path (default rrg.png)')
    args = parser.parse_args(argv)

    prices, bench = load_folder(args.folder, args.benchmark, args.tickers)
    results = {}
    for ticker, price in prices.items():
        r = compute_rrg(price, bench, window=args.window, momentum_window=args.momentum_window)
        if r.empty:
            print('Skipping %s: not enough overlapping history' % ticker)
            continue
        results[ticker] = r

    if not results:
        raise SystemExit('Nothing to plot.')

    title = 'Relative Rotation Graph vs %s (%s)' % (args.benchmark, os.path.basename(os.path.normpath(args.folder)))
    plot_rrg(results, args.out, trail=args.trail, title=title)
    print('Saved %d securities to %s' % (len(results), args.out))
    for ticker, r in results.items():
        last = r.iloc[-1]
        print('%-8s RS-Ratio %7.2f  RS-Momentum %7.2f  %s' % (ticker, last['rs_ratio'], last['rs_momentum'], last['quadrant']))
    return results

if __name__ == '__main__':
    main()
