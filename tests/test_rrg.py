import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import rrg

TAG = '2026-10-10'

def _dates(n):
    return pd.bdate_range('2024-01-01', periods=n)

def _series_pair(n=260, lead_days=120):
    # Benchmark: gentle noisy uptrend. Stock: same, but it outperforms strongly over the last `lead_days`.
    rng = np.random.default_rng(0)
    dates = _dates(n)
    bench = 100 * np.exp(np.cumsum(0.0004 + 0.008 * rng.standard_normal(n)))
    rel = np.zeros(n)
    rel[-lead_days:] = np.linspace(0, 0.6, lead_days)
    stock = bench * np.exp(rel)
    return pd.Series(stock, index=dates), pd.Series(bench, index=dates)

def test_quadrant_boundaries():
    assert rrg.quadrant(101, 101) == 'Leading'
    assert rrg.quadrant(101, 99) == 'Weakening'
    assert rrg.quadrant(99, 99) == 'Lagging'
    assert rrg.quadrant(99, 101) == 'Improving'

def test_outperformer_ends_in_leading():
    stock, bench = _series_pair()
    out = rrg.compute_rrg(stock, bench)
    last = out.iloc[-1]
    assert last['rs_ratio'] > 100
    assert last['rs_momentum'] > 100
    assert last['quadrant'] == 'Leading'

def test_underperformer_ends_in_lagging():
    stock, bench = _series_pair()
    # Mirror the relative move so the stock falls behind over the last period
    rel = np.log(stock / bench)
    stock_down = bench * np.exp(2 * rel.iloc[-1] - rel)
    out = rrg.compute_rrg(stock_down, bench)
    last = out.iloc[-1]
    assert last['rs_ratio'] < 100
    assert last['quadrant'] in ('Lagging', 'Improving')

def test_output_is_aligned_and_centred():
    stock, bench = _series_pair()
    out = rrg.compute_rrg(stock, bench, window=10)
    assert out.notna().all().all()
    assert out.index.is_monotonic_increasing
    assert abs(out['rs_ratio'].mean() - 100) < 2
    assert abs(out['rs_momentum'].mean() - 100) < 2

def test_load_closes_handles_epoch_ms_and_date_strings(tmp_path):
    ms = tmp_path / 'ms.csv'
    ms.write_text('1704067200000,10.0\n1704153600000,11.0\n')
    s = rrg.load_closes(str(ms))
    assert list(s.values) == [10.0, 11.0]
    assert s.index[0] == pd.Timestamp('2024-01-01')

    txt = tmp_path / 'txt.csv'
    txt.write_text('2024-01-02,12.5\n2024-01-01,12.0\n')
    s = rrg.load_closes(str(txt))
    assert s.index.is_monotonic_increasing
    assert s.iloc[-1] == 12.5

def test_cli_end_to_end(tmp_path):
    folder = tmp_path / TAG
    folder.mkdir()
    stock, bench = _series_pair()
    def write(name, series):
        with open(folder / ('%s-%s.csv' % (name, TAG)), 'w') as f:
            for d, v in series.items():
                f.write('%d,%.6f\n' % (int(d.timestamp() * 1000), v))
    write('SPY', bench)
    write('AAPL', stock)
    write('FLAT', bench * 1.0)
    out_png = tmp_path / 'rrg.png'
    results = rrg.main(['--folder', str(folder), '--benchmark', 'SPY', '--out', str(out_png)])
    assert out_png.exists() and out_png.stat().st_size > 0
    assert set(results) == {'AAPL', 'FLAT'}
    assert results['AAPL'].iloc[-1]['quadrant'] == 'Leading'

def test_missing_benchmark_raises(tmp_path):
    folder = tmp_path / TAG
    folder.mkdir()
    with pytest.raises(FileNotFoundError):
        rrg.load_folder(str(folder), 'SPY')
