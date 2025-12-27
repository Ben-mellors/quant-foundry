import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------
# CONFIG (edit these)
# ----------------------------
START = "2015-01-01"
END = None  # None = up to today

# Risk-on (growth) UCITS tickers (LSE)
RISK_ON = ["CNDX.L", "SEMI.L", "IITU.L", "IUMF.L"]  # Nasdaq100, Semis, US Tech sector, US Momentum factor

# Defensive UCITS tickers (LSE)
DEFENSIVE = ["IGLS.L", "ERNS.L"]  # short gilts 0-5y, GBP ultrashort bond

BENCHMARK = "CNDX.L"  # compare vs this buy&hold

LOOKBACK_MOM = 252        # ~12 months trading days
MA_TREND = 200            # trend filter window
VOL_WINDOW = 20           # realised vol window (days)
TARGET_ANN_VOL = 0.12     # 12% annual target vol
COST_BPS = 2.0            # 2 bps per unit turnover (0.02%)
CASH_YIELD_ANN = 0.03     # 3% annual cash yield (rough, simple)
REBALANCE_WEEKDAY = 4     # Friday = 4

# Grid search space (feel free to shrink/expand)
TOP_N_CHOICES = [1, 2]
THRESH_CHOICES = [0.00, 0.01, 0.02, 0.03]  # 0%..3% absolute momentum threshold

np.set_printoptions(suppress=True, precision=4)

# ----------------------------
# Helpers
# ----------------------------
def robust_download(tickers, start, end=None):
    """
    Returns a DataFrame of prices (one column per ticker).
    Tries Adj Close, falls back to Close.
    Handles yfinance returning MultiIndex columns.
    """
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
    if raw is None or len(raw) == 0:
        raise RuntimeError("No data returned by yfinance. Check tickers or internet.")

    # If multiple tickers, yfinance usually gives MultiIndex columns: (Field, Ticker)
    def get_field(df, field):
        if isinstance(df.columns, pd.MultiIndex):
            if field in df.columns.get_level_values(0):
                out = df[field].copy()
                return out
            return None
        # Single ticker -> simple columns
        if field in df.columns:
            return df[[field]].copy()
        return None

    px = get_field(raw, "Adj Close")
    if px is None:
        px = get_field(raw, "Close")
    if px is None:
        raise RuntimeError("Couldn't find Adj Close or Close in downloaded data.")

    # If single column, rename it to ticker
    if not isinstance(px, pd.DataFrame):
        px = pd.DataFrame(px)

    # Ensure columns are tickers (for single ticker case)
    if len(tickers) == 1:
        px.columns = tickers
    else:
        # Sometimes px columns come as tickers already - good
        pass

    px = px.dropna(how="all")
    return px

def sharpe_ratio(daily_ret, ann_factor=252):
    x = pd.Series(daily_ret).dropna()
    if x.std(ddof=0) == 0 or len(x) < 5:
        return np.nan
    return np.sqrt(ann_factor) * x.mean() / x.std(ddof=0)

def max_drawdown(equity_curve):
    eq = pd.Series(equity_curve).dropna()
    if len(eq) == 0:
        return np.nan
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return dd.min()

def last_trading_day_each_month(idx):
    # Boolean mask for last trading day in each month
    s = pd.Series(idx, index=idx)
    return s.groupby([idx.year, idx.month]).transform("max") == idx

def is_weekday(idx, wd):
    return np.array([d.weekday() == wd for d in idx], dtype=bool)

# ----------------------------
# Core strategy
# ----------------------------
def run_dual_momentum_vol_target(prices, top_n=1, threshold=0.01):
    """
    Dual momentum:
    - Monthly: pick top_n risk-on assets by 12m momentum, BUT only if:
        - momentum >= threshold
        - price >= MA200 (trend filter)
      else go defensive (equal weight across DEFENSIVE assets)
    - Weekly: vol targeting to TARGET_ANN_VOL (cap 100% exposure)
    Costs: COST_BPS per unit turnover when weights change.
    Cash: leftover weight goes to cash, earns CASH_YIELD_ANN (simple daily).
    """
    tickers = list(prices.columns)
    idx = prices.index

    rebal_month = last_trading_day_each_month(idx)
    rebal_week = is_weekday(idx, REBALANCE_WEEKDAY)

    # daily returns
    ret = prices.pct_change().to_numpy(dtype=float)

    # momentum and trend signals computed on prices
    px = prices.to_numpy(dtype=float)
    mom = np.full_like(px, np.nan)
    ma = np.full_like(px, np.nan)

    for j in range(px.shape[1]):
        s = pd.Series(px[:, j], index=idx)
        mom[:, j] = (s / s.shift(LOOKBACK_MOM) - 1.0).to_numpy(dtype=float)
        ma[:, j] = s.rolling(MA_TREND).mean().to_numpy(dtype=float)

    # Map tickers to indices
    t2i = {t: i for i, t in enumerate(tickers)}
    risk_on_idx = [t2i[t] for t in RISK_ON if t in t2i]
    def_idx = [t2i[t] for t in DEFENSIVE if t in t2i]

    if len(risk_on_idx) < 2:
        raise RuntimeError("Not enough risk-on tickers found in price data.")
    if len(def_idx) < 1:
        raise RuntimeError("No defensive tickers found in price data.")

    n = len(idx)
    w = np.zeros((n, px.shape[1]), dtype=float)  # weights per day
    cash_w = np.zeros(n, dtype=float)

    # Start fully defensive until enough history
    current_w = np.zeros(px.shape[1], dtype=float)
    current_w[def_idx] = 1.0 / len(def_idx)

    target_daily_vol = TARGET_ANN_VOL / np.sqrt(252)
    cash_daily = CASH_YIELD_ANN / 252.0
    cost = COST_BPS / 10000.0  # bps -> decimal

    for i in range(n):
        # Monthly: choose which assets are "on"
        if rebal_month[i] and i > max(LOOKBACK_MOM, MA_TREND):
            m = mom[i, risk_on_idx]
            trend_ok = px[i, risk_on_idx] >= ma[i, risk_on_idx]
            elig = (m >= threshold) & trend_ok & np.isfinite(m)

            if np.any(elig):
                elig_idx = np.array(risk_on_idx)[elig]
                elig_m = m[elig]
                # pick top_n by momentum
                pick = elig_idx[np.argsort(elig_m)[-top_n:]]
                # inverse vol weights among picks
                # compute vol from last VOL_WINDOW days returns
                window = ret[max(0, i - VOL_WINDOW + 1): i + 1, :]
                vols = []
                for j in pick:
                    rj = pd.Series(window[:, j]).dropna()
                    vols.append(rj.std(ddof=0) if len(rj) > 3 else np.nan)
                vols = np.array(vols, dtype=float)
                vols = np.where((vols <= 0) | ~np.isfinite(vols), np.nan, vols)

                inv = 1.0 / vols
                if np.all(~np.isfinite(inv)):
                    base = np.ones(len(pick)) / len(pick)
                else:
                    inv = np.where(np.isfinite(inv), inv, 0.0)
                    base = inv / inv.sum() if inv.sum() > 0 else np.ones(len(pick)) / len(pick)

                current_w[:] = 0.0
                current_w[pick] = base
            else:
                # Go defensive
                current_w[:] = 0.0
                current_w[def_idx] = 1.0 / len(def_idx)

        # Weekly: scale total exposure to hit target vol (cap at 100%)
        scaled_w = current_w.copy()
        if rebal_week[i] and i > VOL_WINDOW:
            window = ret[i - VOL_WINDOW + 1: i + 1, :]
            # portfolio realised vol estimate
            port_r = np.nansum(window * scaled_w, axis=1)
            pr = pd.Series(port_r).dropna()
            realised = pr.std(ddof=0) if len(pr) > 5 else np.nan
            if realised and np.isfinite(realised) and realised > 0:
                scale = min(1.0, target_daily_vol / realised)  # NO leverage
                scaled_w = scaled_w * scale

        # cash is whatever isn't allocated (can happen due to scaling)
        cw = max(0.0, 1.0 - float(np.nansum(scaled_w)))
        cash_w[i] = cw
        w[i, :] = scaled_w

    # Apply costs + cash yield in equity curve
    eq = np.ones(n, dtype=float)
    daily_ret = np.zeros(n, dtype=float)

    prev_w = w[0].copy()
    prev_cash = cash_w[0]

    for i in range(1, n):
        r = ret[i, :]
        # portfolio return from risky + defensive assets
        pr = float(np.nansum(prev_w * r)) + prev_cash * cash_daily

        # turnover cost when weights change
        turnover = float(np.nansum(np.abs(w[i, :] - prev_w))) + abs(cash_w[i] - prev_cash)
        pr -= turnover * cost

        daily_ret[i] = pr
        eq[i] = eq[i - 1] * (1.0 + pr)

        prev_w = w[i].copy()
        prev_cash = cash_w[i]

    return daily_ret, eq, w, cash_w

# ----------------------------
# Main
# ----------------------------
def main():
    universe = sorted(list(set(RISK_ON + DEFENSIVE + [BENCHMARK])))
    print("\nUK UCITS universe:", universe)

    px = robust_download(universe, START, END)

    # Keep only dates where we have all prices (clean comparison)
    px = px.dropna(how="any")
    if len(px) < 400:
        raise RuntimeError("Not enough overlapping history after aligning tickers.")

    # Benchmarks
    bench_px = px[BENCHMARK]
    bench_ret = bench_px.pct_change().fillna(0.0).to_numpy(dtype=float)
    bench_eq = np.cumprod(1.0 + bench_ret)

    ew_ret = px.pct_change().fillna(0.0).mean(axis=1).to_numpy(dtype=float)
    ew_eq = np.cumprod(1.0 + ew_ret)

    print("\nBENCHMARKS")
    print(f"Window: {px.index[0].date()} -> {px.index[-1].date()}")
    print(f"{BENCHMARK} B&H final={bench_eq[-1]:.2f}x | sharpe={sharpe_ratio(bench_ret):.2f} | maxDD={max_drawdown(bench_eq):.2%}")
    print(f"EW  B&H final={ew_eq[-1]:.2f}x | sharpe={sharpe_ratio(ew_ret):.2f} | maxDD={max_drawdown(ew_eq):.2%}")

    print("\nHold-each-asset:")
    for t in universe:
        r = px[t].pct_change().fillna(0.0).to_numpy(dtype=float)
        e = np.cumprod(1.0 + r)
        print(f"- {t:6s} final={e[-1]:.2f}x | sharpe={sharpe_ratio(r):.2f} | maxDD={max_drawdown(e):.2%}")

    # Search for winners vs benchmark on all 3 metrics
    bench_final = bench_eq[-1]
    bench_sh = sharpe_ratio(bench_ret)
    bench_dd = max_drawdown(bench_eq)

    winners = []
    for top_n in TOP_N_CHOICES:
        for thr in THRESH_CHOICES:
            strat_ret, strat_eq, w, cash_w = run_dual_momentum_vol_target(px, top_n=top_n, threshold=thr)
            sf = strat_eq[-1]
            ss = sharpe_ratio(strat_ret)
            sd = max_drawdown(strat_eq)
            avg_exp = float(np.nanmean(np.nansum(w, axis=1)))

            beats = (sf > bench_final) and (ss > bench_sh) and (sd > bench_dd)  # dd is negative, "greater" means less bad
            name = f"DM top{top_n} thr={int(thr*100)}%"

            row = dict(name=name, final=sf, sharpe=ss, maxdd=sd, avgexp=avg_exp, beats=beats)
            if beats:
                winners.append(row)

    print("\n--- SEARCH RESULTS (must beat benchmark on final+Sharpe+maxDD) ---")
    if not winners:
        print("No winners found in this small grid. Expand universe or allow leverage, or relax one constraint.")
    else:
        winners = sorted(winners, key=lambda d: d["sharpe"], reverse=True)
        print(f"FOUND {len(winners)} winners. Best by Sharpe:")
        best = winners[0]
        print(f"{best['name']} | final={best['final']:.2f}x | sharpe={best['sharpe']:.2f} | maxDD={best['maxdd']:.2%} | avgExp={best['avgexp']:.2f}")

        print("\nTop 10 winners:")
        for r in winners[:10]:
            print(f"- {r['name']:14s} final={r['final']:.2f}x | sharpe={r['sharpe']:.2f} | maxDD={r['maxdd']:.2%} | avgExp={r['avgexp']:.2f}")

if __name__ == "__main__":
    main()

