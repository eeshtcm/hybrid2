# ========= YOUR IMPORTS (as you already have) =========
import numpy as np
import pandas as pd
from PDScripts.bloomberg.pdblp import BLP     # or from blp.remote import BLP if that's your env
import datetime as dt
# import matplotlib.pyplot as plt  # only if you want to plot

blp = BLP()

# ========= 1) DATA FETCH (kept compatible with your code) =========
def get_spots_df(tickers, lookback='-20y'):
    """
    Fetch PX_LAST timeseries for the input tickers.
    Works for indices and single-name/FX/commodities. 
    Returns a DataFrame indexed by date with one column per ticker.
    """
    data = []
    blp = BLP()
    for tick in tickers:
        if 'Index' in tick:
            # Indices usually don't need adjustment flags
            aux_df = blp.bdh([tick,], ['PX_LAST'], startDate=lookback)
        else:
            # Single names / FX / Commodities: include adjustments
            aux_df = blp.bdh([tick,], ['PX_LAST'], startDate=lookback,
                             adjustmentNormal=True, adjustmentAbnormal=True, adjustmentSplit=True)
        aux_df.columns = [tick]
        data.append(aux_df.copy(True))

    aggr_df = pd.concat(data, axis=1)
    aggr_df.columns = tickers
    return aggr_df

# ========= 2) CLASSIFICATION & TRANSFORMS =========
# Asset class tags we will use:
#   "Equity", "FX", "Commodity", "Rates"

def log_returns(series: pd.Series) -> pd.Series:
    out = np.log(series).diff()
    out.name = f"{series.name}_logret"
    return out

def level_changes(series: pd.Series, to_bps: bool = True) -> pd.Series:
    """
    For rates series quoted in % (typical on Bloomberg), we use *level changes*.
    If to_bps=True, convert percentage point changes to basis points (×100).
    """
    out = series.diff()
    if to_bps:
        out = out * 100.0  # 1.00% → 1.00 *100 = 100 bps change
    out.name = f"{series.name}_dlevel"
    return out

def transform_by_asset_class(series: pd.Series, asset_class: str, rates_in_bps: bool = True) -> pd.Series:
    if asset_class in ("Equity", "FX", "Commodity"):
        return log_returns(series)
    elif asset_class == "Rates":
        return level_changes(series, to_bps=rates_in_bps)
    else:
        raise ValueError(f"Unknown asset class: {asset_class}")

# Optional: simple heuristic classifier if you don’t pass a mapping
def guess_asset_class(ticker: str) -> str:
    t = ticker.lower()
    if 'curncy' in t:
        # FX tickers like 'EURUSD Curncy', 'XAU Curncy'
        return 'FX' if 'xau' not in t and 'xag' not in t and 'xpt' not in t else 'Commodity'
    if 'cmdty' in t:
        return 'Commodity'
    if 'comdty' in t:
        return 'Commodity'
    if 'ussw' in t or 'usgg' in t or 'govt' in t or 'swap' in t or 'rate' in t or 'yld' in t:
        return 'Rates'
    return 'Equity'  # default

# ========= 3) CORRELATION CORE =========
def _prep_pair(df_prices: pd.DataFrame,
               a: str, b: str,
               classes: dict | None = None,
               rates_in_bps: bool = True,
               winsorize: tuple[float, float] | None = (0.01, 0.99)) -> pd.DataFrame:
    """
    - Picks asset classes (from `classes` mapping or heuristics)
    - Applies the correct transform (log/log or log/dLevel)
    - Aligns & cleans the two series
    """
    if classes is None:
        classes = {}
    cls_a = classes.get(a, guess_asset_class(a))
    cls_b = classes.get(b, guess_asset_class(b))

    # align
    sub = df_prices[[a, b]].dropna().copy()
    # transforms
    ta = transform_by_asset_class(sub[a], cls_a, rates_in_bps=rates_in_bps)
    tb = transform_by_asset_class(sub[b], cls_b, rates_in_bps=rates_in_bps)

    X = pd.concat([ta, tb], axis=1).dropna()

    # optional winsorization — trims tail shocks so correlation isn’t dominated by outliers
    if winsorize is not None:
        lq, uq = winsorize
        X.iloc[:, 0] = X.iloc[:, 0].clip(X.iloc[:, 0].quantile(lq), X.iloc[:, 0].quantile(uq))
        X.iloc[:, 1] = X.iloc[:, 1].clip(X.iloc[:, 1].quantile(lq), X.iloc[:, 1].quantile(uq))
        X = X.dropna()

    return X

def unconditional_corr(df_prices: pd.DataFrame,
                       a: str, b: str,
                       classes: dict | None = None,
                       rates_in_bps: bool = True,
                       winsorize: tuple[float, float] | None = (0.01, 0.99)) -> dict:
    """
    Full-sample Pearson correlation of the *transformed* pair.
    Returns dict with rho, n, and 95% Fisher-z CI.
    """
    X = _prep_pair(df_prices, a, b, classes, rates_in_bps, winsorize)
    r = float(X.iloc[:, 0].corr(X.iloc[:, 1]))
    n = len(X)
    out = {"rho": r, "n": n}

    if n > 10 and abs(r) < 0.9999:
        # Fisher z CI
        z = 0.5 * np.log((1 + r) / (1 - r))
        se = 1 / np.sqrt(n - 3)
        z_lo, z_hi = z - 1.96*se, z + 1.96*se
        r_lo = (np.exp(2*z_lo) - 1) / (np.exp(2*z_lo) + 1)
        r_hi = (np.exp(2*z_hi) - 1) / (np.exp(2*z_hi) + 1)
        out.update({"rho_lo95": float(r_lo), "rho_hi95": float(r_hi)})
    return out

def rolling_corr(df_prices: pd.DataFrame,
                 a: str, b: str,
                 window: int = 126,
                 min_periods: int | None = None,
                 classes: dict | None = None,
                 rates_in_bps: bool = True,
                 winsorize: tuple[float, float] | None = (0.01, 0.99)) -> pd.Series:
    """
    Rolling Pearson correlation of the transformed pair.
    window = number of obs (126 ≈ 6m, 252 ≈ 1y).
    """
    if min_periods is None:
        min_periods = max(20, window // 3)
    X = _prep_pair(df_prices, a, b, classes, rates_in_bps, winsorize)
    rc = X.iloc[:, 0].rolling(window, min_periods=min_periods).corr(X.iloc[:, 1])
    rc.name = f"rolling_corr_{a.replace(' ','_')}_{b.replace(' ','_')}"
    return rc

# ========= 4) SHORTCUT WRAPPERS (exact pairs from the brief) =========
def corr_equity_fx(df_prices, equity_tkr, fx_tkr, **kwargs):
    classes = kwargs.pop("classes", {equity_tkr: "Equity", fx_tkr: "FX"})
    return unconditional_corr(df_prices, equity_tkr, fx_tkr, classes=classes, **kwargs)

def corr_equity_cmdty(df_prices, equity_tkr, cmdty_tkr, **kwargs):
    classes = kwargs.pop("classes", {equity_tkr: "Equity", cmdty_tkr: "Commodity"})
    return unconditional_corr(df_prices, equity_tkr, cmdty_tkr, classes=classes, **kwargs)

def corr_equity_rates(df_prices, equity_tkr, rates_tkr, **kwargs):
    classes = kwargs.pop("classes", {equity_tkr: "Equity", rates_tkr: "Rates"})
    return unconditional_corr(df_prices, equity_tkr, rates_tkr, classes=classes, **kwargs)

# ========= 5) EXAMPLES WITH YOUR BLOOMBERG CALLS =========
# Example 1: Equity–Commodity (SPX vs Gold)
tickers = ['SPX Index', 'XAU Curncy']  # (you already use these)
df_px = get_spots_df(tickers, '-25y')

# Unconditional
res_eq_cmdty = corr_equity_cmdty(df_px, 'SPX Index', 'XAU Curncy')
print("Equity–Commodity (SPX vs XAU) correlation:", res_eq_cmdty)

# Rolling (6-month window)
rc_eq_cmdty = rolling_corr(df_px, 'SPX Index', 'XAU Curncy',
                           window=126,
                           classes={'SPX Index':'Equity','XAU Curncy':'Commodity'})
print("Rolling corr tail (SPX/XAU):")
print(rc_eq_cmdty.dropna().tail())

# Example 2: Equity–FX (SPX vs EURUSD)
tickers2 = ['SPX Index', 'EURUSD Curncy']
df_px2 = get_spots_df(tickers2, '-25y')

res_eq_fx = corr_equity_fx(df_px2, 'SPX Index', 'EURUSD Curncy')
print("Equity–FX (SPX vs EURUSD) correlation:", res_eq_fx)

# Example 3: Equity–Rates (SPX vs US 2y swap or 2y yield)
# Choose one rates ticker you actually use, e.g. 'USSW2 Curncy' or 'USGG2YR Index'
tickers3 = ['SPX Index', 'USGG2YR Index']  # 2y UST yield (%)
df_px3 = get_spots_df(tickers3, '-25y')

res_eq_rates = corr_equity_rates(df_px3, 'SPX Index', 'USGG2YR Index', rates_in_bps=True)
print("Equity–Rates (SPX vs USGG2YR) correlation:", res_eq_rates)
