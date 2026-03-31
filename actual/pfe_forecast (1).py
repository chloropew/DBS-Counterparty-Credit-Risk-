"""
Time-series driven Potential Future Exposure (PFE) estimation for a European option.

Features:
  * fits GBM dynamics to historical closes,
  * optionally enriches daily drift using a sentiment factor,
  * simulates Monte Carlo paths for exposure profiling,
  * supports custom exposure date windows,
  * reports PFE, EE, and supporting CCR metrics.

Example:
    python pfe_forecast.py --data-path marketprice_with_sentiment.csv --strike 2750 --risk-free-rate 0.025 \
        --maturity-days 90 --horizon-days 60 --paths 20000 --quantile 0.975 \
        --exposure-start 2024-08-21 --exposure-end 2024-09-26
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


TRADING_DAYS = 252


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PFE forecast for a European option using historical prices.")
    parser.add_argument("--data-path", default="marketprice_with_sentiment.csv", help="CSV with Date,Close columns.")
    parser.add_argument(
        "--sentiment-column",
        default="Sentiment_Score",
        help="Sentiment column name. Use empty string to ignore sentiment adjustments.",
    )
    parser.add_argument("--strike", type=float, default=None, help="Option strike price. Defaults to last close if omitted.")
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.02,
        help="Annualised continuously compounded risk-free rate (e.g. 0.02 for 2%%).",
    )
    parser.add_argument(
        "--maturity-days",
        type=int,
        default=90,
        help="Days until option maturity (business days).",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=60,
        help="Forecast/business horizon in days for exposure profile.",
    )
    parser.add_argument(
        "--paths",
        type=int,
        default=10000,
        help="Number of Monte Carlo scenarios.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.95,
        help="Quantile for Potential Future Exposure (e.g. 0.95).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--save-exposures",
        default=None,
        help="Optional CSV path to persist per-date exposure statistics.",
    )
    parser.add_argument(
        "--exposure-start",
        default=None,
        help="Optional first business date (YYYY-MM-DD) for the exposure profile.",
    )
    parser.add_argument(
        "--exposure-end",
        default=None,
        help="Optional last business date (YYYY-MM-DD) for the exposure profile. Requires --exposure-start.",
    )
    return parser.parse_args(argv)


def parse_business_date(value: str, label: str) -> pd.Timestamp:
    try:
        ts = pd.to_datetime(value)
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise ValueError(f"Could not parse {label}: {value}") from exc

    if pd.isna(ts):
        raise ValueError(f"Could not parse {label}: {value}")

    ts = ts.normalize()
    if ts.weekday() >= 5:
        ts = (ts + BDay(1)).normalize()
    return ts


@dataclass
class MarketStats:
    spot: float
    annual_drift: float
    annual_vol: float
    daily_mean: float
    daily_vol: float
    intercept: float
    residual_std: float
    sentiment_beta: Optional[float] = None
    sentiment_mean: Optional[float] = None
    sentiment_std: Optional[float] = None


def load_market_data(path: str, sentiment_column: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("Expected columns 'Date' and 'Close' in the input dataset.")

    if sentiment_column and sentiment_column not in df.columns:
        raise ValueError(f"Sentiment column '{sentiment_column}' not found in dataset.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
    df = df.set_index("Date")

    if sentiment_column:
        df[sentiment_column] = pd.to_numeric(df[sentiment_column], errors="coerce")

    return df


def estimate_gbm_parameters(df: pd.DataFrame, sentiment_column: Optional[str]) -> MarketStats:
    close_series = df["Close"]
    log_returns = np.log(close_series / close_series.shift(1)).dropna()
    if log_returns.empty:
        raise ValueError("Insufficient data to compute log returns.")

    sentiment_beta = None
    sentiment_mean = None
    sentiment_std = None
    intercept = float(log_returns.mean())
    residual_std = float(log_returns.std(ddof=1))

    if sentiment_column and sentiment_column in df.columns:
        sentiment_series = df.loc[log_returns.index, sentiment_column].astype(float).dropna()
        aligned_returns = log_returns.loc[sentiment_series.index]
        if not aligned_returns.empty:
            X = np.column_stack([np.ones(len(sentiment_series)), sentiment_series.values])
            y = aligned_returns.values
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
            intercept = float(coeffs[0])
            sentiment_beta = float(coeffs[1])
            fitted = X @ coeffs
            residuals = y - fitted
            residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else residual_std
            sentiment_mean = float(sentiment_series.mean())
            sentiment_std = float(sentiment_series.std(ddof=1)) if len(sentiment_series) > 1 else 0.0
            log_returns = aligned_returns  # use aligned sample for statistics when sentiment is present

    daily_mean = float(log_returns.mean())
    daily_vol = float(log_returns.std(ddof=1))
    if sentiment_beta is not None and sentiment_mean is not None:
        daily_mean = intercept + sentiment_beta * sentiment_mean
    else:
        sentiment_mean = None
        sentiment_std = None
        residual_std = daily_vol

    residual_std = float(max(residual_std, 1e-12))
    daily_vol = float(max(daily_vol, 0.0))

    annual_drift = daily_mean * TRADING_DAYS
    annual_vol = daily_vol * math.sqrt(TRADING_DAYS)

    if annual_vol <= 0:
        raise ValueError("Volatility is non-positive; check input data variability.")

    return MarketStats(
        spot=float(close_series.iloc[-1]),
        annual_drift=annual_drift,
        annual_vol=annual_vol,
        daily_mean=daily_mean,
        daily_vol=daily_vol,
        intercept=intercept,
        residual_std=residual_std,
        sentiment_beta=sentiment_beta,
        sentiment_mean=sentiment_mean,
        sentiment_std=sentiment_std,
    )


def simulate_gbm_paths(
    stats: MarketStats,
    horizon_days: int,
    paths: int,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(size=(paths, horizon_days))

    sentiments = None
    base_increment = stats.daily_mean
    residual_std = max(stats.residual_std, 0.0)
    if stats.sentiment_beta is not None and stats.sentiment_std is not None:
        sentiment_mean = stats.sentiment_mean if stats.sentiment_mean is not None else 0.0
        sentiment_std = max(stats.sentiment_std, 0.0)
        sentiments = rng.normal(
            loc=sentiment_mean,
            scale=sentiment_std,
            size=(paths, horizon_days),
        )
        sentiment_effect = stats.sentiment_beta * (sentiments - sentiment_mean)
        increments = base_increment + sentiment_effect + residual_std * noise
    else:
        increments = base_increment + residual_std * noise

    log_factors = np.cumsum(increments, axis=1)
    path_matrix = np.empty((paths, horizon_days + 1), dtype=float)
    path_matrix[:, 0] = stats.spot
    path_matrix[:, 1:] = stats.spot * np.exp(log_factors)
    return path_matrix, sentiments


def norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0))) if np.isscalar(x) else 0.5 * (1.0 + erf_vectorized(x))


def erf_vectorized(x: np.ndarray) -> np.ndarray:
    vec_erf = np.vectorize(math.erf)
    return vec_erf(x)


def black_scholes_call(spot: np.ndarray, strike: float, time_to_expiry: float, rate: float, vol: float) -> np.ndarray:
    if time_to_expiry <= 0:
        intrinsic = np.maximum(spot - strike, 0.0)
        return intrinsic

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol ** 2) * time_to_expiry) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t

    Nd1 = norm_cdf(d1)
    Nd2 = norm_cdf(d2)
    discounted_strike = strike * math.exp(-rate * time_to_expiry)
    return spot * Nd1 - discounted_strike * Nd2


def exposure_profile(
    paths: np.ndarray,
    strike: float,
    rate: float,
    vol: float,
    dt: float,
    maturity_steps: int,
    quantile: float,
    sentiments: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    num_steps = paths.shape[1] - 1
    exposures = []
    for step in range(1, num_steps + 1):
        time_index = step * dt
        time_to_expiry = max(maturity_steps * dt - step * dt, 0.0)
        spot_slice = paths[:, step]
        if step > maturity_steps:
            positive_values = np.zeros_like(spot_slice)
        else:
            option_values = black_scholes_call(spot_slice, strike, time_to_expiry, rate, vol)
            positive_values = np.maximum(option_values, 0.0)

        entry = {
            "step": step,
            "time_years": time_index,
            "mean_spot": float(np.mean(spot_slice)),
            "ee": float(np.mean(positive_values)),
            "pfe": float(np.quantile(positive_values, quantile)),
            "exposure_std": float(np.std(positive_values, ddof=1)),
        }
        if sentiments is not None:
            entry["mean_sentiment"] = float(np.mean(sentiments[:, step - 1]))

        exposures.append(entry)

    profile_df = pd.DataFrame(exposures)
    return profile_df


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    dt = 1.0 / TRADING_DAYS

    sentiment_column = (args.sentiment_column or "").strip()
    sentiment_column = sentiment_column if sentiment_column else None

    market_df = load_market_data(args.data_path, sentiment_column)

    exposure_end: Optional[pd.Timestamp] = None
    if args.exposure_end and not args.exposure_start:
        raise ValueError("--exposure-end requires --exposure-start.")

    if args.exposure_start:
        desired_start = parse_business_date(args.exposure_start, "--exposure-start")
    else:
        desired_start = (market_df.index[-1] + BDay(1)).normalize()

    if args.exposure_end:
        exposure_end = parse_business_date(args.exposure_end, "--exposure-end")
        if exposure_end < desired_start:
            raise ValueError("Exposure end date must be on or after exposure start date.")
    else:
        exposure_end = None

    if desired_start <= market_df.index[0]:
        raise ValueError("Exposure start must be after the first observation in the dataset.")

    if desired_start <= market_df.index[-1]:
        idx = market_df.index.searchsorted(desired_start)
        if idx == 0:
            raise ValueError("Exposure start must be after the first observation in the dataset.")
        anchor_date = market_df.index[idx - 1]
        calibration_df = market_df.iloc[: idx]
    else:
        anchor_date = market_df.index[-1]
        calibration_df = market_df

    if len(calibration_df) < 2:
        raise ValueError("Not enough observations prior to the exposure window to calibrate the model.")

    stats = estimate_gbm_parameters(calibration_df, sentiment_column)
    strike = stats.spot if args.strike is None else args.strike

    maturity_steps = max(int(round(args.maturity_days)), 1)
    maturity_steps = min(max(maturity_steps, 1), 5 * TRADING_DAYS)  # cap to five trading years

    if exposure_end is not None:
        business_dates = pd.bdate_range(start=desired_start, end=exposure_end, freq="B")
        horizon_steps = len(business_dates)
    else:
        horizon_steps = int(round(args.horizon_days))
        if horizon_steps <= 0:
            raise ValueError("Forecast horizon must be positive.")
        business_dates = pd.bdate_range(start=desired_start, periods=horizon_steps, freq="B")

    if horizon_steps <= 0:
        raise ValueError("Exposure window contains no business days.")

    paths, sentiments = simulate_gbm_paths(
        stats=stats,
        horizon_days=horizon_steps,
        paths=args.paths,
        seed=args.seed,
    )

    profile = exposure_profile(
        paths=paths,
        strike=strike,
        rate=args.risk_free_rate,
        vol=stats.annual_vol,
        dt=dt,
        maturity_steps=maturity_steps,
        quantile=args.quantile,
        sentiments=sentiments,
    )

    # Attach calendar dates for readability.
    profile["date"] = business_dates[: len(profile)]

    option_today = black_scholes_call(
        np.array([stats.spot]),
        strike=strike,
        time_to_expiry=maturity_steps * dt,
        rate=args.risk_free_rate,
        vol=stats.annual_vol,
    )[0]

    epe = float(profile["ee"].mean())
    peak_pfe = float(profile["pfe"].max())
    peak_row = profile.loc[profile["pfe"].idxmax()]

    print("\n=== Market data ===")
    print(
        f"Sample size: {len(calibration_df):d} observations from {calibration_df.index[0].date()} to {calibration_df.index[-1].date()}"
    )
    print(f"Anchor date (last observation): {anchor_date.date()}")
    print(f"Last close (spot): {stats.spot:,.4f}")
    print(f"Annualised drift: {stats.annual_drift:.4%}")
    print(f"Annualised volatility: {stats.annual_vol:.4%}")
    if stats.sentiment_beta is not None:
        print(f"Sentiment beta (per day): {stats.sentiment_beta:.6f}")
        if stats.sentiment_mean is not None:
            print(f"Sentiment mean/std: {stats.sentiment_mean:.4f} / {stats.sentiment_std:.4f}")

    if exposure_end is not None:
        horizon_label = f"{len(business_dates)} business days (custom window)"
    else:
        horizon_label = f"{len(business_dates)} business days"

    print("\n=== Option & simulation setup ===")
    print(f"European call strike: {strike:,.4f}")
    print(f"Risk-free rate: {args.risk_free_rate:.2%}")
    print(f"Maturity: {args.maturity_days} business days (~{args.maturity_days/ TRADING_DAYS:.2f} years)")
    print(f"Forecast horizon: {horizon_label}")
    print(f"Monte Carlo scenarios: {args.paths:,d}")
    print(f"Current option PV (Black-Scholes): {option_today:,.4f}")
    print(
        f"Exposure window: {business_dates[0].date()} to {business_dates[-1].date()} "
        f"({len(business_dates)} business days)"
    )

    print("\n=== Exposure profile ===")
    display_cols = ["date", "mean_spot"]
    if "mean_sentiment" in profile.columns:
        display_cols.append("mean_sentiment")
    display_cols.extend(["ee", "pfe", "exposure_std"])
    print(profile[display_cols].to_string(index=False, justify="center", float_format=lambda x: f"{x:,.4f}"))

    print("\n=== Counterparty credit risk metrics ===")
    print(f"PFE quantile: {args.quantile:.1%}")
    print(f"Peak PFE: {peak_pfe:,.4f} on {peak_row['date'].date()}")
    print(f"Expected Positive Exposure (EPE): {epe:,.4f}")

    if args.save_exposures:
        profile.to_csv(args.save_exposures, index=False)
        print(f"\nExposure profile saved to {args.save_exposures}")


if __name__ == "__main__":
    main()
