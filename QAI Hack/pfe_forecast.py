#!/usr/bin/env python3
"""
Time-series driven Potential Future Exposure (PFE) estimation for a European option.

Given a historical marketplace price series, this script:
  * estimates Geometric Brownian Motion (GBM) parameters from log-returns,
  * simulates future price paths,
  * values a European option along the path,
  * derives counterparty credit risk metrics including PFE and expected exposure.

Example:
    python pfe_forecast.py \
        --data-path marketprice_structured.csv \
        --strike 2750 \
        --risk-free-rate 0.025 \
        --maturity-days 90 \
        --horizon-days 60 \
        --paths 20000 \
        --quantile 0.97
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PFE forecast for a European option using historical prices.")
    parser.add_argument("--data-path", default="marketprice_structured.csv", help="CSV with Date,Close columns.")
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
    return parser.parse_args(argv)


@dataclass
class MarketStats:
    annual_drift: float
    annual_vol: float
    spot: float


def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("Expected columns 'Date' and 'Close' in the input dataset.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
    df = df.set_index("Date")
    return df


def estimate_gbm_parameters(series: pd.Series) -> MarketStats:
    log_returns = np.log(series / series.shift(1)).dropna()
    if log_returns.empty:
        raise ValueError("Insufficient data to compute log returns.")

    annual_drift = log_returns.mean() * TRADING_DAYS
    annual_vol = log_returns.std(ddof=1) * math.sqrt(TRADING_DAYS)

    if annual_vol <= 0:
        raise ValueError("Volatility is non-positive; check input data variability.")

    return MarketStats(annual_drift=annual_drift, annual_vol=annual_vol, spot=series.iloc[-1])


def simulate_gbm_paths(
    spot: float,
    annual_drift: float,
    annual_vol: float,
    horizon_days: int,
    paths: int,
    dt: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    drift = (annual_drift - 0.5 * annual_vol ** 2) * dt
    diffusion = annual_vol * math.sqrt(dt)

    increments = drift + diffusion * rng.standard_normal(size=(paths, horizon_days))
    log_factors = np.cumsum(increments, axis=1)
    path_matrix = np.empty((paths, horizon_days + 1), dtype=float)
    path_matrix[:, 0] = spot
    path_matrix[:, 1:] = spot * np.exp(log_factors)
    return path_matrix


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

        exposures.append(
            {
                "step": step,
                "time_years": time_index,
                "mean_spot": float(np.mean(spot_slice)),
                "ee": float(np.mean(positive_values)),
                "pfe": float(np.quantile(positive_values, quantile)),
                "exposure_std": float(np.std(positive_values, ddof=1)),
            }
        )

    profile_df = pd.DataFrame(exposures)
    return profile_df


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    dt = 1.0 / TRADING_DAYS

    prices = load_prices(args.data_path)
    stats = estimate_gbm_parameters(prices["Close"])

    strike = stats.spot if args.strike is None else args.strike

    if args.horizon_days <= 0:
        raise ValueError("Forecast horizon must be positive.")

    maturity_steps = max(int(round(args.maturity_days)), 1)
    horizon_steps = int(round(args.horizon_days))
    maturity_steps = min(max(maturity_steps, 1), 5 * TRADING_DAYS)  # cap to five trading years

    paths = simulate_gbm_paths(
        spot=stats.spot,
        annual_drift=stats.annual_drift,
        annual_vol=stats.annual_vol,
        horizon_days=horizon_steps,
        paths=args.paths,
        dt=dt,
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
    )

    # Attach calendar dates for readability.
    start_date = prices.index[-1]
    profile["date"] = pd.bdate_range(start=start_date + pd.Timedelta(days=1), periods=len(profile), freq="B")

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
    print(f"Sample size: {len(prices):d} observations from {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Last close (spot): {stats.spot:,.4f}")
    print(f"Annualised drift: {stats.annual_drift:.4%}")
    print(f"Annualised volatility: {stats.annual_vol:.4%}")

    print("\n=== Option & simulation setup ===")
    print(f"European call strike: {strike:,.4f}")
    print(f"Risk-free rate: {args.risk_free_rate:.2%}")
    print(f"Maturity: {args.maturity_days} business days (~{args.maturity_days/ TRADING_DAYS:.2f} years)")
    print(f"Forecast horizon: {args.horizon_days} business days")
    print(f"Monte Carlo scenarios: {args.paths:,d}")
    print(f"Current option PV (Black-Scholes): {option_today:,.4f}")

    print("\n=== Exposure profile ===")
    display_cols = ["date", "mean_spot", "ee", "pfe", "exposure_std"]
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

