"""Latent merit distributions for NEET (UG) 2023 from published category quantiles.

General is fixed at N(0, 1). The latent thresholds for 137 and 107 marks, and each other category's mean and standard
deviation, are fitted to the category shares at or above those marks (binomial likelihood: these counts are large, so
they act as near-exact constraints). General and EWS give one quantile each, so the EWS spread gets a prior centred on
its JEE (Advanced) estimate. The top-50 composition is not used in the fit; it is the out-of-sample check on the tail,
which is where seats are decided and where the fit is an extrapolation from the middle of the distribution.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import norm

from .neet_data import CATEGORIES, NEET_2023, quantile_targets

JEE_SPREAD_PRIOR = {"GEN-EWS": 1.04, "OBC-NCL": 1.09, "SC": 1.21, "ST": 1.22}
PRIOR_LOG_SD = 0.15


def _unpack(x):
    z137, z107 = x[0], x[1]
    mean = np.concatenate([[0.0], x[2:6]])
    sd = np.concatenate([[1.0], np.exp(x[6:10])])
    return z137, z107, mean, sd


def fit_neet(data=NEET_2023):
    targets = quantile_targets(data)
    appeared = data["appeared"]

    def negative_log_posterior(x):
        z137, z107, mean, sd = _unpack(x)
        if z107 >= z137:
            return 1e12
        value = 0.0
        above_137 = 1 - ndtr((z137 - mean) / sd)
        for index in range(len(CATEGORIES)):
            observed = targets["share_at_or_above_137"][index]
            value -= appeared[index] * (observed * np.log(above_137[index] + 1e-300) + (1 - observed) * np.log(1 - above_137[index] + 1e-300))
        for category, observed in targets["share_at_or_above_107"].items():
            index = CATEGORIES.index(category)
            above_107 = 1 - ndtr((z107 - mean[index]) / sd[index])
            value -= appeared[index] * (observed * np.log(above_107 + 1e-300) + (1 - observed) * np.log(1 - above_107 + 1e-300))
        band = sum(appeared[i] * (ndtr((z137 - mean[i]) / sd[i]) - ndtr((z107 - mean[i]) / sd[i])) for i in (0, 1))
        value += 0.5 * ((band - targets["general_and_ews_between_107_and_136"]) / 500.0) ** 2
        for category, prior in JEE_SPREAD_PRIOR.items():
            value += 0.5 * ((np.log(sd[CATEGORIES.index(category)]) - np.log(prior)) / PRIOR_LOG_SD) ** 2
        return value

    x = np.array([0.0, -0.25, 0.3, 0.0, -0.2, -0.4] + [np.log(v) for v in JEE_SPREAD_PRIOR.values()])
    for tolerance in (1e-4, 1e-7):
        x = minimize(negative_log_posterior, x, method="Nelder-Mead", options={"maxiter": 40000, "maxfev": 40000, "xatol": tolerance, "fatol": tolerance}).x
    z137, z107, mean, sd = _unpack(x)
    return {"z137": float(z137), "z107": float(z107), "mean": mean, "sd": sd, "candidates": appeared.copy()}


def top_ranks_check(fit, top=50):
    """Expected category counts among the top-ranked candidates, against the published list."""
    grid = np.linspace(-2, 8, 20001)
    exceed = (fit["candidates"] * (1 - ndtr((grid[:, None] - fit["mean"]) / fit["sd"]))).sum(axis=1)
    z_top = float(np.interp(-top, -exceed, grid))
    expected = fit["candidates"] * (1 - ndtr((z_top - fit["mean"]) / fit["sd"]))
    return {"expected": dict(zip(CATEGORIES, np.round(expected, 2).tolist())), "observed": NEET_2023["top_50"]}
