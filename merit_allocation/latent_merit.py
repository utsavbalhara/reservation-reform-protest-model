"""A latent merit scale for JEE (Advanced) candidates, fitted to the category of every candidate on the Common Rank
List (CRL).

Each category's candidates are modelled as normally distributed on a common merit scale, with the General (GEN)
category fixed at mean 0 and standard deviation 1. For given parameters, the expected number of candidates above a
point z is G(z) = sum over categories of N_c (1 - Phi((z - mu_c) / sigma_c)), so CRL rank r sits at the z where
G(z) = r - 1/2, and the probability that the candidate at that rank belongs to category c is proportional to
N_c phi_c(z). The fit maximises the likelihood of the observed categories at all CRL ranks.

Checks (validate_fit):
1. In sample: category shares by CRL rank decile, observed against fitted.
2. Out of sample: the GEN-EWS and OBC-NCL rank lists end at the same aggregate-marks cutoff, as do the SC and ST lists,
   so the latent cutoffs implied by each list's size should agree within each pair.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from .jee_advanced_data import CATEGORIES, appeared_by_category, read_rank_table

GRID = np.linspace(-6.0, 9.0, 6001)


@dataclass
class LatentMeritFit:
    year: int
    candidates: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    log_likelihood: float
    crl_size: int
    imputed_counts: bool

    def exceedance(self, z):
        z = np.asarray(z, float)[..., None]
        return (self.candidates * (1 - ndtr((z - self.mean) / self.sd))).sum(axis=-1)

    def latent_at_rank(self, ranks):
        counts = self.exceedance(GRID)
        return np.interp(-np.log(np.asarray(ranks, float) - 0.5), -np.log(np.maximum(counts, 1e-300)), GRID)

    def composition(self, z):
        z = np.asarray(z, float)[..., None]
        density = self.candidates * np.exp(-0.5 * ((z - self.mean) / self.sd) ** 2) / self.sd
        return density / density.sum(axis=-1, keepdims=True)


def _unpack(x):
    return np.concatenate([[0.0], x[:4]]), np.concatenate([[1.0], np.exp(x[4:])])


def fit_year(year: int) -> LatentMeritFit:
    ranks, categories, _ = read_rank_table(year)
    candidates, imputed = appeared_by_category(year)

    def negative_log_likelihood(x):
        mean, sd = _unpack(x)
        trial = LatentMeritFit(year, candidates, mean, sd, 0.0, len(ranks), imputed)
        probabilities = trial.composition(trial.latent_at_rank(ranks))[np.arange(len(ranks)), categories]
        return -np.sum(np.log(np.maximum(probabilities, 1e-300)))

    x = np.array([0.0, -0.3, -1.0, -1.3, 0.0, 0.0, 0.0, 0.0])
    for tolerance in (1e-4, 1e-6):
        x = minimize(negative_log_likelihood, x, method="Nelder-Mead",
                     options={"maxiter": 20000, "maxfev": 20000, "xatol": tolerance, "fatol": tolerance}).x
    mean, sd = _unpack(x)
    return LatentMeritFit(year, candidates, mean, sd, -float(negative_log_likelihood(x)), len(ranks), imputed)


def validate_fit(fit: LatentMeritFit) -> dict:
    ranks, categories, list_sizes = read_rank_table(fit.year)
    fitted = fit.composition(fit.latent_at_rank(ranks))
    by_decile = []
    for index, members in enumerate(np.array_split(np.arange(len(ranks)), 10)):
        observed = np.bincount(categories[members], minlength=len(CATEGORIES)) / len(members)
        by_decile.append({"decile": index + 1, "observed": np.round(observed, 4).tolist(), "fitted": np.round(fitted[members].mean(axis=0), 4).tolist()})
    worst = max(max(abs(o - f) for o, f in zip(row["observed"], row["fitted"])) for row in by_decile)
    implied = {}
    for category, size in list_sizes.items():
        c = CATEGORIES.index(category)
        count_above = fit.candidates[c] * (1 - ndtr((GRID - fit.mean[c]) / fit.sd[c]))
        implied[category] = float(np.interp(-size, -count_above, GRID))
    return {"composition_by_decile": by_decile, "largest_decile_share_error": round(float(worst), 4),
            "latent_cutoff_implied_by_list_size": {key: round(value, 3) for key, value in implied.items()},
            "crl_latent_cutoff": round(float(fit.latent_at_rank([len(ranks)])[0]), 3),
            "pair_gaps": {"GEN-EWS minus OBC-NCL": round(implied["GEN-EWS"] - implied["OBC-NCL"], 3),
                          "SC minus ST": round(implied["SC"] - implied["ST"], 3)}}
