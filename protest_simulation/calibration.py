"""Recalibrate the mean threshold so a specification's baseline hits the turnout target.

Every structural variant, and every point of the material-versus-symbolic sweep, changes how much turnout a given
threshold produces. Comparing lever effects across variants is only fair if each variant is first put back on the
same calibration target. The median peak is taken over paired runs (seeds 1000 + r and 5000 + r), so it moves
almost monotonically with the threshold and bisection converges.
"""
import numpy as np

from .monte_carlo import run_paired_monte_carlo
from .parameter_uncertainty import draw_plausible_world

CALIBRATION_TARGET_PEAK_LAKH = (20.0, 100.0)
CALIBRATION_TARGET_MIDPOINT_LAKH = float(np.sqrt(CALIBRATION_TARGET_PEAK_LAKH[0] * CALIBRATION_TARGET_PEAK_LAKH[1]))


def median_peak_lakh(population, base_parameters, run_count, interventions=(), world_sampler=draw_plausible_world, workers=None):
    runs = run_paired_monte_carlo(population, interventions, run_count, base_parameters=base_parameters, workers=workers,
                                  world_sampler=world_sampler)
    return float(np.median(runs.peak_day_protesters)) / 1e5


def calibrate_mean_threshold(population, base_parameters, target_lakh=CALIBRATION_TARGET_MIDPOINT_LAKH, run_count=20,
                             bracket=(3.0, 11.0), tolerance=0.005, world_sampler=draw_plausible_world, workers=None):
    """Return (threshold, achieved median peak in lakh, evaluations). Works on log turnout."""
    evaluations = []

    def peak_at(threshold):
        parameters = base_parameters.copy()
        parameters.mean_participation_threshold = float(threshold)
        peak = median_peak_lakh(population, parameters, run_count, world_sampler=world_sampler, workers=workers)
        evaluations.append((float(threshold), peak))
        return peak

    low, high = bracket
    if peak_at(low) < target_lakh or peak_at(high) > target_lakh:
        raise ValueError(f"target {target_lakh} lakh not bracketed by thresholds {bracket}: {evaluations}")
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if peak_at(middle) > target_lakh:
            low = middle
        else:
            high = middle
    # Take whichever end lands closer to the target on the log scale.
    candidates = [(abs(np.log(peak / target_lakh)), threshold, peak) for threshold, peak in evaluations if low - 1e-9 <= threshold <= high + 1e-9]
    _, threshold, peak = min(candidates)
    return threshold, peak, evaluations
