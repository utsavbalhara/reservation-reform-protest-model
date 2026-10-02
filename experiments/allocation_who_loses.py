"""Who loses IIT seats under a merged income-only pool, read from the stored allocation draws.

Uses only results/merged_pool_allocation.json (no new simulation). For each kept draw of the income overlay, in the main
case (reserved-first order, pool 59.5%), it computes:
  - the seats lost by SC and ST candidates, split into those above and below the income line;
  - who receives the seats that change hands;
  - how the below-line SC and ST losses vary with the unobserved income overlay (Spearman correlation with each
    category's above-line share and with the merit gradient).
Writes results/allocation_who_loses.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RESULTS_FOLDER = Path(__file__).resolve().parent.parent / "results"
SEGMENTS = ("SC_above", "SC_below", "ST_above", "ST_below", "OBC-NCL_above", "OBC-NCL_below", "GEN-EWS", "GEN_below", "GEN_above")


def interval(values):
    values = np.asarray(values, float)
    return [round(float(np.percentile(values, q)), 4) for q in (5, 50, 95)]


def main():
    data = json.loads((RESULTS_FOLDER / "merged_pool_allocation.json").read_text())
    draws = data["draws"]
    change = {s: np.array([d["main_case"]["reform"][s] - d["main_case"]["status_quo"][s] for d in draws]) for s in SEGMENTS}
    sc_st_loss_below = -(change["SC_below"] + change["ST_below"])
    sc_st_loss_above = -(change["SC_above"] + change["ST_above"])
    share_below = sc_st_loss_below / (sc_st_loss_below + sc_st_loss_above)
    gains = {s: np.maximum(change[s], 0) for s in SEGMENTS}
    total_moved = sum(np.maximum(-change[s], 0) for s in SEGMENTS)
    output = {
        "draws": len(draws),
        "seats_lost_by_sc_st_below_line": interval(sc_st_loss_below),
        "seats_lost_by_sc_st_above_line": interval(sc_st_loss_above),
        "share_of_sc_st_seat_loss_below_line": interval(share_below),
        "share_of_draws_below_line_loss_exceeds_above_line_loss": round(float(np.mean(sc_st_loss_below > sc_st_loss_above)), 3),
        "share_of_moved_seats_received": {s: interval(gains[s] / total_moved) for s in SEGMENTS if gains[s].max() > 0},
        "seats_moved": interval(total_moved),
        "share_of_draws_sc_below_loses_half_or_more": round(float(np.mean(
            [(d["main_case"]["reform"]["SC_below"] / d["main_case"]["status_quo"]["SC_below"]) <= 0.5 for d in draws])), 3),
    }
    # How the below-line losses depend on the unobserved income overlay.
    sc_below_percent = np.array([d["main_case"]["reform"]["SC_below"] / d["main_case"]["status_quo"]["SC_below"] - 1 for d in draws]) * 100
    drivers = {}
    for category in ("GEN", "OBC-NCL", "SC", "ST"):
        drivers[f"above_share_{category}"] = round(float(spearmanr([d["overlay"]["above_share"][category] for d in draws], sc_below_percent).correlation), 3)
    drivers["merit_gradient"] = round(float(spearmanr([d["overlay"]["merit_gradient"] for d in draws], sc_below_percent).correlation), 3)
    output["spearman_with_sc_below_percent_change"] = drivers
    output["sc_below_percent_change_by_year"] = {str(year): interval([p for p, d in zip(sc_below_percent, draws) if d["year"] == year])
                                                 for year in sorted({d["year"] for d in draws})}
    fits = data["latent_fits"]
    output["latent_means_by_year"] = {year: dict(zip(fit["categories"], fit["mean"])) for year, fit in fits.items()}
    path = RESULTS_FOLDER / "allocation_who_loses.json"
    path.write_text(json.dumps(output, indent=1))
    print(json.dumps(output, indent=1))


if __name__ == "__main__":
    main()
