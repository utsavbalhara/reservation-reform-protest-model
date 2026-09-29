from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FOLDER = REPOSITORY_ROOT / "results"
FIGURES_FOLDER = REPOSITORY_ROOT / "figures"
VIDEOS_FOLDER = REPOSITORY_ROOT / "videos"

SURFACE = "#fcfcfb"
INK = "#1d2327"
SECONDARY_INK = "#52514e"
MUTED_INK = "#8a8984"
GRID_LINE = "#e4e3df"

GROUP_COLOURS = {"SC": "#2a78d6", "ST": "#eb6834", "OBC": "#1baf7a", "General": "#eda100"}
CATEGORY_COLOURS = {"reference": "#6f6e69", "lever": "#2a78d6", "package": "#4a3aa7", "suppression": "#eb6834"}
CATEGORY_NAMES = {"reference": "Reference", "lever": "Single lever", "package": "Package", "suppression": "Suppression (tested for backfire)"}
HIGHLIGHT_SCENARIO_COLOURS = {
    "baseline": "#6f6e69",
    "consensus_commission": "#2a78d6",
    "heavy_policing": "#eb6834",
    "hybrid_caste_subquotas": "#1baf7a",
}

TURNOUT_COLOUR_MAP = LinearSegmentedColormap.from_list(
    "neighbourhood_turnout",
    ["#f1f0ec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
)

MATERIAL_LOSS_COLOUR = "#eb6834"
SYMBOLIC_THREAT_COLOUR = "#2a78d6"


def apply_house_style():
    plt.rcParams.update({
        "font.family": ["TeX Gyre Heros", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9.5,
        "axes.labelcolor": SECONDARY_INK,
        "axes.edgecolor": GRID_LINE,
        "axes.linewidth": 0.8,
        "axes.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID_LINE,
        "grid.linewidth": 0.6,
        "xtick.color": SECONDARY_INK,
        "ytick.color": SECONDARY_INK,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": INK,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "pdf.fonttype": 42,
    })


def save_figure(figure, file_stem: str):
    FIGURES_FOLDER.mkdir(exist_ok=True)
    figure.savefig(FIGURES_FOLDER / f"{file_stem}.png", dpi=200, bbox_inches="tight")
    figure.savefig(FIGURES_FOLDER / f"{file_stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def format_lakh(lakh: float) -> str:
    if lakh >= 100:
        return f"{lakh / 100:.1f} cr"
    if lakh >= 10:
        return f"{lakh:.0f} L"
    return f"{lakh:.1f} L"
