"""Agreement between two human coders of the GDELT relevance sample, and replacement of the language-model labels.

See data/coding/CODING_INSTRUCTIONS.md. Reports Krippendorff's alpha (nominal data, two coders) and Cohen's kappa,
each coder's agreement with the current labels, and writes the disagreements to resolve. With --write and a file of
resolved disagreements, replaces the labels in data/derived/gdelt_relevance_audit.csv with the human ones.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
AUDIT_PATH = REPOSITORY_ROOT / "data" / "derived" / "gdelt_relevance_audit.csv"
CODING_FOLDER = REPOSITORY_ROOT / "data" / "coding"


def krippendorff_alpha_nominal(a, b):
    """Krippendorff's alpha for two coders and nominal labels, on items both coded."""
    pairs = [(x, y) for x, y in zip(a, b) if not (pd.isna(x) or pd.isna(y))]
    values = sorted({v for pair in pairs for v in pair})
    index = {v: i for i, v in enumerate(values)}
    coincidence = np.zeros((len(values), len(values)))
    for x, y in pairs:
        coincidence[index[x], index[y]] += 1
        coincidence[index[y], index[x]] += 1
    n = coincidence.sum()
    marginals = coincidence.sum(axis=1)
    observed = n - np.trace(coincidence)
    expected = (n * n - (marginals ** 2).sum()) / (n - 1)
    return 1.0 - observed / expected if expected else float("nan")


def cohen_kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if not (pd.isna(x) or pd.isna(y))]
    x, y = np.array(pairs).T
    observed = np.mean(x == y)
    expected = sum(np.mean(x == v) * np.mean(y == v) for v in np.unique(np.concatenate([x, y])))
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def load_sheet(path):
    sheet = pd.read_csv(path)
    sheet["relevant"] = pd.to_numeric(sheet["relevant"], errors="coerce")
    return sheet.set_index("url")


def main():
    parser = argparse.ArgumentParser(description="Agreement between two coders of the relevance sample.")
    parser.add_argument("--coder1", type=Path, required=True)
    parser.add_argument("--coder2", type=Path, required=True)
    parser.add_argument("--resolved", type=Path, help="disagreements.csv with a filled 'resolved' column")
    parser.add_argument("--write", action="store_true", help="write the human labels into the audit file")
    arguments = parser.parse_args()
    audit = pd.read_csv(AUDIT_PATH)
    one, two = load_sheet(arguments.coder1), load_sheet(arguments.coder2)
    urls = audit.url.tolist()
    a, b = one.relevant.reindex(urls).to_numpy(), two.relevant.reindex(urls).to_numpy()
    model = audit.relevant.to_numpy()
    both = ~(np.isnan(a) | np.isnan(b))
    print(f"items coded by both: {both.sum()} of {len(urls)}")
    print(f"raw agreement: {np.mean(a[both] == b[both]):.3f}")
    print(f"Krippendorff's alpha (nominal): {krippendorff_alpha_nominal(a, b):.3f}")
    print(f"Cohen's kappa: {cohen_kappa(a, b):.3f}")
    for name, labels in (("coder 1", a), ("coder 2", b)):
        mask = ~np.isnan(labels) & ~np.isnan(model)
        print(f"{name} agreement with the language-model labels: {np.mean(labels[mask] == model[mask]):.3f} "
              f"(kappa {cohen_kappa(labels[mask], model[mask]):.3f})")
    disagreements = audit.loc[both & (a != b), ["episode", "date", "url"]].copy()
    disagreements["coder1"], disagreements["coder2"] = a[both & (a != b)], b[both & (a != b)]
    disagreements["resolved"] = ""
    if not arguments.resolved:
        disagreements.to_csv(CODING_FOLDER / "disagreements.csv", index=False)
        print(f"{len(disagreements)} disagreements written to {CODING_FOLDER / 'disagreements.csv'}")
    if arguments.write:
        resolved = pd.read_csv(arguments.resolved).set_index("url")["resolved"] if arguments.resolved else pd.Series(dtype=float)
        human = np.where(both & (a == b), a, np.nan)
        for i, url in enumerate(urls):
            if url in resolved.index and not pd.isna(resolved[url]) and str(resolved[url]).strip() != "":
                human[i] = float(resolved[url])
        missing = int(np.isnan(human).sum())
        if missing:
            print(f"{missing} items still have no human label; they keep no label (dropped from the precision estimate).")
        audit["relevant"] = human
        audit["coder"] = "two independent human coders from the article text; disagreements resolved by discussion"
        audit.to_csv(AUDIT_PATH, index=False)
        print(f"Wrote human labels to {AUDIT_PATH}")


if __name__ == "__main__":
    main()
