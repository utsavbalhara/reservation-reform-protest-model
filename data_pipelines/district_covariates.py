"""District covariates and boundaries for the v3 protest model.

Covariates (Census 2011, as scraped by pigshell/india-census-2011, files in data/raw/census/):
  - urban_share: urban population over total population (Primary Census Abstract, Urban and Total rows);
  - literacy_rate: literates over the population aged 7 and above (PCA);
  - phone_share: share of households owning a telephone or mobile phone (Houselisting PCA, assets_Tel).
They are joined to data/derived/census2011_district_sc_st.csv on state and district codes and written to
data/derived/census2011_district_covariates.csv.

Boundaries: Census 2011 districts from DataMeet (github.com/datameet/maps, website/docs/data/geojson/dists11.geojson,
CC BY 4.0), downloaded to data/raw/maps/. Coordinates are rounded and thinned, and the result is written to
data/derived/district_boundaries_2011.json keyed by the national Census 2011 district code, for drawing maps only.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RAW = REPOSITORY_ROOT / "data" / "raw"
DERIVED = REPOSITORY_ROOT / "data" / "derived"
CENSUS_SOURCE = "https://raw.githubusercontent.com/pigshell/india-census-2011/master/"
MAP_SOURCE = "https://raw.githubusercontent.com/datameet/maps/master/website/docs/data/geojson/dists11.geojson"


def download(url, path):
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    return path


def covariates():
    pca = pd.read_csv(download(CENSUS_SOURCE + "pca-full.csv", RAW / "census" / "pca-full.csv"))
    total = pca[pca.TRU == "Total"].set_index(["State", "District"])
    urban = pca[pca.TRU == "Urban"].set_index(["State", "District"])
    table = pd.DataFrame({
        "urban_share": (urban.TOT_P.reindex(total.index).fillna(0) / total.TOT_P),
        "literacy_rate": total.P_LIT / (total.TOT_P - total.P_06),
    })
    housing = pd.read_csv(download(CENSUS_SOURCE + "hlpca-total.csv", RAW / "census" / "hlpca-total.csv"), dtype=str)
    housing = housing[housing["Rural/Urban"].str.strip() == "Total"]
    housing = housing.assign(State=housing["State Code"].astype(int), District=housing["District Code"].astype(int))
    phone = housing.set_index(["State", "District"])["assets_Tel"].astype(float) / 100
    table["phone_share"] = phone.groupby(level=[0, 1]).first().reindex(table.index)
    districts = pd.read_csv(DERIVED / "census2011_district_sc_st.csv")
    joined = districts.join(table, on=["state_code", "district_code"])
    for column in ("urban_share", "literacy_rate", "phone_share"):
        missing = joined[column].isna()
        if missing.any():
            # Fill gaps with the state mean, then the national mean.
            joined.loc[missing, column] = joined.groupby("state")[column].transform("mean")[missing]
            joined[column] = joined[column].fillna(joined[column].mean())
            print(f"{column}: {int(missing.sum())} districts filled with state means")
    out = joined[["state_code", "district_code", "district", "state", "urban_share", "literacy_rate", "phone_share"]]
    out.to_csv(DERIVED / "census2011_district_covariates.csv", index=False, float_format="%.4f")
    print(out.describe().round(3))
    return out


def thin(ring, step):
    ring = np.round(np.asarray(ring, float), 3)
    kept = ring[::step]
    if len(kept) < 4:
        kept = ring
    if not np.array_equal(kept[0], kept[-1]):
        kept = np.vstack([kept, kept[:1]])
    return kept.tolist()


def boundaries(step=6):
    geo = json.loads(download(MAP_SOURCE, RAW / "maps" / "dists11.geojson").read_text())
    shapes = {}
    for feature in geo["features"]:
        properties, geometry = feature["properties"], feature["geometry"]
        if properties.get("censuscode") is None:
            continue
        key = str(int(properties["censuscode"]))
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        rings = [thin(polygon[0], step) for polygon in polygons if len(polygon[0]) >= 4]
        shapes.setdefault(key, []).extend(rings)
    path = DERIVED / "district_boundaries_2011.json"
    path.write_text(json.dumps({"source": "DataMeet, Census 2011 districts (CC BY 4.0), thinned", "districts": shapes}, separators=(",", ":")))
    print(f"{len(shapes)} districts, {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    covariates()
    boundaries()
