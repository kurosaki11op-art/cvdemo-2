"""Spatial data-science pipeline: H3 -> PySAL ESDA -> spatial-block CV ablation -> XGBoost -> TreeSHAP -> hazard routing.

Real inputs (from src/geoai/download_stac_and_xbd.py): data/buildings.geojson (+ data/flood.geojson, data/roads.graphml
optional). buildings need `damage_score` (0-4, e.g. xBD labels or Siamese-U-Net mean class). Demographic/elevation
columns (`pop_density`, `elderly_frac`, `elevation`) are used when present.
`synthetic_demo()` exists only for tests/offline smoke runs and is refused by the API unless NEXUS_ALLOW_SYNTHETIC=1.
Geometry is stored in EPSG:4326; distances/weights are computed in a local UTM CRS (e.g. EPSG:32646 for Bangladesh).
"""
import json
from pathlib import Path

import geopandas as gpd
import h3
import networkx as nx
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from esda.moran import Moran, Moran_Local
from libpysal.weights import KNN
from shapely.geometry import LineString, Point, box
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold

H3_RES = 9
STAGES = {
    "1_geoai_only": ["flood_frac", "ndwi_post", "d_ndwi"],
    "2_plus_terrain_demographics": ["flood_frac", "ndwi_post", "d_ndwi", "elevation", "pop_density", "elderly_frac"],
    "3_plus_spatial_context": ["flood_frac", "ndwi_post", "d_ndwi", "elevation", "pop_density", "elderly_frac",
                               "nbr_flood_mean"],
}
HAZARD_PENALTY = 1000.0


def synthetic_demo(n=600, seed=0, lon=90.40, lat=23.78):
    rng = np.random.default_rng(seed)
    xy = rng.normal(0, 0.006, (n, 2)) + [lon, lat]
    d = np.hypot(xy[:, 0] - lon - 0.002, xy[:, 1] - lat) / 0.006
    flood = np.clip(1.2 - d + rng.normal(0, .15, n), 0, 1)
    g = gpd.GeoDataFrame({
        "flood_frac": flood, "ndwi_post": flood * .6 - .2 + rng.normal(0, .05, n),
        "d_ndwi": flood * .5 + rng.normal(0, .05, n), "elevation": 8 - 6 * flood + rng.normal(0, .5, n),
        "pop_density": rng.gamma(4, 1500, n), "elderly_frac": rng.beta(2, 12, n)},
        geometry=[Point(p).buffer(0.00008) for p in xy], crs="EPSG:4326")
    g["damage_score"] = np.clip(4 * flood + 0.0001 * g.pop_density * .3 + rng.normal(0, .4, n), 0, 4)
    flood_poly = gpd.GeoDataFrame(geometry=[box(lon - .004, lat - .005, lon + .006, lat + .005)], crs="EPSG:4326")
    return g, flood_poly


def _centroids(g, utm):
    c = g.to_crs(utm).geometry.centroid
    return np.c_[c.x, c.y]


def _block_cv(X, y, xy, k=5, spatial=True, seed=0):
    model = lambda: xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=.08, subsample=.9, random_state=seed)
    if spatial:  # K-Means blocks on coordinates -> no train/test spatial leakage
        groups = KMeans(k, n_init=5, random_state=seed).fit_predict(xy)
        splits = GroupKFold(k).split(X, y, groups)
    else:
        splits = KFold(k, shuffle=True, random_state=seed).split(X)
    return float(np.mean([r2_score(y[te], model().fit(X[tr], y[tr]).predict(X[te])) for tr, te in splits]))


def road_graph(buildings, flood, roads_path="data/roads.graphml", synthetic=False):
    """Road graph in EPSG:4326 with `length` (m, UTM) and hazard-penalised `weight`."""
    import osmnx as ox
    if Path(roads_path).exists():
        G = ox.load_graphml(roads_path)
    elif synthetic:
        G = nx.MultiDiGraph(crs="EPSG:4326"); minx, miny, maxx, maxy = buildings.total_bounds
        xs, ys = np.linspace(minx, maxx, 14), np.linspace(miny, maxy, 14)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                G.add_node((i, j), x=x, y=y)
                for di, dj in ((1, 0), (0, 1)):
                    if i + di < 14 and j + dj < 14:
                        G.add_edge((i, j), (i + di, j + dj)); G.add_edge((i + di, j + dj), (i, j))
    else:
        minx, miny, maxx, maxy = buildings.total_bounds
        G = ox.graph_from_bbox(bbox=(minx, miny, maxx, maxy), network_type="drive")
        ox.save_graphml(G, roads_path)
    utm = buildings.estimate_utm_crs()
    hazard = flood.to_crs(utm).union_all()
    for u, v, k, d in G.edges(keys=True, data=True):
        geom = d.get("geometry") or LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])
        line = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(utm).iloc[0]
        d["length"] = line.length
        d["blocked"] = bool(line.intersects(hazard))
        d["weight"] = d["length"] * (HAZARD_PENALTY if d["blocked"] else 1.0)
    return G


class Pipeline:
    def __init__(self, buildings, flood, synthetic=False, roads_path="data/roads.graphml"):
        self.synthetic = synthetic
        self.b, self.flood = buildings.to_crs("EPSG:4326").reset_index(drop=True), flood.to_crs("EPSG:4326")
        self.utm = self.b.estimate_utm_crs()
        for c, v in (("elevation", np.nan), ("pop_density", np.nan), ("elderly_frac", np.nan)):
            if c not in self.b:
                self.b[c] = v
        if "damage_score" not in self.b:
            raise ValueError("buildings need a `damage_score` column (xBD labels or Siamese U-Net output)")
        self.G = road_graph(self.b, self.flood, roads_path, synthetic)
        self.run()

    def run(self):
        b, utm = self.b, self.utm
        xy = _centroids(b, utm)
        cen = b.to_crs(utm).geometry.centroid.to_crs(4326)
        b["lon"], b["lat"] = cen.x, cen.y
        b["h3"] = [h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(b.lat, b.lon)]
        # ESDA: KNN weights on metric coordinates
        W = KNN.from_array(xy, k=8); W.transform = "r"
        y = b.damage_score.to_numpy(float)
        mi = Moran(y, W, permutations=999); lisa = Moran_Local(y, W, permutations=499, seed=1)
        b["lisa_q"] = lisa.q; b["lisa_p"] = lisa.p_sim
        b["lisa_cluster"] = np.where(lisa.p_sim < .05, np.array(["", "HH", "LH", "LL", "HL"])[lisa.q], "ns")
        b["nbr_flood_mean"] = [b.flood_frac.to_numpy()[W.neighbors[i]].mean() for i in range(len(b))]
        self.moran = {"I": float(mi.I), "expected_I": float(mi.EI), "z": float(mi.z_sim), "p": float(mi.p_sim), "n": len(b),
                      "weights": "KNN(k=8), row-standardised, UTM metres", "lisa_counts": b.lisa_cluster.value_counts().to_dict()}
        # Ablation under random vs spatial-block CV
        self.ablation = []
        for name, cols in STAGES.items():
            X = b[cols].fillna(b[cols].median()).to_numpy(float)
            self.ablation.append({"stage": name, "features": cols,
                                  "r2_random_cv": _block_cv(X, y, xy, spatial=False), "r2_spatial_block_cv": _block_cv(X, y, xy)})
        # Final model, TreeSHAP
        cols = STAGES["3_plus_spatial_context"]
        X = b[cols].fillna(b[cols].median())
        self.model = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=.08, subsample=.9, random_state=0).fit(X, y)
        sv = shap.TreeExplainer(self.model).shap_values(X)
        self.cols, self.shap_values = cols, sv
        b["triage"] = self.model.predict(X)
        self.shap_global = sorted(({"feature": c, "mean_abs_shap": float(np.abs(sv[:, i]).mean())}
                                   for i, c in enumerate(cols)), key=lambda r: -r["mean_abs_shap"])
        self.hex_agg = b.groupby("h3").agg(n=("triage", "size"), triage=("triage", "mean"), damage=("damage_score", "mean")).reset_index()

    def save_model(self, path="data/triage.ubj"):
        self.model.save_model(path)

    def buildings_geojson(self):
        keep = ["damage_score", "triage", "flood_frac", "lisa_cluster", "h3", "pop_density", "elevation"]
        out = self.b[keep + ["geometry"]].copy()
        out["id"] = out.index
        return json.loads(out.to_json())

    def flood_geojson(self):
        return json.loads(self.flood.to_json())

    def shap_for(self, i):
        return {"building": int(i), "base_value": float(np.ravel(shap.TreeExplainer(self.model).expected_value)[0]),
                "contributions": [{"feature": c, "value": float(self.b[c].iloc[i]) if pd.notna(self.b[c].iloc[i]) else None,
                                   "shap": float(self.shap_values[i, j])} for j, c in enumerate(self.cols)]}

    def route(self, building_id, origin=None):
        """Hazard-penalised Dijkstra from a depot (default: lowest-flood corner) to the target building."""
        G = self.G
        tgt = self.b.iloc[building_id]
        nodes = pd.DataFrame([(n, d["x"], d["y"]) for n, d in G.nodes(data=True)], columns=["n", "x", "y"])
        near = lambda lon, lat: nodes.n.iloc[int(np.argmin((nodes.x - lon) ** 2 + (nodes.y - lat) ** 2))]
        if origin is None:
            cand = self.b.nsmallest(max(5, len(self.b) // 20), "flood_frac")  # safest buildings, farthest from target
            safe = cand.iloc[int(np.argmax((cand.lon - tgt.lon) ** 2 + (cand.lat - tgt.lat) ** 2))]
            origin = (safe.lon, safe.lat)
        s, t = near(*origin), near(tgt.lon, tgt.lat)
        path = nx.shortest_path(G, s, t, weight="weight")
        edges = [min(G[u][v].values(), key=lambda d: d["weight"]) for u, v in zip(path[:-1], path[1:])]
        coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in path]
        return {"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"length_m": float(sum(e["length"] for e in edges)),
                               "hazard_edges": int(sum(e["blocked"] for e in edges)), "target": int(building_id),
                               "crs": "EPSG:4326 (lengths in " + str(self.utm) + ")"}}
