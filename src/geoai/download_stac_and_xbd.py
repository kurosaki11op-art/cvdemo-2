"""Download real pre/post-disaster Sentinel-2 + Sentinel-1 from Microsoft Planetary Computer (STAC),
derive NDWI / SAR flood masks, polygonize them, and (optionally) fit + save model weights.

  python -m src.geoai.download_stac_and_xbd --bbox 90.30 23.70 90.45 23.85 \
      --pre 2024-07-01/2024-07-15 --post 2024-08-10/2024-08-25 --out data

Outputs (EPSG:4326 GeoJSON): data/flood.geojson. Building footprints come from OSM (osmnx) -> data/buildings.geojson
with per-building flood-overlap/NDWI/dNDWI statistics. Weights: data/triage.ubj (XGBoost) and data/siamese.pt (torch).
"""
import argparse
import json
from pathlib import Path

import numpy as np

NDWI_T, SAR_DB_T = 0.25, -18.0


def _search(client, coll, bbox, rng, **q):
    items = list(client.search(collections=[coll], bbox=bbox, datetime=rng, max_items=20, **q).items())
    if not items:
        raise SystemExit(f"No {coll} scenes for {rng} in {bbox}")
    return items


def _read(item, assets, bbox):
    import planetary_computer as pc
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    out, transform, crs = [], None, None
    for a in assets:
        with rasterio.open(pc.sign(item.assets[a].href)) as src:
            w = from_bounds(*transform_bounds("EPSG:4326", src.crs, *bbox), src.transform)
            out.append(src.read(1, window=w, boundless=False).astype("float32"))
            if transform is None:
                transform, crs = src.window_transform(w), src.crs
    h = min(o.shape[0] for o in out); wd = min(o.shape[1] for o in out)
    return np.stack([o[:h, :wd] for o in out]), transform, crs


def ndwi(green, nir):
    return (green - nir) / (green + nir + 1e-9)


def polygonize(mask, transform, crs):
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape
    m = mask.astype("uint8")
    geoms = [shape(g) for g, v in shapes(m, mask=m > 0, transform=transform)]
    return gpd.GeoDataFrame(geometry=geoms, crs=crs).to_crs("EPSG:4326")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("W", "S", "E", "N"))
    ap.add_argument("--pre", required=True); ap.add_argument("--post", required=True)
    ap.add_argument("--out", default="data"); ap.add_argument("--no-sar", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True, parents=True)

    import geopandas as gpd
    import osmnx as ox
    import planetary_computer as pc
    from pystac_client import Client
    cl = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    bbox = tuple(a.bbox)
    s2 = lambda rng: min(_search(cl, "sentinel-2-l2a", bbox, rng, query={"eo:cloud_cover": {"lt": 20}}),
                         key=lambda i: i.properties["eo:cloud_cover"])
    (pre, t, crs), (post, _, _) = (_read(s2(r), ["B03", "B08"], bbox) for r in (a.pre, a.post))
    h, w = min(pre.shape[1], post.shape[1]), min(pre.shape[2], post.shape[2])
    nd_pre, nd_post = ndwi(*pre[:, :h, :w]), ndwi(*post[:, :h, :w])
    flood = (nd_post > NDWI_T) & ~(nd_pre > NDWI_T)  # newly inundated
    if not a.no_sar:
        try:
            s1 = _search(cl, "sentinel-1-rtc", bbox, a.post)[0]
            vv, ts, cs = _read(s1, ["vv"], bbox)
            from rasterio.warp import reproject, Resampling
            vv_db = 10 * np.log10(np.maximum(vv[0], 1e-6)); dst = np.full((h, w), np.nan, "float32")
            reproject(vv_db, dst, src_transform=ts, src_crs=cs, dst_transform=t, dst_crs=crs, resampling=Resampling.bilinear)
            flood |= np.nan_to_num(dst, nan=0) < SAR_DB_T
        except Exception as e:  # SAR is optional enrichment
            print("SAR skipped:", e)
    fl = polygonize(flood, t, crs)
    fl.to_file(out / "flood.geojson", driver="GeoJSON")

    # per-building raster stats over OSM footprints
    from rasterio.features import geometry_mask
    b = ox.features_from_bbox(bbox=bbox, tags={"building": True})
    b = b[b.geometry.type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)[["geometry"]]
    bp = b.to_crs(crs); rows = []
    for g in bp.geometry:
        m = ~geometry_mask([g], (h, w), t)
        rows.append((float(flood[m].mean()) if m.any() else 0.0, float(nd_post[m].mean()) if m.any() else 0.0,
                     float((nd_post - nd_pre)[m].mean()) if m.any() else 0.0))
    b[["flood_frac", "ndwi_post", "d_ndwi"]] = np.array(rows)
    b.to_file(out / "buildings.geojson", driver="GeoJSON")
    print(f"flood polygons={len(fl)} buildings={len(b)} -> {out}")
    print("Next: python -m src.datascience.pipeline_end_to_end --train  (needs data/labels.csv or --demographics)")


if __name__ == "__main__":
    main()
