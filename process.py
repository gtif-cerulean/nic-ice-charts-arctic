import os
import io
import zipfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from datetime import date as date_cls
from collections import defaultdict

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

# Config

PARQUET_PATH = os.getenv("PARQUET_PATH", "geojson_assets.parquet")
GROUPED_PARQUET_PATH = os.getenv("GROUPED_PARQUET_PATH", "daily_items.parquet")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "geojsons"))

START_DATE = datetime.strptime(os.getenv("START_DATE", "2025-09-01"), "%Y-%m-%d").date()
END_DATE = datetime.strptime(os.getenv("END_DATE", "2025-09-08"), "%Y-%m-%d").date()

ASSET_BASE_URL_GEOJSON = os.getenv(
    "ASSET_BASE_URL_GEOJSON", "http://127.0.0.1:9091/geojsons"
)
STYLE_URL = os.getenv(
    "STYLE_URL",
    "https://raw.githubusercontent.com/gtif-cerulean/assets/refs/heads/main/styles/nic_arctic_ice_charts_style.json",
)

# USNIC prd prefix (e.g., "30") - 26 is arctic
USNIC_PREFIX = os.getenv("USNIC_PREFIX", "26")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from shapely.validation import make_valid as _make_valid
except Exception:
    try:
        from shapely import make_valid as _make_valid
    except Exception:
        _make_valid = None


# Helpers
def date_iter(d0, d1):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def usnic_zip_url(d):
    # prd = <prefix><mmddYYYY> e.g., 30 + 10032025 -> 3010032025
    return f"https://usicecenter.gov/File/DownloadArchive?prd={USNIC_PREFIX}{d.strftime('%m%d%Y')}"


def download_zip_bytes(url: str) -> bytes:
    import requests

    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def extract_first_shp(zip_bytes: bytes, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(target_dir)
    shp_files = list(target_dir.rglob("*.shp"))
    if not shp_files:
        raise FileNotFoundError("No .shp found inside archive")
    return shp_files[0]


def load_existing_parquet(path: Path) -> gpd.GeoDataFrame:
    if path.exists():
        return gpd.read_parquet(path)
    cols = [
        "type",
        "stac_version",
        "id",
        "datetime",
        "geometry",
        "bbox",
        "assets",
        "links",
    ]
    return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs="EPSG:4326")


def add_style_link(row):
    if not STYLE_URL:
        return row.get("links", [])
    assets = row.get("assets", {})
    asset_keys = list(assets.keys()) if isinstance(assets, dict) else []
    links = [
        link for link in (row.get("links", []) or []) if link.get("rel") != "style"
    ]
    if asset_keys:
        links.append(
            {
                "rel": "style",
                "href": f"{STYLE_URL}",
                "type": "text/vector-styles",
                "asset:keys": asset_keys,
            }
        )
    return links


def to_ll_repair(gdf_src: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject to EPSG:4326 and repair geometries."""
    # if gdf_src.crs is None:
    #     gdf_src = gdf_src.set_crs(3031, allow_override=True)
    gdf = gdf_src.to_crs(3413)
    # gdf = gdf_src
    if _make_valid:
        gdf["geometry"] = gdf.geometry.apply(_make_valid)
    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: geom if geom is None or geom.is_valid else geom.buffer(0)
    )
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    return gdf


def env_box_bounds(gdf_ll: gpd.GeoDataFrame):
    """Return (geometry_box, bbox_list) safely from total_bounds."""
    minx, miny, maxx, maxy = gdf_ll.total_bounds
    return box(minx, miny, maxx, maxy), [
        float(minx),
        float(miny),
        float(maxx),
        float(maxy),
    ]


def create_stac_item(date, id_, assets, asset_type):
    if not assets:
        return {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": id_,
            "datetime": pd.to_datetime(date),
            "geometry": None,
            "bbox": None,
            "assets": {},
            "links": [],
            "properties": {
                "description": "Error downloading or processing assets",
                "invalid": True,
            },
        }

    bxs = [a["geometry"].bounds for a in assets]
    minx = min(b[0] for b in bxs)
    miny = min(b[1] for b in bxs)
    maxx = max(b[2] for b in bxs)
    maxy = max(b[3] for b in bxs)
    geom = box(minx, miny, maxx, maxy)

    return {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": id_,
        "datetime": pd.to_datetime(date),
        "geometry": geom,
        "bbox": [minx, miny, maxx, maxy],
        "assets": {
            f"asset_{i}": {"href": a["url"], "type": asset_type, "roles": ["data"]}
            for i, a in enumerate(assets)
        },
        "links": [],
    }


def merge_items_per_day(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    merged = []
    # Exclude invalids if present
    if "properties" in df.columns:
        df = df[
            ~df["properties"].apply(
                lambda props: (
                    props.get("invalid", False) if isinstance(props, dict) else False
                )
            )
        ]

    for id_, group in df.groupby("id"):
        # Safe envelope
        minx, miny, maxx, maxy = group.total_bounds
        geom = box(minx, miny, maxx, maxy)

        # Flatten and rekey assets
        assets_list = []
        for asset_dict in group["assets"]:
            if isinstance(asset_dict, dict):
                assets_list.extend(asset_dict.values())
        rekeyed_assets = {f"asset_{i}": a for i, a in enumerate(assets_list)}

        # Collect links arrays (some rows may have [] / None)
        links = []
        for link_set in group.get("links", []):
            if isinstance(link_set, list):
                links.extend(link_set)

        date = pd.to_datetime(group["datetime"].iloc[0])
        merged.append(
            {
                "type": "Feature",
                "stac_version": "1.0.0",
                "id": id_,
                "geometry": geom,
                "bbox": [float(minx), float(miny), float(maxx), float(maxy)],
                "datetime": date,
                "assets": rekeyed_assets,
                "links": links,
            }
        )

    return gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")


# Main
def main():
    print(
        f"Fetching USNIC shapefiles from {START_DATE} to {END_DATE} (prefix={USNIC_PREFIX})"
    )
    tmp_root = OUTPUT_DIR / "_tmp_usnic"
    tmp_root.mkdir(parents=True, exist_ok=True)

    new_records = []
    grouped_items = defaultdict(list)

    # Load existing (to preserve schema & allow incremental runs)
    existing_items = load_existing_parquet(Path(PARQUET_PATH))
    existing_item_ids = (
        set(existing_items["id"].astype(str)) if not existing_items.empty else set()
    )

    for d in date_iter(START_DATE, END_DATE):
        if d >= date_cls.today():
            print(f"🛑 Reached current date ({d}), stopping.")
            break
        item_id = f"USNIC_{d.strftime('%Y%m%d')}"
        if item_id in existing_item_ids:
            continue

        url = usnic_zip_url(d)
        try:
            zbytes = download_zip_bytes(url)
            shp_path = extract_first_shp(zbytes, tmp_root / item_id)
        except Exception as e:
            print(f"✖️  Failed to get shapefile for {d}: {e}")
            # do not append invalid rows (prevents Arrow struct errors)
            continue

        try:
            gdf_src = gpd.read_file(shp_path)
            if gdf_src.empty or gdf_src.geometry.isna().all():
                print(f"✖️  Shapefile has no valid geometry for {d}")
                continue

            gdf_ll = to_ll_repair(gdf_src)
            if gdf_ll.empty:
                print(f"✖️  All geometries became empty/invalid after repair for {d}")
                continue

            # Persist GeoJSON for the asset href
            out_geojson = OUTPUT_DIR / f"{item_id}.geojson"
            # In your to_ll_repair function or right before saving, add:
            gdf_ll["geometry"] = gdf_ll.geometry.simplify(
                tolerance=0.05, preserve_topology=True
            )
            gdf_ll.to_file(
                out_geojson,
                driver="GeoJSON",
                engine="pyogrio",
                layer_options={"COORDINATE_PRECISION": 0},
            )

            # Envelope box per item
            geom_box, bbox_list = env_box_bounds(gdf_ll)

            asset_url = f"{ASSET_BASE_URL_GEOJSON}/{item_id}.geojson"
            # Per-item feature (matches original schema)
            new_records.append(
                create_stac_item(
                    d,
                    item_id,
                    [{"url": asset_url, "geometry": geom_box}],
                    "application/geo+json",
                )
            )
            # For grouped per-date
            grouped_items[d].append({"url": asset_url, "geometry": geom_box})

        except Exception as e:
            print(f"✖️  Error processing {d}: {e}")
            continue

    # -------- Save individual items (PARQUET_PATH) --------
    if new_records:
        df_new = gpd.GeoDataFrame(new_records, crs="EPSG:4326")
        # filter out invalids just in case
        if "properties" in df_new.columns:
            df_new = df_new[
                ~df_new["properties"].apply(
                    lambda p: p.get("invalid", False) if isinstance(p, dict) else False
                )
            ]
            df_new = df_new.drop(columns=["properties"], errors="ignore")

        if existing_items.empty:
            items_out = df_new
        else:
            # align columns (keep original order if file exists)
            for c in existing_items.columns:
                if c not in df_new.columns:
                    df_new[c] = pd.NA
            df_new = df_new[existing_items.columns]
            items_out = pd.concat([existing_items, df_new], ignore_index=True)

        items_out.to_parquet(PARQUET_PATH)
        print(f"✅ Saved {len(df_new)} new records to {PARQUET_PATH}")
        print(f"Copying {PARQUET_PATH} to {OUTPUT_DIR}")
        shutil.copy(PARQUET_PATH, OUTPUT_DIR)
    else:
        print("No new individual geojson assets to save.")

    # -------- Save grouped daily items (GROUPED_PARQUET_PATH) --------
    grouped_existing = load_existing_parquet(Path(GROUPED_PARQUET_PATH))
    grouped_records = []
    for date, assets in grouped_items.items():
        grouped_records.append(
            create_stac_item(
                date, date.strftime("%Y-%m-%d"), assets, "application/geo+json"
            )
        )
    if grouped_records:
        grouped_df = gpd.GeoDataFrame(grouped_records, crs="EPSG:4326")
        merged = (
            pd.concat([grouped_existing, grouped_df], ignore_index=True)
            if not grouped_existing.empty
            else grouped_df
        )
    else:
        merged = grouped_existing

    if not merged.empty:
        final = merge_items_per_day(merged)
        final["links"] = final.apply(add_style_link, axis=1)
        final.to_parquet(GROUPED_PARQUET_PATH)
        print(
            f"✅ Saved grouped items to {GROUPED_PARQUET_PATH} ({len(final)} total items)"
        )
        print(f"Copying {GROUPED_PARQUET_PATH} to {OUTPUT_DIR}")
        shutil.copy(GROUPED_PARQUET_PATH, OUTPUT_DIR)
    else:
        print("No grouped items to save.")


if __name__ == "__main__":
    main()
