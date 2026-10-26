"""Pure allocation engine for the Warehouse Atlas Streamlit dashboard.

The only required input columns are:
  orderId, warehouseLocation_Coordinates, customerLocation_Coordinates

Distances are great-circle kilometres, NOT drive distances.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "orderId",
    "warehouseLocation_Coordinates",
    "customerLocation_Coordinates",
)
COORD_PRECISION = 5  # absorb tiny variations in repeated warehouse coordinates
EARTH_RADIUS_KM = 6371.0088

# Lat/lon separated by a comma, semicolon or pipe; optional matching brackets.
_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_COORD_PATTERN = re.compile(
    rf"^\s*[\(\[\{{]?\s*({_FLOAT})\s*[,;|]\s*({_FLOAT})\s*[\)\]\}}]?\s*$"
)

@dataclass(frozen=True)
class Prepared:
    orders: pd.DataFrame
    rejected: pd.DataFrame
    warehouses: pd.DataFrame
    notes: tuple[str, ...]


def format_coordinate(lat: float, lon: float) -> str:
    return f"({lat:.5f}, {lon:.5f})"


def _parse_coordinates(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    matched = series.astype("string").str.extract(_COORD_PATTERN)
    lat = pd.to_numeric(matched[0], errors="coerce")
    lon = pd.to_numeric(matched[1], errors="coerce")
    is_valid = lat.between(-90, 90) & lon.between(-180, 180)
    lat = lat.where(is_valid).round(COORD_PRECISION)
    lon = lon.where(is_valid).round(COORD_PRECISION)
    return lat, lon


def prepare_orders(raw: pd.DataFrame) -> Prepared:
    """Validate, split malformed rows, infer the candidate warehouse locations."""
    missing = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
            + ". Required headers are: " + ", ".join(REQUIRED_COLUMNS)
            + ". Coordinate values containing commas must be enclosed in double quotes in the CSV."
        )
    if raw.empty:
        raise ValueError("CSV has no order rows.")

    orders = raw[list(REQUIRED_COLUMNS)].copy().reset_index(drop=True)
    orders.insert(0, "source_row", np.arange(2, len(orders) + 2))
    orders["orderId"] = orders["orderId"].astype("string").str.strip()

    orders["warehouse_lat"], orders["warehouse_lon"] = _parse_coordinates(
        orders["warehouseLocation_Coordinates"]
    )
    orders["customer_lat"], orders["customer_lon"] = _parse_coordinates(
        orders["customerLocation_Coordinates"]
    )

    has_id = orders["orderId"].notna() & orders["orderId"].ne("")
    wh_ok = orders["warehouse_lat"].notna()
    cust_ok = orders["customer_lat"].notna()
    valid_mask = has_id & wh_ok & cust_ok

    rejected = orders.loc[~valid_mask, ["source_row", *REQUIRED_COLUMNS]].copy()
    if not rejected.empty:
        # Prevent mixed numpy/string extension assignment issues via pure list.
        rejected["reason"] = [
            "; ".join(
                [
                    *( [] if bool(has_id.iloc[idx]) else ["Missing orderId"] ),
                    *( [] if bool(wh_ok.iloc[idx]) else ["Invalid warehouse coordinate"] ),
                    *( [] if bool(cust_ok.iloc[idx]) else ["Invalid customer coordinate"] ),
                ]
            )
            for idx in rejected.index
        ]

    orders = orders.loc[valid_mask].reset_index(drop=True)
    if orders.empty:
        raise ValueError("No valid orders after checking coordinates and orderId.")

    warehouses = (
        orders[["warehouse_lat", "warehouse_lon"]]
        .drop_duplicates()
        .sort_values(["warehouse_lon", "warehouse_lat"], kind="stable")
        .reset_index(drop=True)
    )
    warehouses.insert(0, "warehouse_id", [f"WH-{i:02d}" for i in range(1, len(warehouses) + 1)])
    warehouses["coordinates"] = [
        format_coordinate(lat, lon)
        for lat, lon in zip(warehouses.warehouse_lat, warehouses.warehouse_lon)
    ]
    warehouses["display_name"] = warehouses["warehouse_id"] + " · " + warehouses["coordinates"]
    orders = orders.merge(
        warehouses[["warehouse_id", "warehouse_lat", "warehouse_lon"]],
        on=["warehouse_lat", "warehouse_lon"], how="left", validate="many_to_one"
    ).rename(columns={"warehouse_id": "original_warehouse"})

    notes = []
    if not rejected.empty:
        notes.append(f"{len(rejected):,} invalid row(s) excluded from optimisation; download the rejection report to inspect them.")
    if len(warehouses) == 1:
        notes.append("Only one distinct warehouse is present. The CSV cannot reveal alternative warehouses that never appear as an original assignment.")
    if len(warehouses) > 200:
        raise ValueError(f"Found {len(warehouses):,} distinct warehouse coordinates. This is unusually high for the allocation tool; check coordinate precision and clean GPS jitter in warehouse locations.")
    if len(warehouses) > 30:
        notes.append(f"Detected {len(warehouses):,} unique warehouse coordinates. Check for GPS jitter or duplicate locations. Coordinates are deduplicated to {COORD_PRECISION} decimal places.")
    if orders["orderId"].duplicated().any():
        notes.append("Some orderId values repeat. Each CSV row is treated as an independently assignable delivery record.")
    return Prepared(orders=orders, rejected=rejected, warehouses=warehouses, notes=tuple(notes))


def haversine_matrix(
    src_lat: np.ndarray, src_lon: np.ndarray, dst_lat: np.ndarray, dst_lon: np.ndarray
) -> np.ndarray:
    """Pairwise distance in km with broadcasting; rows are customers, columns warehouses."""
    alat = np.radians(np.asarray(src_lat, dtype=np.float64)).reshape(-1, 1)
    alon = np.radians(np.asarray(src_lon, dtype=np.float64)).reshape(-1, 1)
    blat = np.radians(np.asarray(dst_lat, dtype=np.float64)).reshape(1, -1)
    blon = np.radians(np.asarray(dst_lon, dtype=np.float64)).reshape(1, -1)
    a = np.sin((alat - blat) / 2) ** 2 + np.cos(alat) * np.cos(blat) * np.sin((alon - blon) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def optimise_assignments(prepared: Prepared, chunk_size: int = 4096) -> pd.DataFrame:
    """Return one output per valid input row; optimise nearest original warehouse.

    O(orders x warehouses) time; O(chunk_size x warehouses) working memory.
    For tied distances, prefer the originally assigned warehouse.
    """
    orders = prepared.orders
    warehouses = prepared.warehouses
    n = len(orders)
    m = len(warehouses)
    if m == 0 or n == 0:
        raise ValueError("No warehouses or orders to optimise.")

    best_idx = np.empty(n, dtype=np.int32)
    best_km = np.empty(n, dtype=np.float64)
    original_km = np.empty(n, dtype=np.float64)
    original_idx = orders["original_warehouse"].map(
        {wid: i for i, wid in enumerate(warehouses["warehouse_id"])}
    ).to_numpy(dtype=np.int32)

    warehouse_lats = warehouses["warehouse_lat"].to_numpy()
    warehouse_lons = warehouses["warehouse_lon"].to_numpy()
    cust_lats = orders["customer_lat"].to_numpy()
    cust_lons = orders["customer_lon"].to_numpy()

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        matrix = haversine_matrix(
            cust_lats[start:end], cust_lons[start:end], warehouse_lats, warehouse_lons
        )
        rows = np.arange(end - start)
        current_idx = original_idx[start:end]
        current_km = matrix[rows, current_idx]
        selected = matrix.argmin(axis=1)
        nearest_km = matrix[rows, selected]
        # Preserve current selection when distances differ only by numerical noise.
        tied_current = current_km - nearest_km < 1e-8
        selected[tied_current] = current_idx[tied_current]
        best_idx[start:end] = selected
        best_km[start:end] = matrix[rows, selected]
        original_km[start:end] = current_km

    result = orders.copy()
    result["optimised_warehouse"] = warehouses["warehouse_id"].to_numpy()[best_idx]
    result["optimised_warehouse_lat"] = warehouse_lats[best_idx]
    result["optimised_warehouse_lon"] = warehouse_lons[best_idx]
    result["optimisedWarehouse_Coordinates"] = [
        format_coordinate(lat, lon)
        for lat, lon in zip(result["optimised_warehouse_lat"], result["optimised_warehouse_lon"])
    ]
    result["original_distance_km"] = original_km
    result["optimised_distance_km"] = best_km
    result["distance_saved_km"] = np.maximum(0.0, original_km - best_km)
    result["assignment_status"] = np.where(
        result["optimised_warehouse"] == result["original_warehouse"], "Unchanged", "Reassigned"
    )
    return result


def spatial_grid(result: pd.DataFrame, cell_km: float = 5.0) -> pd.DataFrame:
    """Aggregate ALL customer rows into spatial cells (no point dropping/sampling).

    Each cell reports mean original/optimised/saved distance and sum of savings.
    This is a local tangent-plane approximation used only for visual binning.
    """
    if result.empty:
        return pd.DataFrame()
    lat = result["customer_lat"].to_numpy(dtype=float)
    lon = result["customer_lon"].to_numpy(dtype=float)
    lat0, lon0 = float(lat.min()), float(lon.min())
    lat_step = cell_km / 111.195
    lon_step = cell_km / max(111.195 * abs(np.cos(np.radians(float(np.mean(lat))))), 0.001)
    bin_x = np.floor((lon - lon0) / lon_step).astype(int)
    bin_y = np.floor((lat - lat0) / lat_step).astype(int)
    work = pd.DataFrame({
        "bin_x": bin_x,
        "bin_y": bin_y,
        "original": result["original_distance_km"].to_numpy(),
        "optimised": result["optimised_distance_km"].to_numpy(),
        "saved": result["distance_saved_km"].to_numpy(),
    })
    aggregate = work.groupby(["bin_x", "bin_y"], as_index=False).agg(
        orders=("original", "size"),
        mean_original_km=("original", "mean"),
        mean_optimised_km=("optimised", "mean"),
        mean_saved_km=("saved", "mean"),
        total_saved_km=("saved", "sum"),
    )
    aggregate["west"] = lon0 + aggregate["bin_x"] * lon_step
    aggregate["east"] = aggregate["west"] + lon_step
    aggregate["south"] = lat0 + aggregate["bin_y"] * lat_step
    aggregate["north"] = aggregate["south"] + lat_step
    aggregate["center_lon"] = (aggregate["west"] + aggregate["east"]) / 2
    aggregate["center_lat"] = (aggregate["south"] + aggregate["north"]) / 2
    aggregate["polygon"] = [
        [[west, south], [east, south], [east, north], [west, north]]
        for west, east, south, north in zip(
            aggregate["west"], aggregate["east"], aggregate["south"], aggregate["north"]
        )
    ]
    return aggregate


def make_demo(count: int = 100, seed: int = 42) -> pd.DataFrame:
    """Reproducible Greater Sydney illustration, not real customer records."""
    rng = np.random.default_rng(seed)
    # Wetherill Park, Alexandria, Hornsby (approximate coordinates)
    warehouses = np.array([
        [-33.8475, 150.9006],
        [-33.9118, 151.1940],
        [-33.7058, 151.0989],
    ])
    # Clustered across major Sydney suburban demand corridors.
    clusters = np.array([
        [-33.8145, 151.0017],  # Parramatta
        [-33.8990, 151.2080],  # inner southeast
        [-33.7632, 151.1049],  # northern suburbs
        [-33.9250, 150.9150],  # Liverpool
        [-33.8670, 151.0270],  # inner west
        [-33.8720, 151.2520],  # eastern suburbs
        [-33.7680, 150.9060],  # Blacktown
        [-33.9580, 151.1350],  # southern suburbs
    ])
    probs = np.array([.22, .15, .14, .14, .12, .06, .11, .06])
    homes = clusters[rng.choice(len(clusters), size=count, p=probs)]
    customers = homes + rng.normal(size=(count, 2)) * np.array([0.022, 0.029])
    customers[:, 0] = np.clip(customers[:, 0], -34.15, -33.52)
    customers[:, 1] = np.clip(customers[:, 1], 150.64, 151.45)

    # Start with nearest geography, then intentionally retain some legacy/wrong allocations.
    distance = haversine_matrix(customers[:, 0], customers[:, 1], warehouses[:, 0], warehouses[:, 1])
    assigned = distance.argmin(axis=1)
    imperfect = rng.random(count) < .43
    alternative = (assigned + rng.integers(1, 3, size=count)) % 3
    assigned[imperfect] = alternative[imperfect]
    return pd.DataFrame({
        "orderId": [f"ORD-{i+1:06d}" for i in range(count)],
        "warehouseLocation_Coordinates": [format_coordinate(*warehouses[i]) for i in assigned],
        "customerLocation_Coordinates": [format_coordinate(lat, lon) for lat, lon in customers],
    })


def output_csv(result: pd.DataFrame, minimal: bool = False) -> bytes:
    """Get optimized output as UTF-8 CSV bytes suitable for Streamlit download."""
    if minimal:
        out = result[list(REQUIRED_COLUMNS)].copy()
        out["warehouseLocation_Coordinates"] = result["optimisedWarehouse_Coordinates"]
    else:
        fields = [
            *REQUIRED_COLUMNS,
            "original_warehouse", "optimised_warehouse", "optimisedWarehouse_Coordinates",
            "original_distance_km", "optimised_distance_km", "distance_saved_km", "assignment_status",
        ]
        out = result[fields].copy()
        for name in ("original_distance_km", "optimised_distance_km", "distance_saved_km"):
            out[name] = out[name].round(3)
    return out.to_csv(index=False).encode("utf-8-sig")
