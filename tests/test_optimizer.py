import io

import numpy as np
import pandas as pd
import pytest

from optimizer import (
    REQUIRED_COLUMNS, haversine_matrix, make_demo, optimise_assignments,
    output_csv, prepare_orders, spatial_grid,
)


def test_demo_is_reproducible_and_has_three_columns():
    a = make_demo(100, seed=42)
    pd.testing.assert_frame_equal(a, make_demo(100, seed=42))
    assert list(a.columns) == list(REQUIRED_COLUMNS)


def test_optimiser_never_makes_distance_worse_and_preserves_records():
    raw = make_demo(5000)
    prepared = prepare_orders(raw)
    out = optimise_assignments(prepared, chunk_size=17)
    assert len(out) == 5000
    assert len(prepared.warehouses) == 3
    assert out.orderId.nunique() == 5000
    assert np.all(out.optimised_distance_km <= out.original_distance_km + 1e-8)
    assert np.allclose(out.distance_saved_km, out.original_distance_km - out.optimised_distance_km)
    assert (out.assignment_status == "Reassigned").sum() > 0


def test_known_nearest_selection():
    raw = pd.DataFrame({
        "orderId": ["A", "B", "C"],
        "warehouseLocation_Coordinates": ["(0, 0)", "(0, 2)", "(0, 0)"],
        "customerLocation_Coordinates": ["(0, 2.1)", "(0, 0.1)", "(0, 0.01)"],
    })
    out = optimise_assignments(prepare_orders(raw))
    assert out.assignment_status.tolist() == ["Reassigned", "Reassigned", "Unchanged"]
    assert out.optimisedWarehouse_Coordinates.tolist() == ["(0.00000, 2.00000)", "(0.00000, 0.00000)", "(0.00000, 0.00000)"]


def test_invalid_rows_reported_and_omitted():
    raw = pd.DataFrame({
        "orderId": ["ok", "bad", "", "bad lat", "piped"],
        "warehouseLocation_Coordinates": ["(-33.8, 151)", "bad", "(-33.8,151)", "(101, 0)", "[-33.8|151.0]"],
        "customerLocation_Coordinates": ["(-33.9,151.2)", "(-33.9,151.2)", "(-33.9,151.2)", "(-33.9,151.2)", "[-33.9|151.2]"],
    })
    prep = prepare_orders(raw)
    assert prep.orders.orderId.tolist() == ["ok", "piped"]
    assert len(prep.rejected) == 3
    assert "reason" in prep.rejected.columns


def test_header_is_strict_and_empty_fails():
    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_orders(pd.DataFrame({"orderID": ["foo"]}))
    with pytest.raises(ValueError, match="no order rows"):
        prepare_orders(pd.DataFrame(columns=REQUIRED_COLUMNS))


def test_single_warehouse_has_zero_savings():
    raw = pd.DataFrame({
        "orderId": ["1", "2"],
        "warehouseLocation_Coordinates": ["(0,0)", "(0,0)"],
        "customerLocation_Coordinates": ["(0,1)", "(0,2)"],
    })
    prep = prepare_orders(raw)
    assert len(prep.warehouses) == 1
    out = optimise_assignments(prep)
    assert out.distance_saved_km.sum() == 0


def test_zero_distance_tie_keeps_original():
    raw = pd.DataFrame({
        "orderId": ["1", "2"],
        "warehouseLocation_Coordinates": ["(0,0)", "(0,2)"],
        "customerLocation_Coordinates": ["(0,1)", "(0,1)"],
    })
    out = optimise_assignments(prepare_orders(raw))
    assert out.assignment_status.tolist() == ["Unchanged", "Unchanged"]


def test_haversine_scale_is_correct():
    distance = haversine_matrix(np.array([0]), np.array([0]), np.array([0]), np.array([1]))
    assert distance.shape == (1, 1)
    assert 111 < distance[0, 0] < 112


def test_heatmap_accounts_for_every_order_and_preserves_savings():
    out = optimise_assignments(prepare_orders(make_demo(25000)))
    grid = spatial_grid(out, cell_km=5)
    assert grid.orders.sum() == 25000
    assert abs(grid.total_saved_km.sum() - out.distance_saved_km.sum()) < 1e-6
    assert grid.polygon.map(len).eq(4).all()


def test_export_formats():
    out = optimise_assignments(prepare_orders(make_demo(100)))
    normal = pd.read_csv(io.BytesIO(output_csv(out)))
    replacement = pd.read_csv(io.BytesIO(output_csv(out, minimal=True)))
    assert "optimisedWarehouse_Coordinates" in normal.columns
    assert list(replacement.columns) == list(REQUIRED_COLUMNS)
    assert replacement["warehouseLocation_Coordinates"].equals(normal["optimisedWarehouse_Coordinates"])


def test_100k_smoke():
    out = optimise_assignments(prepare_orders(make_demo(100000)))
    assert len(out) == 100000
    assert (out.distance_saved_km >= 0).all()
