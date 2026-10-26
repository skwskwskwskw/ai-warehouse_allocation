"""Warehouse Atlas | Streamlit warehouse-selection optimisation dashboard.

Launch: streamlit run app.py
"""
from __future__ import annotations

import csv
import io
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import pydeck as pdk

from optimizer import make_demo, optimise_assignments, output_csv, prepare_orders, spatial_grid

st.set_page_config(
    page_title="Warehouse Atlas | Allocation Optimisation",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Carefully styled to remain legible in either Streamlit theme.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --atlas-blue:#3669e8; --atlas-teal:#159b92; --atlas-ink:#14243d; }
html, body, [class*="css"], [data-testid="stAppViewContainer"] {font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"] {background:linear-gradient(160deg, rgba(240,246,255,0.48), transparent 48%);}
[data-testid="stSidebar"] {background:linear-gradient(165deg,#11223d 0%,#172d48 85%);}
[data-testid="stSidebar"] * {color:#eaf0fa !important;}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] small {color:#c4d2e3 !important;}
[data-testid="stSidebar"] [data-baseweb="select"] *,[data-testid="stSidebar"] [data-baseweb="input"] * {color:#14243d !important;}
[data-testid="stSidebar"] [data-testid="stFileUploader"] {border:1px solid #68849c;border-radius:10px;}
[data-testid="stSidebar"] button {color:#14243d !important;}
.block-container {padding-top:1.3rem; padding-bottom:2.5rem; max-width:1510px;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif;letter-spacing:-0.025em;}
.hero {padding:27px 32px;border-radius:18px;background:linear-gradient(110deg,#142844,#194774 64%,#236b82);color:#f3f8ff;margin:4px 0 18px;position:relative;overflow:hidden;}
.hero:after {content:"";position:absolute;right:-36px;top:-65px;width:300px;height:300px;background:radial-gradient(circle,rgba(83,214,202,.22),transparent 70%);border-radius:100%;}
.hero .kicker {font-size:11px;letter-spacing:.19em;font-weight:800;color:#80dfd3;text-transform:uppercase;margin-bottom:7px;}
.hero h1 {margin:0;font-size:clamp(28px,3vw,40px);font-weight:700;color:white;line-height:1.12;}
.hero p {margin:12px 0 0;font-size:14px;color:#d0e2f1;max-width:770px;line-height:1.55;}
.hero .tag {display:inline-block;margin-top:15px;font-size:11px;border:1px solid rgba(187,222,247,.36);border-radius:20px;padding:5px 11px;color:#d4eeff;}
.card {border:1px solid rgba(136,158,187,.22);background:var(--secondary-background-color, #fff);padding:17px 19px;border-radius:14px;min-height:120px;box-shadow:0 2px 11px rgba(20,46,83,.035);margin-bottom:8px;}
.card .eyebrow {color:#76859c;font-size:11px;font-weight:800;letter-spacing:.07em;text-transform:uppercase;}
.card .number {color:var(--text-color,#14243d);font-size:clamp(23px,2.2vw,31px);font-weight:800;letter-spacing:-.045em;font-family:'Space Grotesk',sans-serif;margin-top:6px;}
.card .foot {font-size:12px;color:#77869c;margin-top:3px;}
.card.featured {background:linear-gradient(125deg,#e8f7f2,#eaf7ff);border-color:#abdfd4;}
.card.featured .number {color:#11766e;}
.section-tag {font-weight:800;color:#5b718c;font-size:12px;letter-spacing:.10em;text-transform:uppercase;margin:20px 0 7px;}
.footnote {font-size:12px;color:#74869c;line-height:1.6;}
.warehouse-dot {display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;}
.legend-box {padding:9px 13px;border-radius:9px;background:rgba(148,163,184,.09);font-size:12px;line-height:2.1;margin:5px 0 12px;}
.map-heading {font-size:16px;font-weight:800;margin:7px 0 4px;color:var(--text-color,#14243d);}
[data-testid="stMetric"] {padding:8px;}
[data-testid="stTabs"] button {font-size:13px;}
@media (max-width:900px){.hero {padding:21px 18px;}.card{min-height:110px;}}
</style>
""", unsafe_allow_html=True)

COLORS = [
    [53, 110, 231, 225], [20, 157, 146, 225], [242, 163, 59, 225],
    [158, 93, 228, 225], [240, 111, 107, 225], [44, 170, 205, 225],
    [166, 170, 62, 225], [224, 102, 166, 225], [98, 121, 158, 225],
    [67, 174, 104, 225],
]
CARTO_STYLE = pdk.map_styles.CARTO_LIGHT


def wh_color(index: int) -> list[int]:
    return COLORS[index % len(COLORS)]


def css_color(index: int) -> str:
    rgb = wh_color(index)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def metric_card(label: str, value: str, detail: str, featured: bool = False) -> None:
    css = "card featured" if featured else "card"
    st.markdown(
        f'<div class="{css}"><div class="eyebrow">{label}</div><div class="number">{value}</div>'
        f'<div class="foot">{detail}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def analyse(data: pd.DataFrame):
    prepared = prepare_orders(data)
    result = optimise_assignments(prepared)
    return prepared, result


@st.cache_data(show_spinner=False)
def get_grid(result: pd.DataFrame, cell_km: float) -> pd.DataFrame:
    return spatial_grid(result, cell_km)


def calc_view(result: pd.DataFrame, warehouses: pd.DataFrame) -> pdk.ViewState:
    la = pd.concat([result["customer_lat"], warehouses["warehouse_lat"]], ignore_index=True)
    lo = pd.concat([result["customer_lon"], warehouses["warehouse_lon"]], ignore_index=True)
    latspread, lonspread = float(la.max() - la.min()), float(lo.max() - lo.min())
    width = max(lonspread, latspread * 1.5, 0.01)
    zoom = float(np.clip(math.log2(210 / width), 1.0, 12.5))
    return pdk.ViewState(latitude=float((la.min() + la.max()) / 2),
                         longitude=float((lo.min() + lo.max()) / 2), zoom=zoom, pitch=0)


def warehouse_layer(warehouses: pd.DataFrame) -> pdk.Layer:
    records = warehouses.copy()
    records["color"] = [wh_color(i) for i in range(len(records))]
    records["label"] = records["warehouse_id"] + " · " + records["coordinates"]
    records["tooltip_text"] = "Warehouse " + records["label"]
    return pdk.Layer(
        "ScatterplotLayer", data=records,
        get_position="[warehouse_lon, warehouse_lat]", get_fill_color="color",
        get_line_color=[255,255,255,240], line_width_min_pixels=2.5,
        radius_min_pixels=9, radius_max_pixels=17, get_radius=200,
        stroked=True, filled=True, pickable=True, auto_highlight=True,
    )


def coverage_map(result: pd.DataFrame, warehouses: pd.DataFrame, after: bool,
                 show_connections: bool, view: pdk.ViewState, max_points: int,
                 max_connections: int = 180) -> tuple[pdk.Deck, int, int]:
    use = result if len(result) <= max_points else result.sample(n=max_points, random_state=42)
    source_col = "optimised_warehouse" if after else "original_warehouse"
    color_lookup = {wid: wh_color(i) for i, wid in enumerate(warehouses["warehouse_id"])}
    points = use[["orderId", "customer_lon", "customer_lat", "original_distance_km",
                  "optimised_distance_km", "distance_saved_km", "assignment_status", source_col]].copy()
    points["warehouse_display"] = points[source_col]
    points["color"] = points[source_col].map(color_lookup)
    points["distance_display"] = np.round(
        points["optimised_distance_km" if after else "original_distance_km"], 2)
    points["tooltip_text"] = (points["orderId"].astype(str) + " · " + points["warehouse_display"]
                              + " · " + points["distance_display"].astype(str) + " km · "
                              + points["assignment_status"].astype(str))
    layers = []
    arc_n = 0
    if show_connections:
        # Show largest improvements first, not all lines: clarity > spaghetti.
        lines = use.nlargest(max_connections, "distance_saved_km").copy()
        lines["from_lat"] = lines["optimised_warehouse_lat" if after else "warehouse_lat"]
        lines["from_lon"] = lines["optimised_warehouse_lon" if after else "warehouse_lon"]
        arc_n = len(lines)
        layers.append(pdk.Layer(
            "LineLayer", data=lines,
            get_source_position="[from_lon, from_lat]",
            get_target_position="[customer_lon, customer_lat]",
            get_color=[69, 110, 158, 95] if after else [232, 116, 95, 92],
            get_width=1.5, width_min_pixels=1,
            pickable=False,
        ))
    layers.extend([
        pdk.Layer("ScatterplotLayer", data=points,
                  get_position="[customer_lon, customer_lat]", get_fill_color="color",
                  radius_min_pixels=2.5, radius_max_pixels=8, get_radius=90,
                  pickable=True, auto_highlight=True, opacity=0.86),
        warehouse_layer(warehouses)
    ])
    return pdk.Deck(
        layers=layers, initial_view_state=view, map_style=CARTO_STYLE,
        tooltip={"text": "{tooltip_text}",
                 "style": {"backgroundColor": "#16273c", "color": "#ffffff", "fontSize": "12px"}},
    ), len(points), arc_n


def ramp(value: float, limit: float, kind: str = "distance") -> list[int]:
    """Stable color ramp with a shared denominator between compared maps."""
    t = float(np.clip(value / max(limit, 1e-9), 0, 1))
    # Low = mint, middle = blue, high = orange/red. Red denotes more kilometres.
    stops = ([(28,165,143), (69,129,211), (244,159,64), (201,67,72)]
             if kind == "distance" else
             [(235,242,248), (82,167,217), (51,117,205), (20,63,137)])
    pos = t * (len(stops) - 1)
    k = min(int(pos), len(stops) - 2)
    frac = pos - k
    a, b = stops[k], stops[k + 1]
    return [int(a[i] + (b[i] - a[i]) * frac) for i in range(3)] + [190]


def grid_map(grid: pd.DataFrame, warehouses: pd.DataFrame, view: pdk.ViewState,
             value_col: str, scale_max: float, kind: str = "distance") -> pdk.Deck:
    points = grid.copy()
    points["color"] = [ramp(v, scale_max, kind) for v in points[value_col]]
    for col in ["mean_original_km", "mean_optimised_km", "mean_saved_km", "total_saved_km"]:
        points[col] = points[col].round(2)
    points["tooltip_text"] = (points["orders"].astype(str) + " orders · Baseline avg "
                              + points["mean_original_km"].astype(str) + " km · Optimised avg "
                              + points["mean_optimised_km"].astype(str) + " km · Total saved "
                              + points["total_saved_km"].astype(str) + " km")
    layers = [
        pdk.Layer("PolygonLayer", data=points,
                  get_polygon="polygon", get_fill_color="color",
                  get_line_color=[255,255,255,80], line_width_min_pixels=0.3,
                  stroked=True, filled=True, pickable=True, auto_highlight=True),
        warehouse_layer(warehouses),
    ]
    return pdk.Deck(
        layers=layers, initial_view_state=view, map_style=CARTO_STYLE,
        tooltip={"text": "{tooltip_text}",
                 "style": {"backgroundColor": "#16273c", "color": "#ffffff", "fontSize": "12px"}},
    )


def legend(warehouses: pd.DataFrame) -> None:
    # Clear legend at reasonable warehouse counts; high counts aren't a palette guarantee.
    blocks = []
    for idx, row in warehouses.head(12).iterrows():
        blocks.append(f'<span style="display:inline-block;margin-right:20px"><i class="warehouse-dot" '
                      f'style="background:{css_color(idx)}"></i>{row.warehouse_id}</span>')
    if len(warehouses) > 12:
        blocks.append(f"+{len(warehouses)-12} further warehouses (colours repeat)")
    st.markdown('<div class="legend-box">' + "".join(blocks) + "</div>", unsafe_allow_html=True)


def chart_movements(result: pd.DataFrame, warehouses: pd.DataFrame) -> None:
    before = result["original_warehouse"].value_counts()
    after = result["optimised_warehouse"].value_counts()
    ids = warehouses["warehouse_id"].tolist()
    fig = go.Figure()
    fig.add_bar(name="Original", x=ids, y=[int(before.get(i, 0)) for i in ids],
                marker_color="#a5b8d0", marker_line_width=0)
    fig.add_bar(name="Optimised", x=ids, y=[int(after.get(i, 0)) for i in ids],
                marker_color="#218b85", marker_line_width=0)
    fig.update_layout(
        barmode="group", height=330, margin=dict(t=18, b=18, l=18, r=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.14, x=0),
        yaxis=dict(title="Order count", showgrid=True, gridcolor="rgba(148,163,184,.19)"),
        xaxis=dict(title=""), font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})


def chart_savings(result: pd.DataFrame, warehouses: pd.DataFrame) -> None:
    saved = result.groupby("original_warehouse")["distance_saved_km"].sum()
    ids = warehouses["warehouse_id"].tolist()
    fig = go.Figure(go.Bar(
        x=[float(saved.get(i, 0)) for i in ids], y=ids,
        orientation="h", marker=dict(color=[css_color(i) for i in range(len(ids))]),
        text=[f"{saved.get(i,0):,.0f} km" for i in ids], textposition="auto",
    ))
    fig.update_layout(
        height=330, margin=dict(t=16,b=18,l=18,r=15),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="Kilometres saved", showgrid=True, gridcolor="rgba(148,163,184,.19)"),
        showlegend=False, font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})


# --- Sidebar: data source and controllable visual load ---
with st.sidebar:
    st.markdown("### ◈ WAREHOUSE ATLAS")
    st.caption("ALLOCATION INTELLIGENCE / V1")
    st.divider()
    data_source = st.radio("DATA SOURCE", ["Sample Sydney data", "Upload CSV"], index=0)
    if data_source == "Sample Sydney data":
        demo_n = st.select_slider("Sample orders", options=[100, 500, 1_000, 5_000, 10_000, 25_000], value=100)
        data = make_demo(count=demo_n)
        st.download_button("↓ Download this sample CSV", data=data.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"sydney_{demo_n}_orders.csv", mime="text/csv", use_container_width=True)
    else:
        uploaded = st.file_uploader("Upload allocation records", type=["csv"],
                                    help="Upload a UTF-8 CSV using the three exact column names.")
        if uploaded is None:
            st.info("Upload a CSV to begin, or choose the sample dataset.")
            with st.expander("CSV schema", expanded=True):
                st.code('orderId,warehouseLocation_Coordinates,customerLocation_Coordinates\n'
                        'ORD001,"(-33.85, 150.90)","(-33.87, 151.21)"\n'
                        'ORD002,"(-33.91, 151.19)","(-33.78, 150.91)"', language="csv")
            st.stop()
        try:
            raw_bytes = uploaded.getvalue()
            text = raw_bytes.decode("utf-8-sig")
            csv_rows = csv.reader(io.StringIO(text))
            csv_header = next(csv_rows, None)
            if not csv_header:
                raise ValueError("The CSV is empty.")
            wrong_lines = [line_num for line_num, values in enumerate(csv_rows, start=2)
                           if len(values) != len(csv_header)]
            if wrong_lines:
                raise ValueError(f"Rows {wrong_lines[:8]} have the wrong number of fields. "
                                 "Quote coordinate strings containing commas, e.g. \"(-33.85, 150.90)\".")
            data = pd.read_csv(io.BytesIO(raw_bytes), dtype="string", encoding="utf-8-sig")
        except Exception as exc:
            st.error(f"Cannot read uploaded CSV: {exc}")
            st.stop()
    st.divider()
    st.markdown("**MAP CONTROLS**")
    max_points = st.select_slider("Maximum individual map points", options=[1_000, 3_000, 8_000, 15_000, 25_000], value=8_000)
    connections = st.checkbox("Show example warehouse links", value=False,
                              help="Draw up to 180 of the highest-savings order links to avoid a dense tangle.")
    grid_size = st.slider("Heatmap cell size (km)", min_value=1, max_value=20, value=5, step=1)
    st.caption("Heatmaps aggregate every valid row. Only individual markers are sampled when over the map limit.")
    st.divider()
    st.caption("MODEL ASSUMPTIONS")
    st.caption("All inferred warehouses can fulfil all orders. Distances are Haversine (straight-line kilometres), not road travel.")

try:
    prepared, result = analyse(data)
except Exception as exc:
    st.error(f"Unable to process dataset: {exc}")
    st.stop()

warehouses = prepared.warehouses
n = len(result)
reassigned_n = int(result["assignment_status"].eq("Reassigned").sum())
baseline_km = float(result["original_distance_km"].sum())
optimal_km = float(result["optimised_distance_km"].sum())
saved_km = max(0.0, baseline_km - optimal_km)
pct = (saved_km / baseline_km * 100) if baseline_km > 0 else 0.0
view = calc_view(result, warehouses)

st.markdown("""
<div class="hero">
  <div class="kicker">Network design / Geo-allocation</div>
  <h1>Fulfil smarter. Travel less.</h1>
  <p>Measure the cost of your current warehouse assignments, reallocate each order to its nearest eligible warehouse, and see exactly where distance disappears.</p>
  <span class="tag">◉ Warehouse selection only &nbsp; · &nbsp; No vehicle routing</span>
</div>""", unsafe_allow_html=True)

if data_source == "Sample Sydney data":
    st.caption(f"DEMO DATA  ·  {n:,} synthetic Sydney orders  ·  {len(warehouses)} inferred warehouses  ·  Seed 42")
else:
    st.caption(f"UPLOADED FILE  ·  {n:,} valid records  ·  {len(warehouses):,} inferred warehouse coordinates")

for message in prepared.notes:
    st.warning(message)

kpis = st.columns(5, gap="small")
with kpis[0]: metric_card("Baseline distance", f"{baseline_km:,.1f}", "km · current assignments")
with kpis[1]: metric_card("Optimised distance", f"{optimal_km:,.1f}", "km · nearest warehouse")
with kpis[2]: metric_card("Distance saved", f"{saved_km:,.1f}", "km · reduction", featured=True)
with kpis[3]: metric_card("Improvement", f"{pct:.1f}%", "of original total", featured=True)
with kpis[4]: metric_card("Orders reassigned", f"{reassigned_n:,}", f"of {n:,} valid orders")

st.markdown('<div class="section-tag">01 / Geographical comparison</div>', unsafe_allow_html=True)
legend(warehouses)
map_tabs = st.tabs(["Allocation maps", "Distance heatmaps", "Savings hotspots", "Order explorer"])

with map_tabs[0]:
    st.caption("Customer markers are coloured by their assigned warehouse. Both maps use the same centre and zoom. Hover over points to inspect orders.")
    before, after = st.columns(2, gap="medium")
    before_map, before_count, before_arcs = coverage_map(result, warehouses, False, connections, view, max_points)
    after_map, after_count, after_arcs = coverage_map(result, warehouses, True, connections, view, max_points)
    with before:
        st.markdown('<div class="map-heading">01 · Existing warehouse allocation</div>', unsafe_allow_html=True)
        st.pydeck_chart(before_map, height=450, use_container_width=True, key="map_before")
        st.caption(f"{before_count:,} customer markers · original warehouse assignment")
    with after:
        st.markdown('<div class="map-heading">02 · Optimised warehouse allocation</div>', unsafe_allow_html=True)
        st.pydeck_chart(after_map, height=450, use_container_width=True, key="map_after")
        st.caption(f"{after_count:,} customer markers · nearest warehouse selection")
    if n > max_points:
        st.info(f"For map responsiveness, individual markers are a reproducible sample of {max_points:,} from {n:,} orders. Every order is still optimised and included in all KPIs, heatmaps, and exports.")
    if connections:
        st.caption(f"Only up to {before_arcs} highest-potential-saving connections are drawn per map.")

with map_tabs[1]:
    st.caption("Each coloured grid square displays the average warehouse-to-customer distance for ALL orders inside it. Both views share exactly the same colour scale. Green/blue = shorter, orange/red = longer.")
    grid = get_grid(result, float(grid_size))
    # Percentile dampens single extreme long-haul outliers but both maps share a scale.
    max_scale = max(float(np.quantile(result["original_distance_km"], .95)), 0.01)
    x, y = st.columns(2, gap="medium")
    with x:
        st.markdown('<div class="map-heading">Before · average distance per cell</div>', unsafe_allow_html=True)
        st.pydeck_chart(grid_map(grid, warehouses, view, "mean_original_km", max_scale),
                        height=450, use_container_width=True, key="heat_before")
    with y:
        st.markdown('<div class="map-heading">After · average distance per cell</div>', unsafe_allow_html=True)
        st.pydeck_chart(grid_map(grid, warehouses, view, "mean_optimised_km", max_scale),
                        height=450, use_container_width=True, key="heat_after")
    st.caption(f"{len(grid):,} occupied geographic cells · {grid_size} km approximate grid resolution · {n:,} records aggregated · Shared colour scale: 0–{max_scale:.1f} km (95th percentile of baseline; values above this saturate).")

with map_tabs[2]:
    st.caption("Blue intensity shows cumulative kilometres saved across all orders in each grid square. Brighter cells represent the biggest combined improvement opportunities, not simply the most customers.")
    grid = get_grid(result, float(grid_size))
    upper = max(float(np.quantile(grid["total_saved_km"], .95)), 0.01)
    st.pydeck_chart(grid_map(grid, warehouses, view, "total_saved_km", upper, kind="savings"),
                    height=480, use_container_width=True, key="savings_map")
    st.caption(f"Sum of grid savings = {grid['total_saved_km'].sum():,.1f} km · Colour saturation at {upper:,.1f} km saved per cell (95th percentile).")
    hotspots = grid.nlargest(8, "total_saved_km")[["center_lat", "center_lon", "orders", "total_saved_km", "mean_saved_km"]].copy()
    st.markdown("**Highest-impact geographical cells**")
    st.dataframe(hotspots.rename(columns={"center_lat":"Latitude", "center_lon":"Longitude", "orders":"Orders", "total_saved_km":"Total saved (km)", "mean_saved_km":"Saved / order (km)"}).round(2),
                 hide_index=True, use_container_width=True)

with map_tabs[3]:
    f1, f2, f3 = st.columns([1, 1, 2], gap="small")
    with f1:
        warehouse_filter = st.selectbox("Original warehouse", ["All", *warehouses["warehouse_id"].tolist()])
    with f2:
        status_filter = st.selectbox("Assignment status", ["All", "Reassigned", "Unchanged"])
    with f3:
        search = st.text_input("Find orderId", placeholder="Search part of an order ID", max_chars=100)
    display = result
    if warehouse_filter != "All":
        display = display.loc[display["original_warehouse"].eq(warehouse_filter)]
    if status_filter != "All":
        display = display.loc[display["assignment_status"].eq(status_filter)]
    if search:
        display = display.loc[display["orderId"].astype("string").str.contains(search, regex=False, case=False, na=False)]
    st.caption(f"{len(display):,} matching rows. Showing up to 2,000 rows in the browser; downloads include all matching records.")
    columns = ["orderId", "original_warehouse", "optimised_warehouse", "original_distance_km",
               "optimised_distance_km", "distance_saved_km", "assignment_status"]
    st.dataframe(display[columns].head(2000), use_container_width=True, hide_index=True,
                 column_config={
                     "original_distance_km":st.column_config.NumberColumn("Original km",format="%.2f"),
                     "optimised_distance_km":st.column_config.NumberColumn("Optimised km",format="%.2f"),
                     "distance_saved_km":st.column_config.NumberColumn("Saved km",format="%.2f"),
                 })
    st.download_button("↓ Export filtered rows", data=output_csv(display), file_name="filtered_warehouse_allocations.csv",
                       mime="text/csv", use_container_width=False)

st.markdown('<div class="section-tag">02 / Allocation insights</div>', unsafe_allow_html=True)
first, second = st.columns(2, gap="medium")
with first:
    st.markdown("**Order distribution by warehouse**")
    chart_movements(result, warehouses)
with second:
    st.markdown("**Distance saved by original warehouse**")
    chart_savings(result, warehouses)

st.markdown('<div class="section-tag">03 / Export optimisation results</div>', unsafe_allow_html=True)
exp1, exp2 = st.columns(2, gap="medium")
with exp1:
    st.markdown("**Detailed optimisation audit**")
    st.caption("Original + new coordinates, original/optimised distances, total improvement per order, and reassignment status.")
    st.download_button("↓ Download detailed CSV", data=output_csv(result),
                       file_name="warehouse_optimisation_full.csv", mime="text/csv",
                       use_container_width=True, type="primary")
with exp2:
    st.markdown("**Drop-in replacement CSV**")
    st.caption("The same original three headers, with warehouseLocation_Coordinates replaced by the optimised assignment.")
    st.download_button("↓ Download optimised 3-column CSV", data=output_csv(result, minimal=True),
                       file_name="warehouse_assignments_optimised.csv", mime="text/csv",
                       use_container_width=True)

with st.expander("Data contract, limitations and methodology"):
    st.markdown("""
**Required CSV columns (case-sensitive):**

`orderId`, `warehouseLocation_Coordinates`, `customerLocation_Coordinates`

**Accepted coordinate format:** latitude then longitude, such as `"(-33.84750, 150.90060)"`.
Because coordinates contain commas, each coordinate value **must be quoted** in a CSV.

**Warehouse discovery:** Every distinct origin coordinate in the CSV defines one possible warehouse.
GPS differences smaller than 0.00001 degrees are rounded together. A warehouse never mentioned in any original row **cannot** be discovered and therefore cannot be selected.

**Optimisation assumptions:** Every listed warehouse can fulfil any order, with no capacity, inventory, or product restrictions. Each order is considered independently, and the objective is the sum of one-way warehouse-to-customer great-circle distances. There is no delivery routing or vehicle consolidation.

**Large datasets:** The distance calculation runs in bounded-memory chunks. Individual markers are capped by the visualisation control, but every valid order contributes to totals, assignments and spatial heatmaps. Geographic cells are aggregated before rendering for responsiveness.

**Limitations:** Haversine is not driving distance. The model does not predict delivery time or monetary shipping cost. If warehouses have different inventory or transport tariffs, a constraint-aware solver will be required.
""")

if not prepared.rejected.empty:
    with st.expander(f"Rejected input records ({len(prepared.rejected):,})"):
        st.dataframe(prepared.rejected.head(500), hide_index=True, use_container_width=True)
        st.download_button("↓ Download invalid row report", prepared.rejected.to_csv(index=False).encode("utf-8-sig"),
                           file_name="warehouse_invalid_rows.csv", mime="text/csv")

st.markdown('<div class="footnote">WAREHOUSE ATLAS · Geographic reallocation analysis · All displayed distances are straight-line estimates. Synthetic samples are illustrative, not operational shipments.</div>', unsafe_allow_html=True)
