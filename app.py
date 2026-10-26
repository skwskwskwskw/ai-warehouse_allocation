"""Warehouse Atlas | Streamlit warehouse-selection optimisation dashboard.

Launch: streamlit run app.py
"""
from __future__ import annotations

import csv
import io
import math
from typing import NamedTuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from optimizer import make_demo, optimise_assignments, output_csv, prepare_orders, spatial_grid

st.set_page_config(
    page_title="Warehouse Atlas | Allocation Optimisation",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&display=swap');
:root {
  --atlas-ink:#14243d; --atlas-muted:#5f6f86; --atlas-faint:#8693a6;
  --atlas-line:rgba(136,158,187,.30); --atlas-card:#ffffff; --atlas-soft:#eef2f8;
  --atlas-teal:#218b85; --atlas-teal-dark:#11766e; --atlas-before:#a5b8d0;
}
.block-container {padding-top:2.6rem; padding-bottom:3rem; max-width:1320px;}
.hero h1, .result h2, .section h3, .empty h3, .brand-name {font-family:'Space Grotesk',sans-serif !important;letter-spacing:-0.02em;}
[data-testid="stSidebar"] {border-right:1px solid var(--atlas-line);}
[data-testid="stTabs"] button p {font-size:14px;font-weight:600;}

.brand {margin:-8px 0 18px;}
.brand-name {font-family:'Space Grotesk',sans-serif;font-size:21px;font-weight:700;color:var(--atlas-ink);}
.brand-sub {font-size:13px;color:var(--atlas-muted);margin-top:2px;}
.side-step {font-size:12px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:var(--atlas-muted);margin:6px 0 4px;}

.hero {padding:28px 32px 26px;border-radius:18px;background:linear-gradient(110deg,#142844,#194774 64%,#236b82);color:#f3f8ff;margin:0 0 16px;position:relative;overflow:hidden;}
.hero:after {content:"";position:absolute;right:-40px;top:-70px;width:300px;height:300px;background:radial-gradient(circle,rgba(83,214,202,.22),transparent 70%);border-radius:100%;}
.hero .kicker {font-size:11px;letter-spacing:.19em;font-weight:800;color:#80dfd3;text-transform:uppercase;margin-bottom:8px;}
.hero h1 {margin:0;padding:0;font-size:clamp(26px,2.8vw,36px);font-weight:700;color:#ffffff;line-height:1.15;}
.hero p {margin:10px 0 0;font-size:15px;color:#d0e2f1;max-width:760px;line-height:1.55;}
.steps {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:22px;position:relative;z-index:1;}
.step {background:rgba(255,255,255,.07);border:1px solid rgba(187,222,247,.22);border-radius:12px;padding:13px 15px;font-size:13px;line-height:1.45;color:#c9dcec;}
.step b {display:block;color:#ffffff;font-size:14px;margin-bottom:3px;}
.step .num {display:inline-flex;width:22px;height:22px;border-radius:50%;background:#80dfd3;color:#11223d;font-weight:800;font-size:12px;align-items:center;justify-content:center;margin-bottom:8px;}

.chips {display:flex;flex-wrap:wrap;gap:8px;margin:0 0 16px;}
.chip {font-size:12px;padding:4px 11px;border-radius:999px;background:var(--atlas-soft);color:var(--atlas-muted);font-weight:600;}

.eyebrow {color:var(--atlas-faint);font-size:11px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;}
.result {display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:36px;align-items:center;padding:26px 30px;border:1px solid var(--atlas-line);border-radius:16px;background:var(--atlas-card);box-shadow:0 2px 12px rgba(20,46,83,.04);margin-bottom:12px;}
.result h2 {margin:6px 0 8px;padding:0;font-size:clamp(23px,2.3vw,30px);color:var(--atlas-ink);line-height:1.2;}
.result h2 em {font-style:normal;color:var(--atlas-teal-dark);}
.result p {margin:0;color:var(--atlas-muted);font-size:14px;line-height:1.55;}
.bar-row {display:grid;grid-template-columns:82px minmax(0,1fr) 84px;gap:12px;align-items:center;margin:12px 0;}
.bar-label {font-size:13px;font-weight:700;color:var(--atlas-ink);}
.bar-track {height:18px;border-radius:9px;background:var(--atlas-soft);overflow:hidden;}
.bar-fill {height:100%;border-radius:9px;}
.bar-fill.before {background:var(--atlas-before);}
.bar-fill.after {background:var(--atlas-teal);}
.bar-value {font-size:13px;font-weight:700;text-align:right;font-variant-numeric:tabular-nums;color:var(--atlas-ink);}

.card {border:1px solid var(--atlas-line);background:var(--atlas-card);padding:16px 19px;border-radius:14px;box-shadow:0 2px 11px rgba(20,46,83,.035);margin-bottom:8px;}
.card .number {color:var(--atlas-ink);font-size:clamp(22px,2vw,28px);font-weight:700;letter-spacing:-.03em;font-family:'Space Grotesk',sans-serif;margin-top:6px;font-variant-numeric:tabular-nums;}
.card .foot {font-size:12.5px;color:var(--atlas-muted);margin-top:2px;}
.card.featured {background:linear-gradient(125deg,#e8f7f2,#eaf7ff);border-color:#abdfd4;}
.card.featured .number {color:var(--atlas-teal-dark);}

.section {margin:38px 0 10px;}
.section h3 {margin:4px 0 4px;padding:0;font-size:22px;color:var(--atlas-ink);}
.section p {margin:0;color:var(--atlas-muted);font-size:14px;line-height:1.55;max-width:860px;}

.legend-box {display:flex;flex-wrap:wrap;gap:6px 22px;padding:10px 14px;border-radius:10px;background:var(--atlas-soft);font-size:12.5px;color:var(--atlas-muted);margin:4px 0 12px;}
.legend-box b {color:var(--atlas-ink);margin-right:4px;}
.warehouse-dot {display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px;vertical-align:-1px;border:2px solid #ffffff;box-shadow:0 0 0 1px rgba(20,36,61,.18);}
.small-dot {display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:0;}
.swatch-line {display:inline-block;width:20px;height:3px;border-radius:2px;margin-right:7px;vertical-align:3px;}
.map-heading {font-size:15px;font-weight:700;margin:6px 0 6px;color:var(--atlas-ink);}
.map-heading span {font-weight:500;color:var(--atlas-muted);}
.scale-box {max-width:420px;margin:10px 0 4px;}
.scale-title {font-size:12px;font-weight:700;color:var(--atlas-ink);margin-bottom:5px;}
.gradient {height:10px;border-radius:5px;}
.scale {display:flex;justify-content:space-between;font-size:11.5px;color:var(--atlas-muted);margin-top:4px;}

.empty {padding:26px 30px;border:1px dashed var(--atlas-line);border-radius:16px;background:var(--atlas-card);}
.empty h3 {margin:0 0 6px;padding:0;color:var(--atlas-ink);}
.empty p {margin:0;color:var(--atlas-muted);font-size:14px;}
.footnote {font-size:12px;color:var(--atlas-faint);line-height:1.6;margin-top:22px;}
@media (max-width:900px){
  .hero {padding:22px 18px;}
  .steps, .result {grid-template-columns:1fr;}
  .result {gap:18px;padding:22px 20px;}
}
</style>
""", unsafe_allow_html=True)

COLORS = [
    [53, 110, 231, 225], [20, 157, 146, 225], [242, 163, 59, 225],
    [158, 93, 228, 225], [240, 111, 107, 225], [44, 170, 205, 225],
    [166, 170, 62, 225], [224, 102, 166, 225], [98, 121, 158, 225],
    [67, 174, 104, 225],
]
DISTANCE_STOPS = [(28, 165, 143), (150, 200, 98), (244, 190, 64), (236, 125, 60), (201, 67, 72)]
SAVINGS_STOPS = [(235, 242, 248), (82, 167, 217), (51, 117, 205), (20, 63, 137)]
BEFORE_LINK = [214, 90, 90, 110]
AFTER_LINK = [33, 139, 133, 200]
UNCHANGED_POINT = [148, 163, 184, 150]
CARTO_STYLE = pdk.map_styles.CARTO_LIGHT
TOOLTIP = {"text": "{tooltip_text}",
           "style": {"backgroundColor": "#16273c", "color": "#ffffff", "fontSize": "12px"}}
MAP_HEIGHT = 430
MAX_CHANGE_LINKS = 150
SAMPLE_SOURCE = "Sample Sydney orders"
UPLOAD_SOURCE = "Upload my own CSV"
CSV_EXAMPLE = ('orderId,warehouseLocation_Coordinates,customerLocation_Coordinates\n'
               'ORD001,"(-33.85, 150.90)","(-33.87, 151.21)"\n'
               'ORD002,"(-33.91, 151.19)","(-33.78, 150.91)"')


class MapSettings(NamedTuple):
    max_points: int
    show_links: bool
    cell_km: int


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def wh_color(index: int) -> list[int]:
    return COLORS[index % len(COLORS)]


def css_color(index: int) -> str:
    rgb = wh_color(index)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def metric_card(label: str, value: str, detail: str, featured: bool = False) -> None:
    css = "card featured" if featured else "card"
    html(f'<div class="{css}"><div class="eyebrow">{label}</div><div class="number">{value}</div>'
         f'<div class="foot">{detail}</div></div>')


def section_heading(step: str, title: str, text: str) -> None:
    html(f'<div class="section"><div class="eyebrow">{step}</div><h3>{title}</h3><p>{text}</p></div>')


@st.cache_data(show_spinner=False)
def analyse(data: pd.DataFrame):
    prepared = prepare_orders(data)
    result = optimise_assignments(prepared)
    return prepared, result


@st.cache_data(show_spinner=False)
def get_grid(result: pd.DataFrame, cell_km: float) -> pd.DataFrame:
    return spatial_grid(result, cell_km)


def read_upload(uploaded) -> pd.DataFrame:
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
    return pd.read_csv(io.BytesIO(raw_bytes), dtype="string", encoding="utf-8-sig")


# --- Maps ---

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
    records["tooltip_text"] = "Warehouse " + records["warehouse_id"] + " · " + records["coordinates"]
    return pdk.Layer(
        "ScatterplotLayer", data=records,
        get_position="[warehouse_lon, warehouse_lat]", get_fill_color="color",
        get_line_color=[255, 255, 255, 240], line_width_min_pixels=2.5,
        radius_min_pixels=9, radius_max_pixels=17, get_radius=200,
        stroked=True, filled=True, pickable=True, auto_highlight=True,
    )


def deck(layers: list[pdk.Layer], view: pdk.ViewState) -> pdk.Deck:
    return pdk.Deck(layers=layers, initial_view_state=view, map_style=CARTO_STYLE, tooltip=TOOLTIP)


def sample_rows(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    return frame if len(frame) <= limit else frame.sample(n=limit, random_state=42)


def coverage_map(result: pd.DataFrame, warehouses: pd.DataFrame, after: bool,
                 show_links: bool, view: pdk.ViewState, max_points: int,
                 max_links: int = 180) -> tuple[pdk.Deck, int]:
    use = sample_rows(result, max_points)
    source_col = "optimised_warehouse" if after else "original_warehouse"
    color_lookup = {wid: wh_color(i) for i, wid in enumerate(warehouses["warehouse_id"])}
    points = use[["orderId", "customer_lon", "customer_lat", "original_distance_km",
                  "optimised_distance_km", "assignment_status", source_col]].copy()
    points["color"] = points[source_col].map(color_lookup)
    distance = np.round(points["optimised_distance_km" if after else "original_distance_km"], 2)
    points["tooltip_text"] = (points["orderId"].astype(str) + " · ships from " + points[source_col]
                              + " · " + distance.astype(str) + " km")
    layers = []
    if show_links:
        lines = use.nlargest(max_links, "distance_saved_km").copy()
        lines["from_lat"] = lines["optimised_warehouse_lat" if after else "warehouse_lat"]
        lines["from_lon"] = lines["optimised_warehouse_lon" if after else "warehouse_lon"]
        layers.append(pdk.Layer(
            "LineLayer", data=lines,
            get_source_position="[from_lon, from_lat]",
            get_target_position="[customer_lon, customer_lat]",
            get_color=AFTER_LINK if after else BEFORE_LINK,
            get_width=1.5, width_min_pixels=1, pickable=False,
        ))
    layers.extend([
        pdk.Layer("ScatterplotLayer", data=points,
                  get_position="[customer_lon, customer_lat]", get_fill_color="color",
                  radius_min_pixels=2.5, radius_max_pixels=8, get_radius=90,
                  pickable=True, auto_highlight=True, opacity=0.86),
        warehouse_layer(warehouses),
    ])
    return deck(layers, view), len(points)


def changes_map(result: pd.DataFrame, warehouses: pd.DataFrame, view: pdk.ViewState,
                max_points: int) -> tuple[pdk.Deck, int]:
    color_lookup = {wid: wh_color(i) for i, wid in enumerate(warehouses["warehouse_id"])}
    reassigned = result.loc[result["assignment_status"].eq("Reassigned")]
    links = reassigned.nlargest(MAX_CHANGE_LINKS, "distance_saved_km")

    unchanged = sample_rows(result.loc[result["assignment_status"].eq("Unchanged")], max_points).copy()
    unchanged["tooltip_text"] = (unchanged["orderId"].astype(str) + " · stays with "
                                 + unchanged["original_warehouse"])

    moved = sample_rows(reassigned, max_points).copy()
    moved["color"] = moved["optimised_warehouse"].map(color_lookup)
    moved["tooltip_text"] = (moved["orderId"].astype(str) + " · " + moved["original_warehouse"]
                             + " → " + moved["optimised_warehouse"] + " · saves "
                             + moved["distance_saved_km"].round(2).astype(str) + " km")

    layers = [
        pdk.Layer("LineLayer", data=links,
                  get_source_position="[warehouse_lon, warehouse_lat]",
                  get_target_position="[customer_lon, customer_lat]",
                  get_color=BEFORE_LINK, get_width=1.5, width_min_pixels=1),
        pdk.Layer("LineLayer", data=links,
                  get_source_position="[optimised_warehouse_lon, optimised_warehouse_lat]",
                  get_target_position="[customer_lon, customer_lat]",
                  get_color=AFTER_LINK, get_width=2, width_min_pixels=1.5),
        pdk.Layer("ScatterplotLayer", data=unchanged,
                  get_position="[customer_lon, customer_lat]", get_fill_color=UNCHANGED_POINT,
                  radius_min_pixels=2, radius_max_pixels=6, get_radius=70,
                  pickable=True, auto_highlight=True),
        pdk.Layer("ScatterplotLayer", data=moved,
                  get_position="[customer_lon, customer_lat]", get_fill_color="color",
                  get_line_color=[255, 255, 255, 220], line_width_min_pixels=1, stroked=True,
                  radius_min_pixels=3.5, radius_max_pixels=9, get_radius=110,
                  pickable=True, auto_highlight=True),
        warehouse_layer(warehouses),
    ]
    return deck(layers, view), len(links)


def ramp(value: float, limit: float, stops: list[tuple[int, int, int]]) -> list[int]:
    t = float(np.clip(value / max(limit, 1e-9), 0, 1))
    pos = t * (len(stops) - 1)
    k = min(int(pos), len(stops) - 2)
    frac = pos - k
    a, b = stops[k], stops[k + 1]
    return [int(a[i] + (b[i] - a[i]) * frac) for i in range(3)] + [190]


def grid_map(grid: pd.DataFrame, warehouses: pd.DataFrame, view: pdk.ViewState,
             value_col: str, scale_max: float, stops: list[tuple[int, int, int]]) -> pdk.Deck:
    cells = grid.copy()
    cells["color"] = [ramp(v, scale_max, stops) for v in cells[value_col]]
    for col in ["mean_original_km", "mean_optimised_km", "total_saved_km"]:
        cells[col] = cells[col].round(2)
    cells["tooltip_text"] = (cells["orders"].astype(str) + " orders · today avg "
                             + cells["mean_original_km"].astype(str) + " km · optimised avg "
                             + cells["mean_optimised_km"].astype(str) + " km · total saved "
                             + cells["total_saved_km"].astype(str) + " km")
    layers = [
        pdk.Layer("PolygonLayer", data=cells,
                  get_polygon="polygon", get_fill_color="color",
                  get_line_color=[255, 255, 255, 80], line_width_min_pixels=0.3,
                  stroked=True, filled=True, pickable=True, auto_highlight=True),
        warehouse_layer(warehouses),
    ]
    return deck(layers, view)


def gradient_legend(title: str, stops: list[tuple[int, int, int]], low: str, high: str) -> None:
    colours = ", ".join(f"rgb({r},{g},{b})" for r, g, b in stops)
    html(f'<div class="scale-box"><div class="scale-title">{title}</div>'
         f'<div class="gradient" style="background:linear-gradient(90deg, {colours})"></div>'
         f'<div class="scale"><span>{low}</span><span>{high}</span></div></div>')


def warehouse_legend(result: pd.DataFrame, warehouses: pd.DataFrame) -> None:
    before = result["original_warehouse"].value_counts()
    after = result["optimised_warehouse"].value_counts()
    items = []
    for idx, row in warehouses.head(12).iterrows():
        wid = row.warehouse_id
        items.append(f'<span><i class="warehouse-dot" style="background:{css_color(idx)}"></i>'
                     f'<b>{wid}</b>{int(before.get(wid, 0)):,} → {int(after.get(wid, 0)):,} orders</span>')
    if len(warehouses) > 12:
        items.append(f"<span>+{len(warehouses) - 12} more warehouses (colours repeat)</span>")
    html('<div class="legend-box">' + "".join(items) + "</div>")


# --- Charts ---

def style_figure(fig: go.Figure, height: int) -> None:
    fig.update_layout(
        height=height, margin=dict(t=30, b=16, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Source Sans 3, Source Sans Pro, sans-serif", color="#14243d", size=13),
    )


def chart_movements(result: pd.DataFrame, warehouses: pd.DataFrame) -> None:
    before = result["original_warehouse"].value_counts()
    after = result["optimised_warehouse"].value_counts()
    ids = warehouses["warehouse_id"].tolist()
    fig = go.Figure()
    fig.add_bar(name="Today", x=ids, y=[int(before.get(i, 0)) for i in ids],
                marker_color="#a5b8d0", marker_line_width=0, texttemplate="%{y:,}", textposition="outside")
    fig.add_bar(name="Optimised", x=ids, y=[int(after.get(i, 0)) for i in ids],
                marker_color="#218b85", marker_line_width=0, texttemplate="%{y:,}", textposition="outside")
    style_figure(fig, 320)
    fig.update_layout(
        barmode="group", bargap=0.3, legend=dict(orientation="h", y=1.12, x=0),
        yaxis=dict(title="Orders", showgrid=True, gridcolor="rgba(148,163,184,.19)"),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def chart_savings(result: pd.DataFrame, warehouses: pd.DataFrame) -> None:
    saved = result.groupby("original_warehouse")["distance_saved_km"].sum()
    ids = warehouses["warehouse_id"].tolist()
    fig = go.Figure(go.Bar(
        x=[float(saved.get(i, 0)) for i in ids], y=ids,
        orientation="h", marker=dict(color=[css_color(i) for i in range(len(ids))]),
        text=[f"{saved.get(i, 0):,.0f} km" for i in ids], textposition="auto",
    ))
    style_figure(fig, 320)
    fig.update_layout(
        showlegend=False, bargap=0.35,
        yaxis=dict(autorange="reversed", title=""),
        xaxis=dict(title="Kilometres saved", showgrid=True, gridcolor="rgba(148,163,184,.19)"),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def transfer_table(result: pd.DataFrame) -> None:
    moved = result.loc[result["assignment_status"].eq("Reassigned")]
    if moved.empty:
        st.caption("No orders were reassigned.")
        return
    table = (moved.groupby(["original_warehouse", "optimised_warehouse"], as_index=False)
             .agg(orders=("orderId", "size"), saved_km=("distance_saved_km", "sum"),
                  mean_saved_km=("distance_saved_km", "mean"))
             .sort_values("orders", ascending=False))
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "original_warehouse": st.column_config.TextColumn("From (today)"),
        "optimised_warehouse": st.column_config.TextColumn("To (optimised)"),
        "orders": st.column_config.NumberColumn("Orders moved", format="%d"),
        "saved_km": st.column_config.NumberColumn("Total km saved", format="%.1f"),
        "mean_saved_km": st.column_config.NumberColumn("Km saved per order", format="%.2f"),
    })


# --- Page sections ---

def render_sidebar() -> tuple[str, pd.DataFrame | None, str | None, MapSettings]:
    data, error = None, None
    with st.sidebar:
        html('<div class="brand"><div class="brand-name">◈ Warehouse Atlas</div>'
             '<div class="brand-sub">Warehouse allocation optimiser</div></div>')
        html('<div class="side-step">Orders to analyse</div>')
        source = st.radio("Orders to analyse", [SAMPLE_SOURCE, UPLOAD_SOURCE], label_visibility="collapsed")
        if source == SAMPLE_SOURCE:
            demo_n = st.select_slider("Number of sample orders",
                                      options=[100, 500, 1_000, 5_000, 10_000, 25_000], value=1_000)
            data = make_demo(count=demo_n)
            st.download_button("Download this sample as CSV", data=data.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"sydney_{demo_n}_orders.csv", mime="text/csv",
                               icon=":material/download:", type="tertiary")
        else:
            uploaded = st.file_uploader("Upload a CSV", type=["csv"],
                                        help="A UTF-8 CSV with the three column names shown below.")
            st.caption("Needs the columns `orderId`, `warehouseLocation_Coordinates` and "
                       "`customerLocation_Coordinates`.")
            if uploaded is not None:
                try:
                    data = read_upload(uploaded)
                except Exception as exc:
                    error = str(exc)

        st.divider()
        with st.expander("Map display settings"):
            max_points = st.select_slider("Maximum dots per map",
                                          options=[1_000, 3_000, 8_000, 15_000, 25_000], value=8_000,
                                          help="Larger datasets show a random sample of dots. "
                                               "Totals, heatmaps and downloads always use every order.")
            show_links = st.checkbox("Draw warehouse links on the before/after maps", value=False,
                                     help="Draws up to 180 links for the orders with the biggest savings.")
            cell_km = st.slider("Heatmap square size (km)", min_value=1, max_value=20, value=5, step=1)
        st.caption("Distances are straight-line kilometres, not road distance. "
                   "Every warehouse is assumed able to serve every order.")
    return source, data, error, MapSettings(max_points, show_links, cell_km)


def render_hero() -> None:
    html('<div class="hero">'
         '<div class="kicker">Warehouse allocation optimiser</div>'
         '<h1>Ship every order from its nearest warehouse</h1>'
         '<p>Each order is shipped from a warehouse today. This tool checks every order against every '
         'warehouse, moves it to the closest one, and shows how much delivery distance that saves.</p>'
         '<div class="steps">'
         '<div class="step"><span class="num">1</span><b>Load orders</b>'
         'Each row has an order, the warehouse it ships from today, and the customer location.</div>'
         '<div class="step"><span class="num">2</span><b>Measure distances</b>'
         'Work out the straight-line distance from every customer to every warehouse.</div>'
         '<div class="step"><span class="num">3</span><b>Pick the nearest</b>'
         'Reassign each order that has a closer warehouse. Keep the rest as they are.</div>'
         '</div></div>')


def render_upload_help(error: str | None) -> None:
    if error:
        st.error(f"We couldn't read that CSV. {error}")
    html('<div class="empty"><h3>Upload your orders to begin</h3>'
         '<p>Use the sidebar to upload a CSV, or switch back to the sample Sydney orders. '
         'The file needs exactly these three columns. Write each coordinate as latitude, longitude '
         'in brackets, inside double quotes:</p></div>')
    st.code(CSV_EXAMPLE, language="csv")


def render_dataset_chips(source: str, n: int, warehouse_count: int, rejected_count: int) -> None:
    chips = ["Sample Sydney data" if source == SAMPLE_SOURCE else "Your uploaded data",
             f"{n:,} orders", f"{warehouse_count:,} warehouses found"]
    if rejected_count:
        chips.append(f"{rejected_count:,} invalid rows skipped")
    html('<div class="chips">' + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div>")


def render_result(result: pd.DataFrame) -> None:
    n = len(result)
    reassigned_n = int(result["assignment_status"].eq("Reassigned").sum())
    baseline_km = float(result["original_distance_km"].sum())
    optimal_km = float(result["optimised_distance_km"].sum())
    saved_km = max(0.0, baseline_km - optimal_km)
    pct = (saved_km / baseline_km * 100) if baseline_km > 0 else 0.0
    after_width = (optimal_km / baseline_km * 100) if baseline_km > 0 else 0.0

    if reassigned_n == 0:
        headline = "Every order already ships from its nearest warehouse"
        summary = "There is nothing to reassign, so the total distance stays the same."
    else:
        headline = f"Delivery distance drops <em>{pct:.1f}%</em>"
        summary = (f"Moving {reassigned_n:,} of {n:,} orders to a closer warehouse saves "
                   f"{saved_km:,.1f} km of straight-line distance.")
    html(f'<div class="result"><div><div class="eyebrow">The result</div><h2>{headline}</h2><p>{summary}</p></div>'
         '<div>'
         f'<div class="bar-row"><span class="bar-label">Today</span><div class="bar-track">'
         f'<div class="bar-fill before" style="width:100%"></div></div>'
         f'<span class="bar-value">{baseline_km:,.0f} km</span></div>'
         f'<div class="bar-row"><span class="bar-label">Optimised</span><div class="bar-track">'
         f'<div class="bar-fill after" style="width:{after_width:.1f}%"></div></div>'
         f'<span class="bar-value">{optimal_km:,.0f} km</span></div>'
         '</div></div>')

    cards = st.columns(3, gap="small")
    with cards[0]:
        metric_card("Distance saved", f"{saved_km:,.1f} km", f"{pct:.1f}% less than today", featured=True)
    with cards[1]:
        metric_card("Orders reassigned", f"{reassigned_n:,}",
                    f"of {n:,} orders ({reassigned_n / n * 100:.0f}%) move to a closer warehouse")
    with cards[2]:
        metric_card("Average distance per order", f"{baseline_km / n:,.1f} → {optimal_km / n:,.1f} km",
                    "today → optimised")


def render_maps(result: pd.DataFrame, warehouses: pd.DataFrame, settings: MapSettings) -> None:
    section_heading("Step 1", "See the difference on the map",
                    "Each dot is a customer, coloured by the warehouse that serves it. "
                    "Large outlined dots are the warehouses. Hover over any dot for details.")
    warehouse_legend(result, warehouses)
    view = calc_view(result, warehouses)
    n = len(result)
    tabs = st.tabs(["Before vs after", "What changed", "Distance heatmap", "Savings hotspots"])

    with tabs[0]:
        before_col, after_col = st.columns(2, gap="medium")
        before_map, shown = coverage_map(result, warehouses, False, settings.show_links, view, settings.max_points)
        after_map, _ = coverage_map(result, warehouses, True, settings.show_links, view, settings.max_points)
        with before_col:
            html('<div class="map-heading">Today <span>· current warehouse for each order</span></div>')
            st.pydeck_chart(before_map, height=MAP_HEIGHT, key="map_before")
        with after_col:
            html('<div class="map-heading">Optimised <span>· nearest warehouse for each order</span></div>')
            st.pydeck_chart(after_map, height=MAP_HEIGHT, key="map_after")
        if n > settings.max_points:
            st.caption(f"Showing a random sample of {shown:,} of {n:,} orders to keep the maps fast. "
                       "All figures and downloads include every order.")

    with tabs[1]:
        changes, link_count = changes_map(result, warehouses, view, settings.max_points)
        html('<div class="legend-box">'
             f'<span><i class="swatch-line" style="background:rgba({",".join(map(str, BEFORE_LINK[:3]))},.75)"></i>'
             'Before: link to current warehouse</span>'
             f'<span><i class="swatch-line" style="background:rgb({",".join(map(str, AFTER_LINK[:3]))})"></i>'
             'After: link to nearest warehouse</span>'
             f'<span><i class="small-dot" style="background:rgb({",".join(map(str, UNCHANGED_POINT[:3]))})"></i>'
             'Order that stays with its warehouse</span></div>')
        st.pydeck_chart(changes, height=MAP_HEIGHT + 40, key="map_changes")
        st.caption(f"Links are drawn for the {link_count:,} reassigned orders with the biggest savings. "
                   "Coloured dots are reassigned orders, in the colour of their new warehouse.")

    with tabs[2]:
        grid = get_grid(result, float(settings.cell_km))
        max_scale = max(float(np.quantile(result["original_distance_km"], .95)), 0.01)
        gradient_legend("Average distance from warehouse to customer in each square",
                        DISTANCE_STOPS, "0 km (short)", f"{max_scale:.0f}+ km (long)")
        left, right = st.columns(2, gap="medium")
        with left:
            html('<div class="map-heading">Today</div>')
            st.pydeck_chart(grid_map(grid, warehouses, view, "mean_original_km", max_scale, DISTANCE_STOPS),
                            height=MAP_HEIGHT, key="heat_before")
        with right:
            html('<div class="map-heading">Optimised</div>')
            st.pydeck_chart(grid_map(grid, warehouses, view, "mean_optimised_km", max_scale, DISTANCE_STOPS),
                            height=MAP_HEIGHT, key="heat_after")
        st.caption(f"Each square is about {settings.cell_km} km wide and includes every order inside it. "
                   "Both maps use the same colour scale, so greener means shorter trips.")

    with tabs[3]:
        grid = get_grid(result, float(settings.cell_km))
        upper = max(float(np.quantile(grid["total_saved_km"], .95)), 0.01)
        gradient_legend("Total kilometres saved in each square", SAVINGS_STOPS,
                        "0 km", f"{upper:,.0f}+ km")
        st.pydeck_chart(grid_map(grid, warehouses, view, "total_saved_km", upper, SAVINGS_STOPS),
                        height=MAP_HEIGHT + 40, key="savings_map")
        st.caption("Darker squares are where reassigning orders saves the most distance in total.")
        hotspots = grid.nlargest(8, "total_saved_km").copy()
        hotspots["area"] = [f"({lat:.3f}, {lon:.3f})"
                            for lat, lon in zip(hotspots["center_lat"], hotspots["center_lon"])]
        st.markdown("**Top 8 areas by distance saved**")
        st.dataframe(hotspots[["area", "orders", "total_saved_km", "mean_saved_km"]],
                     hide_index=True, width="stretch", column_config={
                         "area": st.column_config.TextColumn("Area centre (lat, lon)"),
                         "orders": st.column_config.NumberColumn("Orders", format="%d"),
                         "total_saved_km": st.column_config.NumberColumn("Total km saved", format="%.1f"),
                         "mean_saved_km": st.column_config.NumberColumn("Km saved per order", format="%.2f"),
                     })


def render_insights(result: pd.DataFrame, warehouses: pd.DataFrame) -> None:
    section_heading("Step 2", "Where the savings come from",
                    "How the workload shifts between warehouses, and which moves save the most distance.")
    first, second = st.columns(2, gap="large")
    with first:
        st.markdown("**Orders per warehouse, today vs optimised**")
        chart_movements(result, warehouses)
    with second:
        st.markdown("**Distance saved, by the warehouse orders leave**")
        chart_savings(result, warehouses)
    st.markdown("**Where reassigned orders move**")
    transfer_table(result)


def render_explorer(result: pd.DataFrame, warehouses: pd.DataFrame) -> None:
    section_heading("Step 3", "Look up individual orders",
                    "Filter by warehouse or status, or search for an order ID.")
    f1, f2, f3 = st.columns([1, 1, 2], gap="small")
    with f1:
        warehouse_filter = st.selectbox("Warehouse today", ["All", *warehouses["warehouse_id"].tolist()])
    with f2:
        status_filter = st.selectbox("Status", ["All", "Reassigned", "Unchanged"])
    with f3:
        search = st.text_input("Order ID", placeholder="Search part of an order ID", max_chars=100)
    display = result
    if warehouse_filter != "All":
        display = display.loc[display["original_warehouse"].eq(warehouse_filter)]
    if status_filter != "All":
        display = display.loc[display["assignment_status"].eq(status_filter)]
    if search:
        display = display.loc[display["orderId"].astype("string").str.contains(search, regex=False, case=False, na=False)]
    columns = ["orderId", "original_warehouse", "optimised_warehouse", "original_distance_km",
               "optimised_distance_km", "distance_saved_km", "assignment_status"]
    st.dataframe(display[columns].head(2000), width="stretch", hide_index=True, column_config={
        "orderId": st.column_config.TextColumn("Order"),
        "original_warehouse": st.column_config.TextColumn("Warehouse today"),
        "optimised_warehouse": st.column_config.TextColumn("Optimised warehouse"),
        "original_distance_km": st.column_config.NumberColumn("Km today", format="%.2f"),
        "optimised_distance_km": st.column_config.NumberColumn("Km optimised", format="%.2f"),
        "distance_saved_km": st.column_config.NumberColumn("Km saved", format="%.2f"),
        "assignment_status": st.column_config.TextColumn("Status"),
    })
    shown_note = " The table shows the first 2,000." if len(display) > 2000 else ""
    st.caption(f"{len(display):,} matching orders.{shown_note}")
    st.download_button("Download matching orders", data=output_csv(display),
                       file_name="filtered_warehouse_allocations.csv", mime="text/csv",
                       icon=":material/download:")


def render_downloads(result: pd.DataFrame) -> None:
    section_heading("Step 4", "Download the results", "Both files include every valid order.")
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        st.markdown("**Full report**")
        st.caption("Today's and the optimised warehouse for each order, both distances, km saved and status.")
        st.download_button("Download full report", data=output_csv(result),
                           file_name="warehouse_optimisation_full.csv", mime="text/csv",
                           icon=":material/download:", type="primary", width="stretch")
    with right, st.container(border=True):
        st.markdown("**Updated input file**")
        st.caption("Same three columns as your input, with each warehouse replaced by the optimised one.")
        st.download_button("Download updated file", data=output_csv(result, minimal=True),
                           file_name="warehouse_assignments_optimised.csv", mime="text/csv",
                           icon=":material/download:", width="stretch")


def render_methodology(rejected: pd.DataFrame) -> None:
    st.write("")
    with st.expander("How it works, and its limits"):
        st.markdown("""
**Input:** a CSV with the columns `orderId`, `warehouseLocation_Coordinates` and `customerLocation_Coordinates` (case-sensitive).
Coordinates are latitude then longitude, such as `"(-33.84750, 150.90060)"`.
Each coordinate contains a comma, so it **must be quoted** in the CSV.

**Finding warehouses:** every distinct warehouse coordinate in the file counts as one warehouse.
Coordinates are rounded to 5 decimal places so tiny GPS differences merge.
A warehouse that never appears in the file can't be chosen.

**Choosing the nearest:** each order is checked on its own against every warehouse, using straight-line (Haversine) distance.
If the current warehouse is already the closest, or tied for closest, the order stays where it is.

**Not included:** road distance, delivery time, shipping cost, warehouse capacity, stock levels, product restrictions, or combining deliveries into routes.
If those matter, a solver that handles constraints is needed.

**Large files:** distances are calculated in chunks to keep memory use bounded.
Maps show a sample of dots above the limit set in the sidebar, but every order counts in totals, heatmaps and downloads.
""")
    if not rejected.empty:
        with st.expander(f"Rows skipped because they were invalid ({len(rejected):,})"):
            st.dataframe(rejected.head(500), hide_index=True, width="stretch")
            st.download_button("Download invalid rows", rejected.to_csv(index=False).encode("utf-8-sig"),
                               file_name="warehouse_invalid_rows.csv", mime="text/csv",
                               icon=":material/download:")


# --- Page ---

source, data, upload_error, settings = render_sidebar()
render_hero()

if data is None:
    render_upload_help(upload_error)
    st.stop()

try:
    prepared, result = analyse(data)
except Exception as exc:
    render_upload_help(f"Unable to process this dataset: {exc}")
    st.stop()

warehouses = prepared.warehouses
render_dataset_chips(source, len(result), len(warehouses), len(prepared.rejected))
for message in prepared.notes:
    st.warning(message)

render_result(result)
render_maps(result, warehouses, settings)
render_insights(result, warehouses)
render_explorer(result, warehouses)
render_downloads(result)
render_methodology(prepared.rejected)

html('<div class="footnote">Warehouse Atlas · All distances are straight-line estimates. '
     'The sample Sydney orders are made up for illustration.</div>')
