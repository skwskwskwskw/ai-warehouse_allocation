# Warehouse Atlas

A polished Streamlit app to optimise **warehouse selection only**, using each customer's original warehouse and location. It does not plan vehicle routes or delivery stops.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (normally `http://localhost:8501`). Choose the built-in 100 / 1,000 / 5,000 / 25,000 Sydney records or upload your own CSV.

## CSV input: exactly three required headers

```csv
orderId,warehouseLocation_Coordinates,customerLocation_Coordinates
ORD001,"(-33.84750, 150.90060)","(-33.87000, 151.21000)"
ORD002,"(-33.91180, 151.19400)","(-33.78000, 150.91000)"
ORD003,"(-33.70580, 151.09890)","(-33.75000, 151.15000)"
```

- Coordinates must be formatted as `(latitude, longitude)` using **decimal degrees** and a comma (or a semicolon/pipe) as a delimiter.
- **Quote coordinate values** that contain commas in your CSV, as in the example.
- The header names are case-sensitive. Order IDs are preserved as strings, including leading zeros.
- The app infers the candidate warehouse list from distinct original warehouse coordinates. Any warehouse absent from the input cannot be selected. A coordinate deduplication step rounds to five decimal places.
- All available warehouses are assumed to be able to fulfil every order. **No** capacity, inventory, product compatibility, cross-border, timing, road network, or different freight tariffs are considered.
- Malformed rows are excluded with a downloadable reason report. Valid rows are processed and exported.

## How optimisation works

1. Collect all distinct original warehouse locations.
2. Calculate Haversine (great-circle, not driving) distance between each valid customer's location and every candidate warehouse.
3. Choose the warehouse with the lowest distance, retaining the original warehouse if the distances tie.
4. Calculate the new summed distance, distance saved and percentage improvement.

The independent nearest-warehouse assignment is mathematically optimal **for this exact unconstrained distance objective**. Because the model assumes one direct warehouse-to-customer distance per order, it does not account for transport consolidation or carrier service levels.

## Maps and large datasets

- **Allocation maps:** before and after, colour-coded by warehouse. Individual markers are sampled above a user-selected limit (default 8,000), but all orders are fully optimised.
- **Distance heatmaps:** geographic cells contain mean original distance and mean optimised distance on a shared colour scale. Grid aggregation includes **every valid order**, regardless of marker sampling.
- **Savings hotspots:** spatial cells coloured by total kilometres saved across their assigned orders; shows where efficiency improvements concentrate.
- **Order explorer:** filters, searchable IDs, downloadable matching rows.
- **Charts:** original and optimised warehouse allocations, and savings by original warehouse.

Data processing is vectorised, with chunked distance calculations to keep memory bounded. 100,000 synthetic records were smoke-tested locally through the optimisation and heatmap engine. Interactive browser performance depends on your environment and GPU; adjust the marker cap and heatmap cell size when needed.

## Output CSVs

- `warehouse_optimisation_full.csv`: original 3 fields + original warehouse label, optimised label and coordinates, old/new distance, savings and status.
- `warehouse_assignments_optimised.csv`: exactly the original 3 headers, with the warehouse column rewritten to the optimised coordinate.

## Tests

```bash
pytest -q tests/
```

## Tech stack

Streamlit · pandas · NumPy · pydeck / deck.gl · Plotly. Maps use third-party Carto basemap tiles and therefore normally require internet access in the user's browser; the distance calculation itself is offline.
