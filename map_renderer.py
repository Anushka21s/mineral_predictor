"""
map_renderer.py
"""

import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd
import numpy as np

MINERAL_COLORS = {
    "Gold":      "#F4B942",
    "Copper":    "#C45E2A",
    "Iron":      "#8B3A3A",
    "Lithium":   "#4E9FD1",
    "Manganese": "#6A5ACD",
    "Bauxite":   "#8FBC8F",
}

def _base_map() -> folium.Map:
    return folium.Map(
        location=[22.5, 82.5],
        zoom_start=5,
        tiles="CartoDB positron",
        attr="© OpenStreetMap contributors, © CARTO",
    )

def build_site_map(df: pd.DataFrame) -> folium.Map:
    m = _base_map()
    title_html = (
        '<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);'
        'z-index:1000;background:white;padding:8px 16px;border-radius:8px;'
        'box-shadow:0 2px 6px rgba(0,0,0,.25);font-family:sans-serif;font-size:14px;">'
        '🗺️ India Mineral Survey Sites</div>'
    )
    m.get_root().html.add_child(folium.Element(title_html))
    groups = {mineral: folium.FeatureGroup(name=mineral) for mineral in MINERAL_COLORS}
    for _, row in df.iterrows():
        mineral = row["mineral"]
        color   = MINERAL_COLORS.get(mineral, "#888888")
        popup_html = (
            f"<b>{row['site_name']}</b><br>"
            f"Mineral : {mineral}<br>"
            f"State   : {row['state']}<br>"
            f"Deposit : {row['dep_type']}<br>"
            f"Status  : {row['dev_stat']}<br>"
            f"Au ppb  : {row['au_ppb']:.1f} | Fe %: {row['fe_pct']:.1f}<br>"
            f"Elevation: {row['elevation_m']:.0f} m"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5, color=color, fill=True, fill_color=color,
            fill_opacity=0.75, weight=1,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{mineral} – {row['site_name']}",
        ).add_to(groups[mineral])
    for g in groups.values():
        g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def build_prediction_map(pred_df: pd.DataFrame,
                         mineral_gradients: dict = None) -> folium.Map:
    m = _base_map()
    _default_gradients = {
        "Gold":      {0.2: "#fff7aa", 0.5: "#f4b942", 1.0: "#b8730a"},
        "Copper":    {0.2: "#f7dfd0", 0.5: "#c45e2a", 1.0: "#7a2a00"},
        "Iron":      {0.2: "#f0c0c0", 0.5: "#8b3a3a", 1.0: "#3a0000"},
        "Lithium":   {0.2: "#cce8f7", 0.5: "#4e9fd1", 1.0: "#003f6b"},
        "Manganese": {0.2: "#ddd8f5", 0.5: "#6a5acd", 1.0: "#2e006b"},
        "Bauxite":   {0.2: "#d8f0d8", 0.5: "#8fbc8f", 1.0: "#2e6b2e"},
    }
    gradients = mineral_gradients or _default_gradients
    title_html = (
        '<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);'
        'z-index:1000;background:white;padding:8px 16px;border-radius:8px;'
        'box-shadow:0 2px 6px rgba(0,0,0,.25);font-family:sans-serif;font-size:14px;">'
        '🔬 ML Predicted Mineral Hotspots</div>'
    )
    m.get_root().html.add_child(folium.Element(title_html))
    minerals = pred_df["predicted_mineral"].unique()
    for mineral in minerals:
        color    = MINERAL_COLORS.get(mineral, "#888")
        gradient = gradients.get(mineral, {0.2: "#aaa", 0.5: color, 1.0: "#000"})
        sub      = pred_df[pred_df["predicted_mineral"] == mineral]
        prob_col = f"prob_{mineral}"
        
       # Only use points where this mineral has meaningful probability
       heat_sub = pred_df[pred_df[prob_col] >= 50][["latitude","longitude", prob_col]]
        heat_data = heat_sub.values.tolist()
        if heat_data:
            HeatMap(
                heat_data,
                name=f"{mineral} heatmap",
                min_opacity=0.3,
                max_val=float(pred_df[prob_col].max()),
                radius=18,
                blur=20,
                gradient=gradient,
                show=True,
            ).add_to(m)
        
        top = sub.nlargest(8, prob_col)
        fg  = folium.FeatureGroup(name=f"{mineral} markers", show=False)
        for _, row in top.iterrows():
            popup_html = (
                f"<b>Predicted: {mineral}</b><br>"
                f"Confidence : {row['confidence']:.1f}%<br>"
                f"Lat/Lon : {row['latitude']:.3f}, {row['longitude']:.3f}"
            )
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=f"{mineral} ({row['confidence']:.0f}%)",
                icon=folium.Icon(color="white", icon_color=color,
                                 icon="circle", prefix="fa"),
            ).add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def build_state_summary_map(df: pd.DataFrame) -> folium.Map:
    from data_generator import STATE_INFO
    m = _base_map()
    state_counts = df.groupby("state").size().reset_index(name="count")
    max_count    = state_counts["count"].max()
    for _, row in state_counts.iterrows():
        state = row["state"]
        if state not in STATE_INFO:
            continue
        lat, lon    = STATE_INFO[state]
        radius      = 6 + 24 * (row["count"] / max_count)
        top_mineral = df[df["state"] == state]["mineral"].value_counts().idxmax()
        color       = MINERAL_COLORS.get(top_mineral, "#888")
        folium.CircleMarker(
            location=[lat, lon], radius=radius,
            color=color, fill=True, fill_color=color,
            fill_opacity=0.55, weight=1.5,
            tooltip=f"{state}: {row['count']} sites | Top: {top_mineral}",
            popup=folium.Popup(
                f"<b>{state}</b><br>Sites: {row['count']}<br>"
                f"Dominant: {top_mineral}", max_width=180),
        ).add_to(m)
    return m
