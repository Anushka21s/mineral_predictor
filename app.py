"""
app.py  –  Geospatial Mineral Prediction Dashboard
----------------------------------------------------
Run with:
    streamlit run app.py

Changes v2:
  - Removed dataset size slider → fixed at 2000 samples
  - Added "🔮 Predict a Location" page with lat/lon/elevation/deposit inputs
    showing full probability breakdown for all minerals
  - Probability bar chart on Overview
  - Gold heatmap now uses a distinct yellow-orange gradient per mineral
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Geospatial Mineral Prediction",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ─────────────────────────────────────────────────────────────
MINERAL_COLORS = {
    "Gold":      "#F4B942",
    "Copper":    "#C45E2A",
    "Iron":      "#8B3A3A",
    "Lithium":   "#4E9FD1",
    "Manganese": "#6A5ACD",
    "Bauxite":   "#8FBC8F",
}

# Per-mineral distinct heatmap gradients (fixes uninformative Gold heatmap)
MINERAL_GRADIENTS = {
    "Gold":      {0.2: "#fff7aa", 0.5: "#f4b942", 1.0: "#b8730a"},
    "Copper":    {0.2: "#f7dfd0", 0.5: "#c45e2a", 1.0: "#7a2a00"},
    "Iron":      {0.2: "#f0c0c0", 0.5: "#8b3a3a", 1.0: "#3a0000"},
    "Lithium":   {0.2: "#cce8f7", 0.5: "#4e9fd1", 1.0: "#003f6b"},
    "Manganese": {0.2: "#ddd8f5", 0.5: "#6a5acd", 1.0: "#2e006b"},
    "Bauxite":   {0.2: "#d8f0d8", 0.5: "#8fbc8f", 1.0: "#2e6b2e"},
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #1a1a2e; }
  [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
  .section-header {
    font-size: 1.5rem; font-weight: 700;
    color: #1a1a2e; padding: 8px 0 4px 0;
    border-bottom: 2px solid #4E9FD1; margin-bottom: 16px;
  }
  .prob-bar-wrap {
    margin: 6px 0; font-family: sans-serif;
  }
  .prob-label {
    display: flex; justify-content: space-between;
    font-size: 0.9rem; margin-bottom: 3px;
  }
  .prob-track {
    background: #e9ecef; border-radius: 6px; height: 22px;
    overflow: hidden;
  }
  .prob-fill {
    height: 100%; border-radius: 6px;
    display: flex; align-items: center;
    padding-left: 8px; color: white;
    font-size: 0.8rem; font-weight: 600;
    min-width: 32px; transition: width 0.4s ease;
  }
  .predict-card {
    background: #f8f9fa; border-radius: 12px;
    padding: 20px; border-left: 5px solid;
    margin-bottom: 12px;
  }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔬 Training ML model on 2000 survey points…")
def load_model():
    from model import train_models
    return train_models(2000)          # ← fixed at 2000, no slider


@st.cache_data(show_spinner="🗺️ Generating prediction grid…")
def load_prediction_map(_rf_pipeline, _le, feat_names):
    from model import predict_map
    return predict_map(_rf_pipeline, _le, feat_names)


def render_probability_bars(proba_dict: dict):
    """Render coloured horizontal probability bars for each mineral."""
    sorted_items = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
    html = ""
    for mineral, prob in sorted_items:
        color  = MINERAL_COLORS.get(mineral, "#888")
        pct    = round(prob * 100, 1)
        width  = max(pct, 3)          # min visible width
        html  += f"""
        <div class="prob-bar-wrap">
          <div class="prob-label">
            <span><b>{mineral}</b></span>
            <span>{pct}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill"
                 style="width:{width}%;background:{color};">
              {pct}%
            </div>
          </div>
        </div>"""
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⛏️ Mineral Predictor")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "📊 Dataset Explorer",
         "🤖 Model & Metrics",
         "🗺️ Prediction Map",
         "📈 Feature Importance",
         "📍 State Analysis",
         "🔮 Predict a Location"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    sel_minerals = st.multiselect(
        "Filter minerals (charts only)",
        list(MINERAL_COLORS.keys()),
        default=list(MINERAL_COLORS.keys()),
    )
    st.markdown("---")
    st.markdown(
        "<small>2000 survey points · Random Forest · Folium</small>",
        unsafe_allow_html=True,
    )

# ── Load model (always 2000 samples) ─────────────────────────────────────────
df, rf_pipeline, xgb_pipeline, metrics, feature_names, le = load_model()

# Apply mineral filter for charts (not for model)
df_filtered = df[df["mineral"].isin(sel_minerals)] if sel_minerals else df


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown(
        '<div class="section-header">Geospatial Mineral Prediction — India</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        This dashboard uses **Machine Learning** to predict regions in India
        with high mineral occurrence probability based on:

        | Feature category | Variables |
        |---|---|
        | 🌍 Geospatial | Latitude, Longitude, Elevation |
        | ⚗️ Geochemical | Au (ppb), Cu (ppm), Fe (%), Li (ppm), Mn (%), Al (%) |
        | 🪨 Geological | Deposit type, Soil pH, Fault proximity (km) |
        | 🏭 Operational | Oper. type, Dev. status, Production size |

        **Minerals predicted:** Gold · Copper · Iron · Lithium · Manganese · Bauxite

        The dataset is synthetically generated to mirror the structure of the
        real **India Mineral Ores** dataset (Kaggle), preserving authentic
        geochemical signatures for each mineral type.
        """)

    with col2:
        st.markdown("#### Dataset at a glance")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Survey Sites",    "2,000")
            st.metric("Features Used",   len(feature_names))
        with m2:
            st.metric("Mineral Classes", len(df["mineral"].unique()))
            st.metric("Indian States",   df["state"].nunique())

        st.markdown("#### Model performance")
        st.metric("RF Accuracy",   f"{metrics['rf_accuracy']*100:.1f}%")
        st.metric("RF F1 Score",   f"{metrics['rf_f1']*100:.1f}%")
        st.metric("5-fold CV Acc", f"{metrics['rf_cv_accuracy']*100:.1f}%")

    st.markdown("---")
    st.markdown("#### Mineral distribution across sites")
    counts = df_filtered["mineral"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 3))
    bars = ax.bar(counts.index, counts.values,
                  color=[MINERAL_COLORS.get(m, "#888") for m in counts.index],
                  edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Site count"); ax.set_xlabel("Mineral")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Explorer":
    st.markdown('<div class="section-header">📊 Dataset Explorer</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Raw Data", "Geochemical Profiles",
                                 "Geological Attributes"])

    with tab1:
        col_filter = st.multiselect(
            "Select columns to display",
            df_filtered.columns.tolist(),
            default=["site_name","state","mineral","latitude","longitude",
                     "elevation_m","au_ppb","cu_ppm","fe_pct","dep_type","dev_stat"],
        )
        st.dataframe(df_filtered[col_filter].head(200),
                     use_container_width=True, height=400)
        st.caption(f"Showing first 200 of {len(df_filtered)} rows")

    with tab2:
        st.markdown("#### Geochemical signature by mineral")
        geochem_cols = ["au_ppb","cu_ppm","fe_pct","li_ppm","mn_pct","al_pct"]
        selected_elem = st.selectbox("Element", geochem_cols)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        minerals_present = [m for m in sel_minerals
                            if m in df_filtered["mineral"].unique()]
        data_by_mineral  = [df_filtered[df_filtered["mineral"]==m][selected_elem].values
                            for m in minerals_present]
        bp = axes[0].boxplot(data_by_mineral, labels=minerals_present,
                             patch_artist=True,
                             medianprops=dict(color="white", linewidth=2))
        for patch, mineral in zip(bp["boxes"], minerals_present):
            patch.set_facecolor(MINERAL_COLORS.get(mineral, "#888"))
        axes[0].set_title(f"{selected_elem} distribution by mineral")
        axes[0].set_ylabel(selected_elem)
        axes[0].tick_params(axis="x", rotation=15)
        axes[0].spines[["top","right"]].set_visible(False)

        sc = axes[1].scatter(df_filtered["longitude"], df_filtered["latitude"],
                             c=df_filtered[selected_elem], cmap="YlOrRd",
                             s=10, alpha=0.6)
        plt.colorbar(sc, ax=axes[1], label=selected_elem)
        axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
        axes[1].set_title(f"{selected_elem} spatial distribution (India)")
        axes[1].spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Deposit type distribution**")
            dep_counts = df_filtered["dep_type"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.barh(dep_counts.index, dep_counts.values,
                    color="#4E9FD1", edgecolor="white")
            ax.set_xlabel("Count")
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            st.markdown("**Development status breakdown**")
            dev_counts = df_filtered["dev_stat"].value_counts()
            colors_pie = ["#F4B942","#4E9FD1","#8B3A3A","#6A5ACD","#8FBC8F"]
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(dev_counts.values, labels=dev_counts.index,
                   autopct="%1.1f%%", colors=colors_pie[:len(dev_counts)],
                   startangle=90,
                   wedgeprops=dict(linewidth=0.5, edgecolor="white"))
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("**Correlation heatmap — geochemical + spatial features**")
        num_cols = ["latitude","longitude","elevation_m",
                    "au_ppb","cu_ppm","fe_pct","li_ppm","mn_pct","al_pct",
                    "soil_ph","fault_dist_km"]
        corr = df_filtered[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8})
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL & METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model & Metrics":
    st.markdown('<div class="section-header">🤖 Model Training & Evaluation</div>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RF Accuracy",      f"{metrics['rf_accuracy']*100:.2f}%")
    col2.metric("RF F1 (weighted)", f"{metrics['rf_f1']*100:.2f}%")
    col3.metric("5-fold CV Acc",    f"{metrics['rf_cv_accuracy']*100:.2f}%")
    if metrics["xgb_available"]:
        col4.metric("XGBoost Acc",  f"{metrics['xgb']['accuracy']*100:.2f}%")
    else:
        col4.metric("XGBoost", "Not installed")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix",
                                 "Classification Report",
                                 "Model Comparison"])

    with tab1:
        cm      = metrics["rf_confusion"]
        classes = metrics["classes"]
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
               xticklabels=classes, yticklabels=classes,
               xlabel="Predicted", ylabel="Actual",
               title="Confusion Matrix — Random Forest")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        thresh = cm.max() / 2
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i,j] > thresh else "black",
                        fontsize=11)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        report      = metrics["rf_report"]
        report_rows = []
        for label, vals in report.items():
            if label in ("accuracy","macro avg","weighted avg"):
                continue
            if isinstance(vals, dict):
                report_rows.append({
                    "Mineral":   label,
                    "Precision": f"{vals['precision']:.3f}",
                    "Recall":    f"{vals['recall']:.3f}",
                    "F1-Score":  f"{vals['f1-score']:.3f}",
                    "Support":   int(vals["support"]),
                })
        st.dataframe(pd.DataFrame(report_rows), use_container_width=True)
        wa = report.get("weighted avg", {})
        st.info(
            f"Weighted avg — Precision: {wa.get('precision',0):.3f}  |  "
            f"Recall: {wa.get('recall',0):.3f}  |  "
            f"F1: {wa.get('f1-score',0):.3f}"
        )

    with tab3:
        models = ["Random Forest"]
        accs   = [metrics["rf_accuracy"]]
        f1s    = [metrics["rf_f1"]]
        if metrics["xgb_available"]:
            models.append("XGBoost")
            accs.append(metrics["xgb"]["accuracy"])
            f1s.append(metrics["xgb"]["f1"])
        x = np.arange(len(models))
        fig, ax = plt.subplots(figsize=(6, 4))
        bars1 = ax.bar(x - 0.2, [a*100 for a in accs], 0.35,
                       label="Accuracy", color="#4E9FD1")
        bars2 = ax.bar(x + 0.2, [f*100 for f in f1s],  0.35,
                       label="F1 Score", color="#F4B942")
        for bar in list(bars1) + list(bars2):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%",
                    ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(models)
        ax.set_ylim(0, 110); ax.set_ylabel("Score (%)")
        ax.set_title("Model Comparison")
        ax.legend(); ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION MAP  (fixed Gold heatmap gradient)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Prediction Map":
    st.markdown('<div class="section-header">🗺️ Mineral Prediction Map</div>',
                unsafe_allow_html=True)

    map_type = st.radio(
        "Select map type",
        ["Survey sites", "ML prediction heatmap", "State bubble map"],
        horizontal=True,
    )

    from map_renderer import (build_site_map, build_prediction_map,
                               build_state_summary_map)

    if map_type == "Survey sites":
        st.info("Actual survey sites coloured by mineral type. "
                "Click any marker for details.")
        m = build_site_map(df_filtered)

    elif map_type == "ML prediction heatmap":
        st.info("Random Forest predictions across a geospatial grid of India. "
                "Each mineral has its own colour gradient — intensity = confidence.")
        pred_df = load_prediction_map(rf_pipeline, le, feature_names)
        m = build_prediction_map(pred_df, MINERAL_GRADIENTS)

    else:
        st.info("State-level bubbles: size = number of sites, "
                "colour = dominant mineral.")
        m = build_state_summary_map(df_filtered)

    components.html(m._repr_html_(), height=580, scrolling=False)

    st.markdown("**Legend**")
    cols = st.columns(len(MINERAL_COLORS))
    for col, (mineral, color) in zip(cols, MINERAL_COLORS.items()):
        col.markdown(
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<div style="width:14px;height:14px;border-radius:50%;'
            f'background:{color};"></div><span>{mineral}</span></div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Feature Importance":
    st.markdown('<div class="section-header">📈 Feature Importance</div>',
                unsafe_allow_html=True)

    feat_df = metrics["feat_imp"]
    top_n   = st.slider("Show top N features", 10, 40, 20)
    top_df  = feat_df.head(top_n)

    fig, ax = plt.subplots(figsize=(9, top_n * 0.32 + 1))
    colors  = ["#F4B942" if any(k in f for k in
                                ["au_","cu_","fe_","li_","mn_","al_"])
               else "#4E9FD1" if f in ("latitude","longitude","elevation_m")
               else "#8FBC8F"
               for f in top_df["feature"]]
    ax.barh(top_df["feature"][::-1], top_df["importance"][::-1],
            color=colors[::-1], edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Gini importance")
    ax.set_title(f"Top {top_n} features — Random Forest")
    ax.spines[["top","right"]].set_visible(False)
    patches = [
        mpatches.Patch(color="#F4B942", label="Geochemical"),
        mpatches.Patch(color="#4E9FD1", label="Geospatial"),
        mpatches.Patch(color="#8FBC8F", label="Geological / Metadata"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    with st.expander("Full feature importance table"):
        st.dataframe(feat_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📍 State Analysis":
    st.markdown('<div class="section-header">📍 State-wise Mineral Analysis</div>',
                unsafe_allow_html=True)

    state_counts = df_filtered.groupby("state").size().reset_index(name="Sites")
    state_min    = (df_filtered.groupby("state")["mineral"]
                    .agg(lambda x: x.value_counts().idxmax())
                    .reset_index(name="Dominant mineral"))
    state_df     = state_counts.merge(state_min, on="state")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sites per state**")
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = state_counts.sort_values("Sites", ascending=False)
        ax.barh(sc["state"], sc["Sites"], color="#4E9FD1", edgecolor="white")
        ax.set_xlabel("Number of sites")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Dominant mineral per state**")
        fig, ax = plt.subplots(figsize=(6, 5))
        sc2 = state_df.sort_values("Sites", ascending=False)
        bar_colors = [MINERAL_COLORS.get(m,"#888") for m in sc2["Dominant mineral"]]
        ax.barh(sc2["state"], sc2["Sites"], color=bar_colors, edgecolor="white")
        ax.set_xlabel("Number of sites")
        patches = [mpatches.Patch(color=c, label=m)
                   for m,c in MINERAL_COLORS.items()]
        ax.legend(handles=patches, fontsize=7, loc="lower right", ncol=2)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("**Mineral heatmap — state × mineral site counts**")
    pivot = df_filtered.groupby(["state","mineral"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                linewidths=0.4, cbar_kws={"label": "Site count"},
                annot_kws={"size": 9})
    ax.set_xlabel("Mineral"); ax.set_ylabel("State")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("**Detailed state summary table**")
    full_state = (
        df_filtered.groupby("state")
          .agg(
            Total_sites=("site_name","count"),
            Dominant_mineral=("mineral", lambda x: x.value_counts().idxmax()),
            Avg_elevation=("elevation_m","mean"),
            Avg_Au_ppb=("au_ppb","mean"),
            Avg_Fe_pct=("fe_pct","mean"),
            Unique_minerals=("mineral","nunique"),
          )
          .round(2).reset_index()
          .sort_values("Total_sites", ascending=False)
    )
    st.dataframe(full_state, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT A LOCATION  (NEW)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict a Location":
    st.markdown('<div class="section-header">🔮 Predict Mineral at a Location</div>',
                unsafe_allow_html=True)

    st.markdown(
        "Enter geological survey parameters for any location in India. "
        "The Random Forest model will return the **probability of each mineral** "
        "being present at that site."
    )

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("predict_form"):
        st.markdown("#### 🌍 Geospatial Parameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            inp_lat  = st.number_input("Latitude",
                                       min_value=8.0, max_value=37.0,
                                       value=21.0, step=0.1,
                                       help="India range: 8° – 37° N")
        with c2:
            inp_lon  = st.number_input("Longitude",
                                       min_value=68.0, max_value=97.5,
                                       value=82.0, step=0.1,
                                       help="India range: 68° – 97.5° E")
        with c3:
            inp_elev = st.number_input("Elevation (m)",
                                       min_value=10, max_value=2500,
                                       value=400, step=10)

        st.markdown("#### ⚗️ Geochemical Parameters")
        c4, c5, c6 = st.columns(3)
        with c4:
            inp_au = st.number_input("Gold (Au) ppb",   min_value=0.0, value=50.0,  step=5.0)
            inp_cu = st.number_input("Copper (Cu) ppm", min_value=0.0, value=30.0,  step=5.0)
        with c5:
            inp_fe = st.number_input("Iron (Fe) %",     min_value=0.0, max_value=80.0, value=5.0,  step=0.5)
            inp_li = st.number_input("Lithium (Li) ppm",min_value=0.0, value=10.0,  step=1.0)
        with c6:
            inp_mn = st.number_input("Manganese (Mn) %",min_value=0.0, max_value=50.0, value=1.0,  step=0.1)
            inp_al = st.number_input("Aluminium (Al) %",min_value=0.0, max_value=40.0, value=3.0,  step=0.5)

        st.markdown("#### 🪨 Geological Parameters")
        c7, c8, c9 = st.columns(3)
        with c7:
            inp_ph    = st.slider("Soil pH", 4.0, 9.0, 6.5, 0.1)
            inp_fault = st.slider("Fault distance (km)", 0.1, 20.0, 4.0, 0.1)
        with c8:
            inp_dep   = st.selectbox("Deposit type",
                                     ["Placer","Vein","Skarn",
                                      "Porphyry","Sedimentary","Laterite"])
            inp_oper  = st.selectbox("Operation type",
                                     ["Surface","Underground","Unknown"])
        with c9:
            inp_com   = st.selectbox("Composition type",
                                     ["Metallic","Non-metallic","Atomic"])
            inp_dev   = st.selectbox("Development status",
                                     ["Producer","Past Producer",
                                      "Prospect","Occurrence","Plant"])
            inp_prod  = st.selectbox("Production size",
                                     ["Large","Medium","Small","Unknown"])
            inp_state = st.selectbox("State", [
                "Andhra Pradesh","Assam","Bihar","Chhattisgarh","Goa",
                "Gujarat","Himachal Pradesh","Jharkhand","Karnataka","Kerala",
                "Madhya Pradesh","Maharashtra","Manipur","Nagaland","Odisha",
                "Punjab","Rajasthan","Tamil Nadu","Uttar Pradesh","West Bengal",
            ])

        submitted = st.form_submit_button("🔍 Predict Mineral Probabilities",
                                          use_container_width=True)

    # ── Prediction ────────────────────────────────────────────────────────────
    if submitted:
        from data_generator import generate_dataset, encode_features

        # Build a 1-row dataframe matching the training schema
        ref_df = generate_dataset(200)
        _, _, ref_feat_names, _ = encode_features(ref_df)

        ref_enc = pd.get_dummies(
            ref_df[["dep_type","oper_type","com_type","dev_stat",
                    "prod_size","state",
                    "latitude","longitude","elevation_m",
                    "au_ppb","cu_ppm","fe_pct","li_ppm",
                    "mn_pct","al_pct","soil_ph","fault_dist_km"]],
            columns=["dep_type","oper_type","com_type","dev_stat","prod_size","state"]
        )
        row = ref_enc.median().copy()

        # Fill in user values
        row["latitude"]     = inp_lat
        row["longitude"]    = inp_lon
        row["elevation_m"]  = inp_elev
        row["au_ppb"]       = inp_au
        row["cu_ppm"]       = inp_cu
        row["fe_pct"]       = inp_fe
        row["li_ppm"]       = inp_li
        row["mn_pct"]       = inp_mn
        row["al_pct"]       = inp_al
        row["soil_ph"]      = inp_ph
        row["fault_dist_km"]= inp_fault

        # One-hot: set correct category flags
        for cat, val, prefix in [
            ("dep_type",  inp_dep,   "dep_type"),
            ("oper_type", inp_oper,  "oper_type"),
            ("com_type",  inp_com,   "com_type"),
            ("dev_stat",  inp_dev,   "dev_stat"),
            ("prod_size", inp_prod,  "prod_size"),
            ("state",     inp_state, "state"),
        ]:
            for col in row.index:
                if col.startswith(prefix + "_"):
                    row[col] = 0.0
            key = f"{prefix}_{val}"
            if key in row.index:
                row[key] = 1.0

        # Align to training features
        input_df = pd.DataFrame([row])
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0.0
        input_df = input_df[feature_names]

        proba   = rf_pipeline.predict_proba(input_df)[0]
        classes = le.classes_
        proba_dict = dict(zip(classes, proba))

        top_mineral = max(proba_dict, key=proba_dict.get)
        top_color   = MINERAL_COLORS.get(top_mineral, "#888")
        top_pct     = round(proba_dict[top_mineral] * 100, 1)

        # ── Result display ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown(
                f'<div class="predict-card" style="border-color:{top_color};">'
                f'<div style="font-size:2.5rem;text-align:center;">⛏️</div>'
                f'<div style="text-align:center;font-size:1.3rem;font-weight:700;'
                f'color:{top_color};">{top_mineral}</div>'
                f'<div style="text-align:center;font-size:2rem;font-weight:800;">'
                f'{top_pct}%</div>'
                f'<div style="text-align:center;font-size:0.8rem;color:#666;">'
                f'Top predicted mineral</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Location:** {inp_lat:.2f}°N, {inp_lon:.2f}°E")
            st.markdown(f"**Elevation:** {inp_elev} m")
            st.markdown(f"**Deposit:** {inp_dep}")
            st.markdown(f"**State:** {inp_state}")

        with res_col2:
            st.markdown("#### Probability breakdown — all minerals")
            render_probability_bars(proba_dict)

        # Mini bar chart
        st.markdown("---")
        st.markdown("#### Visual comparison")
        sorted_minerals = sorted(proba_dict, key=proba_dict.get, reverse=True)
        sorted_probs    = [proba_dict[m]*100 for m in sorted_minerals]
        bar_colors      = [MINERAL_COLORS.get(m,"#888") for m in sorted_minerals]

        fig, ax = plt.subplots(figsize=(8, 3))
        bars = ax.bar(sorted_minerals, sorted_probs, color=bar_colors,
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, sorted_probs):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=10)
        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, max(sorted_probs) * 1.2)
        ax.set_title("Mineral occurrence probability at entered location")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Interpretation note
        st.info(
            f"**Interpretation:** Based on the geochemical and geological parameters "
            f"entered, this location shows the strongest signature for **{top_mineral}** "
            f"({top_pct}% confidence). "
            f"The model considers {len(feature_names)} features learned from "
            f"2,000 survey points across India."
        )
