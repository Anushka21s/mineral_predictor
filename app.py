"""
app.py  –  Geospatial Mineral Prediction Dashboard
----------------------------------------------------
Run with:
    streamlit run app.py

Sections:
  0. Sidebar controls
  1. Home / Overview
  2. Dataset Explorer
  3. ML Model Training & Metrics
  4. Prediction Map
  5. Feature Importance
  6. State-wise Analysis
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

# ── Colour palette (match map_renderer) ───────────────────────────────────────
MINERAL_COLORS = {
    "Gold":      "#F4B942",
    "Copper":    "#C45E2A",
    "Iron":      "#8B3A3A",
    "Lithium":   "#4E9FD1",
    "Manganese": "#6A5ACD",
    "Bauxite":   "#8FBC8F",
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #1a1a2e; }
  [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
  .metric-card {
    background: #f8f9fa; border-radius: 10px;
    padding: 16px; text-align: center;
    border-left: 4px solid #4E9FD1;
  }
  .metric-value { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
  .metric-label { font-size: 0.85rem; color: #666; margin-top: 4px; }
  .section-header {
    font-size: 1.5rem; font-weight: 700;
    color: #1a1a2e; padding: 8px 0 4px 0;
    border-bottom: 2px solid #4E9FD1; margin-bottom: 16px;
  }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔬 Training ML model…")
def load_model(n_samples):
    from model import train_models
    return train_models(n_samples)


@st.cache_data(show_spinner="🗺️ Generating prediction grid…")
def load_prediction_map(_rf_pipeline, _le, feat_names):
    from model import predict_map
    return predict_map(_rf_pipeline, _le, feat_names)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⛏️ Mineral Predictor")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Dataset Explorer",
         "🤖 Model & Metrics", "🗺️ Prediction Map",
         "📈 Feature Importance", "📍 State Analysis"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Settings**")
    n_samples = st.slider("Dataset size", 600, 2000, 1200, 100)
    sel_minerals = st.multiselect(
        "Filter minerals",
        ["Gold", "Copper", "Iron", "Lithium", "Manganese", "Bauxite"],
        default=["Gold", "Copper", "Iron", "Lithium", "Manganese", "Bauxite"],
    )

    st.markdown("---")
    st.markdown(
        "<small>Built with Streamlit · scikit-learn · Folium · GeoPandas</small>",
        unsafe_allow_html=True,
    )

# ── Load model ────────────────────────────────────────────────────────────────
df, rf_pipeline, xgb_pipeline, metrics, feature_names, le = load_model(n_samples)

# Apply mineral filter
if sel_minerals:
    df = df[df["mineral"].isin(sel_minerals)]


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
            st.metric("Survey Sites",   f"{len(df):,}")
            st.metric("Features Used",  len(feature_names))
        with m2:
            st.metric("Mineral Classes", len(df["mineral"].unique()))
            st.metric("Indian States",  df["state"].nunique())

        st.markdown("#### Model performance")
        st.metric("RF Accuracy",  f"{metrics['rf_accuracy']*100:.1f}%")
        st.metric("RF F1 Score",  f"{metrics['rf_f1']*100:.1f}%")
        st.metric("5-fold CV Acc", f"{metrics['rf_cv_accuracy']*100:.1f}%")

    st.markdown("---")
    st.markdown("#### Mineral distribution across sites")
    counts = df["mineral"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 3))
    bars = ax.bar(counts.index, counts.values,
                  color=[MINERAL_COLORS.get(m, "#888") for m in counts.index],
                  edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Site count")
    ax.set_xlabel("Mineral")
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
            df.columns.tolist(),
            default=["site_name","state","mineral","latitude","longitude",
                     "elevation_m","au_ppb","cu_ppm","fe_pct","dep_type",
                     "dev_stat"],
        )
        st.dataframe(df[col_filter].head(200), use_container_width=True,
                     height=400)
        st.caption(f"Showing first 200 of {len(df)} rows")

    with tab2:
        st.markdown("#### Geochemical signature by mineral")
        geochem_cols = ["au_ppb","cu_ppm","fe_pct","li_ppm","mn_pct","al_pct"]
        selected_elem = st.selectbox("Element", geochem_cols)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Boxplot
        data_by_mineral = [
            df[df["mineral"] == m][selected_elem].values
            for m in sel_minerals if m in df["mineral"].unique()
        ]
        bp = axes[0].boxplot(
            data_by_mineral,
            labels=[m for m in sel_minerals if m in df["mineral"].unique()],
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
        )
        for patch, mineral in zip(bp["boxes"],
                                   [m for m in sel_minerals
                                    if m in df["mineral"].unique()]):
            patch.set_facecolor(MINERAL_COLORS.get(mineral, "#888"))
        axes[0].set_title(f"{selected_elem} distribution by mineral")
        axes[0].set_ylabel(selected_elem)
        axes[0].tick_params(axis="x", rotation=15)
        axes[0].spines[["top","right"]].set_visible(False)

        # Scatter: lat/lon coloured by element intensity
        sc = axes[1].scatter(
            df["longitude"], df["latitude"],
            c=df[selected_elem], cmap="YlOrRd", s=10, alpha=0.6,
        )
        plt.colorbar(sc, ax=axes[1], label=selected_elem)
        axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
        axes[1].set_title(f"{selected_elem} spatial distribution (India)")
        axes[1].spines[["top","right"]].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Deposit type distribution**")
            dep_counts = df["dep_type"].value_counts()
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.barh(dep_counts.index, dep_counts.values,
                    color="#4E9FD1", edgecolor="white")
            ax.set_xlabel("Count")
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("**Development status breakdown**")
            dev_counts = df["dev_stat"].value_counts()
            colors_pie = ["#F4B942","#4E9FD1","#8B3A3A","#6A5ACD","#8FBC8F"]
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.pie(dev_counts.values, labels=dev_counts.index,
                   autopct="%1.1f%%", colors=colors_pie[:len(dev_counts)],
                   startangle=90, wedgeprops=dict(linewidth=0.5, edgecolor="white"))
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("**Correlation heatmap — geochemical + spatial features**")
        num_cols = ["latitude","longitude","elevation_m",
                    "au_ppb","cu_ppm","fe_pct","li_ppm","mn_pct","al_pct",
                    "soil_ph","fault_dist_km"]
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8})
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL & METRICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model & Metrics":
    st.markdown('<div class="section-header">🤖 Model Training & Evaluation</div>',
                unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RF Accuracy",    f"{metrics['rf_accuracy']*100:.2f}%")
    col2.metric("RF F1 (weighted)", f"{metrics['rf_f1']*100:.2f}%")
    col3.metric("5-fold CV Acc",  f"{metrics['rf_cv_accuracy']*100:.2f}%")
    if metrics["xgb_available"]:
        col4.metric("XGBoost Acc", f"{metrics['xgb']['accuracy']*100:.2f}%")
    else:
        col4.metric("XGBoost", "Not installed")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Confusion Matrix",
                                 "Classification Report",
                                 "Model Comparison"])

    with tab1:
        cm = metrics["rf_confusion"]
        classes = metrics["classes"]
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set(
            xticks=range(len(classes)), yticks=range(len(classes)),
            xticklabels=classes, yticklabels=classes,
            xlabel="Predicted", ylabel="Actual",
            title="Confusion Matrix — Random Forest",
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        thresh = cm.max() / 2
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        report = metrics["rf_report"]
        report_rows = []
        for label, vals in report.items():
            if label in ("accuracy", "macro avg", "weighted avg"):
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

        # Weighted avg
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
        bars2 = ax.bar(x + 0.2, [f*100 for f in f1s], 0.35,
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
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION MAP
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
                "Click any marker for details. Use layer control to toggle minerals.")
        m = build_site_map(df)

    elif map_type == "ML prediction heatmap":
        st.info("Random Forest prediction across a geospatial grid of India. "
                "Heat intensity = model confidence. Toggle layers to compare minerals.")
        pred_df = load_prediction_map(rf_pipeline, le, feature_names)
        m = build_prediction_map(pred_df)

    else:
        st.info("State-level bubbles: size = number of sites, "
                "colour = dominant mineral.")
        m = build_state_summary_map(df)

    # Render map
    map_html = m._repr_html_()
    components.html(map_html, height=580, scrolling=False)

    # Legend
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
    colors  = ["#F4B942" if "au_" in f or "cu_" in f or "fe_" in f
                          or "li_" in f or "mn_" in f or "al_" in f
               else "#4E9FD1" if f in ("latitude","longitude","elevation_m")
               else "#8FBC8F"
               for f in top_df["feature"]]
    bars = ax.barh(top_df["feature"][::-1],
                   top_df["importance"][::-1],
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
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    with st.expander("Full feature importance table"):
        st.dataframe(feat_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📍 State Analysis":
    st.markdown('<div class="section-header">📍 State-wise Mineral Analysis</div>',
                unsafe_allow_html=True)

    # Sites per state
    state_counts = df.groupby("state").size().reset_index(name="Sites")
    state_min    = (df.groupby("state")["mineral"]
                    .agg(lambda x: x.value_counts().idxmax())
                    .reset_index(name="Dominant mineral"))
    state_df     = state_counts.merge(state_min, on="state")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Sites per state**")
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = state_counts.sort_values("Sites", ascending=False)
        ax.barh(sc["state"], sc["Sites"],
                color="#4E9FD1", edgecolor="white")
        ax.set_xlabel("Number of sites")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("**Dominant mineral per state**")
        fig, ax = plt.subplots(figsize=(6, 5))
        sc2 = state_df.sort_values("Sites", ascending=False)
        bar_colors = [MINERAL_COLORS.get(m, "#888")
                      for m in sc2["Dominant mineral"]]
        ax.barh(sc2["state"], sc2["Sites"],
                color=bar_colors, edgecolor="white")
        ax.set_xlabel("Number of sites")
        patches = [mpatches.Patch(color=c, label=m)
                   for m, c in MINERAL_COLORS.items()]
        ax.legend(handles=patches, fontsize=7,
                  loc="lower right", ncol=2)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("**Mineral heatmap — state × mineral site counts**")
    pivot = (df.groupby(["state","mineral"])
               .size().unstack(fill_value=0))
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd",
                ax=ax, linewidths=0.4, cbar_kws={"label": "Site count"},
                annot_kws={"size": 9})
    ax.set_xlabel("Mineral"); ax.set_ylabel("State")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("**Detailed state summary table**")
    full_state = (
        df.groupby("state")
          .agg(
            Total_sites=("site_name","count"),
            Dominant_mineral=("mineral", lambda x: x.value_counts().idxmax()),
            Avg_elevation=("elevation_m","mean"),
            Avg_Au_ppb=("au_ppb","mean"),
            Avg_Fe_pct=("fe_pct","mean"),
            Unique_minerals=("mineral","nunique"),
          )
          .round(2)
          .reset_index()
          .sort_values("Total_sites", ascending=False)
    )
    st.dataframe(full_state, use_container_width=True)
