"""
model.py
--------
Trains a Random Forest (primary) and XGBoost (comparison) classifier
on the generated geological survey dataset.

Exposes:
  train_models()  → returns fitted pipeline + metrics dict
  predict_map()   → predict mineral probability for a grid of India coords
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from data_generator import generate_dataset, encode_features, MINERALS

# ── Try importing XGBoost; fall back gracefully ───────────────────────────────
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def train_models(n_samples: int = 1200, test_size: float = 0.2):
    """
    Generate data, train RF + XGBoost, return:
      df          – raw dataframe
      rf_pipeline – fitted sklearn Pipeline (scaler + RF)
      xgb_pipeline– fitted XGBoost pipeline (or None)
      metrics     – dict with all evaluation numbers
      feature_names, label_encoder
    """
    df = generate_dataset(n_samples)
    X, y, feature_names, le = encode_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_pred  = rf_pipeline.predict(X_test)
    rf_proba = rf_pipeline.predict_proba(X_test)

    rf_acc   = accuracy_score(y_test, rf_pred)
    rf_f1    = f1_score(y_test, rf_pred, average="weighted")
    rf_cv    = cross_val_score(rf_pipeline, X_train, y_train, cv=5,
                               scoring="accuracy").mean()
    rf_cm    = confusion_matrix(y_test, rf_pred)
    rf_report= classification_report(y_test, rf_pred,
                                     target_names=le.classes_, output_dict=True)

    # Feature importances (from the RF inside the pipeline)
    importances = rf_pipeline.named_steps["clf"].feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_pipeline = None
    xgb_metrics  = {}
    if XGBOOST_AVAILABLE:
        xgb_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42,
            ))
        ])
        xgb_pipeline.fit(X_train, y_train)
        xgb_pred = xgb_pipeline.predict(X_test)
        xgb_metrics = {
            "accuracy": accuracy_score(y_test, xgb_pred),
            "f1":       f1_score(y_test, xgb_pred, average="weighted"),
        }

    metrics = {
        "rf_accuracy":     rf_acc,
        "rf_f1":           rf_f1,
        "rf_cv_accuracy":  rf_cv,
        "rf_confusion":    rf_cm,
        "rf_report":       rf_report,
        "feat_imp":        feat_imp_df,
        "xgb":             xgb_metrics,
        "xgb_available":   XGBOOST_AVAILABLE,
        "classes":         le.classes_,
        "y_test":          y_test,
        "rf_pred":         rf_pred,
        "rf_proba":        rf_proba,
    }

    return df, rf_pipeline, xgb_pipeline, metrics, feature_names, le


def predict_map(rf_pipeline, le, feature_names,
                grid_lat_steps: int = 35,
                grid_lon_steps: int = 35):
    """
    Predict the most-likely mineral and its probability for a grid of
    points covering India, returning a GeoDataFrame-friendly DataFrame.

    Uses median values for all non-spatial features so the map shows
    pure geospatial variation in predictions.
    """
    from data_generator import (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
                                 generate_dataset, encode_features)

    # Fit a reference dataset to know column order & fill medians
    ref_df = generate_dataset(600)
    _, _, ref_feat_names, _ = encode_features(ref_df)

    # One-hot encoded reference — compute column medians
    ref_enc = pd.get_dummies(
        ref_df[["dep_type","oper_type","com_type","dev_stat",
                "prod_size","state",
                "latitude","longitude","elevation_m",
                "au_ppb","cu_ppm","fe_pct","li_ppm",
                "mn_pct","al_pct","soil_ph","fault_dist_km"]],
        columns=["dep_type","oper_type","com_type",
                 "dev_stat","prod_size","state"]
    )
    col_medians = ref_enc.median()

    # Build grid
    lats = np.linspace(LAT_MIN + 0.5, LAT_MAX - 0.5, grid_lat_steps)
    lons = np.linspace(LON_MIN + 0.5, LON_MAX - 0.5, grid_lon_steps)
    grid_lats, grid_lons = np.meshgrid(lats, lons)
    grid_lats = grid_lats.ravel()
    grid_lons = grid_lons.ravel()

    # Build a feature matrix for grid points
    rows = []
    for lat, lon in zip(grid_lats, grid_lons):
        row = col_medians.copy()
        row["latitude"]  = lat
        row["longitude"] = lon
        rows.append(row)

    grid_df = pd.DataFrame(rows)

    # Align columns to training feature names
    for col in feature_names:
        if col not in grid_df.columns:
            grid_df[col] = 0
    grid_df = grid_df[feature_names]

    X_grid = grid_df.values.astype(float)
    proba  = rf_pipeline.predict_proba(X_grid)        # (n_points, n_classes)
    pred_idx = np.argmax(proba, axis=1)
    pred_mineral = le.inverse_transform(pred_idx)
    pred_prob    = proba[np.arange(len(proba)), pred_idx]

    result = pd.DataFrame({
        "latitude":         grid_lats,
        "longitude":        grid_lons,
        "predicted_mineral": pred_mineral,
        "confidence":       np.round(pred_prob * 100, 1),
    })
    # Add per-class probability columns
    for i, cls in enumerate(le.classes_):
        result[f"prob_{cls}"] = np.round(proba[:, i] * 100, 1)

    return result


if __name__ == "__main__":
    df, rf, xgb, metrics, feat_names, le = train_models()
    print(f"RF  accuracy : {metrics['rf_accuracy']:.3f}")
    print(f"RF  F1       : {metrics['rf_f1']:.3f}")
    print(f"RF  CV acc   : {metrics['rf_cv_accuracy']:.3f}")
    if metrics["xgb_available"]:
        print(f"XGB accuracy : {metrics['xgb']['accuracy']:.3f}")
    print("\nTop-10 features:")
    print(metrics["feat_imp"].head(10))
