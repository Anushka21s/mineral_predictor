"""
data_generator.py
-----------------
Generates a realistic synthetic dataset modelled on the
'Mineral ores round the world' Kaggle dataset (India subset).

Real columns used as inspiration:
  site_name, state, latitude, longitude,
  commod1/2/3, dep_type, com_type, oper_type,
  dev_stat, prod_size

We engineer additional geochemical / geological features that a real
survey would contain, giving the ML model meaningful signals to learn.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)

# ── India bounding box (approximate) ─────────────────────────────────────────
LAT_MIN, LAT_MAX = 8.0,  37.0
LON_MIN, LON_MAX = 68.0, 97.5

# ── Target classes ────────────────────────────────────────────────────────────
MINERALS = ["Gold", "Copper", "Iron", "Lithium", "Manganese", "Bauxite"]

# ── Indian states with rough centroids ───────────────────────────────────────
STATE_INFO = {
    "Odisha":           (20.5, 84.9),
    "Jharkhand":        (23.6, 85.3),
    "Chhattisgarh":     (21.3, 81.6),
    "Rajasthan":        (27.0, 74.2),
    "Karnataka":        (15.3, 75.7),
    "Andhra Pradesh":   (15.9, 79.7),
    "Maharashtra":      (19.7, 75.7),
    "Madhya Pradesh":   (22.9, 78.6),
    "Goa":              (15.3, 74.0),
    "Tamil Nadu":       (11.1, 78.6),
    "West Bengal":      (22.5, 87.6),
    "Assam":            (26.2, 92.9),
    "Gujarat":          (22.3, 71.2),
    "Uttar Pradesh":    (26.8, 80.9),
    "Bihar":            (25.1, 85.3),
}

# ── Deposit types / operation types ──────────────────────────────────────────
DEP_TYPES  = ["Placer", "Vein", "Skarn", "Porphyry", "Sedimentary", "Laterite"]
OPER_TYPES = ["Surface", "Underground", "Unknown"]
COM_TYPES  = ["Metallic", "Non-metallic", "Atomic"]
DEV_STATS  = ["Producer", "Past Producer", "Prospect", "Occurrence", "Plant"]
PROD_SIZES = ["Large", "Medium", "Small", "Unknown"]

# ── Mineral ↔ feature affinity (domain knowledge priors) ─────────────────────
# Each mineral has a characteristic geochemical + geological signature.
MINERAL_PRIORS = {
    # mineral   : (elev_mean, elev_std,
    #              au_ppb_mean, cu_ppm_mean, fe_pct_mean,
    #              li_ppm_mean, mn_pct_mean, al_pct_mean,
    #              soil_ph_mean, faults_km_mean,
    #              preferred_dep_types,
    #              preferred_states)
    "Gold":      (650, 200, 1800, 30,  4,  10, 0.2, 2,  6.5, 3,
                  ["Vein", "Placer"],
                  ["Karnataka", "Andhra Pradesh", "Rajasthan"]),
    "Copper":    (500, 150, 20,   800, 6,  15, 0.3, 3,  6.8, 4,
                  ["Porphyry", "Skarn"],
                  ["Rajasthan", "Jharkhand", "Madhya Pradesh"]),
    "Iron":      (300, 120, 5,    25,  58, 8,  0.5, 4,  6.2, 5,
                  ["Sedimentary"],
                  ["Odisha", "Jharkhand", "Chhattisgarh", "Goa"]),
    "Lithium":   (900, 250, 8,    20,  3,  420, 0.1, 5, 7.2, 2,
                  ["Vein", "Placer"],
                  ["Rajasthan", "Karnataka", "Andhra Pradesh"]),
    "Manganese": (200, 100, 3,    15,  8,  5,  28,  3,  6.0, 6,
                  ["Sedimentary", "Laterite"],
                  ["Odisha", "Maharashtra", "Madhya Pradesh"]),
    "Bauxite":   (400, 130, 4,    10,  5,  8,  0.4, 32, 5.8, 3,
                  ["Laterite"],
                  ["Odisha", "Jharkhand", "Chhattisgarh", "Gujarat"]),
}


def _sample_state(mineral: str) -> str:
    """Return a state biased towards the mineral's preferred geology."""
    preferred = MINERAL_PRIORS[mineral][11]
    all_states = list(STATE_INFO.keys())
    weights = np.array([5.0 if s in preferred else 1.0 for s in all_states])
    weights /= weights.sum()
    return rng.choice(all_states, p=weights)


def _jitter(center: float, spread: float = 2.0) -> float:
    return float(center + rng.normal(0, spread))


def generate_dataset(n_samples: int = 1200) -> pd.DataFrame:
    """
    Generate a synthetic India mineral survey dataset.

    Each row represents one geological survey site with:
    - Geospatial attributes  (lat/lon, elevation, state)
    - Geochemical attributes (Au/Cu/Fe/Li/Mn/Al concentrations)
    - Geological attributes  (deposit type, fault proximity, soil pH)
    - Operational metadata   (oper_type, dev_stat, prod_size)
    - Target label           (primary mineral commodity)
    """
    rows = []
    mineral_counts = {m: 0 for m in MINERALS}

    for i in range(n_samples):
        mineral = rng.choice(MINERALS)
        mineral_counts[mineral] += 1

        p = MINERAL_PRIORS[mineral]
        (elev_mu, elev_sd,
         au_mu, cu_mu, fe_mu,
         li_mu, mn_mu, al_mu,
         ph_mu, fault_mu,
         dep_preferred, _) = p

        state = _sample_state(mineral)
        state_lat, state_lon = STATE_INFO[state]

        # Geospatial
        lat = np.clip(_jitter(state_lat, 1.8), LAT_MIN, LAT_MAX)
        lon = np.clip(_jitter(state_lon, 1.8), LON_MIN, LON_MAX)
        elevation = float(np.clip(rng.normal(elev_mu, elev_sd), 10, 2500))

        # Geochemical (add noise; ensure non-negative)
        au_ppb   = float(np.clip(rng.normal(au_mu, au_mu * 0.5),   0, None))
        cu_ppm   = float(np.clip(rng.normal(cu_mu, cu_mu * 0.5),   0, None))
        fe_pct   = float(np.clip(rng.normal(fe_mu, fe_mu * 0.3),   0, 80))
        li_ppm   = float(np.clip(rng.normal(li_mu, li_mu * 0.5),   0, None))
        mn_pct   = float(np.clip(rng.normal(mn_mu, mn_mu * 0.4),   0, 50))
        al_pct   = float(np.clip(rng.normal(al_mu, al_mu * 0.35),  0, 40))
        soil_ph  = float(np.clip(rng.normal(ph_mu, 0.6),           4.0, 9.0))
        faults_km= float(np.clip(rng.normal(fault_mu, 1.5),        0.1, 20))

        # Deposit type (biased)
        all_deps = DEP_TYPES
        dep_weights = np.array([4.0 if d in dep_preferred else 1.0 for d in all_deps])
        dep_weights /= dep_weights.sum()
        dep_type = rng.choice(all_deps, p=dep_weights)

        # Metadata
        oper_type = rng.choice(OPER_TYPES, p=[0.45, 0.45, 0.10])
        com_type  = rng.choice(COM_TYPES,  p=[0.70, 0.20, 0.10])
        dev_stat  = rng.choice(DEV_STATS,  p=[0.20, 0.15, 0.40, 0.20, 0.05])
        prod_size = rng.choice(PROD_SIZES, p=[0.15, 0.30, 0.40, 0.15])
        site_name = f"Site_{i+1:04d}"

        rows.append({
            "site_name":   site_name,
            "state":       state,
            "latitude":    round(lat, 5),
            "longitude":   round(lon, 5),
            "elevation_m": round(elevation, 1),
            "au_ppb":      round(au_ppb, 2),       # gold concentration
            "cu_ppm":      round(cu_ppm, 2),       # copper concentration
            "fe_pct":      round(fe_pct, 2),       # iron %
            "li_ppm":      round(li_ppm, 2),       # lithium concentration
            "mn_pct":      round(mn_pct, 2),       # manganese %
            "al_pct":      round(al_pct, 2),       # aluminium % (bauxite)
            "soil_ph":     round(soil_ph, 2),
            "fault_dist_km": round(faults_km, 2),
            "dep_type":    dep_type,
            "oper_type":   oper_type,
            "com_type":    com_type,
            "dev_stat":    dev_stat,
            "prod_size":   prod_size,
            "mineral":     mineral,                # ← target
        })

    df = pd.DataFrame(rows)
    return df


def encode_features(df: pd.DataFrame):
    """
    One-hot encode categorical columns and return
    (X, y, feature_names, label_encoder).
    """
    cat_cols = ["dep_type", "oper_type", "com_type", "dev_stat",
                "prod_size", "state"]
    num_cols = ["latitude", "longitude", "elevation_m",
                "au_ppb", "cu_ppm", "fe_pct", "li_ppm",
                "mn_pct", "al_pct", "soil_ph", "fault_dist_km"]

    df_enc = pd.get_dummies(df[cat_cols + num_cols], columns=cat_cols)
    X = df_enc.values.astype(float)
    feature_names = list(df_enc.columns)

    le = LabelEncoder()
    y  = le.fit_transform(df["mineral"])

    return X, y, feature_names, le


if __name__ == "__main__":
    df = generate_dataset(1200)
    print(df.head())
    print("\nMineral distribution:\n", df["mineral"].value_counts())
    print("\nShape:", df.shape)
