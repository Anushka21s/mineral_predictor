"""
data_generator.py  v3
---------------------
Geologically realistic synthetic dataset for India mineral prediction.

Key fix: each mineral now has TIGHTLY BOUNDED, non-overlapping geochemical
signatures so the ML model learns genuinely distinct spatial patterns.

Real geological facts encoded:
  Iron     → Odisha/Jharkhand/Chhattisgarh belt, very high Fe%, low Au/Cu
  Gold     → Karnataka/Andhra schist belts, very high Au ppb, Vein deposits
  Copper   → Rajasthan/Jharkhand, high Cu ppm, Porphyry/Skarn deposits
  Manganese→ Odisha/Maharashtra, high Mn%, coastal/plateau low elevation
  Bauxite  → Odisha/Jharkhand laterite plateaus, very high Al%, Laterite dep
  Lithium  → Rajasthan pegmatites, very high Li ppm, high elevation, acidic
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

SEED = 42
rng  = np.random.default_rng(SEED)

LAT_MIN, LAT_MAX = 8.0,  37.0
LON_MIN, LON_MAX = 68.0, 97.5

MINERALS = ["Gold", "Copper", "Iron", "Lithium", "Manganese", "Bauxite"]

# ── State centroids ───────────────────────────────────────────────────────────
STATE_INFO = {
    "Odisha":          (20.5, 84.9),
    "Jharkhand":       (23.6, 85.3),
    "Chhattisgarh":    (21.3, 81.6),
    "Rajasthan":       (27.0, 74.2),
    "Karnataka":       (15.3, 75.7),
    "Andhra Pradesh":  (15.9, 79.7),
    "Maharashtra":     (19.7, 75.7),
    "Madhya Pradesh":  (22.9, 78.6),
    "Goa":             (15.3, 74.0),
    "Tamil Nadu":      (11.1, 78.6),
    "West Bengal":     (22.5, 87.6),
    "Assam":           (26.2, 92.9),
    "Gujarat":         (22.3, 71.2),
    "Uttar Pradesh":   (26.8, 80.9),
    "Bihar":           (25.1, 85.3),
}

DEP_TYPES  = ["Placer", "Vein", "Skarn", "Porphyry", "Sedimentary", "Laterite"]
OPER_TYPES = ["Surface", "Underground", "Unknown"]
COM_TYPES  = ["Metallic", "Non-metallic", "Atomic"]
DEV_STATS  = ["Producer", "Past Producer", "Prospect", "Occurrence", "Plant"]
PROD_SIZES = ["Large", "Medium", "Small", "Unknown"]

# ── Tightly separated mineral signatures ─────────────────────────────────────
# Format: (elev_mu, elev_sd,
#           au_ppb_mu, au_ppb_sd,     ← Gold tracer
#           cu_ppm_mu, cu_ppm_sd,     ← Copper tracer
#           fe_pct_mu, fe_pct_sd,     ← Iron tracer
#           li_ppm_mu, li_ppm_sd,     ← Lithium tracer
#           mn_pct_mu, mn_pct_sd,     ← Manganese tracer
#           al_pct_mu, al_pct_sd,     ← Bauxite/Aluminium tracer
#           soil_ph_mu, fault_km_mu,
#           preferred_dep_types, preferred_states)
MINERAL_PRIORS = {
    # Iron: very high Fe%, virtually no Au/Li/Mn/Al signal
    # Dominates Odisha-Jharkhand-Chhattisgarh-Goa belt
    "Iron": (
        280, 80,          # elevation: low plateau
        2,   1,           # au_ppb: near zero
        20,  8,           # cu_ppm: low
        62,  4,           # fe_pct: VERY HIGH (>55% is iron ore grade)
        5,   2,           # li_ppm: negligible
        0.3, 0.1,         # mn_pct: trace
        2,   0.8,         # al_pct: low
        6.1, 5.5,         # soil_ph, fault_km
        ["Sedimentary"],
        ["Odisha", "Jharkhand", "Chhattisgarh", "Goa"],
    ),
    # Gold: extremely high Au ppb, Vein/Placer deposit, high elevation
    # Kolar/Karnataka schist belts, Andhra Pradesh, Rajasthan
    "Gold": (
        720, 180,         # elevation: higher terrain
        2200, 300,        # au_ppb: VERY HIGH signature
        25,  10,          # cu_ppm: low
        4,   1.5,         # fe_pct: low
        8,   3,           # li_ppm: low
        0.15, 0.05,       # mn_pct: near zero
        1.5, 0.5,         # al_pct: low
        6.5, 2.8,         # soil_ph, fault_km: near faults
        ["Vein", "Placer"],
        ["Karnataka", "Andhra Pradesh", "Rajasthan"],
    ),
    # Copper: high Cu ppm, Porphyry/Skarn, medium elevation
    # Rajasthan (Khetri), Jharkhand (Singhbhum), MP
    "Copper": (
        480, 120,         # elevation: mid-level
        15,  5,           # au_ppb: low
        850, 120,         # cu_ppm: VERY HIGH
        6,   2,           # fe_pct: low-moderate
        12,  4,           # li_ppm: trace
        0.25, 0.08,       # mn_pct: trace
        2.5, 0.8,         # al_pct: low
        6.8, 4.2,         # soil_ph, fault_km
        ["Porphyry", "Skarn"],
        ["Rajasthan", "Jharkhand", "Madhya Pradesh"],
    ),
    # Manganese: high Mn%, low elevation coastal/plateau
    # Odisha (Koraput), Maharashtra (Nagpur), MP
    "Manganese": (
        190, 70,          # elevation: low lying
        3,   1,           # au_ppb: near zero
        12,  4,           # cu_ppm: low
        7,   2,           # fe_pct: low-moderate
        4,   1.5,         # li_ppm: trace
        32,  4,           # mn_pct: VERY HIGH
        2,   0.6,         # al_pct: low
        5.9, 6.2,         # soil_ph: slightly acidic, fault_km
        ["Sedimentary", "Laterite"],
        ["Odisha", "Maharashtra", "Madhya Pradesh"],
    ),
    # Bauxite: very high Al%, Laterite deposits, plateau surface
    # Odisha (Koraput/Kalahandi), Jharkhand, Chhattisgarh, Gujarat
    "Bauxite": (
        420, 100,         # elevation: laterite plateau
        3,   1,           # au_ppb: near zero
        8,   3,           # cu_ppm: low
        5,   1.5,         # fe_pct: low (Al replaces Fe)
        6,   2,           # li_ppm: trace
        0.3, 0.1,         # mn_pct: trace
        34,  3,           # al_pct: VERY HIGH (>28% = bauxite ore)
        5.7, 3.5,         # soil_ph: acidic laterite, fault_km
        ["Laterite"],
        ["Odisha", "Jharkhand", "Chhattisgarh", "Gujarat"],
    ),
    # Lithium: very high Li ppm, pegmatite/Vein, high elevation, acidic
    # Rajasthan (Degana), Karnataka (Marlagalla), Andhra
    "Lithium": (
        950, 200,         # elevation: high terrain pegmatites
        6,   2,           # au_ppb: trace
        18,  6,           # cu_ppm: trace
        3,   1,           # fe_pct: very low
        450, 60,          # li_ppm: VERY HIGH
        0.08, 0.03,       # mn_pct: near zero
        4,   1.2,         # al_pct: moderate (feldspar association)
        7.3, 2.2,         # soil_ph: slightly alkaline, fault_km
        ["Vein", "Placer"],
        ["Rajasthan", "Karnataka", "Andhra Pradesh"],
    ),
}

# State → dominant mineral mapping for geospatially realistic prediction
STATE_DOMINANT = {
    "Odisha":          "Iron",
    "Jharkhand":       "Iron",
    "Chhattisgarh":    "Iron",
    "Goa":             "Iron",
    "Karnataka":       "Gold",
    "Andhra Pradesh":  "Gold",
    "Rajasthan":       "Copper",
    "Madhya Pradesh":  "Manganese",
    "Maharashtra":     "Manganese",
    "Gujarat":         "Bauxite",
    "Bihar":           "Copper",
    "Tamil Nadu":      "Bauxite",
    "West Bengal":     "Iron",
    "Assam":           "Lithium",
    "Uttar Pradesh":   "Lithium",
}


def _sample_mineral_for_state(state: str) -> str:
    """
    Return a mineral biased strongly towards the geological reality
    of that Indian state.
    """
    dominant = STATE_DOMINANT.get(state, "Iron")
    all_minerals = MINERALS
    # 65% chance of dominant mineral, 35% split among others
    weights = np.array([
        6.5 if m == dominant else 0.7
        for m in all_minerals
    ])
    weights /= weights.sum()
    return rng.choice(all_minerals, p=weights)


def _sample_state(mineral: str) -> str:
    """Return a state strongly biased toward where this mineral occurs."""
    preferred = MINERAL_PRIORS[mineral][-1]   # last element = preferred states
    all_states = list(STATE_INFO.keys())
    weights = np.array([8.0 if s in preferred else 0.5 for s in all_states])
    weights /= weights.sum()
    return rng.choice(all_states, p=weights)


def generate_dataset(n_samples: int = 2000) -> pd.DataFrame:
    rows = []
    for i in range(n_samples):
        mineral = rng.choice(MINERALS)
        p = MINERAL_PRIORS[mineral]
        (elev_mu, elev_sd,
         au_mu,  au_sd,
         cu_mu,  cu_sd,
         fe_mu,  fe_sd,
         li_mu,  li_sd,
         mn_mu,  mn_sd,
         al_mu,  al_sd,
         ph_mu, fault_mu,
         dep_preferred, _) = p

        state = _sample_state(mineral)
        state_lat, state_lon = STATE_INFO[state]

        lat       = float(np.clip(state_lat + rng.normal(0, 1.2), LAT_MIN, LAT_MAX))
        lon       = float(np.clip(state_lon + rng.normal(0, 1.2), LON_MIN, LON_MAX))
        elevation = float(np.clip(rng.normal(elev_mu, elev_sd), 10, 2500))

        # Tight geochemical signals — low noise to keep classes well separated
        au_ppb    = float(np.clip(rng.normal(au_mu,  au_sd),  0, 5000))
        cu_ppm    = float(np.clip(rng.normal(cu_mu,  cu_sd),  0, 2000))
        fe_pct    = float(np.clip(rng.normal(fe_mu,  fe_sd),  0, 80))
        li_ppm    = float(np.clip(rng.normal(li_mu,  li_sd),  0, 800))
        mn_pct    = float(np.clip(rng.normal(mn_mu,  mn_sd),  0, 50))
        al_pct    = float(np.clip(rng.normal(al_mu,  al_sd),  0, 40))
        soil_ph   = float(np.clip(rng.normal(ph_mu,  0.4),    4.0, 9.0))
        fault_km  = float(np.clip(rng.normal(fault_mu, 1.2),  0.1, 20))

        dep_weights = np.array([4.0 if d in dep_preferred else 0.5
                                 for d in DEP_TYPES])
        dep_weights /= dep_weights.sum()
        dep_type  = rng.choice(DEP_TYPES,  p=dep_weights)
        oper_type = rng.choice(OPER_TYPES, p=[0.45, 0.45, 0.10])
        com_type  = rng.choice(COM_TYPES,  p=[0.70, 0.20, 0.10])
        dev_stat  = rng.choice(DEV_STATS,  p=[0.20, 0.15, 0.40, 0.20, 0.05])
        prod_size = rng.choice(PROD_SIZES, p=[0.15, 0.30, 0.40, 0.15])

        rows.append({
            "site_name":     f"Site_{i+1:04d}",
            "state":         state,
            "latitude":      round(lat,       5),
            "longitude":     round(lon,       5),
            "elevation_m":   round(elevation, 1),
            "au_ppb":        round(au_ppb,    2),
            "cu_ppm":        round(cu_ppm,    2),
            "fe_pct":        round(fe_pct,    2),
            "li_ppm":        round(li_ppm,    2),
            "mn_pct":        round(mn_pct,    2),
            "al_pct":        round(al_pct,    2),
            "soil_ph":       round(soil_ph,   2),
            "fault_dist_km": round(fault_km,  2),
            "dep_type":      dep_type,
            "oper_type":     oper_type,
            "com_type":      com_type,
            "dev_stat":      dev_stat,
            "prod_size":     prod_size,
            "mineral":       mineral,
        })

    return pd.DataFrame(rows)


def encode_features(df: pd.DataFrame):
    cat_cols = ["dep_type","oper_type","com_type","dev_stat","prod_size","state"]
    num_cols = ["latitude","longitude","elevation_m",
                "au_ppb","cu_ppm","fe_pct","li_ppm",
                "mn_pct","al_pct","soil_ph","fault_dist_km"]
    df_enc       = pd.get_dummies(df[cat_cols + num_cols], columns=cat_cols)
    X            = df_enc.values.astype(float)
    feature_names= list(df_enc.columns)
    le           = LabelEncoder()
    y            = le.fit_transform(df["mineral"])
    return X, y, feature_names, le


if __name__ == "__main__":
    df = generate_dataset(2000)
    print(df["mineral"].value_counts())
    print("\nIron  Fe% mean:", df[df.mineral=="Iron"]["fe_pct"].mean().round(1))
    print("Gold  Au ppb mean:", df[df.mineral=="Gold"]["au_ppb"].mean().round(1))
    print("Bauxite Al% mean:", df[df.mineral=="Bauxite"]["al_pct"].mean().round(1))
    print("Manganese Mn% mean:", df[df.mineral=="Manganese"]["mn_pct"].mean().round(1))
    print("Lithium Li ppm mean:", df[df.mineral=="Lithium"]["li_ppm"].mean().round(1))
    print("Copper Cu ppm mean:", df[df.mineral=="Copper"]["cu_ppm"].mean().round(1))
