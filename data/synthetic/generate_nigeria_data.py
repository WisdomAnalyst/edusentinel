"""
Synthetic Nigeria Education Dataset Generator
Produces realistic but entirely synthetic data for all 36 states + FCT
across 774 LGAs, covering out-of-school children indicators, dropout
risk factors, and geospatial coordinates.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from loguru import logger

SEED = 42
rng = np.random.default_rng(SEED)

# ── Nigeria administrative data ───────────────────────────────────────────────

STATES = {
    "Abia": {"zone": "SE", "center": (5.45, 7.52), "conflict_risk": 0.1},
    "Adamawa": {"zone": "NE", "center": (9.33, 12.40), "conflict_risk": 0.7},
    "Akwa Ibom": {"zone": "SS", "center": (5.00, 7.93), "conflict_risk": 0.2},
    "Anambra": {"zone": "SE", "center": (6.21, 7.06), "conflict_risk": 0.2},
    "Bauchi": {"zone": "NE", "center": (10.31, 9.84), "conflict_risk": 0.6},
    "Bayelsa": {"zone": "SS", "center": (4.77, 6.07), "conflict_risk": 0.3},
    "Benue": {"zone": "NC", "center": (7.34, 8.75), "conflict_risk": 0.5},
    "Borno": {"zone": "NE", "center": (11.85, 13.16), "conflict_risk": 0.95},
    "Cross River": {"zone": "SS", "center": (5.87, 8.60), "conflict_risk": 0.2},
    "Delta": {"zone": "SS", "center": (5.54, 5.90), "conflict_risk": 0.3},
    "Ebonyi": {"zone": "SE", "center": (6.27, 8.01), "conflict_risk": 0.2},
    "Edo": {"zone": "SS", "center": (6.34, 5.63), "conflict_risk": 0.2},
    "Ekiti": {"zone": "SW", "center": (7.62, 5.22), "conflict_risk": 0.1},
    "Enugu": {"zone": "SE", "center": (6.86, 7.50), "conflict_risk": 0.2},
    "FCT Abuja": {"zone": "NC", "center": (9.07, 7.40), "conflict_risk": 0.1},
    "Gombe": {"zone": "NE", "center": (10.28, 11.17), "conflict_risk": 0.5},
    "Imo": {"zone": "SE", "center": (5.57, 7.06), "conflict_risk": 0.2},
    "Jigawa": {"zone": "NW", "center": (12.22, 9.56), "conflict_risk": 0.4},
    "Kaduna": {"zone": "NW", "center": (10.52, 7.44), "conflict_risk": 0.6},
    "Kano": {"zone": "NW", "center": (12.00, 8.52), "conflict_risk": 0.4},
    "Katsina": {"zone": "NW", "center": (12.99, 7.61), "conflict_risk": 0.5},
    "Kebbi": {"zone": "NW", "center": (12.45, 4.20), "conflict_risk": 0.3},
    "Kogi": {"zone": "NC", "center": (7.80, 6.74), "conflict_risk": 0.3},
    "Kwara": {"zone": "NC", "center": (8.50, 4.55), "conflict_risk": 0.2},
    "Lagos": {"zone": "SW", "center": (6.52, 3.38), "conflict_risk": 0.1},
    "Nasarawa": {"zone": "NC", "center": (8.54, 8.32), "conflict_risk": 0.4},
    "Niger": {"zone": "NC", "center": (9.93, 5.60), "conflict_risk": 0.3},
    "Ogun": {"zone": "SW", "center": (7.16, 3.35), "conflict_risk": 0.1},
    "Ondo": {"zone": "SW", "center": (7.25, 5.19), "conflict_risk": 0.1},
    "Osun": {"zone": "SW", "center": (7.56, 4.56), "conflict_risk": 0.1},
    "Oyo": {"zone": "SW", "center": (7.85, 3.93), "conflict_risk": 0.1},
    "Plateau": {"zone": "NC", "center": (9.22, 9.52), "conflict_risk": 0.5},
    "Rivers": {"zone": "SS", "center": (4.78, 6.99), "conflict_risk": 0.3},
    "Sokoto": {"zone": "NW", "center": (13.06, 5.24), "conflict_risk": 0.4},
    "Taraba": {"zone": "NE", "center": (7.87, 11.36), "conflict_risk": 0.6},
    "Yobe": {"zone": "NE", "center": (12.29, 11.44), "conflict_risk": 0.8},
    "Zamfara": {"zone": "NW", "center": (12.17, 6.66), "conflict_risk": 0.8},
}

ZONE_POVERTY = {"NW": 0.82, "NE": 0.75, "NC": 0.45, "SE": 0.28, "SS": 0.35, "SW": 0.20}
ZONE_GENDER_GAP = {"NW": 0.40, "NE": 0.38, "NC": 0.20, "SE": 0.08, "SS": 0.10, "SW": 0.06}
ZONE_DISABILITY_PREV = {"NW": 0.08, "NE": 0.09, "NC": 0.06, "SE": 0.05, "SS": 0.05, "SW": 0.04}


def _lga_count(state: str) -> int:
    counts = {
        "Kano": 44, "Borno": 27, "Niger": 25, "Kaduna": 23, "Bauchi": 20,
        "Oyo": 33, "Ondo": 18, "Sokoto": 23, "Benue": 23, "Zamfara": 14,
        "Adamawa": 21, "Gombe": 11, "Taraba": 16, "Yobe": 17, "Nasarawa": 13,
        "Plateau": 17, "Lagos": 20, "Ogun": 20, "FCT Abuja": 6,
    }
    return counts.get(state, rng.integers(10, 20))


def generate_lga_dataset() -> pd.DataFrame:
    rows = []
    lga_id = 1

    for state, meta in STATES.items():
        zone = meta["zone"]
        lat_c, lon_c = meta["center"]
        conflict = meta["conflict_risk"]
        poverty_base = ZONE_POVERTY[zone]
        gender_gap_base = ZONE_GENDER_GAP[zone]
        disability_prev = ZONE_DISABILITY_PREV[zone]
        n_lgas = _lga_count(state)

        for lga_idx in range(n_lgas):
            lat = lat_c + rng.uniform(-0.8, 0.8)
            lon = lon_c + rng.uniform(-0.8, 0.8)

            poverty_rate = float(np.clip(
                poverty_base + rng.normal(0, 0.08), 0.05, 0.98
            ))
            conflict_score = float(np.clip(
                conflict + rng.normal(0, 0.10), 0.0, 1.0
            ))
            gender_gap = float(np.clip(
                gender_gap_base + rng.normal(0, 0.05), 0.0, 0.60
            ))
            distance_km = float(np.clip(rng.exponential(4.5), 0.5, 30.0))
            teacher_pupil_ratio = float(np.clip(rng.normal(1 / 50, 1 / 120), 1 / 100, 1 / 15))
            disability_rate = float(np.clip(rng.normal(disability_prev, 0.01), 0.01, 0.20))
            school_density = float(np.clip(rng.normal(0.8, 0.3), 0.05, 3.0))
            literacy_rate = float(np.clip(
                1.0 - poverty_base * 0.8 + rng.normal(0, 0.08), 0.05, 0.98
            ))
            water_sanitation = float(np.clip(
                (1 - poverty_rate) * 0.9 + rng.normal(0, 0.05), 0.02, 0.99
            ))
            population_0_14 = int(rng.integers(8_000, 250_000))
            children_6_14 = int(population_0_14 * rng.uniform(0.40, 0.55))

            # Compute out-of-school rate as weighted function of drivers
            oos_rate = (
                0.30 * poverty_rate +
                0.20 * conflict_score +
                0.15 * gender_gap +
                0.15 * min(distance_km / 15, 1.0) +
                0.10 * disability_rate * 5 +
                0.10 * (1 - school_density / 3) +
                rng.normal(0, 0.04)
            )
            oos_rate = float(np.clip(oos_rate, 0.02, 0.92))
            oos_count = int(children_6_14 * oos_rate)

            dominant_driver = max(
                {
                    "poverty": poverty_rate * 0.30,
                    "conflict": conflict_score * 0.20,
                    "gender": gender_gap * 0.15,
                    "distance": min(distance_km / 15, 1.0) * 0.15,
                    "disability": disability_rate * 5 * 0.10,
                    "school_supply": (1 - school_density / 3) * 0.10,
                },
                key=lambda k: {
                    "poverty": poverty_rate * 0.30,
                    "conflict": conflict_score * 0.20,
                    "gender": gender_gap * 0.15,
                    "distance": min(distance_km / 15, 1.0) * 0.15,
                    "disability": disability_rate * 5 * 0.10,
                    "school_supply": (1 - school_density / 3) * 0.10,
                }[k],
            )

            # Risk tier
            if oos_rate >= 0.60:
                risk_tier = "Critical"
            elif oos_rate >= 0.40:
                risk_tier = "High"
            elif oos_rate >= 0.20:
                risk_tier = "Medium"
            else:
                risk_tier = "Low"

            rows.append({
                "lga_id": lga_id,
                "lga_name": f"{state}_LGA_{lga_idx + 1:02d}",
                "state": state,
                "geopolitical_zone": zone,
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "population_0_14": population_0_14,
                "children_6_14": children_6_14,
                "oos_count": oos_count,
                "oos_rate": round(oos_rate, 4),
                "poverty_rate": round(poverty_rate, 4),
                "conflict_score": round(conflict_score, 4),
                "gender_gap": round(gender_gap, 4),
                "distance_to_school_km": round(distance_km, 2),
                "teacher_pupil_ratio": round(teacher_pupil_ratio, 5),
                "disability_rate": round(disability_rate, 4),
                "school_density_per_1000": round(school_density, 3),
                "literacy_rate": round(literacy_rate, 4),
                "water_sanitation_access": round(water_sanitation, 4),
                "dominant_driver": dominant_driver,
                "risk_tier": risk_tier,
            })
            lga_id += 1

    df = pd.DataFrame(rows)
    logger.info(f"Generated LGA dataset: {len(df)} records across {df['state'].nunique()} states")
    return df


def generate_children_dataset(lga_df: pd.DataFrame, n_children: int = 15_000) -> pd.DataFrame:
    """Individual-level synthetic child records for dropout prediction."""
    rows = []
    lga_sample = lga_df.sample(n=n_children, replace=True, random_state=SEED).reset_index(drop=True)

    for i, lga in lga_sample.iterrows():
        gender = rng.choice(["M", "F"], p=[0.50, 0.50])
        age = int(rng.integers(6, 16))
        grade = max(1, min(9, age - 5 + int(rng.integers(-1, 2))))

        attendance_rate = float(np.clip(
            (1 - lga["poverty_rate"] * 0.4 - lga["conflict_score"] * 0.3 +
             rng.normal(0, 0.10)), 0.0, 1.0
        ))
        if gender == "F":
            attendance_rate = float(np.clip(attendance_rate - lga["gender_gap"] * 0.5, 0.0, 1.0))

        math_score = float(np.clip(
            50 + (attendance_rate - 0.5) * 60 + rng.normal(0, 12), 0, 100
        ))
        literacy_score = float(np.clip(
            50 + (attendance_rate - 0.5) * 55 + rng.normal(0, 12), 0, 100
        ))

        household_income_usd_day = float(np.clip(
            rng.exponential(1.5 * (1 - lga["poverty_rate"]) + 0.5), 0.1, 20.0
        ))
        if lga["poverty_rate"] > 0.5:
            _raw_p = np.array([
                lga["poverty_rate"] * 0.5,
                0.30,
                max(0.05, 0.30 - lga["poverty_rate"] * 0.2),
                max(0.02, 0.25 - lga["poverty_rate"] * 0.3),
            ])
            _edu_p = (_raw_p / _raw_p.sum()).tolist()
        else:
            _edu_p = [0.10, 0.25, 0.35, 0.30]
        parent_edu_level = int(rng.choice([0, 1, 2, 3], p=_edu_p))

        disability = rng.random() < lga["disability_rate"]
        distance = float(np.clip(lga["distance_to_school_km"] + rng.normal(0, 1.0), 0.2, 30.0))
        conflict_displaced = rng.random() < lga["conflict_score"] * 0.3
        school_fee_burden = float(np.clip(rng.normal(lga["poverty_rate"] * 0.8, 0.15), 0, 1))
        sibling_count = int(np.clip(rng.poisson(4 + lga["poverty_rate"] * 3), 0, 14))
        has_birth_cert = rng.random() > lga["poverty_rate"] * 0.6
        meal_programme = rng.random() > 0.6

        # Dropout label (1 = dropped out / never enrolled)
        dropout_score = (
            0.25 * (1 - attendance_rate) +
            0.15 * lga["poverty_rate"] +
            0.15 * lga["conflict_score"] +
            0.10 * (distance / 15) +
            0.10 * (1 if disability else 0) +
            0.08 * (1 - min(math_score, literacy_score) / 100) +
            0.07 * (1 if conflict_displaced else 0) +
            0.05 * school_fee_burden +
            0.03 * (0 if has_birth_cert else 1) +
            0.02 * (0 if meal_programme else 1) +
            rng.normal(0, 0.05)
        )
        if gender == "F":
            dropout_score += lga["gender_gap"] * 0.15
        dropout = int(dropout_score > rng.uniform(0.35, 0.55))

        rows.append({
            "child_id": i + 1,
            "lga_id": int(lga["lga_id"]),
            "state": lga["state"],
            "geopolitical_zone": lga["geopolitical_zone"],
            "gender": gender,
            "age": age,
            "grade_level": grade,
            "attendance_rate": round(attendance_rate, 3),
            "math_score": round(math_score, 1),
            "literacy_score": round(literacy_score, 1),
            "household_income_usd_day": round(household_income_usd_day, 2),
            "parent_edu_level": parent_edu_level,
            "sibling_count": sibling_count,
            "distance_to_school_km": round(distance, 2),
            "disability": int(disability),
            "conflict_displaced": int(conflict_displaced),
            "school_fee_burden": round(school_fee_burden, 3),
            "has_birth_certificate": int(has_birth_cert),
            "meal_programme_access": int(meal_programme),
            "poverty_rate_lga": round(float(lga["poverty_rate"]), 4),
            "conflict_score_lga": round(float(lga["conflict_score"]), 4),
            "teacher_pupil_ratio": round(float(lga["teacher_pupil_ratio"]), 5),
            "dropout": dropout,
        })

    df = pd.DataFrame(rows)
    logger.info(
        f"Generated children dataset: {len(df)} records, "
        f"{df['dropout'].mean():.1%} dropout rate"
    )
    return df


def main():
    out_dir = Path(__file__).parent.parent / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    lga_df = generate_lga_dataset()
    lga_path = out_dir / "nigeria_lga_education_indicators.csv"
    lga_df.to_csv(lga_path, index=False)
    logger.success(f"Saved LGA dataset → {lga_path}")

    children_df = generate_children_dataset(lga_df, n_children=20_000)
    child_path = out_dir / "nigeria_children_dropout_dataset.csv"
    children_df.to_csv(child_path, index=False)
    logger.success(f"Saved children dataset → {child_path}")

    # Summary JSON for dashboard
    summary = {
        "total_lgas": int(len(lga_df)),
        "total_states": int(lga_df["state"].nunique()),
        "total_oos_children": int(lga_df["oos_count"].sum()),
        "national_oos_rate": round(float(lga_df["oos_rate"].mean()), 4),
        "critical_lgas": int((lga_df["risk_tier"] == "Critical").sum()),
        "high_risk_lgas": int((lga_df["risk_tier"] == "High").sum()),
        "children_records": int(len(children_df)),
        "dropout_rate": round(float(children_df["dropout"].mean()), 4),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.success(f"Summary: {summary}")


if __name__ == "__main__":
    main()
