"""
step0_data_adapter.py

NOTE: This file is NOT part of the active CMS replication pipeline.
      For the NYC Congestion Pricing SubPop replication, start with
      step1_cms_adapter.py, which handles all data processing directly
      from CMS 2022 (cms_survey_distributions.csv, demographics, weights).

      step0 is an experimental adapter for PUMS-only subgroup construction.
      LODES mode is not implemented and will raise NotImplementedError.

----------------------------------------------------------------------
Flexible data adapter that normalizes PUMS data into a unified subgroup
format for exploratory use.  Only the --source pums mode is functional.

Outputs (under OUT_DIR):
    subgroup_weights.csv         - (attribute, group, weight, count) pairs
    demographics_congestion.csv  - SubPOP demographics definition file
    congestion_steering_prompts.json - steering prompts for all 3 formats

Data source decision matrix:
    ┌─────────────────────────────────────────────────────────────────┐
    │ Source    │ Demographics        │ Geography/Commute             │
    │──────────│─────────────────────│───────────────────────────────│
    │ PUMS     │ age, sex, race,     │ ❌ no PUMA, no commute mode,  │
    │ (yours)  │ education, income,  │    no work location           │
    │          │ vehicles, HH size   │                               │
    │──────────│─────────────────────│───────────────────────────────│
    │ LODES    │ age (3 bins),       │ ✅ home block, work block,    │
    │          │ earnings (3 bins)   │    (no commute mode)          │
    │──────────│─────────────────────│───────────────────────────────│
    │ Merged   │ ✅ all PUMS demos + │ ✅ LODES geography +          │
    │          │                     │    PUMS vehicles as proxy     │
    │──────────│─────────────────────│───────────────────────────────│
    │ Full PUMS│ ✅ all demos        │ ✅ PUMA + POWPUMA + JWTR +    │
    │ (ideal)  │                     │    JWMNP (need more columns)  │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    # With your current PUMS parquet (no geography):
    python step0_data_adapter.py --source pums \
        --pums_path pums_demographics_ny_2024.parquet

    # With LODES only (current pipeline):
    python step0_data_adapter.py --source lodes \
        --lodes_files ny_od_main_JT03_2023.csv nj_od_main_JT03_2023.csv ct_od_main_JT03_2023.csv

    # With merged PUMS+LODES (when professor decides):
    python step0_data_adapter.py --source merged \
        --pums_path pums_demographics_ny_2024.parquet \
        --lodes_files ny_od_main_JT03_2023.csv

    # With full PUMS (if you get commute columns later):
    python step0_data_adapter.py --source pums_full \
        --pums_path pums_with_commute.parquet
"""

import argparse
import json
from pathlib import Path

import pandas as pd


# =========================================================================
# CONFIGURATION: attribute definitions
# =========================================================================

# Master registry: every possible attribute, its source, and binning.
# Active attributes depend on --source flag.
ATTRIBUTE_REGISTRY = {
    # --- From PUMS ---
    "AGE": {
        "source": "pums",
        "pums_col": "AGEP",
        "bins": [18, 29, 49, 64, 200],
        "labels": ["18-29", "30-49", "50-64", "65+"],
        "filter_below": 18,  # exclude minors
    },
    "SEX": {
        "source": "pums",
        "pums_col": "SEX",
        "mapping": {1: "Male", 2: "Female"},
    },
    "RACE": {
        "source": "pums",
        "pums_cols": ["RAC1P", "HISP"],
        "custom_fn": "_bin_race",
        "groups": ["White", "Black", "Asian", "Hispanic", "Other"],
    },
    "EDUCATION": {
        "source": "pums",
        "pums_col": "SCHL",
        "mapping": {
            # SCHL codes → SubPop education groups
            # 0-11: less than HS, 12: HS equiv, 13-15: some college no degree,
            # 16-17: HS diploma, 18-19: some college, 20: associate's,
            # 21: bachelor's, 22-23: master's/professional, 24: doctorate
        },
        "custom_fn": "_bin_education",
        "groups": [
            "Less than high school",
            "High school graduate",
            "Some college, no degree",
            "Associate's degree",
            "College graduate/some postgrad",
            "Postgraduate",
        ],
    },
    "INCOME": {
        "source": "pums",
        "pums_col": "HINCP",
        "bins": [0, 30000, 50000, 75000, 100000, float("inf")],
        "labels": [
            "Less than $30,000",
            "$30,000-$50,000",
            "$50,000-$75,000",
            "$75,000-$100,000",
            "$100,000 or more",
        ],
    },
    "VEHICLES": {
        "source": "pums",
        "pums_col": "VEH",
        "custom_fn": "_bin_vehicles",
        "groups": ["No vehicle", "1 vehicle", "2+ vehicles"],
    },
    "HOUSEHOLD_SIZE": {
        "source": "pums",
        "pums_col": "NP",
        "custom_fn": "_bin_household_size",
        "groups": ["Lives alone", "2 people", "3-4 people", "5+ people"],
    },
    # --- From LODES ---
    "HOME_REGION": {
        "source": "lodes",
        "groups": [
            "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island",
            "NY outside NYC", "NJ", "CT",
        ],
    },
    "WORK_REGION": {
        "source": "lodes",
        "groups": [
            "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island",
            "NY outside NYC", "NJ", "CT",
        ],
    },
    # --- From full PUMS (if commute columns available) ---
    "COMMUTE_MODE": {
        "source": "pums_commute",
        "pums_col": "JWTR",
        "custom_fn": "_bin_commute_mode",
        "groups": [
            "Drive alone", "Carpool", "Public transit",
            "Walk/bike", "Work from home",
        ],
    },
    "COMMUTE_TIME": {
        "source": "pums_commute",
        "pums_col": "JWMNP",
        "bins": [0, 15, 30, 45, 60, 999],
        "labels": [
            "Less than 15 min", "15-29 min", "30-44 min",
            "45-59 min", "60+ min",
        ],
    },
}

# Which attributes are active for each data source mode
SOURCE_ATTRIBUTES = {
    "pums":      ["AGE", "SEX", "RACE", "EDUCATION", "INCOME", "VEHICLES", "HOUSEHOLD_SIZE"],
    "lodes":     ["HOME_REGION", "WORK_REGION"],  # age & earnings already in LODES but coarser
    "merged":    ["AGE", "SEX", "RACE", "EDUCATION", "INCOME", "VEHICLES", "HOUSEHOLD_SIZE",
                  "HOME_REGION", "WORK_REGION"],
    "pums_full": ["AGE", "SEX", "RACE", "EDUCATION", "INCOME", "VEHICLES", "HOUSEHOLD_SIZE",
                  "COMMUTE_MODE", "COMMUTE_TIME"],
}


# =========================================================================
# STEERING PROMPT TEMPLATES  (all 3 SubPop formats)
# =========================================================================

# For attributes matching SubPop's 12, reuse their exact prompts.
# For new attributes (VEHICLES, HOUSEHOLD_SIZE, COMMUTE_*), write new ones.
STEERING_PROMPTS = [
    # --- Reused from SubPop (adjusted options where needed) ---
    {
        "attribute": "AGE",
        "bio_prompt": "Below you will be asked to provide a short description of your age group and then answer some questions. Description: I am in the age group",
        "qa_prompt": "How old are you?",
        "portray_prompt": "Answer the following question as if you were in the age group of",
        "no_prompt": "",
        "options": "['18-29', '30-49', '50-64', '65+']",
    },
    {
        "attribute": "SEX",
        "bio_prompt": "Below you will be asked to provide a short description of the sex you were assigned at birth and then answer some questions. Description: I identify as",
        "qa_prompt": "What is the sex that you were assigned at birth?",
        "portray_prompt": "Answer the following question as if the sex you were assigned at birth were",
        "no_prompt": "",
        "options": "['Male', 'Female']",
    },
    {
        "attribute": "RACE",
        "bio_prompt": "Below you will be asked to provide a short description of your race or ethnicity and then answer some questions. Description: I am",
        "qa_prompt": "Which race or ethnicity do you identify with?",
        "portray_prompt": "Answer the following question as if your race or ethnicity were",
        "no_prompt": "",
        "options": "['White', 'Black', 'Asian', 'Hispanic', 'Other']",
    },
    {
        "attribute": "EDUCATION",
        "bio_prompt": "Below you will be asked to provide a short description of your current education level and then answer some questions. Description: The highest level of education I have completed is",
        "qa_prompt": "What is the highest level of schooling or degree that you have completed?",
        "portray_prompt": "Answer the following question as if the highest level of education you have completed is",
        "no_prompt": "",
        "options": "['Less than high school', 'High school graduate', 'Some college, no degree', \"Associate's degree\", 'College graduate/some postgrad', 'Postgraduate']",
    },
    {
        "attribute": "INCOME",
        "bio_prompt": "Below you will be asked to provide a short description of your current family income and then answer some questions. Description: Last year, my total family income from all sources, before taxes was",
        "qa_prompt": "Last year, what was your total family income from all sources, before taxes?",
        "portray_prompt": "Answer the following question as if last year, your total family income from all sources, before taxes was",
        "no_prompt": "",
        "options": "['Less than $30,000', '$30,000-$50,000', '$50,000-$75,000', '$75,000-$100,000', '$100,000 or more']",
    },
    {
        "attribute": "HOME_REGION",
        "bio_prompt": "Below you will be asked to provide a short description of the area you currently reside in and then answer some questions. Description: I currently reside in",
        "qa_prompt": "Which area do you currently live in?",
        "portray_prompt": "Answer the following question as if you currently resided in",
        "no_prompt": "",
        "options": "['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island', 'NY outside NYC', 'NJ', 'CT']",
    },
    {
        "attribute": "WORK_REGION",
        "bio_prompt": "Below you will be asked to provide a short description of the area you currently work in and then answer some questions. Description: I currently work in",
        "qa_prompt": "Which area do you currently work in?",
        "portray_prompt": "Answer the following question as if you currently worked in",
        "no_prompt": "",
        "options": "['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island', 'NY outside NYC', 'NJ', 'CT']",
    },
    # --- NEW: commuter/household attributes ---
    {
        "attribute": "VEHICLES",
        "bio_prompt": "Below you will be asked to provide a short description of your household vehicle ownership and then answer some questions. Description: My household has",
        "qa_prompt": "How many vehicles does your household own or lease?",
        "portray_prompt": "Answer the following question as if your household had",
        "no_prompt": "",
        "options": "['No vehicle', '1 vehicle', '2+ vehicles']",
    },
    {
        "attribute": "HOUSEHOLD_SIZE",
        "bio_prompt": "Below you will be asked to provide a short description of your household size and then answer some questions. Description: I live in a household of",
        "qa_prompt": "How many people live in your household, including yourself?",
        "portray_prompt": "Answer the following question as if your household had",
        "no_prompt": "",
        "options": "['Lives alone', '2 people', '3-4 people', '5+ people']",
    },
    {
        "attribute": "COMMUTE_MODE",
        "bio_prompt": "Below you will be asked to describe your primary commuting mode and then answer some questions. Description: My primary mode of commuting to work is",
        "qa_prompt": "What is your primary mode of commuting to work?",
        "portray_prompt": "Answer the following question as if your primary commuting mode were",
        "no_prompt": "",
        "options": "['Drive alone', 'Carpool', 'Public transit', 'Walk/bike', 'Work from home']",
    },
    {
        "attribute": "COMMUTE_TIME",
        "bio_prompt": "Below you will be asked to describe the length of your commute and then answer some questions. Description: My one-way commute to work typically takes",
        "qa_prompt": "How long is your typical one-way commute to work?",
        "portray_prompt": "Answer the following question as if your typical one-way commute to work took",
        "no_prompt": "",
        "options": "['Less than 15 min', '15-29 min', '30-44 min', '45-59 min', '60+ min']",
    },
]


# =========================================================================
# BINNING FUNCTIONS (custom logic for complex mappings)
# =========================================================================

def _bin_race(df: pd.DataFrame) -> pd.Series:
    """Combine RAC1P + HISP into 5 SubPop race/ethnicity groups."""
    race = pd.to_numeric(df["RAC1P"], errors="coerce")
    hisp = pd.to_numeric(df["HISP"], errors="coerce")

    result = pd.Series("Other", index=df.index)
    result[race == 1] = "White"
    result[race == 2] = "Black"
    result.loc[(race == 6)] = "Asian"
    # RAC1P 3=American Indian, 4=Alaska Native, 5=Tribal, 7=Native Hawaiian,
    # 8=Some other, 9=Two or more → Other
    result.loc[race.isin([3, 4, 5, 7, 8, 9])] = "Other"
    # Hispanic overrides race (standard Census practice)
    result[hisp > 1] = "Hispanic"
    return result


def _bin_education(df: pd.DataFrame) -> pd.Series:
    """Map SCHL (25 codes) → 6 SubPop education groups."""
    schl = pd.to_numeric(df["SCHL"], errors="coerce")
    result = pd.Series("Less than high school", index=df.index)

    # SCHL codes (2024 ACS):
    #  0-15: no HS diploma → "Less than high school"
    # 16-17: Regular HS diploma / GED → "High school graduate"
    # 18-19: Some college (< 1 year / 1+ year, no degree) → "Some college, no degree"
    # 20:    Associate's degree → "Associate's degree"
    # 21:    Bachelor's degree → "College graduate/some postgrad"
    # 22-23: Master's / Professional degree → "Postgraduate"
    # 24:    Doctorate → "Postgraduate"

    result[schl.between(16, 17)] = "High school graduate"
    result[schl.between(18, 19)] = "Some college, no degree"
    result[schl == 20] = "Associate's degree"
    result[schl == 21] = "College graduate/some postgrad"
    result[schl.between(22, 24)] = "Postgraduate"
    return result


def _bin_vehicles(df: pd.DataFrame) -> pd.Series:
    """Map VEH → 3 groups. VEH=-1 means GQ (group quarters) → treat as 'No vehicle'."""
    veh = pd.to_numeric(df["VEH"], errors="coerce")
    result = pd.Series("No vehicle", index=df.index)
    result[veh == 1] = "1 vehicle"
    result[veh >= 2] = "2+ vehicles"
    return result


def _bin_household_size(df: pd.DataFrame) -> pd.Series:
    """Map NP (number of persons in household) → 4 groups."""
    np_col = pd.to_numeric(df["NP"], errors="coerce")
    result = pd.Series("Lives alone", index=df.index)
    result[np_col == 2] = "2 people"
    result[np_col.between(3, 4)] = "3-4 people"
    result[np_col >= 5] = "5+ people"
    return result


def _bin_commute_mode(df: pd.DataFrame) -> pd.Series:
    """Map JWTR → 5 commute mode groups (only if column exists).

    Unmapped codes (e.g. motorcycle=7, taxicab=11) are left as NaN so they
    are excluded from subgroup counts rather than creating a spurious 'Other'
    group that does not appear in the COMMUTE_MODE groups list.
    """
    jwtr = pd.to_numeric(df["JWTR"], errors="coerce")
    result = pd.Series(pd.NA, index=df.index, dtype="object")
    result[jwtr == 1] = "Drive alone"           # Car, truck, van - drove alone
    result[jwtr == 2] = "Carpool"                # Car, truck, van - carpooled
    result[jwtr.isin([3, 4, 5, 6])] = "Public transit"  # Bus, streetcar, subway, railroad
    result[jwtr.isin([9, 10])] = "Walk/bike"     # Walked / bicycle
    result[jwtr == 12] = "Work from home"        # Worked from home
    return result


CUSTOM_BINNING_FNS = {
    "_bin_race": _bin_race,
    "_bin_education": _bin_education,
    "_bin_vehicles": _bin_vehicles,
    "_bin_household_size": _bin_household_size,
    "_bin_commute_mode": _bin_commute_mode,
}


# =========================================================================
# CORE ADAPTER: bin PUMS records → (attribute, group) subgroup weights
# =========================================================================

def process_pums(pums_path: str, active_attributes: list[str]) -> pd.DataFrame:
    """
    Load PUMS parquet and compute weighted subgroup counts
    for each active attribute.

    Returns DataFrame with columns: attribute, group, weight, count
    """
    df = pd.read_parquet(pums_path)

    # Convert numeric columns
    for col in df.columns:
        if col != "SERIALNO":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter to adults (18+)
    df = df[df["AGEP"] >= 18].copy()
    print(f"PUMS: {len(df):,} adult records loaded")

    rows = []
    for attr_name in active_attributes:
        attr_def = ATTRIBUTE_REGISTRY.get(attr_name)
        if attr_def is None or attr_def["source"] not in ("pums", "pums_commute"):
            continue

        # Check required columns exist
        required_cols = [attr_def.get("pums_col")]
        if "pums_cols" in attr_def:
            required_cols = attr_def["pums_cols"]
        if any(c not in df.columns for c in required_cols if c is not None):
            print(f"  WARNING: skipping {attr_name} — column(s) {required_cols} not in data")
            continue

        # Bin the attribute
        if "custom_fn" in attr_def:
            binned = CUSTOM_BINNING_FNS[attr_def["custom_fn"]](df)
        elif "mapping" in attr_def:
            binned = df[attr_def["pums_col"]].map(attr_def["mapping"])
        elif "bins" in attr_def:
            binned = pd.cut(
                df[attr_def["pums_col"]],
                bins=attr_def["bins"],
                labels=attr_def["labels"],
                right=True,
            )
        else:
            raise ValueError(f"No binning method for {attr_name}")

        # Drop NaN / unmapped values
        valid = binned.dropna()
        weights = df.loc[valid.index, "PWGTP"]

        # Aggregate weighted counts per group
        group_weights = (
            pd.DataFrame({"group": valid, "weight": weights})
            .groupby("group")
            .agg(weight=("weight", "sum"), count=("weight", "size"))
            .reset_index()
        )
        group_weights["attribute"] = attr_name

        # Normalize weight to population share
        total_weight = group_weights["weight"].sum()
        group_weights["pop_share"] = group_weights["weight"] / total_weight

        rows.append(group_weights)
        print(f"  {attr_name}: {len(group_weights)} groups, {group_weights['count'].sum():,} records")

    if not rows:
        raise ValueError(
            "No valid PUMS subgroup rows were generated. Check that the requested "
            "attributes exist in the parquet file and contain non-missing values."
        )

    result = pd.concat(rows, ignore_index=True)
    return result[["attribute", "group", "weight", "count", "pop_share"]]


# =========================================================================
# OUTPUT GENERATORS
# =========================================================================

def write_demographics_csv(subgroup_df: pd.DataFrame, out_path: Path):
    """Write demographics_congestion.csv in SubPop format."""
    rows = []
    for attr in subgroup_df["attribute"].unique():
        groups = subgroup_df[subgroup_df["attribute"] == attr]["group"].tolist()
        rows.append({"attribute": attr, "group": str(groups)})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


def write_steering_prompts(active_attributes: list[str], out_path: Path):
    """Write congestion_steering_prompts.json for active attributes only."""
    active = [p for p in STEERING_PROMPTS if p["attribute"] in active_attributes]
    with open(out_path, "w") as f:
        json.dump(active, f, indent=4)
    print(f"Wrote {out_path} ({len(active)} attributes)")


def write_subgroup_weights(subgroup_df: pd.DataFrame, out_path: Path):
    """Write subgroup_weights.csv."""
    subgroup_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(subgroup_df)} rows)")


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Data adapter: PUMS/LODES/merged → SubPop subgroups")
    parser.add_argument(
        "--source",
        choices=["pums", "lodes", "merged", "pums_full"],
        required=True,
        help="Data source mode: determines which attributes are active",
    )
    parser.add_argument("--pums_path", type=str, default="pums_demographics_ny_2024.parquet")
    parser.add_argument(
        "--lodes_files",
        nargs="*",
        default=["ny_od_main_JT03_2023.csv", "nj_od_main_JT03_2023.csv", "ct_od_main_JT03_2023.csv"],
    )
    parser.add_argument("--output_dir", type=str, default="approach2_outputs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    active_attrs = SOURCE_ATTRIBUTES[args.source]
    print(f"\n{'='*60}")
    print(f"Data source: {args.source}")
    print(f"Active attributes: {active_attrs}")
    print(f"{'='*60}\n")

    # ---- Process PUMS-sourced attributes ----
    pums_attrs = [a for a in active_attrs if ATTRIBUTE_REGISTRY.get(a, {}).get("source") in ("pums", "pums_commute")]
    all_subgroups = []

    if pums_attrs:
        pums_subgroups = process_pums(args.pums_path, pums_attrs)
        all_subgroups.append(pums_subgroups)

    # ---- Process LODES-sourced attributes ----
    lodes_attrs = [a for a in active_attrs if ATTRIBUTE_REGISTRY.get(a, {}).get("source") == "lodes"]
    if lodes_attrs:
        raise NotImplementedError(
            f"LODES attributes {lodes_attrs} are not yet implemented in step0.\n"
            "This pipeline uses CMS 2022 data via step1_cms_adapter.py, which does not "
            "require step0. Do not use --source lodes or --source merged."
        )

    # ---- Combine and write outputs ----
    if not all_subgroups:
        raise ValueError(
            f"No subgroup data was generated for source={args.source!r}. "
            "Check the input files and selected source mode."
        )

    subgroup_df = pd.concat(all_subgroups, ignore_index=True)

    write_subgroup_weights(subgroup_df, out_dir / "subgroup_weights.csv")
    write_demographics_csv(subgroup_df, out_dir / "demographics_congestion.csv")
    write_steering_prompts(active_attrs, out_dir / "congestion_steering_prompts.json")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for attr in subgroup_df["attribute"].unique():
        groups = subgroup_df[subgroup_df["attribute"] == attr]
        print(f"  {attr}: {len(groups)} groups")
        for _, row in groups.iterrows():
            print(f"    {row['group']:40s} count={row['count']:>7,}  share={row['pop_share']:.3f}")
    print(f"\nTotal subgroups: {len(subgroup_df)}")
    print(f"Output directory: {out_dir}/")


if __name__ == "__main__":
    main()
