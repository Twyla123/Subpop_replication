"""
step1_cms_adapter.py

Adapts the NYC Citywide Mobility Survey (CMS) 2022 data into the SubPop
pipeline format.  Replaces step1b (question definitions) + step1c (survey
processing) for the CMS use-case.

Pipeline position:
    step0_data_adapter.py   → subgroup_weights / demographics (PUMS-based)
    [THIS FILE]             → CMS-based questions + empirical distributions
    step2_vllm_baselines.py → zero-shot LLM predictions
    step3 / step6           → fine-tuning prep + evaluation (unchanged)

Outputs (under --output_dir):
    cms_questions.json              travel-behavior question bank
    cms_survey_distributions.csv    weighted empirical distributions
                                    per (attribute, group, question)
    cms_subgroup_weights.csv        CMS-derived population weights
    cms_demographics.csv            demographics definition for step2
    cms_steering_prompts.json       steering prompts for QA / BIO / PORTRAY

Usage:
    python step1_cms_adapter.py \
        --cms_csv ../CMS/CMS_merged.csv \
        --output_dir approach2_outputs/cms

NOTE: 23 questions total — 17 train / 3 val / 3 test.
      5 demographic attributes: AGE, GENDER, INCOME, BOROUGH, RACE (17 subgroups).
      Covers commute, mobility, socioeconomic, built-environment, and equity themes.
      Subgroup weights are computed from CMS person_weight (no PUMS needed).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# =========================================================================
# 1.  DEMOGRAPHIC DEFINITIONS
#     cms_col        → column name in CMS_merged.csv
#     valid_codes    → codes to keep (drop 995/996/997/998/999 missing)
#     code_map       → CMS integer → human-readable label
#     steering_*     → text used to build LLM steering prompts
# =========================================================================

CMS_DEMOGRAPHICS: Dict[str, dict] = {
    "AGE": {
        # Use raw 'age' decade-code column for 4 meaningful groups.
        # CMS age codes: 4=18-24, 5=25-34, 6=35-44, 7=45-54, 8=55-64,
        #                9=65-74, 10=75-84, 11=85+
        # Collapsed into 4 groups matching the user guide description.
        "cms_col": "age",
        "valid_codes": [4, 5, 6, 7, 8, 9, 10, 11],
        "code_map": {
            4:  "18-24",
            5:  "25-44",
            6:  "25-44",
            7:  "45-64",
            8:  "45-64",
            9:  "65+",
            10: "65+",
            11: "65+",
        },
        "steering_question": "What is your age group?",
        "steering_options": ["18-24", "25-44", "45-64", "65+"],
        "bio_template": "I am {group} years old.",
        "portray_template": "Portray someone who is {group} years old.",
    },
    "GENDER": {
        "cms_col": "gender",
        "valid_codes": [1, 2],
        "code_map": {1: "Male", 2: "Female"},
        "steering_question": "What is your gender?",
        "steering_options": ["Male", "Female"],
        "bio_template": "I am {group}.",
        "portray_template": "Portray a {group} New Yorker.",
    },
    "INCOME": {
        "cms_col": "income_broad",
        # Code 8 = "Not sure"; 999 = "Prefer not to answer" → excluded
        "valid_codes": [1, 2, 3, 4, 5, 6, 7],
        "code_map": {
            1: "Under $25K",
            2: "$25K-$50K",
            3: "$50K-$75K",
            4: "$75K-$100K",
            5: "$100K-$150K",
            6: "$150K-$200K",
            7: "$200K+",
        },
        # Collapsed into 4 brackets to keep cell sizes adequate
        "group_collapse": {
            "Under $25K":    "Under $50K",
            "$25K-$50K":     "Under $50K",
            "$50K-$75K":     "$50K-$100K",
            "$75K-$100K":    "$50K-$100K",
            "$100K-$150K":   "$100K-$200K",
            "$150K-$200K":   "$100K-$200K",
            "$200K+":        "$200K+",
        },
        "steering_question": "What is your annual household income?",
        "steering_options": ["Under $50K", "$50K-$100K", "$100K-$200K", "$200K+"],
        "bio_template": "My household income is {group} per year.",
        "portray_template": "Portray a New Yorker with a household income of {group}.",
    },
    "BOROUGH": {
        "cms_col": "home_county",
        "valid_codes": [36005, 36047, 36061, 36081, 36085],
        "code_map": {
            36005: "Bronx",
            36047: "Brooklyn",
            36061: "Manhattan",
            36081: "Queens",
            36085: "Staten Island",
        },
        "steering_question": "Which NYC borough do you live in?",
        "steering_options": ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"],
        "bio_template": "I live in {group}, New York City.",
        "portray_template": "Portray a resident of {group}, NYC.",
    },
    "RACE": {
        "cms_col": "r_race",
        # CMS codes: 5=White, 2=Asian, 3=Black, 1=Am.Indian, 4=Pac.Islander,
        #            6=Other, 7=Two or more  (per typical HHTS coding)
        # User guide summarises as White / Non-White; we keep both levels.
        "valid_codes": [1, 2, 3, 4, 5, 6, 7],
        "code_map": {
            1: "Non-White",   # American Indian / Alaska Native
            2: "Non-White",   # Asian
            3: "Non-White",   # Black / African American
            4: "Non-White",   # Native Hawaiian / Pacific Islander
            5: "White",
            6: "Non-White",   # Some other race
            7: "Non-White",   # Two or more races
        },
        "steering_question": "How do you identify racially?",
        "steering_options": ["White", "Non-White"],
        "bio_template": "I identify as {group}.",
        "portray_template": "Portray a {group} New Yorker.",
    },
}


# =========================================================================
# 2.  QUESTION DEFINITIONS
#
#     These are PLACEHOLDER questions derived from available CMS behavioral
#     variables.  Replace / extend once the final question list is confirmed.
#
#     Schema per question:
#       qkey          unique key, passed through to all downstream files
#       cms_col       column in CMS_merged.csv that holds the response
#       question      natural-language question shown to the LLM
#       options       ordered list of human-readable response labels
#       code_map      CMS integer code → one of the labels above
#       filter_codes  codes to exclude (e.g. 995=missing, 996=N/A)
#       is_ordinal    True = use Wasserstein distance; False = TVD
#       ordinal       numeric ordinality weights for Wasserstein (if ordinal)
#       theme         for train/val/test stratification
# =========================================================================

CMS_QUESTIONS: List[dict] = [
    # =========================================================
    # TRAIN questions — model sees these during fine-tuning
    # =========================================================

    # ── Q1: Typical commute mode  [Questionnaire §5.5 WORK_MODE]
    {
        "qkey": "Q_COMMUTE_MODE",
        "cms_col": "r_work_mode",
        # Exact question wording from questionnaire §5.5
        "question": "Currently, how do you typically travel to your primary workplace?",
        "options": [
            "Rail (subway, PATH, commuter rail)",
            "Bus, shuttle, or vanpool",
            "Household vehicle or motorcycle",
            "Walk or jog",
            "Work from home",
            "Bicycle or e-bicycle",
            "Uber/Lyft, taxi, or ferry/other",
        ],
        "code_map": {
            100: "Rail (subway, PATH, commuter rail)",
            102: "Bus, shuttle, or vanpool",
            1:   "Household vehicle or motorcycle",
            103: "Walk or jog",
            105: "Work from home",
            104: "Bicycle or e-bicycle",
            107: "Bicycle or e-bicycle",   # micromobility → grouped with bike
            106: "Uber/Lyft, taxi, or ferry/other",   # ferry
            108: "Uber/Lyft, taxi, or ferry/other",   # taxi/TNC
        },
        "filter_codes": [995],     # 995 = missing / not a worker
        "is_ordinal": False,
        "ordinal": None,
        "theme": "commute",
    },

    # ── Q3: Telework frequency  [Questionnaire §5.4 TELEWORK_FREQ]
    {
        "qkey": "Q_TELEWORK_FREQ",
        "cms_col": "r_telework_freq",
        # Exact question wording from questionnaire §5.4
        "question": "How many days do you work from home or telework?",
        # Questionnaire options: 5+ days/wk, 4 days/wk, 2-3 days/wk,
        #   1 day/wk, 1-3 days/month, Less than monthly, Never
        # Data has 6 codes (1-6).
        "options": [
            "5 or more days a week",
            "4 days a week",
            "2-3 days a week",
            "1 day a week",
            "1-3 days a month",
            "Never",
        ],
        "code_map": {
            1: "5 or more days a week",
            2: "4 days a week",
            3: "2-3 days a week",
            4: "1 day a week",
            5: "1-3 days a month",
            6: "Never",
        },
        "filter_codes": [995, 996],
        "is_ordinal": True,
        "ordinal": [5.0, 4.0, 2.5, 1.0, 0.5, 0.0],
        "theme": "commute",
    },

    # ── Q4: Rideshare (TNC) frequency  [Questionnaire §6.3 TNC_FREQ]
    {
        "qkey": "Q_TNC_FREQ",
        "cms_col": "tnc_freq",
        # Exact question wording from questionnaire §6.3
        "question": "How often do you use Uber, Lyft, or a similar app-based ride service?",
        # Questionnaire options: 5+ days/wk, 4 days/wk, 2-3 days/wk,
        #   1 day/wk, 1-3 days/month, Less than monthly
        # (Note: no "Never" — question is only asked to TNC users from screener §6.2)
        "options": [
            "5 or more days a week",
            "4 days a week",
            "2-3 days a week",
            "1 day a week",
            "1-3 days a month",
            "Less than monthly",
        ],
        "code_map": {
            1: "5 or more days a week",
            2: "4 days a week",
            3: "2-3 days a week",
            4: "1 day a week",
            5: "1-3 days a month",
            6: "Less than monthly",
        },
        "filter_codes": [995],
        "is_ordinal": True,
        "ordinal": [5.0, 4.0, 2.5, 1.0, 0.5, 0.1],
        "theme": "shared_mobility",
    },

    # ── Q5: Biking frequency  [Questionnaire §9.4 BIKE_FREQ]
    {
        "qkey": "Q_BIKE_FREQ",
        "cms_col": "bike_freq",
        # Exact question wording from questionnaire §9.4
        "question": "How often do you use a bicycle outside?",
        # Questionnaire options: 5+ days/wk, 4 days/wk, 2-3 days/wk,
        #   1 day/wk, 1-3 days/month, Less than monthly, Never
        # Data has 6 codes.
        "options": [
            "5 or more days a week",
            "4 days a week",
            "2-3 days a week",
            "1 day a week",
            "1-3 days a month",
            "Less than monthly or never",
        ],
        "code_map": {
            1: "5 or more days a week",
            2: "4 days a week",
            3: "2-3 days a week",
            4: "1 day a week",
            5: "1-3 days a month",
            6: "Less than monthly or never",
        },
        "filter_codes": [995, 996],
        "is_ordinal": True,
        "ordinal": [5.0, 4.0, 2.5, 1.0, 0.5, 0.0],
        "theme": "active_transport",
    },

    # ── Q6: Household vehicle count  [Questionnaire §2.4 NUM_VEHICLES]
    {
        "qkey": "Q_NUM_VEHICLES",
        "cms_col": "r_num_vehicles",
        # Exact question wording from questionnaire §2.4
        "question": "How many registered motor vehicles are in your household?",
        "options": [
            "0 (no vehicles in my household)",
            "1 vehicle",
            "2 vehicles",
            "3 or more vehicles",
        ],
        "code_map": {
            0: "0 (no vehicles in my household)",
            1: "1 vehicle",
            2: "2 vehicles",
            3: "3 or more vehicles",
        },
        "filter_codes": [],
        "is_ordinal": True,
        "ordinal": [0.0, 1.0, 2.0, 3.0],
        "theme": "auto_dependency",
    },

    # ── Q7: Employer WFH requirement  [Questionnaire §2.13 WFH_POLICY]
    {
        "qkey": "Q_WFH_POLICY",
        "cms_col": "wfh_policy",
        # Exact question wording from questionnaire §2.13
        # Key congestion-pricing variable: separates employer-mandated commuting
        # from voluntary commuting (policy constraint vs. personal choice).
        "question": "How many days are you required by your employer to report to your worksite or office each week?",
        # Questionnaire has 7 options; data has 6 codes — last two merged.
        "options": [
            "5 or more days a week",
            "4 days a week",
            "2-3 days a week",
            "1 day a week",
            "1-3 days a month",
            "Less than monthly or never required",
        ],
        "code_map": {
            1: "5 or more days a week",
            2: "4 days a week",
            3: "2-3 days a week",
            4: "1 day a week",
            5: "1-3 days a month",
            6: "Less than monthly or never required",
        },
        "filter_codes": [995, 996],   # 995/996 = not a worker / N/A
        "is_ordinal": True,
        "ordinal": [5.0, 4.0, 2.5, 1.0, 0.5, 0.1],
        "theme": "commute",
    },

    # ── Q7b: Commute frequency  [Questionnaire WORK commute frequency]
    #   Useful transport-behavior addition with strong coverage among workers.
    {
        "qkey": "Q_COMMUTE_FREQ",
        "cms_col": "commute_freq",
        "question": "How often do you commute to your workplace?",
        "options": [
            "5 or more days a week",
            "4 days a week",
            "2-3 days a week",
            "1 day a week",
            "1-3 days a month",
            "Less than monthly",
        ],
        "code_map": {
            1: "5 or more days a week",
            2: "4 days a week",
            3: "2-3 days a week",
            4: "1 day a week",
            5: "1-3 days a month",
            6: "Less than monthly",
        },
        "filter_codes": [995, 996],
        "is_ordinal": True,
        "ordinal": [5.0, 4.0, 2.5, 1.0, 0.5, 0.1],
        "theme": "commute",
    },

    # ── Q7c: Workplace in NYC region  [Questionnaire WORK_IN_REGION]
    #   Added as a simple geography/exposure question with decent worker coverage.
    {
        "qkey": "Q_WORK_IN_REGION",
        "cms_col": "work_in_region",
        "question": "Is your primary workplace located in the New York City region?",
        "options": [
            "Yes, in the New York City region",
            "No, outside the New York City region",
        ],
        "code_map": {
            1: "Yes, in the New York City region",
            0: "No, outside the New York City region",
        },
        "filter_codes": [995, 996],
        "is_ordinal": False,
        "ordinal": None,
        "theme": "commute",
    },

    # ── Q8: Employment status  [Questionnaire §2.1 EMPLOYMENT]
    {
        "qkey": "Q_EMPLOYMENT",
        "cms_col": "employment",
        "question": "What is your current employment status?",
        "options": [
            "Employed full-time (paid)",
            "Employed part-time (paid)",
            "Self-employed",
            "Not employed and not looking for work (e.g., retired, stay-at-home parent)",
            "Unemployed and looking for work",
            "Unpaid volunteer or intern",
            "Employed, but not currently working (e.g., on leave, furloughed)",
        ],
        "code_map": {
            1: "Employed full-time (paid)",
            2: "Employed part-time (paid)",
            3: "Self-employed",
            5: "Not employed and not looking for work (e.g., retired, stay-at-home parent)",
            6: "Unemployed and looking for work",
            7: "Unpaid volunteer or intern",
            8: "Employed, but not currently working (e.g., on leave, furloughed)",
        },
        "filter_codes": [],
        "is_ordinal": False,
        "ordinal": None,
        "theme": "socioeconomic",
    },

    # ── Q9: Education level  [Questionnaire §1 EDUCATION]
    {
        "qkey": "Q_EDUCATION",
        "cms_col": "education",
        "question": "What is the highest level of education you have completed?",
        "options": [
            "Less than high school",
            "High school graduate or GED",
            "Some college, but no degree",
            "Vocational or technical training",
            "Associate degree (2-year college)",
            "Bachelor's degree (4-year college)",
            "Graduate or post-graduate degree",
        ],
        "code_map": {
            1: "Less than high school",
            2: "High school graduate or GED",
            3: "Some college, but no degree",
            4: "Vocational or technical training",
            5: "Associate degree (2-year college)",
            6: "Bachelor's degree (4-year college)",
            7: "Graduate or post-graduate degree",
        },
        "filter_codes": [995, 997, 999],
        "is_ordinal": True,
        "ordinal": [1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0],
        "theme": "socioeconomic",
    },

    # ── Q10: Student status  [Questionnaire §2 STUDENT]
    {
        "qkey": "Q_STUDENT",
        "cms_col": "student",
        "question": "Which of the following best describes your student status?",
        "options": [
            "Not a student",
            "Full-time student, attending some or all classes in-person",
            "Part-time student, attending some or all classes in-person",
            "Part-time student, online classes only",
            "Full-time student, online classes only",
        ],
        "code_map": {
            2: "Not a student",
            0: "Full-time student, attending some or all classes in-person",
            1: "Part-time student, attending some or all classes in-person",
            3: "Part-time student, online classes only",
            4: "Full-time student, online classes only",
        },
        "filter_codes": [995],
        "is_ordinal": False,
        "ordinal": None,
        "theme": "socioeconomic",
    },

    # ── Q11: Job location type  [Questionnaire §2 JOB_TYPE]
    {
        "qkey": "Q_JOB_TYPE",
        "cms_col": "job_type",
        "question": "Which of the following best describes where you typically work?",
        "options": [
            "Go to one fixed work location (outside of home)",
            "Work location regularly varies (different offices or job sites)",
            "Work only from home or remotely (telework or self-employed)",
            "Drive, bike, or travel for work (e.g., driver, delivery, sales)",
            "Hybrid — telework some days and go to a work location other days",
        ],
        "code_map": {
            1: "Go to one fixed work location (outside of home)",
            2: "Work location regularly varies (different offices or job sites)",
            3: "Work only from home or remotely (telework or self-employed)",
            4: "Drive, bike, or travel for work (e.g., driver, delivery, sales)",
            5: "Hybrid — telework some days and go to a work location other days",
        },
        "filter_codes": [995, 996],
        "is_ordinal": False,
        "ordinal": None,
        "theme": "commute",
    },

    # ── Q12: Residence type  [Questionnaire §8 RESIDENCE_TYPE]
    {
        "qkey": "Q_RESIDENCE_TYPE",
        "cms_col": "residence_type",
        "question": "What type of home do you currently live in?",
        "options": [
            "Single-family house (detached)",
            "Rowhouse or townhouse (attached)",
            "Building with 2-4 units (duplex, triplex, or quad)",
            "Building with 5-49 apartments or condos",
            "Building with 50 or more apartments or condos",
            "Senior, age-restricted, or other housing",
        ],
        "code_map": {
            1: "Single-family house (detached)",
            2: "Rowhouse or townhouse (attached)",
            3: "Building with 2-4 units (duplex, triplex, or quad)",
            4: "Building with 5-49 apartments or condos",
            5: "Building with 50 or more apartments or condos",
            6: "Senior, age-restricted, or other housing",
            7: "Senior, age-restricted, or other housing",
            9: "Senior, age-restricted, or other housing",
        },
        "filter_codes": [995, 997],
        "is_ordinal": False,
        "ordinal": None,
        "theme": "built_environment",
    },

    # ── Q13: Primary household language  [Questionnaire §1 PRIMARY_LANGUAGE]
    {
        "qkey": "Q_PRIMARY_LANGUAGE",
        "cms_col": "primary_language",
        "question": "What is the primary language spoken in your household?",
        "options": ["English", "Spanish", "Chinese", "Russian"],
        "code_map": {1: "English", 2: "Spanish", 3: "Chinese", 4: "Russian"},
        "filter_codes": [995, 997],
        "is_ordinal": False,
        "ordinal": None,
        "theme": "equity",
    },

    # ── Q14: Household size  [Questionnaire §1 R_NUM_PEOPLE]
    {
        "qkey": "Q_HOUSEHOLD_SIZE",
        "cms_col": "r_num_people",
        "question": "How many people, including yourself, live in your household?",
        "options": ["1 person", "2 people", "3 people", "4 people", "5 or more people"],
        "code_map": {1: "1 person", 2: "2 people", 3: "3 people", 4: "4 people", 5: "5 or more people"},
        "filter_codes": [],
        "is_ordinal": True,
        "ordinal": [1.0, 2.0, 3.0, 4.0, 5.0],
        "theme": "household",
    },

    # ── Q15: Household bicycle count  [Questionnaire §9 NUM_BICYCLES — grouped]
    {
        "qkey": "Q_NUM_BICYCLES",
        "cms_col": "num_bicycles",
        "question": "How many bicycles does your household have access to?",
        "options": [
            "0 (no bicycles in my household)",
            "1 bicycle",
            "2 bicycles",
            "3 or more bicycles",
        ],
        "code_map": {
            0: "0 (no bicycles in my household)",
            1: "1 bicycle",
            2: "2 bicycles",
            3: "3 or more bicycles",
            4: "3 or more bicycles",
            5: "3 or more bicycles",
            6: "3 or more bicycles",
            7: "3 or more bicycles",
            8: "3 or more bicycles",
        },
        "filter_codes": [995],
        "is_ordinal": True,
        "ordinal": [0.0, 1.0, 2.0, 3.0],
        "theme": "active_transport",
    },

    # ── Q16: Workplace zone  [Derived from §2 work address — WORK_CMS_ZONE, collapsed]
    #   Zone 4 = Manhattan Core (below 60th St) = the congestion pricing toll zone.
    #   Zones collapsed: 1+2→Bronx, 3→N.Manhattan, 5+6+7→Queens, 8+9→Brooklyn, 10+11→SI/outside
    {
        "qkey": "Q_WORK_ZONE",
        "cms_col": "work_cms_zone",
        "question": "Which of the following best describes where your primary workplace is located?",
        "options": [
            "Manhattan Core (below 60th Street — Midtown, Financial District, Lower Manhattan)",
            "Northern Manhattan (above 60th Street)",
            "Bronx",
            "Queens",
            "Brooklyn",
            "Staten Island or outside New York City",
        ],
        "code_map": {
            4:  "Manhattan Core (below 60th Street — Midtown, Financial District, Lower Manhattan)",
            3:  "Northern Manhattan (above 60th Street)",
            1:  "Bronx",
            2:  "Bronx",
            5:  "Queens",
            6:  "Queens",
            7:  "Queens",
            8:  "Brooklyn",
            9:  "Brooklyn",
            10: "Staten Island or outside New York City",
            11: "Staten Island or outside New York City",
        },
        "filter_codes": [995, 996],   # non-workers / N/A
        "is_ordinal": False,
        "ordinal": None,
        "theme": "commute",
    },

    # ── Q17: Number of children in household  [Questionnaire §1 NUM_KIDS]
    #   100% coverage. Families with children are NYC's most car-dependent group
    #   (school drop-off, car seats, activities) — key congestion pricing signal.
    {
        "qkey": "Q_NUM_KIDS",
        "cms_col": "num_kids",
        "question": "How many children under 18 live in your household?",
        "options": [
            "0 (no children)",
            "1 child",
            "2 children",
            "3 or more children",
        ],
        "code_map": {
            0: "0 (no children)",
            1: "1 child",
            2: "2 children",
            3: "3 or more children",
            4: "3 or more children",
            5: "3 or more children",
            6: "3 or more children",
        },
        "filter_codes": [],
        "is_ordinal": True,
        "ordinal": [0.0, 1.0, 2.0, 3.0],
        "theme": "household",
    },

    # ── Q18: Industry sector  [Questionnaire §2 INDUSTRY — collapsed 17→5]
    #   66% coverage (workers only — non-workers filtered by code 995/997).
    #   17 raw codes collapsed to 5 groups ordered by WFH potential:
    #     Healthcare → must commute; Tech/Media/Arts → highest WFH potential.
    #   This is the ONLY variable capturing telework potential by occupation type.
    {
        "qkey": "Q_INDUSTRY",
        "cms_col": "industry",
        "question": "Which of the following best describes the industry you work in?",
        "options": [
            "Healthcare or social services",
            "Education",
            "Finance, professional, or business services",
            "Technology, media, arts, or entertainment",
            "Government, non-profit, retail, hospitality, or other",
        ],
        "code_map": {
            8:  "Healthcare or social services",
            7:  "Education",
            1:  "Finance, professional, or business services",   # financial services
            2:  "Finance, professional, or business services",   # real estate
            4:  "Finance, professional, or business services",   # consulting/legal/marketing
            16: "Technology, media, arts, or entertainment",     # tech/telecom
            17: "Technology, media, arts, or entertainment",     # media
            9:  "Technology, media, arts, or entertainment",     # arts/entertainment
            14: "Government, non-profit, retail, hospitality, or other",  # government
            15: "Government, non-profit, retail, hospitality, or other",  # non-profit
            10: "Government, non-profit, retail, hospitality, or other",  # retail
            5:  "Government, non-profit, retail, hospitality, or other",  # hospitality
            12: "Government, non-profit, retail, hospitality, or other",  # transportation/utilities
            13: "Government, non-profit, retail, hospitality, or other",  # construction
            11: "Government, non-profit, retail, hospitality, or other",  # manufacturing
            3:  "Government, non-profit, retail, hospitality, or other",  # capital goods
            6:  "Government, non-profit, retail, hospitality, or other",  # energy
        },
        "filter_codes": [995, 997],   # 995 = not a worker; 997 = prefer not to say
        "is_ordinal": False,
        "ordinal": None,
        "theme": "socioeconomic",
    },

    # =========================================================
    # VAL questions (3)  — used for tuning / early stopping
    # =========================================================

    # ── Q20: Citi Bike frequency  [Questionnaire §9.13 CITI_BIKE_FREQ]
    {
        "qkey": "Q_CITIBIKE_FREQ",
        "cms_col": "citi_bike_freq",
        # Exact question wording from questionnaire §9.13
        "question": "How frequently do you use Citi Bike?",
        # Exact answer options from questionnaire §9.13:
        # (Data code 1 = Never used, verified by distribution — most common response)
        "options": [
            "I have never used Citi Bike",
            "I use it almost every day",
            "I use it at least once a week",
            "I use it at least once a month",
            "Less than monthly",
            "I have used Citi Bike before, but not in the past 12 months",
        ],
        "code_map": {
            1: "I have never used Citi Bike",
            2: "I use it almost every day",
            3: "I use it at least once a week",
            4: "I use it at least once a month",
            5: "Less than monthly",
            6: "I have used Citi Bike before, but not in the past 12 months",
        },
        "filter_codes": [995],
        "is_ordinal": False,   # Nominal: "never used" ≠ "used before but stopped"
        "ordinal": None,
        "theme": "active_transport",
    },

    # ── Q9: Transit harassment experience  [Questionnaire §10.2 HARASSMENT]
    {
        "qkey": "Q_TRANSIT_SAFETY",
        "cms_col": "harassment",
        # Exact question wording from questionnaire §10.2
        "question": "In the past 12 months, have you seen and/or experienced visual, verbal, or physical harassment or violence when traveling?",
        # Exact answer options from questionnaire §10.2:
        "options": [
            "I have seen harassment or violence when traveling",
            "I have experienced harassment or violence when traveling",
            "I have both seen and experienced harassment or violence when traveling",
            "I have not seen or experienced harassment or violence when traveling",
        ],
        "code_map": {
            1: "I have seen harassment or violence when traveling",
            2: "I have experienced harassment or violence when traveling",
            3: "I have both seen and experienced harassment or violence when traveling",
            4: "I have not seen or experienced harassment or violence when traveling",
        },
        "filter_codes": [995],
        "is_ordinal": False,
        "ordinal": None,
        "theme": "transit_experience",
    },

    # ── Q19: Change in household vehicle count  [Questionnaire §7.6 VEHICLE_CHANGE]
    #   Moved to VAL: household change structure is different from frequency training
    #   questions but less novel than test; gives val loss a non-frequency signal.
    {
        "qkey": "Q_VEHICLE_CHANGE",
        "cms_col": "vehicle_change",
        "question": "In the past two years my household has:",
        "options": [
            "Reduced the number of vehicles it has access to",
            "Increased the number of vehicles it has access to",
            "Not changed the number of vehicles it has access to",
        ],
        "code_map": {
            1: "Reduced the number of vehicles it has access to",
            2: "Increased the number of vehicles it has access to",
            3: "Not changed the number of vehicles it has access to",
        },
        "filter_codes": [995],
        "is_ordinal": False,
        "ordinal": None,
        "theme": "auto_dependency",
    },

    # =========================================================
    # TEST questions (3)  — held out for final evaluation
    # =========================================================

    # ── Q20: Change in household bicycle count  [Questionnaire §9.8 BIKE_CHANGE]
    {
        "qkey": "Q_BIKE_CHANGE",
        "cms_col": "bike_change",
        "question": "In the past two years my household has:",
        "options": [
            "Reduced the number of bicycles it has access to",
            "Increased the number of bicycles it has access to",
            "Not changed the number of bicycles it has access to",
        ],
        "code_map": {
            1: "Reduced the number of bicycles it has access to",
            2: "Increased the number of bicycles it has access to",
            3: "Not changed the number of bicycles it has access to",
        },
        "filter_codes": [995],
        "is_ordinal": False,
        "ordinal": None,
        "theme": "active_transport",
    },

    # ── Q21: Housing tenure  [Questionnaire §8.3 RESIDENCE_RENT_OWN]
    {
        "qkey": "Q_HOUSING_TENURE",
        "cms_col": "residence_rent_own",
        # Exact question wording from questionnaire §8.3
        "question": "Do you own or rent your home?",
        # Exact answer options from questionnaire §8.3 (collapsed to 3):
        "options": [
            "Own or buying (paying a mortgage)",
            "Rent",
            "Other (provided by employer, family, or friend; or other arrangement)",
        ],
        "code_map": {
            1: "Own or buying (paying a mortgage)",
            2: "Rent",
            3: "Other (provided by employer, family, or friend; or other arrangement)",
            4: "Other (provided by employer, family, or friend; or other arrangement)",
            5: "Other (provided by employer, family, or friend; or other arrangement)",
        },
        "filter_codes": [995, 997, 999],   # 999 = Prefer not to answer
        "is_ordinal": False,
        "ordinal": None,
        "theme": "housing",
    },

    # ── Q22: EV purchase consideration  [Questionnaire §7 EV_PURCHASE]
    #   TEST: attitudinal / long-run adaptation question — model trained on
    #   current behavioral frequencies must generalise to future intent.
    #   Code 4 (already bought but no longer primary vehicle, n=6) → merged with code 2.
    {
        "qkey": "Q_EV_PURCHASE",
        "cms_col": "ev_purchase",
        "question": "Have you ever considered purchasing a fully electric vehicle?",
        "options": [
            "Yes, and I may purchase one in the next few years",
            "Yes, but I am not likely to purchase one in the next few years",
            "No, I have not considered purchasing a fully electric vehicle",
        ],
        "code_map": {
            1: "Yes, and I may purchase one in the next few years",
            2: "Yes, but I am not likely to purchase one in the next few years",
            3: "No, I have not considered purchasing a fully electric vehicle",
            4: "Yes, but I am not likely to purchase one in the next few years",
        },
        "filter_codes": [995, 996],   # 995/996 = no vehicle / N/A
        "is_ordinal": False,
        "ordinal": None,
        "theme": "auto_dependency",
    },
]


# =========================================================================
# 3.  MISSING-VALUE SENTINEL CODES (filtered out everywhere)
# =========================================================================
MISSING_CODES = {995, 996, 997, 998, 999}


# =========================================================================
# 4.  STEERING PROMPT BUILDERS  (mirrors step0_data_adapter.py)
# =========================================================================

def _mcq_block(question: str, options: List[str]) -> str:
    """Build a multiple-choice question block with lettered options."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = [question]
    for i, opt in enumerate(options):
        lines.append(f"({letters[i]}) {opt}")
    return "\n".join(lines)


def build_qa_steering(attr_name: str, attr_def: dict, group: str) -> str:
    options = attr_def["steering_options"]
    question = attr_def["steering_question"]
    block = _mcq_block(question, options)
    letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[options.index(group)]
    return f"{block}\nAnswer: ({letter}) {group}"


def build_bio_steering(attr_def: dict, group: str) -> str:
    template = attr_def.get("bio_template", "I am in the {group} group.")
    return template.format(group=group)


def build_portray_steering(attr_def: dict, group: str) -> str:
    template = attr_def.get("portray_template", "Portray someone in the {group} group.")
    return template.format(group=group)


# =========================================================================
# 5.  CORE PROCESSING FUNCTIONS
# =========================================================================

def load_and_filter_cms(cms_path: str) -> pd.DataFrame:
    """Load CMS, keep only diary participants (person_weight > 0)."""
    df = pd.read_csv(cms_path)
    n_total = len(df)
    df = df[df["person_weight"] > 0].copy()
    print(f"CMS: {n_total} rows → {len(df)} participants (person_weight > 0)")
    return df


def apply_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a '_demo_{ATTR}' column for each demographic attribute.
    Rows with missing/invalid codes get NaN.
    """
    for attr_name, attr_def in CMS_DEMOGRAPHICS.items():
        col = attr_def["cms_col"]
        valid = set(attr_def["valid_codes"])
        code_map = attr_def["code_map"]
        collapse = attr_def.get("group_collapse", {})

        # Map CMS code → label
        mapped = df[col].apply(
            lambda v: code_map.get(int(v), None) if pd.notna(v) and int(v) in valid else None
        )

        # Optional collapse (e.g. 7 income bins → 4)
        if collapse:
            mapped = mapped.map(lambda x: collapse.get(x, x) if x is not None else None)

        df[f"_demo_{attr_name}"] = mapped

    return df


def compute_empirical_distributions(
    df: pd.DataFrame,
    min_cell_size: int = 10,
) -> pd.DataFrame:
    """
    For each (attribute, group, question), compute a weighted empirical
    distribution over question response options.

    Returns a DataFrame in SubPop survey_distributions.csv schema:
        qkey | attribute | group | responses | refusal_rate | ordinal | question | options
    """
    rows = []
    warnings_list = []

    for q in CMS_QUESTIONS:
        qkey   = q["qkey"]
        col    = q["cms_col"]
        opts   = q["options"]
        cmap   = q["code_map"]
        filt   = set(q.get("filter_codes", []))
        ordinal = q.get("ordinal") or list(range(1, len(opts) + 1))

        if col not in df.columns:
            print(f"  SKIP {qkey}: column '{col}' not found in CMS data.")
            continue

        for attr_name, attr_def in CMS_DEMOGRAPHICS.items():
            demo_col = f"_demo_{attr_name}"
            groups = [g for g in df[demo_col].dropna().unique()]

            for group in sorted(groups):
                mask = df[demo_col] == group
                sub = df[mask].copy()

                # Filter missing / N/A codes for this question
                sub = sub[sub[col].notna()]
                sub = sub[~sub[col].apply(
                    lambda v: int(v) in filt | MISSING_CODES
                )]

                if len(sub) == 0:
                    continue

                n_valid = len(sub)
                n_total_group = mask.sum()
                refusal_rate = round(1.0 - n_valid / n_total_group, 4) if n_total_group > 0 else 1.0

                if n_valid < min_cell_size:
                    warnings_list.append(
                        f"  WARN {attr_name}={group} | {qkey}: only {n_valid} valid responses"
                    )

                # Weighted counts per option
                opt_counts = {o: 0.0 for o in opts}
                for _, row in sub.iterrows():
                    code = int(row[col])
                    label = cmap.get(code)
                    if label in opt_counts:
                        opt_counts[label] += float(row["person_weight"])

                total_w = sum(opt_counts.values())
                if total_w == 0:
                    continue

                dist = [opt_counts[o] / total_w for o in opts]

                rows.append({
                    "qkey":         qkey,
                    "attribute":    attr_name,
                    "group":        group,
                    "responses":    str(dist),
                    "refusal_rate": refusal_rate,
                    "ordinal":      str(ordinal),
                    "question":     q["question"],
                    "options":      str(opts),
                    "n_respondents": n_valid,
                    "is_ordinal":   q["is_ordinal"],
                    "theme":        q["theme"],
                })

    if warnings_list:
        print("\nCell-size warnings:")
        for w in warnings_list:
            print(w)

    return pd.DataFrame(rows)


def compute_subgroup_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute CMS-derived population weights per (attribute, group).
    Weight = sum of person_weight for that group / total person_weight.
    """
    rows = []
    total_w = df["person_weight"].sum()

    for attr_name, attr_def in CMS_DEMOGRAPHICS.items():
        demo_col = f"_demo_{attr_name}"
        groups = df[demo_col].dropna()

        for group in sorted(groups.unique()):
            mask = df[demo_col] == group
            w = df.loc[mask, "person_weight"].sum()
            rows.append({
                "attribute": attr_name,
                "group":     group,
                "weight":    round(w / total_w, 6),
                "count":     int(mask.sum()),
            })

    return pd.DataFrame(rows)


def build_demographics_csv(weights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the demographics_congestion.csv format expected by step2
    (attribute | group | steering_QA | steering_BIO | steering_PORTRAY).
    """
    rows = []
    for attr_name, attr_def in CMS_DEMOGRAPHICS.items():
        groups = weights_df[weights_df["attribute"] == attr_name]["group"].tolist()
        for group in sorted(groups):
            rows.append({
                "attribute":       attr_name,
                "group":           group,
                "steering_QA":     build_qa_steering(attr_name, attr_def, group),
                "steering_BIO":    build_bio_steering(attr_def, group),
                "steering_PORTRAY": build_portray_steering(attr_def, group),
            })
    return pd.DataFrame(rows)


def build_steering_prompts_json(demographics_df: pd.DataFrame) -> dict:
    """
    Build the congestion_steering_prompts.json expected by step2.
    Structure: { attribute: { group: { QA: ..., BIO: ..., PORTRAY: ... } } }
    """
    prompts: dict = {}
    for _, row in demographics_df.iterrows():
        attr  = row["attribute"]
        group = row["group"]
        prompts.setdefault(attr, {})[group] = {
            "QA":      row["steering_QA"],
            "BIO":     row["steering_BIO"],
            "PORTRAY": row["steering_PORTRAY"],
        }
    return prompts


# =========================================================================
# 6.  MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Adapt CMS 2022 data into SubPop pipeline format"
    )
    parser.add_argument(
        "--cms_csv", type=str,
        default="../CMS/CMS_merged.csv",
        help="Path to CMS_merged.csv",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="approach2_outputs/cms",
        help="Directory to write all outputs",
    )
    parser.add_argument(
        "--min_cell_size", type=int, default=10,
        help="Minimum respondents per (attribute, group, question) cell",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # A. Load & prepare CMS
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP A: Load CMS data")
    print("="*60)
    df = load_and_filter_cms(args.cms_csv)
    df = apply_demographics(df)

    # Demographic coverage report
    for attr_name in CMS_DEMOGRAPHICS:
        demo_col = f"_demo_{attr_name}"
        n_valid = df[demo_col].notna().sum()
        n_total = len(df)
        print(f"  {attr_name}: {n_valid}/{n_total} valid "
              f"({100*n_valid/n_total:.1f}%)")

    # ------------------------------------------------------------------
    # B. Save question bank
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP B: Write question bank")
    print("="*60)
    questions_out = [
        {k: v for k, v in q.items() if k != "code_map"}   # strip internal map
        for q in CMS_QUESTIONS
    ]
    q_path = out_dir / "cms_questions.json"
    with open(q_path, "w") as f:
        json.dump(questions_out, f, indent=2)
    print(f"  Wrote {len(CMS_QUESTIONS)} questions → {q_path}")

    # ------------------------------------------------------------------
    # C. Compute empirical distributions
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP C: Compute empirical distributions")
    print("="*60)
    dist_df = compute_empirical_distributions(df, min_cell_size=args.min_cell_size)
    dist_path = out_dir / "cms_survey_distributions.csv"
    dist_df.to_csv(dist_path, index=False)
    print(f"\n  Wrote {len(dist_df)} (attribute × group × question) distributions → {dist_path}")

    # ------------------------------------------------------------------
    # D. Compute subgroup weights
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP D: Compute subgroup weights")
    print("="*60)
    weights_df = compute_subgroup_weights(df)
    weights_path = out_dir / "cms_subgroup_weights.csv"
    weights_df.to_csv(weights_path, index=False)
    print(f"  Wrote {len(weights_df)} subgroup weights → {weights_path}")

    # ------------------------------------------------------------------
    # E. Build demographics file + steering prompts
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP E: Build demographics + steering prompts")
    print("="*60)
    demo_df = build_demographics_csv(weights_df)
    demo_path = out_dir / "cms_demographics.csv"
    demo_df.to_csv(demo_path, index=False)
    print(f"  Wrote {len(demo_df)} demographic rows → {demo_path}")

    steering = build_steering_prompts_json(demo_df)
    steering_path = out_dir / "cms_steering_prompts.json"
    with open(steering_path, "w") as f:
        json.dump(steering, f, indent=2)
    print(f"  Wrote steering prompts → {steering_path}")

    # ------------------------------------------------------------------
    # F. Build question train/val/test split (required by step3)
    #    25 questions: 19 train / 3 val / 3 test
    #
    #    Split logic (theme-stratified):
    #      train (15) → commute (4) + shared_mobility (1) + active_transport (2)
    #                   + auto_dependency (1) + socioeconomic (3) + built_environment (1)
    #                   + equity (1) + household (1) + commute/work_zone (1)
    #      val   (3)  → active_transport (1) + transit_experience (1)
    #                   + auto_dependency/change (1)
    #      test  (3)  → active_transport/change (1) + housing (1)
    #                   + auto_dependency/long-run intent (1)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP F: Build question split for step3")
    print("="*60)

    TRAIN_QKEYS = [
        # commute behavior
        "Q_COMMUTE_MODE",
        "Q_TELEWORK_FREQ",
        "Q_WFH_POLICY",
        "Q_COMMUTE_FREQ",
        "Q_WORK_IN_REGION",
        "Q_JOB_TYPE",
        # shared mobility
        "Q_TNC_FREQ",
        # active transport
        "Q_BIKE_FREQ",
        "Q_NUM_BICYCLES",
        # auto dependency
        "Q_NUM_VEHICLES",
        # socioeconomic
        "Q_EMPLOYMENT",
        "Q_EDUCATION",
        "Q_STUDENT",
        # built environment
        "Q_RESIDENCE_TYPE",
        # equity
        "Q_PRIMARY_LANGUAGE",
        # household
        "Q_HOUSEHOLD_SIZE",
        # workplace geography (congestion zone exposure)
        "Q_WORK_ZONE",
        # household composition
        "Q_NUM_KIDS",
        # occupation / telework potential
        "Q_INDUSTRY",
    ]
    VAL_QKEYS = [
        "Q_CITIBIKE_FREQ",      # active transport, different from train bike questions
        "Q_TRANSIT_SAFETY",     # attitudinal/safety — structurally unlike frequency train Qs
        "Q_VEHICLE_CHANGE",     # household change — non-frequency structure
    ]
    TEST_QKEYS = [
        "Q_BIKE_CHANGE",        # household asset change — active transport domain
        "Q_HOUSING_TENURE",     # housing economics — furthest from any train question
        "Q_EV_PURCHASE",        # long-run adaptation intent — cross-domain generalisation test
    ]

    # Sanity check: every qkey appears in exactly one split
    all_defined = {q["qkey"] for q in CMS_QUESTIONS}
    all_split   = set(TRAIN_QKEYS) | set(VAL_QKEYS) | set(TEST_QKEYS)
    missing = all_defined - all_split
    extra   = all_split - all_defined
    if missing:
        print(f"  WARNING: qkeys in CMS_QUESTIONS but not split: {missing}")
    if extra:
        print(f"  WARNING: qkeys in split but not CMS_QUESTIONS: {extra}")

    question_split = {
        "train": TRAIN_QKEYS,
        "val":   VAL_QKEYS,
        "test":  TEST_QKEYS,
    }
    split_path = out_dir / "cms_question_split.json"
    with open(split_path, "w") as f:
        json.dump(question_split, f, indent=2)
    print(f"  Train ({len(TRAIN_QKEYS)}): {TRAIN_QKEYS}")
    print(f"  Val   ({len(VAL_QKEYS)}):   {VAL_QKEYS}")
    print(f"  Test  ({len(TEST_QKEYS)}):  {TEST_QKEYS}")
    print(f"  Wrote question split → {split_path}")

    # ------------------------------------------------------------------
    # G. Summary
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"CMS participants:       {len(df)}")
    print(f"Questions defined:      {len(CMS_QUESTIONS)}")
    print(f"Attributes:             {dist_df['attribute'].nunique()}")
    print(f"Total distributions:    {len(dist_df)}")
    print(f"\nPer-attribute breakdown:")
    for attr in sorted(dist_df["attribute"].unique()):
        sub = dist_df[dist_df["attribute"] == attr]
        min_n = sub["n_respondents"].min()
        print(f"  {attr:12s}: {sub['group'].nunique()} groups, "
              f"{len(sub)} distributions, "
              f"min cell n={min_n}")

    print(f"\nNext steps:")
    print(f"  step2: python step2_vllm_baselines.py \\")
    print(f"      --demographics_csv {demo_path} \\")
    print(f"      --steering_json    {steering_path} \\")
    print(f"      --questions_json   {q_path} \\")
    print(f"      --output_dir       {out_dir}")
    print(f"\n  step6: python step6_full_evaluation.py \\")
    print(f"      --ground_truth_csv {dist_path} \\")
    print(f"      --demographics_csv {demo_path} \\")
    print(f"      --weights_csv      {weights_path} \\")
    print(f"      --output_dir       {out_dir}/evaluation")


if __name__ == "__main__":
    main()
