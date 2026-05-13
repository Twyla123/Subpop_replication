"""
step3_prepare_finetuning_data.py

Prepares multi-format fine-tuning data for the CMS dataset, following SubPop's
methodology (scripts/data_generation/prepare_finetuning_data.py).

Generates per-attribute steering prompts (AGE, GENDER, RACE, INCOME, BOROUGH)
across 3 formats (QA, BIO, PORTRAY) with an ALL combined variant, using a question-based
train/val/test split. Supports demographic hold-out for generalization testing.

Outputs (under OUT_DIR):
    cms_QA_{train,val,test}.csv
    cms_BIO_{train,val,test}.csv
    cms_PORTRAY_{train,val,test}.csv
    cms_ALL_{train,val,test}.csv

Usage:
    python step3_prepare_finetuning_data.py \
        --distributions_csv  approach2_outputs/cms/cms_survey_distributions.csv \
        --demographics_csv   approach2_outputs/cms/cms_demographics.csv \
        --steering_json      approach2_outputs/cms/cms_steering_prompts.json \
        --question_split_json approach2_outputs/cms/cms_question_split.json \
        --output_dir         approach2_outputs/cms/finetuning_data

    # With demographic hold-out:
    python step3_prepare_finetuning_data.py \
        --holdout_groups "AGE:18-24" "BOROUGH:Bronx" \
        ...
Note:
  The current CE / WD fine-tuning path consumes `output_dist`, not
  `output_token`. The `output_token` column is still written as an empty
  placeholder so the CSV schema stays aligned with the upstream SubPop tooling.
"""

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_CMS_DIR = "approach2_outputs/cms"

# Add SubPop to path
SUBPOP_ROOT = Path(__file__).parent / "subpop-main"
sys.path.insert(0, str(SUBPOP_ROOT))

from subpop.utils.survey_utils import generate_mcq


# =========================================================================
# CONSTANTS
# =========================================================================

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G"]  # supports up to 7 options


# =========================================================================
# PROMPT BUILDER
# =========================================================================

def build_survey_mcq(question: str, options: list) -> str:
    """Build the survey question MCQ block with question-specific options."""
    return generate_mcq(question_body=question, options=options, add_answer_forcing=True)


# =========================================================================
# CORE: PREPARE DATA FOR ONE PROMPT FORMAT
# =========================================================================

def prepare_data_for_format(
    distributions_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    steering_prompts: dict,
    train_qkeys: list,
    val_qkeys: list,
    test_qkeys: list,
    prompt_format: str,
    holdout_groups: Optional[list] = None,
    seed: int = 42,
) -> tuple:
    """
    Generate train/val/test DataFrames for one prompt format.
    Supports holdout groups for generalization testing.
    """
    np.random.seed(seed)
    holdout_set = set()
    if holdout_groups:
        for hg in holdout_groups:
            attr, group = hg.split(":", 1); holdout_set.add((attr, group))

    train_dist = distributions_df[distributions_df["qkey"].isin(train_qkeys)]
    val_dist   = distributions_df[distributions_df["qkey"].isin(val_qkeys)]
    test_dist  = distributions_df[distributions_df["qkey"].isin(test_qkeys)]
    train_rows, val_rows, test_rows = [], [], []
    row_counter = 0

    for _, demo_row in tqdm(demographics_df.iterrows(), total=len(demographics_df), desc=f"  {prompt_format}"):
        attribute = demo_row["attribute"]
        group     = demo_row["group"]
        if attribute not in steering_prompts:
            raise KeyError(f"No steering prompts found for attribute '{attribute}'. "
                           "Check cms_steering_prompts.json.")
        if group not in steering_prompts[attribute]:
            raise KeyError(f"No steering prompt found for {attribute}='{group}'. "
                           "Check cms_steering_prompts.json.")
        steering_text = steering_prompts[attribute][group][prompt_format]
        is_holdout    = (attribute, group) in holdout_set

        for split_name, split_dist, split_rows in [("train", train_dist, train_rows),
                                                    ("val",   val_dist,   val_rows),
                                                    ("test",  test_dist,  test_rows)]:
            if is_holdout and split_name == "train": continue
            mask = ((split_dist["attribute"] == attribute) &
                    (split_dist["group"] == group))
            for _, dist_row in split_dist[mask].iterrows():
                responses    = ast.literal_eval(dist_row["responses"]) if isinstance(dist_row["responses"], str) else dist_row["responses"]
                options      = ast.literal_eval(dist_row["options"])   if isinstance(dist_row["options"], str)    else dist_row["options"]
                n_options    = len(options)
                survey_mcq   = build_survey_mcq(dist_row["question"], options)
                if prompt_format == "QA":
                    input_prompt = steering_text + "\n\nAnswer the following question keeping in mind your previous answers.\n" + survey_mcq
                else:
                    input_prompt = steering_text + "\n\n" + survey_mcq
                # CMS has no refusal option — output_dist is exactly the n_options distribution
                output_dist   = responses[:n_options]
                row_counter += 1
                if "ordinal" not in dist_row:
                    raise KeyError("Missing 'ordinal' column in distributions data.")
                ordinal = dist_row["ordinal"]
                split_rows.append({"qkey": dist_row["qkey"], "attribute": attribute, "group": group,
                                    "input_prompt": input_prompt, "output_token": "[]",
                                    "output_dist": str(output_dist), "ordinal": ordinal,
                                    "question": dist_row["question"]})
    return pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows)


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare multi-format fine-tuning data (SubPop-style)"
    )
    parser.add_argument("--distributions_csv", type=str,
                        default=f"{DEFAULT_CMS_DIR}/cms_survey_distributions.csv",
                        help="Ground truth distributions (from step1_cms_adapter)")
    parser.add_argument("--demographics_csv", type=str,
                        default=f"{DEFAULT_CMS_DIR}/cms_demographics.csv")
    parser.add_argument("--steering_json", type=str,
                        default=f"{DEFAULT_CMS_DIR}/cms_steering_prompts.json")
    parser.add_argument("--question_split_json", type=str,
                        default=f"{DEFAULT_CMS_DIR}/cms_question_split.json")
    parser.add_argument("--output_dir", type=str, default=f"{DEFAULT_CMS_DIR}/finetuning_data")
    parser.add_argument("--holdout_groups", nargs="*", default=None,
                        help="Groups to hold out from training (format: 'ATTR:group')")
    parser.add_argument("--formats", nargs="+",
                        default=["QA", "BIO", "PORTRAY", "ALL"],
                        choices=["QA", "BIO", "PORTRAY", "ALL"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_shuffle", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    distributions_df = pd.read_csv(args.distributions_csv)
    demographics_df = pd.read_csv(args.demographics_csv)

    with open(args.steering_json, "r") as f:
        steering_prompts = json.load(f)
    with open(args.question_split_json, "r") as f:
        question_split = json.load(f)

    train_qkeys = question_split["train"]
    val_qkeys = question_split["val"]
    test_qkeys = question_split["test"]

    print(f"Distributions: {len(distributions_df)} rows")
    print(f"Demographics: {len(demographics_df)} (attribute, group) pairs")
    print(f"Question split: train={train_qkeys}, val={val_qkeys}, test={test_qkeys}")
    if args.holdout_groups:
        print(f"Holdout groups (excluded from training): {args.holdout_groups}")

    # Process each prompt format
    for fmt in args.formats:
        print(f"\n{'='*60}")
        print(f"FORMAT: {fmt}")
        print(f"{'='*60}")

        if fmt == "ALL":
            # ALL = combine QA + BIO + PORTRAY
            all_train, all_val, all_test = [], [], []
            for sub_fmt in ["QA", "BIO", "PORTRAY"]:
                train_df, val_df, test_df = prepare_data_for_format(
                    distributions_df=distributions_df,
                    demographics_df=demographics_df,
                    steering_prompts=steering_prompts,
                    train_qkeys=train_qkeys,
                    val_qkeys=val_qkeys,
                    test_qkeys=test_qkeys,
                    prompt_format=sub_fmt,
                    holdout_groups=args.holdout_groups,
                    seed=args.seed,
                )
                all_train.append(train_df)
                all_val.append(val_df)
                all_test.append(test_df)

            train_df = pd.concat(all_train, ignore_index=True)
            val_df = pd.concat(all_val, ignore_index=True)
            test_df = pd.concat(all_test, ignore_index=True)
        else:
            train_df, val_df, test_df = prepare_data_for_format(
                distributions_df=distributions_df,
                demographics_df=demographics_df,
                steering_prompts=steering_prompts,
                train_qkeys=train_qkeys,
                val_qkeys=val_qkeys,
                test_qkeys=test_qkeys,
                prompt_format=fmt,
                holdout_groups=args.holdout_groups,
                seed=args.seed,
            )

        # Shuffle unless disabled.
        if not args.no_shuffle:
            train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
            val_df = val_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
            test_df = test_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

        # Save — naming matches CMS convention: cms_{FORMAT}_{split}.csv
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            path = out_dir / f"cms_{fmt}_{split_name}.csv"
            split_df.to_csv(path, index=False)

        print(f"  Train: {len(train_df)} rows")
        print(f"  Val:   {len(val_df)} rows")
        print(f"  Test:  {len(test_df)} rows")

        # Sanity checks
        if len(train_df) > 0:
            train_attrs = train_df["attribute"].nunique()
            train_groups = train_df["group"].nunique()
            train_qkeys_actual = train_df["qkey"].nunique()
            print(f"  Train has {train_attrs} attributes, {train_groups} groups, {train_qkeys_actual} questions")

            # Check for qkey leakage
            train_set = set(train_df["qkey"].unique())
            test_set = set(test_df["qkey"].unique()) if len(test_df) > 0 else set()
            val_set = set(val_df["qkey"].unique()) if len(val_df) > 0 else set()
            if train_set & test_set:
                raise ValueError(f"qkey leakage between train and test: {sorted(train_set & test_set)}")
            if train_set & val_set:
                raise ValueError(f"qkey leakage between train and val: {sorted(train_set & val_set)}")

            # Check holdout groups
            if args.holdout_groups and len(train_df) > 0:
                for hg in args.holdout_groups:
                    attr, group = hg.split(":", 1)
                    leak = train_df[(train_df["attribute"] == attr) & (train_df["group"] == group)]
                    if len(leak) > 0:
                        raise ValueError(f"Holdout group {hg} found in training data.")
                    print(f"  Holdout {hg}: correctly excluded from training")

            # Show sample prompt
            if len(train_df) > 0:
                print(f"\n  Sample prompt ({fmt}):")
                sample = train_df.iloc[0]["input_prompt"]
                # Show first 300 chars
                print(f"  {sample[:300]}...")

    print(f"\n{'='*60}")
    print(f"All formats saved to {out_dir}/")
    print(f"{'='*60}")
    print("Fine-tuning: subpop-main/scripts/experiment/run_finetune.py")


if __name__ == "__main__":
    main()
