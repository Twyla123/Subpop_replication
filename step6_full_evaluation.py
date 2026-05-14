"""
step6_full_evaluation.py

Evaluation framework for the CMS SubPop replication.

Computes, per method and question scope:
  1. Bootstrap confidence intervals on WD/TVD  (SubPop Figure 3)
  2. Per-attribute WD/TVD breakdown            (SubPop Tables 9-10)
  3. Entropy analysis                          (response diversity check)
  4. Inter-group disagreement heatmaps         (SubPop Figure 4)
  5. Ablation comparison table                 (SubPop Table 1 style)
  6. Population-weighted aggregate opinion

Scope-matched outputs:
  evaluation/descriptive_all_rows/   — every question, every (attr, group, qkey) row
                                       disagreement_heatmaps/ (ground truth + per-method)
  evaluation/test_only_fair/         — test questions only (scope-matched comparison
                                       between zero-shot and fine-tuned models, which are
                                       only evaluated on held-out test questions)
                                       disagreement_heatmaps/ (same structure)
  evaluation/                        — ablation_table.csv (both scopes combined)
                                       method_coverage.csv (audit manifest)

Usage:
    python step6_full_evaluation.py \\
        --ground_truth_csv  approach2_outputs/cms/cms_survey_distributions.csv \\
        --questions_json    approach2_outputs/cms/cms_questions.json \\
        --question_split_json approach2_outputs/cms/cms_question_split.json \\
        --predictions_dir   approach2_outputs/cms \\
        --weights_csv       approach2_outputs/cms/cms_subgroup_weights.csv \\
        --output_dir        approach2_outputs/cms/evaluation
"""

import argparse
import ast
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd


DEFAULT_CMS_DIR = "approach2_outputs/cms"

# Add SubPop to path
SUBPOP_ROOT = Path(__file__).parent / "subpop-main"
sys.path.insert(0, str(SUBPOP_ROOT))

from subpop.utils.survey_utils import ordinal_emd


# =========================================================================
# HELPERS
# =========================================================================

def load_ordinal_flags(questions_json_path: str) -> dict:
    """Return {qkey: is_ordinal} from cms_questions.json."""
    with open(questions_json_path, "r") as f:
        questions = json.load(f)
    flags = {}
    for q in questions:
        if "is_ordinal" not in q:
            raise KeyError(
                f"Question '{q['qkey']}' is missing 'is_ordinal' in {questions_json_path}"
            )
        flags[q["qkey"]] = bool(q["is_ordinal"])
    return flags


def tvd(p: list, q: list) -> float:
    """Total Variation Distance for nominal distributions. Range [0, 1]."""
    return sum(abs(pi - qi) for pi, qi in zip(p, q)) / 2


def compute_distance(gt: list, pred: list, ordinal: list, is_ordinal: bool) -> float:
    """Dispatch to ordinal WD (for ordinal questions) or TVD (for nominal questions)."""
    if is_ordinal:
        return ordinal_emd(gt, pred, ordinal)
    else:
        return tvd(gt, pred)


def parse_dist(x):
    """Parse a distribution from string or list.

    Handles two string formats:
      - Python list:  '[0.1, 0.2, 0.7]'   (from step2 zero-shot output)
      - NumPy array:  '[0.1 0.2 0.7]'      (from run_inference.py llm_dist column)
    """
    if isinstance(x, (list, tuple)):
        return list(x)
    if not isinstance(x, str):
        return list(x)
    x = x.strip()
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        # NumPy format: brackets with space-separated values, no commas.
        # Handles both finite floats and nan/inf (e.g. '[nan nan nan]').
        inner = x.lstrip('[').rstrip(']').strip()
        if inner:
            try:
                return [float(v) for v in inner.split()]
            except ValueError:
                pass
        raise ValueError(f"Cannot parse distribution: {x!r}")


def parse_ordinal(x, n_options: int = None):
    """Parse ordinal values from string or list.

    Falls back to [1, 2, ..., n_options] only when x is None or NaN — this
    covers fine-tuned model outputs from run_inference.py, which do not include
    an ordinal column. Malformed non-empty ordinal strings raise ValueError. The
    fallback is used only when ordinal metadata is absent from prediction files.
    """
    if isinstance(x, str):
        try:
            result = ast.literal_eval(x)
            if isinstance(result, list) and len(result) > 0:
                return result
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Cannot parse ordinal values: {x!r}") from exc
    if isinstance(x, list) and len(x) > 0:
        return x
    # Fallback for missing ordinal column (e.g. run_inference.py outputs)
    n = n_options if (n_options is not None and n_options > 0) else 5
    return list(range(1, n + 1))


def load_distributions(csv_path: str) -> pd.DataFrame:
    """Load and parse a distributions CSV.

    Handles two column conventions:
      - 'responses'  : output of step2_vllm_baselines.py (zero-shot files)
      - 'llm_dist'   : output of run_inference.py (fine-tuned / LoRA files)
    """
    df = pd.read_csv(csv_path)
    # Normalise: copy whichever column exists into 'responses'
    if "responses" not in df.columns:
        if "llm_dist" in df.columns:
            df["responses"] = df["llm_dist"]
        else:
            raise KeyError(
                f"{csv_path!r} has neither 'responses' nor 'llm_dist' column. "
                f"Columns found: {list(df.columns)}"
            )
    df["responses_parsed"] = df["responses"].apply(parse_dist)
    # Drop rows where the model produced an all-NaN distribution (inference failure).
    nan_mask = df["responses_parsed"].apply(lambda d: any(np.isnan(v) for v in d))
    if nan_mask.any():
        print(f"  WARNING: {nan_mask.sum()} rows dropped from {csv_path!r} — all-NaN distribution (model inference failure)")
        df = df[~nan_mask].reset_index(drop=True)
    if "ordinal" in df.columns:
        df["ordinal_parsed"] = df.apply(
            lambda r: parse_ordinal(r["ordinal"], n_options=len(parse_dist(r["responses"]))), axis=1
        )
    else:
        df["ordinal_parsed"] = df["responses_parsed"].apply(lambda d: list(range(1, len(d) + 1)))
    return df


def _filter_by_qkeys(df: pd.DataFrame, filter_qkeys: Optional[Set[str]]) -> pd.DataFrame:
    """Return df filtered to filter_qkeys rows, or df unchanged if filter_qkeys is None."""
    if filter_qkeys is None:
        return df
    return df[df["qkey"].isin(filter_qkeys)]


# =========================================================================
# 1. BOOTSTRAP CONFIDENCE INTERVALS
# =========================================================================

def compute_bootstrap_ci(
    ground_truth_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    ordinal_flags: dict = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    filter_qkeys: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Compute bootstrap CI for the configured distance metric (WD or TVD) between
    predicted and ground truth distributions. For each (attribute, group, question)
    pair, resample and compute WD or TVD based on question type.

    filter_qkeys: if provided, restrict analysis to this set of question keys.
    """
    np.random.seed(seed)
    alpha = 1 - confidence

    rows = []
    merged = ground_truth_df.merge(
        predictions_df,
        on=["qkey", "attribute", "group"],
        suffixes=("_gt", "_pred"),
        how="inner",
    )
    merged = _filter_by_qkeys(merged, filter_qkeys)

    for _, row in merged.iterrows():
        gt = parse_dist(row["responses_gt"])
        pred = parse_dist(row["responses_pred"])
        n_options = len(gt)
        ordinal_raw = row.get("ordinal_gt") or row.get("ordinal_pred")
        ordinal = parse_ordinal(ordinal_raw, n_options=n_options)
        if ordinal_flags is None or row["qkey"] not in ordinal_flags:
            raise KeyError(f"Missing is_ordinal metadata for qkey '{row['qkey']}'")
        is_ord = ordinal_flags[row["qkey"]]

        # Point estimate
        wd_point = compute_distance(gt, pred, ordinal, is_ord)

        # Bootstrap: resample from ground truth
        bootstrap_wds = []
        n_sample = 50
        for _ in range(n_bootstrap):
            gt_resample = np.random.multinomial(n_sample, gt)
            gt_resample_dist = (gt_resample / gt_resample.sum()).tolist()
            wd = compute_distance(gt_resample_dist, pred, ordinal, is_ord)
            if not np.isnan(wd):
                bootstrap_wds.append(wd)

        if bootstrap_wds:
            ci_lower = np.percentile(bootstrap_wds, 100 * alpha / 2)
            ci_upper = np.percentile(bootstrap_wds, 100 * (1 - alpha / 2))
            ci_mean = np.mean(bootstrap_wds)
        else:
            ci_lower = ci_upper = ci_mean = np.nan

        rows.append({
            "qkey": row["qkey"],
            "attribute": row["attribute"],
            "group": row["group"],
            "wd_point": wd_point,
            "wd_bootstrap_mean": ci_mean,
            "wd_ci_lower": ci_lower,
            "wd_ci_upper": ci_upper,
        })

    return pd.DataFrame(rows)


# =========================================================================
# 2. PER-ATTRIBUTE DISTANCE BREAKDOWN
# =========================================================================

def per_attribute_wd(
    ground_truth_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    ordinal_flags: dict = None,
    filter_qkeys: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Compute mean distance per attribute (SubPop Tables 9-10).
    Summarizes model error by demographic dimension.

    filter_qkeys: if provided, restrict analysis to this set of question keys.
    """
    merged = ground_truth_df.merge(
        predictions_df,
        on=["qkey", "attribute", "group"],
        suffixes=("_gt", "_pred"),
        how="inner",
    )
    merged = _filter_by_qkeys(merged, filter_qkeys)

    rows = []
    for _, row in merged.iterrows():
        gt = parse_dist(row["responses_gt"])
        pred = parse_dist(row["responses_pred"])
        n_options = len(gt)
        ordinal = parse_ordinal(row.get("ordinal_gt"), n_options=n_options)
        if ordinal_flags is None or row["qkey"] not in ordinal_flags:
            raise KeyError(f"Missing is_ordinal metadata for qkey '{row['qkey']}'")
        is_ord = ordinal_flags[row["qkey"]]
        wd = compute_distance(gt, pred, ordinal, is_ord)
        rows.append({"attribute": row["attribute"], "wd": wd})

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        return pd.DataFrame(columns=["attribute", "mean_wd", "std_wd", "n_pairs"])

    per_attr = detail_df.groupby("attribute")["wd"].agg(["mean", "std", "count"]).reset_index()
    per_attr.columns = ["attribute", "mean_wd", "std_wd", "n_pairs"]
    return per_attr


# =========================================================================
# 3. ENTROPY ANALYSIS
# =========================================================================

def entropy_analysis(
    ground_truth_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    filter_qkeys: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Compare entropy of predicted vs ground truth distributions.
    Checks if model preserves response diversity or collapses.

    filter_qkeys: if provided, restrict analysis to this set of question keys.
    """
    merged = ground_truth_df.merge(
        predictions_df,
        on=["qkey", "attribute", "group"],
        suffixes=("_gt", "_pred"),
        how="inner",
    )
    merged = _filter_by_qkeys(merged, filter_qkeys)

    rows = []
    for _, row in merged.iterrows():
        gt = parse_dist(row["responses_gt"])
        pred = parse_dist(row["responses_pred"])
        n_options = len(gt)

        # Shannon entropy (normalized by log2 of number of options so range is [0,1])
        gt_entropy = -sum(p * np.log2(p + 1e-9) for p in gt) / np.log2(n_options)
        # Use gt n_options for normalization of pred too (same question, same option count)
        pred_entropy = -sum(p * np.log2(p + 1e-9) for p in pred) / np.log2(n_options)

        rows.append({
            "qkey": row["qkey"],
            "attribute": row["attribute"],
            "group": row["group"],
            "entropy_gt": gt_entropy,
            "entropy_pred": pred_entropy,
            "entropy_diff": pred_entropy - gt_entropy,
        })

    return pd.DataFrame(rows)


# =========================================================================
# 4. INTER-GROUP DISAGREEMENT HEATMAPS
# =========================================================================

def intergroup_disagreement(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame = None,
    ordinal_flags: dict = None,
    filter_qkeys: Optional[Set[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    For each attribute, compute pairwise distance between groups on the same question.
    Returns dict of attribute → DataFrame heatmaps.
    Mirrors SubPop Figure 4.

    filter_qkeys: if provided, restrict disagreement computation to this set of
                  question keys (useful for fair comparison on test questions only).
    """
    heatmaps = {}

    for df, label in [(predictions_df, "pred"), (ground_truth_df, "gt")]:
        if df is None:
            continue

        for attr in df["attribute"].unique():
            attr_df = df[df["attribute"] == attr]
            # Apply qkey filter within this attribute's data
            attr_df = _filter_by_qkeys(attr_df, filter_qkeys)

            if attr_df.empty:
                continue

            groups = sorted(attr_df["group"].unique())
            if len(groups) < 2:
                continue

            # Compute average pairwise distance across all filtered questions
            disagreement_matrix = pd.DataFrame(0.0, index=groups, columns=groups)
            count_matrix = pd.DataFrame(0, index=groups, columns=groups)

            for qkey in attr_df["qkey"].unique():
                q_df = attr_df[attr_df["qkey"] == qkey]
                group_dists = {}
                q_ordinal = None
                for _, row in q_df.iterrows():
                    d = parse_dist(row["responses"])
                    group_dists[row["group"]] = d
                    if q_ordinal is None:
                        q_ordinal = parse_ordinal(row.get("ordinal"), n_options=len(d))

                if q_ordinal is None:
                    first_dist = next(iter(group_dists.values())) if group_dists else []
                    q_ordinal = list(range(1, len(first_dist) + 1)) if first_dist else [1, 2, 3, 4, 5]
                if ordinal_flags is None or qkey not in ordinal_flags:
                    raise KeyError(f"Missing is_ordinal metadata for qkey '{qkey}'")
                is_ord = ordinal_flags[qkey]

                for g1, g2 in combinations(groups, 2):
                    if g1 in group_dists and g2 in group_dists:
                        wd = compute_distance(group_dists[g1], group_dists[g2], q_ordinal, is_ord)
                        if not np.isnan(wd):
                            disagreement_matrix.loc[g1, g2] += wd
                            disagreement_matrix.loc[g2, g1] += wd
                            count_matrix.loc[g1, g2] += 1
                            count_matrix.loc[g2, g1] += 1

            # Average
            count_matrix = count_matrix.replace(0, 1)
            avg_disagreement = disagreement_matrix / count_matrix

            heatmaps[f"{attr}_{label}"] = avg_disagreement

    return heatmaps


# =========================================================================
# 5. ABLATION COMPARISON TABLE
# =========================================================================

def build_ablation_table(
    ground_truth_df: pd.DataFrame,
    prediction_files: Dict[str, str],
    bounds_csv: Optional[str] = None,
    ordinal_flags: dict = None,
    filter_qkeys: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Build SubPop Table 1-style ablation table.
    Rows = attributes, Columns = methods.

    filter_qkeys: if provided, only compute distance metrics on this subset of questions.
                  Use this to restrict zero-shot to the same test questions
                  that fine-tuned was evaluated on (fair comparison).
    """
    results = {}

    for method_name, pred_path in prediction_files.items():
        try:
            pred_df = load_distributions(pred_path)
            merged = ground_truth_df.merge(
                pred_df, on=["qkey", "attribute", "group"],
                suffixes=("_gt", "_pred"), how="inner",
            )
            # Restrict to the requested question subset when asked
            merged = _filter_by_qkeys(merged, filter_qkeys)

            method_wds = {}
            for _, row in merged.iterrows():
                gt = parse_dist(row["responses_gt"])
                pred = parse_dist(row["responses_pred"])
                n_options = len(gt)
                ordinal = parse_ordinal(row.get("ordinal_gt"), n_options=n_options)
                if ordinal_flags is None or row["qkey"] not in ordinal_flags:
                    raise KeyError(f"Missing is_ordinal metadata for qkey '{row['qkey']}'")
                is_ord = ordinal_flags[row["qkey"]]
                wd = compute_distance(gt, pred, ordinal, is_ord)
                attr = row["attribute"]
                method_wds.setdefault(attr, []).append(wd)

            # Compute mean per attribute
            for attr, wds in method_wds.items():
                valid_wds = [w for w in wds if not np.isnan(w)]
                if valid_wds:
                    results.setdefault(attr, {})[method_name] = np.mean(valid_wds)

            # Overall
            all_wds = [w for ws in method_wds.values() for w in ws if not np.isnan(w)]
            if all_wds:
                results.setdefault("OVERALL", {})[method_name] = np.mean(all_wds)

        except FileNotFoundError:
            print(f"  WARNING: {pred_path} not found, skipping {method_name}")

    # Add bounds if available
    if bounds_csv and Path(bounds_csv).exists():
        bounds_df = pd.read_csv(bounds_csv)
        for attr in bounds_df["attribute"].unique():
            attr_bounds = bounds_df[bounds_df["attribute"] == attr]
            results.setdefault(attr, {})["Uniform (upper)"] = attr_bounds["bound_uniform_upper"].mean()
            results.setdefault(attr, {})["Bootstrap (lower)"] = attr_bounds["bound_bootstrap_lower_mean"].mean()
        # Overall
        results.setdefault("OVERALL", {})["Uniform (upper)"] = bounds_df["bound_uniform_upper"].mean()
        results.setdefault("OVERALL", {})["Bootstrap (lower)"] = bounds_df["bound_bootstrap_lower_mean"].mean()

    # Convert to DataFrame
    table = pd.DataFrame(results).T
    table.index.name = "Attribute"
    return table


# =========================================================================
# 6. POPULATION-WEIGHTED AGGREGATES
# =========================================================================

def population_weighted_opinion(
    predictions_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    filter_qkeys: Optional[Set[str]] = None,
    ordinal_flags: dict = None,
) -> pd.DataFrame:
    """
    Compute population-weighted aggregate opinion per question.
    Uses the `weight` column from cms_subgroup_weights.csv for population representation.

    filter_qkeys: if provided, restrict to this set of question keys.
    ordinal_flags: {qkey: is_ordinal}. mean_opinion_score is set to NaN for
                   nominal questions because option indices carry no numerical
                   meaning (unordered response categories).
    """
    merged = predictions_df.merge(
        weights_df,
        on=["attribute", "group"],
        how="left",
    )
    merged = _filter_by_qkeys(merged, filter_qkeys)

    rows = []
    for qkey in merged["qkey"].unique():
        q_df = merged[merged["qkey"] == qkey]
        question = q_df.iloc[0].get("question", qkey)
        if ordinal_flags is None or qkey not in ordinal_flags:
            raise KeyError(f"Missing is_ordinal metadata for qkey '{qkey}'")
        is_ordinal = ordinal_flags[qkey]

        # Determine number of options from first valid distribution
        first_dist = parse_dist(q_df.iloc[0]["responses"])
        n_options = len(first_dist)

        # Weighted average distribution across all subgroups
        total_weight = 0.0
        weighted_dist = np.zeros(n_options)

        for _, row in q_df.iterrows():
            dist = parse_dist(row["responses"])
            if len(dist) != n_options:
                raise ValueError(
                    f"Distribution length mismatch for {qkey}: expected {n_options}, got {len(dist)}"
                )
            if "weight" not in row.index or pd.isna(row["weight"]):
                raise KeyError(
                    f"Missing population weight for {row['attribute']}='{row['group']}'. "
                    "Check that cms_subgroup_weights.csv covers all (attribute, group) pairs."
                )
            w = float(row["weight"])
            weighted_dist += np.array(dist) * w
            total_weight += w

        if total_weight > 0:
            weighted_dist /= total_weight

        # Mean score only meaningful for ordinal (Likert) questions.
        # Nominal questions (e.g. commute mode) get NaN — option indices
        # 1,2,3,... have no numeric interpretation there.
        if is_ordinal:
            mean_score = sum((i + 1) * p for i, p in enumerate(weighted_dist))
        else:
            mean_score = float("nan")

        record = {
            "qkey": qkey,
            "question": question,
            "is_ordinal": is_ordinal,
            "n_options": n_options,
            "weighted_distribution": str(weighted_dist.tolist()),
            "mean_opinion_score": mean_score,  # NaN for nominal questions
        }
        # Generic per-option columns: option_1, option_2, ..., option_n
        for i, p in enumerate(weighted_dist):
            record[f"option_{i + 1}"] = p
        rows.append(record)

    return pd.DataFrame(rows)


# =========================================================================
# 7. QUESTION-SUBSET SENSITIVITY ANALYSIS
#    Post-hoc sensitivity analysis, not cross-validation.
#    The model is fixed (trained on one question split). This function
#    evaluates its predictions across non-overlapping question subsets
#    and reports variance. Use test_only_fair/ for unseen-question evaluation.
# =========================================================================

def question_subset_sensitivity(
    ground_truth_df: pd.DataFrame,
    prediction_files: Dict[str, str],
    ordinal_flags: dict = None,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple:
    """
    Post-hoc fold sensitivity analysis over a single fixed model.

    Splits all ground-truth qkeys into n_folds equal groups (deterministic,
    seeded), then evaluates each prediction file on each fold independently.

    Coverage heterogeneity: zero-shot methods have predictions for all
    questions; fine-tuned methods only cover their test questions. Folds are
    built from all ground-truth qkeys, so each method's `n_covered_qkeys` per
    fold may differ. The `pct_covered` column records this coverage. Compare
    methods only when their coverage is similar, or use the common-qkeys section.

    Returns:
        fold_df          : per-(method, fold) rows with coverage columns
        attr_df          : per-(method, fold, attribute) breakdown
        method_qkeys     : dict mapping method_name → set of its qkeys
    """
    np.random.seed(seed)

    all_qkeys = sorted(ground_truth_df["qkey"].unique())
    n = len(all_qkeys)
    indices = np.random.permutation(n)
    fold_assignments = np.array_split(indices, n_folds)
    folds = [[all_qkeys[i] for i in idx] for idx in fold_assignments]

    # Pre-load all prediction files and record their qkey coverage
    method_data: Dict[str, pd.DataFrame] = {}
    method_qkeys: Dict[str, set] = {}
    for method_name, pred_path in prediction_files.items():
        try:
            pdf = load_distributions(pred_path)
            method_data[method_name] = pdf
            method_qkeys[method_name] = set(pdf["qkey"].unique())
        except Exception as exc:
            raise RuntimeError(
                f"Could not load prediction file for {method_name}: {pred_path}"
            ) from exc

    fold_rows = []
    attr_rows = []

    for fold_idx, fold_qkeys in enumerate(folds):
        fold_set = set(fold_qkeys)
        n_fold_q = len(fold_qkeys)

        for method_name, pred_df in method_data.items():
            merged = ground_truth_df.merge(
                pred_df,
                on=["qkey", "attribute", "group"],
                suffixes=("_gt", "_pred"),
                how="inner",
            )
            fold_merged = merged[merged["qkey"].isin(fold_set)]
            if fold_merged.empty:
                continue

            # Per-row distance
            attr_wds: Dict[str, list] = {}
            all_wds = []
            for _, row in fold_merged.iterrows():
                gt      = parse_dist(row["responses_gt"])
                pred    = parse_dist(row["responses_pred"])
                n_opt   = len(gt)
                ordinal = parse_ordinal(row.get("ordinal_gt"), n_options=n_opt)
                if ordinal_flags is None or row["qkey"] not in ordinal_flags:
                    raise KeyError(f"Missing is_ordinal metadata for qkey '{row['qkey']}'")
                is_ord = ordinal_flags[row["qkey"]]
                wd = compute_distance(gt, pred, ordinal, is_ord)
                if not np.isnan(wd):
                    all_wds.append(wd)
                    attr_wds.setdefault(row["attribute"], []).append(wd)

            n_covered = len(fold_merged["qkey"].unique())
            pct_covered = n_covered / n_fold_q if n_fold_q > 0 else 0.0

            if all_wds:
                fold_rows.append({
                    "method":           method_name,
                    "fold":             fold_idx + 1,
                    "fold_qkeys":       "|".join(sorted(fold_qkeys)),
                    "n_fold_questions": n_fold_q,
                    "n_covered_qkeys":  n_covered,
                    "pct_covered":      round(pct_covered, 3),
                    "n_rows":           len(all_wds),
                    "mean_wd":          float(np.mean(all_wds)),
                    "std_wd":           float(np.std(all_wds)),
                    "min_wd":           float(np.min(all_wds)),
                    "max_wd":           float(np.max(all_wds)),
                })

            for attr, wds in attr_wds.items():
                attr_rows.append({
                    "method":    method_name,
                    "fold":      fold_idx + 1,
                    "attribute": attr,
                    "n_rows":    len(wds),
                    "mean_wd":   float(np.mean(wds)),
                })

    return pd.DataFrame(fold_rows), pd.DataFrame(attr_rows), method_qkeys


# =========================================================================
# COVERAGE MANIFEST HELPER
# =========================================================================

def _coverage_row(
    method: str,
    scope: str,
    merged_df: pd.DataFrame,
) -> dict:
    """Build one row for the method_coverage.csv manifest."""
    if merged_df.empty:
        return {
            "method": method,
            "scope": scope,
            "n_rows": 0,
            "n_qkeys": 0,
            "qkeys": "",
            "n_attributes": 0,
            "attributes": "",
        }
    qkeys = sorted(merged_df["qkey"].unique().tolist())
    attrs = sorted(merged_df["attribute"].unique().tolist())
    return {
        "method": method,
        "scope": scope,
        "n_rows": len(merged_df),
        "n_qkeys": len(qkeys),
        "qkeys": "|".join(qkeys),
        "n_attributes": len(attrs),
        "attributes": "|".join(attrs),
    }


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full evaluation framework for the CMS SubPop replication"
    )
    parser.add_argument("--ground_truth_csv", type=str,
                        default=f"{DEFAULT_CMS_DIR}/cms_survey_distributions.csv")
    parser.add_argument("--questions_json", type=str,
                        default=f"{DEFAULT_CMS_DIR}/cms_questions.json",
                        help="Path to cms_questions.json (needed for ordinal vs nominal flag)")
    parser.add_argument("--question_split_json", type=str,
                        default=None,
                        help="Path to cms_question_split.json; enables fair test-only analysis")
    parser.add_argument("--predictions_dir", type=str, default=DEFAULT_CMS_DIR)
    parser.add_argument("--demographics_csv", type=str,
                        default=f"{DEFAULT_CMS_DIR}/cms_demographics.csv")
    parser.add_argument("--weights_csv", type=str,
                        default=f"{DEFAULT_CMS_DIR}/cms_subgroup_weights.csv")
    parser.add_argument("--output_dir", type=str, default=f"{DEFAULT_CMS_DIR}/evaluation")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_sensitivity_folds", type=int, default=5,
        help=(
            "Number of question-subset folds for post-hoc stability analysis "
            "(0 to disable). This is post-hoc sensitivity analysis, not cross-validation; "
            "the model is fixed and folds measure prediction variance across question subsets."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(args.predictions_dir)

    # Create scope-specific output subdirectories
    all_dir = out_dir / "descriptive_all_rows"
    fair_dir = out_dir / "test_only_fair"
    all_dir.mkdir(exist_ok=True)
    fair_dir.mkdir(exist_ok=True)

    # Load ordinal flags (ordinal vs nominal per question)
    ordinal_flags = load_ordinal_flags(args.questions_json)
    n_ordinal = sum(ordinal_flags.values())
    n_nominal = len(ordinal_flags) - n_ordinal
    print(f"Question types: {n_ordinal} ordinal (WD), {n_nominal} nominal (TVD)")

    # Load ground truth
    print("Loading ground truth...")
    gt_df = load_distributions(args.ground_truth_csv)
    print(f"  {len(gt_df)} distributions, {gt_df['attribute'].nunique()} attributes, "
          f"{gt_df['qkey'].nunique()} questions")

    # Discover prediction files
    prediction_files = {}
    for pattern, label in [
        ("distributions_QA.csv", "Zero-shot QA"),
        ("distributions_BIO.csv", "Zero-shot BIO"),
        ("distributions_PORTRAY.csv", "Zero-shot PORTRAY"),
        ("results_finetuned_QA.csv", "Fine-tuned QA"),
        ("results_finetuned_BIO.csv", "Fine-tuned BIO"),
        ("results_finetuned_PORTRAY.csv", "Fine-tuned PORTRAY"),
        # Sequential fine-tuning: pretrain on Pew waves → CMS fine-tune
        ("results_finetuned_QA_seq.csv", "Fine-tuned QA (seq)"),
        ("results_finetuned_BIO_seq.csv", "Fine-tuned BIO (seq)"),
        ("results_finetuned_PORTRAY_seq.csv", "Fine-tuned PORTRAY (seq)"),
    ]:
        path = pred_dir / pattern
        if path.exists():
            prediction_files[label] = str(path)
            print(f"  Found: {label} → {path}")

    if not prediction_files:
        raise FileNotFoundError(f"No prediction files found in {pred_dir}")

    # Load test qkeys for fair-scope analysis
    test_qkeys = None
    split_path = args.question_split_json or str(pred_dir / "cms_question_split.json")
    if Path(split_path).exists():
        with open(split_path) as f:
            split_data = json.load(f)
        test_qkeys = set(split_data.get("test", []))
        print(f"\n  Loaded question split — {len(test_qkeys)} test qkeys: {sorted(test_qkeys)}")
    else:
        raise FileNotFoundError(
            f"cms_question_split.json not found: {split_path}\n"
            "Pass --question_split_json or ensure cms_question_split.json is in predictions_dir."
        )

    # Coverage manifest rows
    coverage_rows = []

    # Load weights once (used in every scope × method iteration)
    if not Path(args.weights_csv).exists():
        raise FileNotFoundError(f"Subgroup weights file not found: {args.weights_csv}")
    weights_df = pd.read_csv(args.weights_csv)

    # -------------------------------------------------------------------------
    # Helper: run all per-method analyses for one scope
    # -------------------------------------------------------------------------
    def run_scope(scope_dir: Path, scope_label: str, qkey_filter: Optional[Set[str]]):
        """
        Run bootstrap CI, per-attribute distance, entropy, and population-weighted
        analyses for all prediction files under the given qkey filter, writing
        results to scope_dir.
        """
        scope_dir.mkdir(exist_ok=True)
        heatmap_dir = scope_dir / "disagreement_heatmaps"
        heatmap_dir.mkdir(exist_ok=True)

        # Ground truth disagreement heatmaps (scoped)
        gt_heatmaps = intergroup_disagreement(
            None, ground_truth_df=gt_df,
            ordinal_flags=ordinal_flags, filter_qkeys=qkey_filter,
        )
        for key, heatmap in gt_heatmaps.items():
            heatmap.to_csv(heatmap_dir / f"{key}.csv")

        for method_name, pred_path in prediction_files.items():
            safe_name = method_name.replace(" ", "_")

            try:
                pred_df = load_distributions(pred_path)
            except Exception as exc:
                raise RuntimeError(f"Could not load prediction file: {pred_path}") from exc

            # Build merge once for coverage stats
            merged_for_cov = gt_df.merge(
                pred_df, on=["qkey", "attribute", "group"],
                suffixes=("_gt", "_pred"), how="inner",
            )
            merged_for_cov = _filter_by_qkeys(merged_for_cov, qkey_filter)
            coverage_rows.append(_coverage_row(method_name, scope_label, merged_for_cov))

            # ── Bootstrap CI ────────────────────────────────────────────────
            ci_df = compute_bootstrap_ci(
                gt_df, pred_df,
                ordinal_flags=ordinal_flags,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
                filter_qkeys=qkey_filter,
            )
            ci_df.to_csv(scope_dir / f"bootstrap_ci_{safe_name}.csv", index=False)

            # ── Per-attribute distance ───────────────────────────────────────
            per_attr = per_attribute_wd(
                gt_df, pred_df,
                ordinal_flags=ordinal_flags,
                filter_qkeys=qkey_filter,
            )
            per_attr.to_csv(scope_dir / f"per_attribute_wd_{safe_name}.csv", index=False)

            # ── Entropy ──────────────────────────────────────────────────────
            ent_df = entropy_analysis(gt_df, pred_df, filter_qkeys=qkey_filter)
            ent_df.to_csv(scope_dir / f"entropy_{safe_name}.csv", index=False)

            # ── Predicted disagreement heatmaps ──────────────────────────────
            pred_heatmaps = intergroup_disagreement(
                pred_df, ordinal_flags=ordinal_flags, filter_qkeys=qkey_filter,
            )
            for key, heatmap in pred_heatmaps.items():
                heatmap.to_csv(heatmap_dir / f"{key}_{safe_name}.csv")

            # ── Population-weighted opinion ───────────────────────────────────
            weighted = population_weighted_opinion(
                pred_df, weights_df,
                filter_qkeys=qkey_filter,
                ordinal_flags=ordinal_flags,
            )
            weighted.to_csv(scope_dir / f"population_weighted_{safe_name}.csv", index=False)

        # Print quick summary for this scope
        if prediction_files:
            print(f"\n  [{scope_label}] Summary:")
            for method_name, pred_path in prediction_files.items():
                safe_name = method_name.replace(" ", "_")
                ci_path = scope_dir / f"bootstrap_ci_{safe_name}.csv"
                if ci_path.exists():
                    ci_df = pd.read_csv(ci_path)
                    if not ci_df.empty:
                        mean_wd = ci_df["wd_point"].mean()
                        ci_lo = ci_df["wd_ci_lower"].mean()
                        ci_hi = ci_df["wd_ci_upper"].mean()
                        n = len(ci_df)
                        print(f"    {method_name:25s}  dist={mean_wd:.4f}  "
                              f"95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]  n={n}"
                              f"  (WD for ordinal Qs, TVD for nominal Qs)")

    # ── Scope 1: all questions ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SCOPE: descriptive_all_rows  (all questions, all rows)")
    print(f"{'='*60}")
    run_scope(all_dir, "all_questions", qkey_filter=None)

    # ── Scope 2: test questions only ────────────────────────────────────────
    if test_qkeys:
        print(f"\n{'='*60}")
        print("SCOPE: test_only_fair  (test questions only — fair comparison)")
        print(f"{'='*60}")
        run_scope(fair_dir, "test_questions", qkey_filter=test_qkeys)

    # ── Ablation table (both scopes combined in one CSV) ────────────────────
    print(f"\n{'='*60}")
    print("ABLATION COMPARISON TABLE (SubPop Table 1 style)")
    print(f"{'='*60}")

    bounds_path = pred_dir / "statistical_bounds.csv"
    bounds_csv_arg = str(bounds_path) if bounds_path.exists() else None

    # Pass 1: all questions
    ablation_all = build_ablation_table(
        gt_df, prediction_files,
        bounds_csv=bounds_csv_arg,
        ordinal_flags=ordinal_flags,
        filter_qkeys=None,
    )
    ablation_all.columns = [
        f"{c} (all Qs)" if c not in ("Uniform (upper)", "Bootstrap (lower)") else c
        for c in ablation_all.columns
    ]

    # Pass 2: test questions only
    if test_qkeys:
        ablation_test = build_ablation_table(
            gt_df, prediction_files,
            bounds_csv=None,
            ordinal_flags=ordinal_flags,
            filter_qkeys=test_qkeys,
        )
        ablation_test.columns = [
            f"{c} (test Qs)" if c not in ("Uniform (upper)", "Bootstrap (lower)") else c
            for c in ablation_test.columns
        ]
        drop_cols = [c for c in ablation_test.columns if c in ("Uniform (upper)", "Bootstrap (lower)")]
        ablation_test = ablation_test.drop(columns=drop_cols, errors="ignore")
        ablation = pd.concat([ablation_all, ablation_test], axis=1)
    else:
        ablation = ablation_all

    ablation.to_csv(out_dir / "ablation_table.csv")
    print(ablation.round(4).to_string())

    # ── Question-subset sensitivity analysis ────────────────────────────────
    if prediction_files and args.n_sensitivity_folds > 0:
        print(f"\n{'='*60}")
        print("QUESTION-SUBSET SENSITIVITY ANALYSIS")
        print("Post-hoc sensitivity analysis with a single fixed model.")
        print("   Measures prediction stability across question subsets.")
        print("   For unseen-question evaluation see: test_only_fair/")
        print(f"   n_folds={args.n_sensitivity_folds}  seed={args.seed}")
        print(f"{'='*60}")

        sens_dir = out_dir / "question_subset_sensitivity"
        sens_dir.mkdir(exist_ok=True)

        fold_df, attr_df, method_qkeys = question_subset_sensitivity(
            gt_df,
            prediction_files,
            ordinal_flags=ordinal_flags,
            n_folds=args.n_sensitivity_folds,
            seed=args.seed,
        )

        if not fold_df.empty:
            fold_df.to_csv(sens_dir / "fold_results.csv", index=False)
            attr_df.to_csv(sens_dir / "fold_results_by_attribute.csv", index=False)

            # ── Coverage heterogeneity check ─────────────────────────────────
            n_gt_qkeys = gt_df["qkey"].nunique()
            coverage_pcts = {
                m: len(qs) / n_gt_qkeys
                for m, qs in method_qkeys.items()
            }
            min_cov = min(coverage_pcts.values())
            max_cov = max(coverage_pcts.values())
            coverage_inconsistent = (max_cov - min_cov) > 0.20

            if coverage_inconsistent:
                print("\n  COVERAGE WARNING: methods cover different question subsets.")
                print("    Across-method mean_wd comparisons in this section are misleading.")
                print("    Use test_only_fair/ for valid cross-method comparisons.")
                for m, pct in sorted(coverage_pcts.items()):
                    n_q = len(method_qkeys.get(m, set()))
                    print(f"    {m:<30} {pct*100:5.1f}% of GT qkeys ({n_q}/{n_gt_qkeys})")

            # ── Per-method summary across folds ──────────────────────────────
            # Only use folds where this method had meaningful coverage (>0 rows)
            summary_rows = []
            for method in fold_df["method"].unique():
                mdf = fold_df[fold_df["method"] == method]
                mean_wd = mdf["mean_wd"].mean()
                std_wd  = mdf["mean_wd"].std()
                n_f     = len(mdf)
                avg_cov = mdf["pct_covered"].mean()
                summary_rows.append({
                    "method":           method,
                    "across_fold_mean": round(float(mean_wd), 6),
                    "across_fold_std":  round(float(std_wd), 6) if not np.isnan(std_wd) else float("nan"),
                    "n_folds":          n_f,
                    "avg_pct_covered":  round(float(avg_cov), 3),
                })
            summary = pd.DataFrame(summary_rows).sort_values("across_fold_mean")
            summary.to_csv(sens_dir / "sensitivity_summary.csv", index=False)

            print("\nPer-method stability (interpret within method only if coverage differs):")
            cov_note = " [avg% of fold qkeys covered]" if coverage_inconsistent else ""
            print(f"  {'Method':<30} {'Mean dist':>9}  {'± Std':>7}  {'Folds':>5}  {'Cov%':>6}{cov_note}")
            print("  " + "-" * 65)
            for _, row in summary.iterrows():
                std_str = f"{row['across_fold_std']:.4f}" if not np.isnan(row["across_fold_std"]) else "   n/a"
                print(f"  {row['method']:<30} {row['across_fold_mean']:>8.4f}  "
                      f"±{std_str:>6}  {int(row['n_folds']):>5}  {row['avg_pct_covered']*100:>5.1f}%")

            # ── Per-fold distance pivot ───────────────────────────────────────
            print(f"\nPer-fold distance breakdown (seed={args.seed}):")
            pivot = fold_df.pivot(index="method", columns="fold", values="mean_wd")
            pivot.columns = [f"fold_{c}" for c in pivot.columns]
            print(pivot.round(4).to_string())
            pivot.to_csv(sens_dir / "fold_pivot.csv")

            # ── Common-qkeys cross-method comparison ─────────────────────────
            if len(method_qkeys) > 1:
                common_qkeys = set.intersection(*method_qkeys.values()) if method_qkeys else set()
                if len(common_qkeys) >= args.n_sensitivity_folds:
                    print(f"\nCommon-qkeys cross-method comparison "
                          f"({len(common_qkeys)} qkeys shared by all methods):")
                    common_summary_rows = []
                    for method in fold_df["method"].unique():
                        mdf = fold_df[fold_df["method"] == method]
                        # Rows where all qkeys in fold are in common_qkeys
                        # proxy: avg pct_covered == 1.0 (meaning fold was fully within common)
                        full_folds = mdf[mdf["pct_covered"] >= 1.0]
                        if full_folds.empty:
                            continue
                        common_summary_rows.append({
                            "method": method,
                            "common_qkeys_mean_wd": float(full_folds["mean_wd"].mean()),
                            "common_qkeys_std_wd":  float(full_folds["mean_wd"].std()),
                            "n_full_folds":         len(full_folds),
                        })
                    if common_summary_rows:
                        common_df = pd.DataFrame(common_summary_rows).sort_values("common_qkeys_mean_wd")
                        common_df.to_csv(sens_dir / "common_qkeys_summary.csv", index=False)
                        print(f"  {'Method':<30} {'Mean dist':>9}  {'± Std':>7}  {'Full folds':>10}")
                        print("  " + "-" * 60)
                        for _, row in common_df.iterrows():
                            std_str = f"{row['common_qkeys_std_wd']:.4f}" if not np.isnan(row["common_qkeys_std_wd"]) else "   n/a"
                            print(f"  {row['method']:<30} {row['common_qkeys_mean_wd']:>8.4f}  "
                                  f"±{std_str:>6}  {int(row['n_full_folds']):>10}")
                elif len(common_qkeys) > 0:
                    print(f"\n  Note: only {len(common_qkeys)} qkeys shared across all methods "
                          f"(need ≥{args.n_sensitivity_folds} for fold analysis) — "
                          f"cross-method comparison skipped.")
                    pd.DataFrame({"common_qkeys": sorted(common_qkeys)}).to_csv(
                        sens_dir / "common_qkeys.csv", index=False)
                else:
                    print("\n  Note: no qkeys shared across all methods — "
                          "cross-method comparison not possible.")

            print(f"\n  Outputs → {sens_dir}/")
            print(f"    fold_results.csv              — per-(method,fold) with coverage cols")
            print(f"    fold_results_by_attribute.csv — per-(method,fold,attribute)")
            print(f"    sensitivity_summary.csv        — mean±std + avg coverage per method")
            print(f"    fold_pivot.csv                 — method × fold distance matrix")
            if len(method_qkeys) > 1:
                print(f"    common_qkeys_summary.csv       — cross-method on shared qkeys only")
        else:
            print("  No data produced (no matching prediction files or ground truth rows).")

    # ── Method coverage manifest ────────────────────────────────────────────
    if coverage_rows:
        coverage_df = pd.DataFrame(coverage_rows)
        # Also add ground truth row
        gt_cov_all = _coverage_row("ground_truth", "all_questions", gt_df)
        if test_qkeys:
            gt_cov_test = _coverage_row("ground_truth", "test_questions",
                                        gt_df[gt_df["qkey"].isin(test_qkeys)])
            coverage_df = pd.concat(
                [pd.DataFrame([gt_cov_all, gt_cov_test]), coverage_df],
                ignore_index=True,
            )
        else:
            coverage_df = pd.concat(
                [pd.DataFrame([gt_cov_all]), coverage_df],
                ignore_index=True,
            )
        coverage_df.to_csv(out_dir / "method_coverage.csv", index=False)
        print(f"\nMethod coverage manifest saved to {out_dir}/method_coverage.csv")
        print(coverage_df[["method", "scope", "n_rows", "n_qkeys", "n_attributes"]].to_string(index=False))

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"All outputs saved to {out_dir}/")
    print(f"\nKey files:")
    print(f"  ablation_table.csv                      — SubPop Table 1 (all Qs + test Qs columns)")
    if coverage_rows:
        print(f"  method_coverage.csv                     — audit: n_rows/n_qkeys per method × scope")
    print(f"  descriptive_all_rows/                   — all questions (full characterization)")
    print(f"    bootstrap_ci_*.csv                    — distance metric with 95% CI")
    print(f"    per_attribute_wd_*.csv                — breakdown by demographic attribute")
    print(f"    entropy_*.csv                         — response diversity analysis")
    print(f"    population_weighted_*.csv             — population-weighted aggregate opinion")
    print(f"    disagreement_heatmaps/                — pairwise inter-group disagreement")
    if test_qkeys:
        print(f"  test_only_fair/                         — test questions only (fair comparison)")
        print(f"    (same file structure as descriptive_all_rows/)")
    if prediction_files and args.n_sensitivity_folds > 0:
        print(f"  question_subset_sensitivity/            — post-hoc stability analysis")
        print(f"    fold_results.csv                      — per-(method, fold) mean distance")
        print(f"    sensitivity_summary.csv               — mean±std across folds per method")
        print(f"    fold_pivot.csv                        — method × fold distance matrix")


if __name__ == "__main__":
    main()
