"""
step6_full_evaluation.py

Phase 7: Comprehensive evaluation framework matching SubPop paper quality.

Upgrades over old step6_analyze_results.ipynb:
  1. Bootstrap confidence intervals on WD  (SubPop Figure 3)
  2. Per-attribute WD breakdown            (SubPop Tables 9-10)
  3. Entropy analysis                      (response diversity check)
  4. Inter-group disagreement heatmaps     (SubPop Figure 4)
  5. Ablation comparison table             (SubPop Table 1 style)
  6. Generalization analysis               (unseen questions + unseen subgroups)
  7. Population-weighted aggregate opinion

Outputs (under OUT_DIR):
    evaluation_summary.csv       - main results table
    per_attribute_wd.csv         - WD breakdown by attribute
    bootstrap_ci.csv             - bootstrap confidence intervals
    entropy_analysis.csv         - entropy of predicted vs ground truth
    disagreement_heatmaps/       - one CSV per attribute
    ablation_table.csv           - full ablation comparison (SubPop Table 1)
    population_weighted.csv      - weighted aggregate opinions

Usage:
    python step6_full_evaluation.py \
        --ground_truth_csv  approach2_outputs/cms/cms_survey_distributions.csv \
        --questions_json    approach2_outputs/cms/cms_questions.json \
        --predictions_dir   approach2_outputs/cms \
        --weights_csv       approach2_outputs/cms/cms_subgroup_weights.csv \
        --output_dir        approach2_outputs/cms/evaluation
"""

import argparse
import ast
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

# Add SubPop to path
SUBPOP_ROOT = Path(__file__).parent / "subpop-main"
sys.path.insert(0, str(SUBPOP_ROOT))

from subpop.utils.survey_utils import ordinal_emd, get_entropy, list_normalize


# =========================================================================
# HELPERS
# =========================================================================

def load_ordinal_flags(questions_json_path: str) -> dict:
    """Return {qkey: is_ordinal} from cms_questions.json."""
    try:
        with open(questions_json_path, "r") as f:
            questions = json.load(f)
        return {q["qkey"]: bool(q.get("is_ordinal", True)) for q in questions}
    except FileNotFoundError:
        print(f"  WARNING: {questions_json_path} not found — treating all questions as ordinal")
        return {}


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
        # NumPy format: brackets with space-separated floats, no commas
        import re
        nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', x)
        if nums:
            return [float(n) for n in nums]
        raise ValueError(f"Cannot parse distribution: {x!r}")


def parse_ordinal(x, n_options: int = None):
    """Parse ordinal values. Falls back to [1, 2, ..., n_options] if parsing fails."""
    if isinstance(x, str):
        try:
            result = ast.literal_eval(x)
            if isinstance(result, list) and len(result) > 0:
                return result
        except Exception:
            pass
    if isinstance(x, list) and len(x) > 0:
        return x
    # Fallback: use n_options if provided, otherwise 5
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
    if "ordinal" in df.columns:
        df["ordinal_parsed"] = df.apply(
            lambda r: parse_ordinal(r["ordinal"], n_options=len(parse_dist(r["responses"]))), axis=1
        )
    else:
        df["ordinal_parsed"] = df["responses_parsed"].apply(lambda d: list(range(1, len(d) + 1)))
    return df


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
) -> pd.DataFrame:
    """
    Compute bootstrap CI for WD between predicted and ground truth distributions.
    For each (attribute, group, question) pair, resample and compute WD.
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

    for _, row in merged.iterrows():
        gt = parse_dist(row["responses_gt"])
        pred = parse_dist(row["responses_pred"])
        n_options = len(gt)
        ordinal_raw = row.get("ordinal_gt") or row.get("ordinal_pred")
        ordinal = parse_ordinal(ordinal_raw, n_options=n_options)
        is_ord = (ordinal_flags or {}).get(row["qkey"], True)

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
# 2. PER-ATTRIBUTE WD BREAKDOWN
# =========================================================================

def per_attribute_wd(
    ground_truth_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    ordinal_flags: dict = None,
) -> pd.DataFrame:
    """
    Compute mean WD per attribute (SubPop Tables 9-10).
    Shows which demographic dimensions the model captures best/worst.
    """
    merged = ground_truth_df.merge(
        predictions_df,
        on=["qkey", "attribute", "group"],
        suffixes=("_gt", "_pred"),
        how="inner",
    )

    rows = []
    for _, row in merged.iterrows():
        gt = parse_dist(row["responses_gt"])
        pred = parse_dist(row["responses_pred"])
        n_options = len(gt)
        ordinal = parse_ordinal(row.get("ordinal_gt"), n_options=n_options)
        is_ord = (ordinal_flags or {}).get(row["qkey"], True)
        wd = compute_distance(gt, pred, ordinal, is_ord)
        rows.append({
            "attribute": row["attribute"],
            "group": row["group"],
            "qkey": row["qkey"],
            "wd": wd,
        })

    detail_df = pd.DataFrame(rows)

    # Aggregate: mean WD per attribute, per group, and overall
    per_attr = detail_df.groupby("attribute")["wd"].agg(["mean", "std", "count"]).reset_index()
    per_attr.columns = ["attribute", "mean_wd", "std_wd", "n_pairs"]

    per_group = detail_df.groupby(["attribute", "group"])["wd"].agg(["mean", "count"]).reset_index()
    per_group.columns = ["attribute", "group", "mean_wd", "n_pairs"]

    return per_attr, per_group, detail_df


# =========================================================================
# 3. ENTROPY ANALYSIS
# =========================================================================

def entropy_analysis(
    ground_truth_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare entropy of predicted vs ground truth distributions.
    Checks if model preserves response diversity or collapses.
    """
    merged = ground_truth_df.merge(
        predictions_df,
        on=["qkey", "attribute", "group"],
        suffixes=("_gt", "_pred"),
        how="inner",
    )

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
) -> Dict[str, pd.DataFrame]:
    """
    For each attribute, compute pairwise WD between groups on the same question.
    Returns dict of attribute → DataFrame heatmaps.
    Mirrors SubPop Figure 4.
    """
    heatmaps = {}

    for df, label in [(predictions_df, "pred"), (ground_truth_df, "gt")]:
        if df is None:
            continue

        for attr in df["attribute"].unique():
            attr_df = df[df["attribute"] == attr]
            groups = sorted(attr_df["group"].unique())

            if len(groups) < 2:
                continue

            # Compute average pairwise WD across all questions
            n_questions = attr_df["qkey"].nunique()
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
                is_ord = (ordinal_flags or {}).get(qkey, True)

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
    filter_qkeys: Optional[set] = None,
) -> pd.DataFrame:
    """
    Build SubPop Table 1-style ablation table.
    Rows = attributes, Columns = methods.

    filter_qkeys: if provided, only compute WD on this subset of questions.
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
            if filter_qkeys is not None:
                merged = merged[merged["qkey"].isin(filter_qkeys)]
            method_wds = {}
            for _, row in merged.iterrows():
                gt = parse_dist(row["responses_gt"])
                pred = parse_dist(row["responses_pred"])
                n_options = len(gt)
                ordinal = parse_ordinal(row.get("ordinal_gt"), n_options=n_options)
                is_ord = (ordinal_flags or {}).get(row["qkey"], True)
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
) -> pd.DataFrame:
    """
    Compute population-weighted aggregate opinion per question.
    Uses PUMS PWGTP weights for proper population representation.
    """
    merged = predictions_df.merge(
        weights_df,
        on=["attribute", "group"],
        how="left",
    )

    rows = []
    for qkey in merged["qkey"].unique():
        q_df = merged[merged["qkey"] == qkey]
        question = q_df.iloc[0].get("question", qkey)

        # Determine number of options from first valid distribution
        first_dist = parse_dist(q_df.iloc[0]["responses"])
        n_options = len(first_dist)

        # Weighted average distribution across all subgroups
        total_weight = 0.0
        weighted_dist = np.zeros(n_options)

        for _, row in q_df.iterrows():
            dist = parse_dist(row["responses"])
            # Guard against mismatched lengths (shouldn't happen within a qkey)
            if len(dist) != n_options:
                dist = dist[:n_options] + [0.0] * max(0, n_options - len(dist))
            w = float(row.get("weight", row.get("pop_share", 1.0)))
            weighted_dist += np.array(dist) * w
            total_weight += w

        if total_weight > 0:
            weighted_dist /= total_weight

        # Mean score on a 1-to-n_options ordinal scale
        mean_score = sum((i + 1) * p for i, p in enumerate(weighted_dist))

        record = {
            "qkey": qkey,
            "question": question,
            "n_options": n_options,
            "weighted_distribution": str(weighted_dist.tolist()),
            "mean_opinion_score": mean_score,
        }
        # Generic per-option columns: option_1, option_2, ..., option_n
        for i, p in enumerate(weighted_dist):
            record[f"option_{i + 1}"] = p
        rows.append(record)

    return pd.DataFrame(rows)


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full evaluation framework (SubPop paper quality)"
    )
    parser.add_argument("--ground_truth_csv", type=str,
                        default="approach2_outputs/cms/cms_survey_distributions.csv")
    parser.add_argument("--questions_json", type=str,
                        default="approach2_outputs/cms/cms_questions.json",
                        help="Path to cms_questions.json (needed for ordinal vs nominal flag)")
    parser.add_argument("--question_split_json", type=str,
                        default=None,
                        help="Path to cms_question_split.json; enables fair test-only ablation columns")
    parser.add_argument("--predictions_dir", type=str, default="approach2_outputs/cms")
    parser.add_argument("--demographics_csv", type=str,
                        default="approach2_outputs/cms/cms_demographics.csv")
    parser.add_argument("--weights_csv", type=str,
                        default="approach2_outputs/cms/cms_subgroup_weights.csv")
    parser.add_argument("--output_dir", type=str, default="approach2_outputs/cms/evaluation")
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(args.predictions_dir)

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
    ]:
        path = pred_dir / pattern
        if path.exists():
            prediction_files[label] = str(path)
            print(f"  Found: {label} → {path}")

    if not prediction_files:
        print("  No prediction files found.")
        print("  Zero-shot results: run step2_vllm_baselines.py --mode zeroshot")
        print("  Fine-tuned results: run subpop-main/scripts/experiment/run_inference.py")
        print("  Will compute evaluation metrics on ground truth only.")

    # === 1. Bootstrap CI ===
    print(f"\n{'='*60}")
    print("1. BOOTSTRAP CONFIDENCE INTERVALS")
    print(f"{'='*60}")

    for method_name, pred_path in prediction_files.items():
        print(f"\n  Computing bootstrap CI for: {method_name}")
        pred_df = load_distributions(pred_path)
        ci_df = compute_bootstrap_ci(gt_df, pred_df, ordinal_flags=ordinal_flags, n_bootstrap=args.n_bootstrap, seed=args.seed)
        ci_path = out_dir / f"bootstrap_ci_{method_name.replace(' ', '_')}.csv"
        ci_df.to_csv(ci_path, index=False)
        print(f"  Mean WD: {ci_df['wd_point'].mean():.4f} "
              f"[{ci_df['wd_ci_lower'].mean():.4f}, {ci_df['wd_ci_upper'].mean():.4f}]")

    # === 2. Per-Attribute WD ===
    print(f"\n{'='*60}")
    print("2. PER-ATTRIBUTE WD BREAKDOWN")
    print(f"{'='*60}")

    for method_name, pred_path in prediction_files.items():
        pred_df = load_distributions(pred_path)
        per_attr, per_group, detail = per_attribute_wd(gt_df, pred_df, ordinal_flags=ordinal_flags)
        per_attr.to_csv(out_dir / f"per_attribute_wd_{method_name.replace(' ', '_')}.csv", index=False)
        print(f"\n  {method_name}:")
        for _, row in per_attr.iterrows():
            print(f"    {row['attribute']:20s} WD={row['mean_wd']:.4f} (std={row['std_wd']:.4f}, n={int(row['n_pairs'])})")

    # === 3. Entropy Analysis ===
    print(f"\n{'='*60}")
    print("3. ENTROPY ANALYSIS")
    print(f"{'='*60}")

    for method_name, pred_path in prediction_files.items():
        pred_df = load_distributions(pred_path)
        ent_df = entropy_analysis(gt_df, pred_df)
        ent_df.to_csv(out_dir / f"entropy_{method_name.replace(' ', '_')}.csv", index=False)
        mean_diff = ent_df["entropy_diff"].mean()
        direction = "more peaked" if mean_diff < 0 else "more diverse"
        print(f"  {method_name}: avg entropy diff = {mean_diff:.4f} (predictions are {direction})")

    # === 4. Inter-Group Disagreement ===
    print(f"\n{'='*60}")
    print("4. INTER-GROUP DISAGREEMENT HEATMAPS")
    print(f"{'='*60}")

    heatmap_dir = out_dir / "disagreement_heatmaps"
    heatmap_dir.mkdir(exist_ok=True)

    # Ground truth disagreement — pass gt_df as ground_truth_df so keys get "_gt" suffix
    gt_heatmaps = intergroup_disagreement(None, ground_truth_df=gt_df, ordinal_flags=ordinal_flags)
    for key, heatmap in gt_heatmaps.items():
        heatmap.to_csv(heatmap_dir / f"{key}.csv")
        print(f"  {key}: {heatmap.shape[0]} groups, max disagreement = {heatmap.max().max():.4f}")

    # Predicted disagreement
    for method_name, pred_path in prediction_files.items():
        pred_df = load_distributions(pred_path)
        pred_heatmaps = intergroup_disagreement(pred_df, ordinal_flags=ordinal_flags)
        for key, heatmap in pred_heatmaps.items():
            safe_name = method_name.replace(" ", "_")
            heatmap.to_csv(heatmap_dir / f"{key}_{safe_name}.csv")

    # === 5. Ablation Table ===
    print(f"\n{'='*60}")
    print("5. ABLATION COMPARISON TABLE (SubPop Table 1)")
    print(f"{'='*60}")

    bounds_path = pred_dir / "statistical_bounds.csv"
    bounds_csv_arg = str(bounds_path) if bounds_path.exists() else None

    # Load test qkeys for fair comparison (zero-shot vs fine-tuned on same Qs)
    test_qkeys = None
    split_path = args.question_split_json or str(pred_dir / "cms_question_split.json")
    if Path(split_path).exists():
        with open(split_path) as f:
            split_data = json.load(f)
        test_qkeys = set(split_data.get("test", []))
        print(f"  Loaded question split — test qkeys: {sorted(test_qkeys)}")
    else:
        print("  NOTE: --question_split_json not found; skipping test-only ablation columns.")

    # Pass 1: all questions (full zero-shot coverage)
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

    # Pass 2: test questions only (fair apples-to-apples with fine-tuned)
    if test_qkeys:
        ablation_test = build_ablation_table(
            gt_df, prediction_files,
            bounds_csv=None,           # bounds already in pass 1
            ordinal_flags=ordinal_flags,
            filter_qkeys=test_qkeys,
        )
        ablation_test.columns = [
            f"{c} (test Qs)" if c not in ("Uniform (upper)", "Bootstrap (lower)") else c
            for c in ablation_test.columns
        ]
        # Drop duplicated bound columns from pass 2 before merging
        drop_cols = [c for c in ablation_test.columns if c in ("Uniform (upper)", "Bootstrap (lower)")]
        ablation_test = ablation_test.drop(columns=drop_cols, errors="ignore")
        ablation = pd.concat([ablation_all, ablation_test], axis=1)
    else:
        ablation = ablation_all

    ablation.to_csv(out_dir / "ablation_table.csv")
    print(ablation.round(4).to_string())

    # === 6. Population-Weighted Aggregates ===
    print(f"\n{'='*60}")
    print("6. POPULATION-WEIGHTED AGGREGATE OPINION")
    print(f"{'='*60}")

    if Path(args.weights_csv).exists():
        weights_df = pd.read_csv(args.weights_csv)
        for method_name, pred_path in prediction_files.items():
            pred_df = load_distributions(pred_path)
            weighted = population_weighted_opinion(pred_df, weights_df)
            safe_name = method_name.replace(" ", "_")
            weighted.to_csv(out_dir / f"population_weighted_{safe_name}.csv", index=False)
            print(f"\n  {method_name}:")
            for _, row in weighted.iterrows():
                print(f"    {row['qkey']}: mean_score={row['mean_opinion_score']:.2f}  "
                      f"({row['question'][:50]}...)")
    else:
        print(f"  WARNING: {args.weights_csv} not found, skipping population-weighted analysis")

    # === Summary ===
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"All outputs saved to {out_dir}/")
    print(f"\nKey files:")
    print(f"  ablation_table.csv            - SubPop Table 1 style comparison")
    print(f"  per_attribute_wd_*.csv        - Tables 9-10 style breakdown")
    print(f"  bootstrap_ci_*.csv            - Figure 3 style CI")
    print(f"  disagreement_heatmaps/        - Figure 4 style heatmaps")
    print(f"  entropy_*.csv                 - diversity analysis")
    print(f"  population_weighted_*.csv     - weighted aggregate opinions")


if __name__ == "__main__":
    main()
