"""
step2_vllm_baselines.py

Phase 4 of the upgraded pipeline: multi-format zero-shot baselines + statistical bounds.

Features:
  - All 3 SubPop steering prompt formats (QA, BIO, PORTRAY)
  - Per-attribute steering (not composite commuter_profile)
  - Auto-fallback from Llama-3.1-8B to Mistral-7B if access is pending
  - Statistical bounds (uniform upper, bootstrap lower)

Outputs (under OUT_DIR):
    distributions_QA.csv      - zero-shot with QA steering
    distributions_BIO.csv     - zero-shot with BIO steering
    distributions_PORTRAY.csv - zero-shot with PORTRAY steering
    statistical_bounds.csv    - uniform upper + bootstrap lower bounds

Usage:
    # Zero-shot in all 3 formats (requires GPU + vLLM):
    python step2_vllm_baselines.py \
        --mode zeroshot \
        --demographics_csv  approach2_outputs/cms/cms_demographics.csv \
        --steering_json     approach2_outputs/cms/cms_steering_prompts.json \
        --questions_json    approach2_outputs/cms/cms_questions.json \
        --output_dir        approach2_outputs/cms

    # Bounds only (--ground_truth_csv is required):
    python step2_vllm_baselines.py \
        --mode bounds \
        --ground_truth_csv  approach2_outputs/cms/cms_survey_distributions.csv \
        --output_dir        approach2_outputs/cms

Supported modes: zeroshot, bounds, all.
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

# =========================================================================
# Import SubPop utilities (add to path if needed)
# =========================================================================
SUBPOP_ROOT = Path(__file__).parent / "subpop-main"
sys.path.insert(0, str(SUBPOP_ROOT))

from subpop.utils.survey_utils import generate_mcq, ordinal_emd, list_normalize


# =========================================================================
# CONSTANTS
# =========================================================================

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G"]
ORDINAL_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0]  # fallback only


# =========================================================================
# METRIC HELPERS
# =========================================================================

def tvd(p: list, q: list) -> float:
    """Total Variation Distance for nominal (unordered) options."""
    return sum(abs(pi - qi) for pi, qi in zip(p, q)) / 2


def load_ordinal_flags(questions_json: str) -> dict:
    """
    Return {qkey: bool} where True = ordinal (use WD), False = nominal (use TVD).
    Defaults to True (ordinal/WD) if is_ordinal key is absent.
    """
    with open(questions_json) as f:
        questions = json.load(f)
    return {q["qkey"]: bool(q.get("is_ordinal", True)) for q in questions}


def bound_distance(gt_dist: list, pred_dist: list,
                   ordinal: list, is_ordinal: bool) -> float:
    """Dispatch to WD (ordinal) or TVD (nominal)."""
    if is_ordinal:
        return ordinal_emd(gt_dist, pred_dist, ordinal)
    return tvd(gt_dist, pred_dist)

# Preferred model → fallback model (used if HuggingFace access is still pending)
MODEL_PREFERENCE = [
    "meta-llama/Llama-3.1-8B",
    "mistralai/Mistral-7B-v0.1",
]


def resolve_model(requested_model: str) -> str:
    """
    Try to load the requested model. If it fails due to gated/pending access,
    fall back through MODEL_PREFERENCE until one succeeds.
    Returns the model name that was successfully resolved.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    candidates = [requested_model]
    # Append fallbacks that aren't already in the list
    for m in MODEL_PREFERENCE:
        if m != requested_model:
            candidates.append(m)

    for model_name in candidates:
        try:
            print(f"  Checking access to {model_name} ...")
            snapshot_download(model_name, ignore_patterns=["*.safetensors", "*.bin"],
                              local_files_only=False)
            print(f"  Access confirmed: using {model_name}")
            return model_name
        except GatedRepoError:
            print(f"  Access PENDING/DENIED for {model_name} — trying next model")
        except RepositoryNotFoundError:
            print(f"  Model not found: {model_name} — trying next model")
        except Exception as e:
            # Network error, quota, etc. — don't silently swallow
            if "gated" in str(e).lower() or "access" in str(e).lower():
                print(f"  Access issue for {model_name}: {e} — trying next model")
            else:
                raise

    raise RuntimeError(
        f"No accessible model found. Tried: {candidates}\n"
        "Request access at https://huggingface.co/meta-llama or use "
        "--model_name mistralai/Mistral-7B-v0.1 directly."
    )


# =========================================================================
# PROMPT BUILDER
# =========================================================================

def build_full_prompt(steering_text: str, prompt_format: str, question: str, options: list) -> str:
    """Append survey question MCQ to the pre-built demographic steering text from step1."""
    survey_mcq = generate_mcq(question_body=question, options=options, add_answer_forcing=True)
    if prompt_format == "QA":
        return steering_text + "\n\nAnswer the following question keeping in mind your previous answers.\n" + survey_mcq
    else:
        return steering_text + "\n\n" + survey_mcq


# =========================================================================
# LOGPROB EXTRACTION (reused from old step2)
# =========================================================================

def extract_distribution_from_logprobs(logprobs_dict: dict, tokenizer, n_options: int = 5) -> list:
    """
    Extract probabilities for A-E (or more) from vLLM logprobs.
    Handles multiple tokenizer surface forms (" A", "A", "\\nA").
    """
    letters = OPTION_LETTERS[:n_options]
    letter_to_ids = {}
    for letter in letters:
        ids = set()
        for variant in [letter, f" {letter}", f"\n{letter}"]:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if encoded:
                ids.add(encoded[-1])
        letter_to_ids[letter] = ids

    raw_logprobs = []
    for letter in letters:
        best_logprob = -float("inf")
        for token_id in letter_to_ids[letter]:
            if token_id in logprobs_dict:
                lp = logprobs_dict[token_id].logprob
                if lp > best_logprob:
                    best_logprob = lp
        if best_logprob == -float("inf"):
            best_logprob = -20.0
        raw_logprobs.append(best_logprob)

    raw_probs = np.exp(raw_logprobs)
    total = raw_probs.sum()
    if total <= 0:
        return [1 / n_options] * n_options
    return (raw_probs / total).tolist()


# =========================================================================
# ZERO-SHOT INFERENCE
# =========================================================================

def run_zero_shot(
    demographics: pd.DataFrame,
    steering_prompts: dict,
    questions: list,
    model_name: str,
    tp_size: int,
    prompt_format: str,
    output_path: Path,
):
    """Run zero-shot inference for one prompt format."""
    from vllm import LLM, SamplingParams

    prompt_records = []
    for _, demo_row in demographics.iterrows():
        attr  = demo_row["attribute"]
        group = demo_row["group"]
        if attr not in steering_prompts:
            print(f"  WARNING: no steering for attribute '{attr}', skipping"); continue
        if group not in steering_prompts[attr]:
            print(f"  WARNING: no steering for {attr}={group}, skipping"); continue
        steering_text = steering_prompts[attr][group][prompt_format]
        for q in questions:
            options   = q.get("options", OPTION_LETTERS[:5])
            n_options = len(options)
            ordinal   = q.get("ordinal") or list(range(1, n_options + 1))
            prompt    = build_full_prompt(steering_text, prompt_format, q["question"], options)
            prompt_records.append({
                "prompt": prompt, "qkey": q["qkey"], "attribute": attr, "group": group,
                "question": q["question"], "options": options, "ordinal": ordinal, "n_options": n_options,
            })

    print(f"  Built {len(prompt_records)} prompts for {prompt_format} format")
    model_name = resolve_model(model_name)  # falls back to Mistral if Llama access pending
    sampling_params = SamplingParams(max_tokens=1, temperature=1.0, logprobs=20)
    llm = LLM(model=model_name, tensor_parallel_size=tp_size, dtype="float16",
              max_model_len=2048, gpu_memory_utilization=0.90, enforce_eager=True, disable_log_stats=True)
    tokenizer = llm.get_tokenizer()
    outputs   = llm.generate([r["prompt"] for r in prompt_records], sampling_params)

    rows = []
    for rec, output in zip(prompt_records, outputs):
        logprobs_dict = output.outputs[0].logprobs[0] if (output.outputs and output.outputs[0].logprobs) else {}
        dist = extract_distribution_from_logprobs(logprobs_dict, tokenizer, rec["n_options"])
        rows.append({"qkey": rec["qkey"], "attribute": rec["attribute"], "group": rec["group"],
                     "responses": str(dist), "refusal_rate": 0.0, "ordinal": str(rec["ordinal"]),
                     "question": rec["question"], "options": str(rec["options"])})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved {len(df)} distributions to {output_path}")

    del llm
    import gc, torch; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return df


# =========================================================================
# STATISTICAL BOUNDS
# =========================================================================

def compute_bounds(
    ground_truth_csv: str,
    output_path: Path,
    questions_json: Optional[str] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
):
    """
    Compute per-row statistical bounds using the correct metric per question type.

    Ordinal questions  → Wasserstein Distance (WD / ordinal_emd)
    Nominal questions  → Total Variation Distance (TVD)

    Outputs metric-agnostic column names so step6 can ingest them directly:
        bound_uniform_upper        — distance(ground_truth, uniform)
        bound_bootstrap_lower_mean — mean distance between resampled gt distributions
        bound_bootstrap_ci_lower   — 2.5th percentile of bootstrap distances
        bound_bootstrap_ci_upper   — 97.5th percentile of bootstrap distances
        metric                     — 'WD' or 'TVD' (for auditing)

    Args:
        questions_json: path to cms_questions.json; if None, all questions are
                        treated as ordinal (WD).  A warning is printed if absent.
    """
    np.random.seed(seed)

    # Load ordinal flags (WD vs TVD per question)
    if questions_json is not None:
        ordinal_flags = load_ordinal_flags(questions_json)
        print(f"  Loaded ordinal flags for {len(ordinal_flags)} questions")
    else:
        ordinal_flags = {}
        print("  WARNING: --questions_json not provided; treating all questions as "
              "ordinal (WD).  Pass --questions_json for metric-consistent bounds.")

    gt_df = pd.read_csv(ground_truth_csv)
    gt_df["responses"] = gt_df["responses"].apply(ast.literal_eval)
    gt_df["ordinal"]   = gt_df["ordinal"].apply(ast.literal_eval)

    rows = []
    for _, row in gt_df.iterrows():
        gt_dist   = row["responses"]
        ordinal   = row["ordinal"]
        qkey      = row["qkey"]
        is_ord    = ordinal_flags.get(qkey, True)   # default to ordinal if unknown
        n_options = len(gt_dist)
        uniform   = [1.0 / n_options] * n_options
        metric    = "WD" if is_ord else "TVD"

        # Uniform upper bound
        upper = bound_distance(gt_dist, uniform, ordinal, is_ord)

        # Bootstrap lower bound — finite-sample noise floor
        boot_vals = []
        n_sample = 50  # simulated respondents per subgroup
        for _ in range(n_bootstrap):
            sample = np.random.multinomial(n_sample, gt_dist)
            sample_dist = (sample / sample.sum()).tolist()
            d = bound_distance(gt_dist, sample_dist, ordinal, is_ord)
            if not np.isnan(d):
                boot_vals.append(d)

        boot_mean  = np.mean(boot_vals)      if boot_vals else 0.0
        boot_lo    = np.percentile(boot_vals, 2.5)  if boot_vals else 0.0
        boot_hi    = np.percentile(boot_vals, 97.5) if boot_vals else 0.0

        rows.append({
            "qkey": qkey,
            "attribute": row["attribute"],
            "group": row["group"],
            "question": row["question"],
            "metric": metric,
            "bound_uniform_upper": upper,
            "bound_bootstrap_lower_mean": boot_mean,
            "bound_bootstrap_ci_lower": boot_lo,
            "bound_bootstrap_ci_upper": boot_hi,
        })

    bounds_df = pd.DataFrame(rows)
    bounds_df.to_csv(output_path, index=False)
    print(f"  Saved {len(bounds_df)} bounds to {output_path}")

    for m in ["WD", "TVD"]:
        sub = bounds_df[bounds_df["metric"] == m]
        if len(sub):
            print(f"\n  [{m}] {len(sub)} questions")
            print(f"    Uniform upper bound (mean): {sub['bound_uniform_upper'].mean():.4f}")
            print(f"    Bootstrap lower bound (mean): {sub['bound_bootstrap_lower_mean'].mean():.4f}")
            print(f"    Bootstrap 95% CI: [{sub['bound_bootstrap_ci_lower'].mean():.4f}, "
                  f"{sub['bound_bootstrap_ci_upper'].mean():.4f}]")

    return bounds_df


# =========================================================================
# MAIN
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-format zero-shot baselines and statistical bounds via vLLM"
    )
    parser.add_argument("--mode", type=str, default="zeroshot",
                        choices=["zeroshot", "bounds", "all"],
                        help="zeroshot: run LLM inference in QA/BIO/PORTRAY formats; "
                             "bounds: compute uniform upper + bootstrap lower bounds; "
                             "all: run both")
    parser.add_argument("--demographics_csv", type=str,
                        default="approach2_outputs/cms/cms_demographics.csv")
    parser.add_argument("--steering_json", type=str,
                        default="approach2_outputs/cms/cms_steering_prompts.json")
    parser.add_argument("--questions_json", type=str,
                        default="approach2_outputs/cms/cms_questions.json")
    parser.add_argument("--ground_truth_csv", type=str, default=None,
                        help="Ground truth distributions CSV (from step1_cms_adapter)")
    parser.add_argument("--output_dir", type=str, default="approach2_outputs/cms")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B",
                        help="Model to use. Falls back to Mistral-7B if Llama access is pending.")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--formats", nargs="+", default=["QA", "BIO", "PORTRAY"],
                        choices=["QA", "BIO", "PORTRAY"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    demographics = pd.read_csv(args.demographics_csv)

    with open(args.steering_json, "r") as f:
        steering_prompts = json.load(f)
    with open(args.questions_json, "r") as f:
        questions = json.load(f)

    print(f"Demographics: {len(demographics)} (attribute, group) pairs")
    print(f"Steering prompts: {len(steering_prompts)} attributes")
    print(f"Questions: {len(questions)}")
    print(f"Prompt formats: {args.formats}")

    # ---- Zero-shot in all formats ----
    if args.mode in ("zeroshot", "all"):
        for fmt in args.formats:
            print(f"\n{'='*60}")
            print(f"ZERO-SHOT {fmt}")
            print(f"{'='*60}")
            run_zero_shot(
                demographics=demographics,
                steering_prompts=steering_prompts,
                questions=questions,
                model_name=args.model_name,
                tp_size=args.tp_size,
                prompt_format=fmt,
                output_path=out_dir / f"distributions_{fmt}.csv",
            )

    # ---- Statistical bounds ----
    if args.mode in ("bounds", "all"):
        gt_csv = args.ground_truth_csv
        if gt_csv is None:
            raise ValueError(
                "--ground_truth_csv is required for bounds mode.\n"
                "Pass the CMS empirical distributions: "
                "--ground_truth_csv approach2_outputs/cms/cms_survey_distributions.csv"
            )
        print(f"\n{'='*60}")
        print(f"STATISTICAL BOUNDS")
        print(f"{'='*60}")
        compute_bounds(
            ground_truth_csv=gt_csv,
            output_path=out_dir / "statistical_bounds.csv",
            questions_json=args.questions_json,
            seed=args.seed,
        )

    print("\nStep 2 (baselines) complete.")


if __name__ == "__main__":
    main()
