# NYC Congestion Pricing — LLM-Based Population Opinion Simulation
## Complete Progress Report

**Author:** Xinxin (Twyla) Zhang
**Project Period:** March 2026 – May 2026
**Status (as of 2026-05-06):** Pipeline complete with Llama-3.1-8B; full evaluation across 7 method variants; prompt format bugs found and fixed May 6 (BIO/PORTRAY results from April run are invalid and must be re-run); Qwen2.5-7B added to notebook (not yet run); awaiting real survey data collection (target: 500 NYC respondents)

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Research Question & Scientific Goal](#2-research-question--scientific-goal)
3. [Methodological Origin: The SubPop Paper](#3-methodological-origin-the-subpop-paper)
4. [Phase A — Initial Pipeline (LODES + Mistral-7B)](#4-phase-a--initial-pipeline-lodes--mistral-7b)
5. [Phase B — Pipeline Upgrade Plan (April 6)](#5-phase-b--pipeline-upgrade-plan-april-6)
6. [Phase C — Pivot to CMS Real Survey Data (April 19)](#6-phase-c--pivot-to-cms-real-survey-data-april-19)
7. [Phase D — Multi-Format Runs and First Evaluation (April 20)](#7-phase-d--multi-format-runs-and-first-evaluation-april-20)
8. [Phase E — Question Set Expansion (5 → 23)](#8-phase-e--question-set-expansion-5--23)
9. [Phase F — Final Evaluation Run (May 4)](#9-phase-f--final-evaluation-run-may-4)
10. [Phase G — Bug Fixes and Notebook Rewrite (May 6)](#10-phase-g--bug-fixes-and-notebook-rewrite-may-6)
11. [Current Pipeline — Step-by-Step](#11-current-pipeline--step-by-step)
12. [Final Results](#12-final-results)
13. [Code Architecture & File Map](#13-code-architecture--file-map)
14. [Decisions, Reversals, and Why](#14-decisions-reversals-and-why)
15. [Known Limitations](#15-known-limitations)
16. [Planned Next Steps](#16-planned-next-steps)
17. [Open Questions & Risks](#17-open-questions--risks)
18. [Timeline Summary](#18-timeline-summary)

---

## 1. Executive Summary

This project investigates whether large language models, conditioned only on demographic descriptions, can predict how different NYC demographic subgroups respond to questions about congestion pricing and transit behavior. The work adapts the SubPop method (Suh et al., 2024) — originally built on national US Pew/ATP/GSS surveys — to NYC-specific data and questions.

**The core prediction target** is, for each (demographic subgroup, survey question) pair, the full probability distribution over multiple-choice answer options — not the single most likely answer.

**The journey across two months has gone through four major decisions:**
1. **Data source pivot:** LODES OD commuter flows → PUMS individual demographics → CMS real survey data (final).
2. **Ground-truth pivot:** LLM zero-shot self-prediction (circular, placeholder) → CMS empirical distributions weighted by `person_weight` (real human responses).
3. **Model pivot:** Mistral-7B-v0.1 (baseline) → Llama-3.1-8B (SubPop's primary model, final).
4. **Evaluation pivot:** Single QA prompt format → ablation over QA / BIO / PORTRAY × Zero-shot / Fine-tuned × all-questions / test-only-fair.

**Final headline numbers (May 4, 2026, on 51 held-out test rows × 3 unseen test questions × 5 demographic attributes):**

| Method | Wasserstein Distance (lower = better) |
|---|---|
| Uniform baseline (upper bound) | 0.328 |
| Bootstrap noise floor (lower bound) | 0.067 |
| Zero-shot QA | 0.308 |
| Zero-shot BIO | 0.308 |
| Zero-shot PORTRAY | 0.305 |
| **Fine-tuned QA** | **0.269** |
| Fine-tuned BIO | 0.283 |
| Fine-tuned PORTRAY | 0.264 |
| Fine-tuned QA (sequential pretrain) | 0.250 |

Fine-tuning gives a meaningful gain (≈12% relative WD reduction in the QA format on unseen questions). The sequential-training variant (Pew → CMS) was attempted in code but was not fully run with real Pew weights; the May 4 numbers labeled "(seq)" came from an earlier configuration of the training cell rather than a true sequential pretrain.

---

## 2. Research Question & Scientific Goal

> **Can a large language model, given only demographic context, predict how different NYC demographic subgroups travel and behave — and do these predictions match empirical CMS survey data?**

The scientific motivation is two-fold:
- **Methodological:** Test whether the SubPop distribution-matching framework generalizes from national political-opinion surveys (Pew/ATP/GSS) to a city-scale travel-behavior survey (CMS). If it does, the technique is a tool for cheaply estimating subgroup-level public opinion *before* running expensive surveys.
- **Substantive:** If the method works, the same machinery can be applied to a forthcoming custom 500-respondent congestion-pricing survey, giving us pre-survey expectations and post-survey debiasing/extension capability.

**What we are NOT doing** (clarifying the scope):
- Not predicting the single most likely answer per group.
- Not generating free-form respondent personas or qualitative justifications.
- Not yet using the model causally (e.g., to predict counterfactual policy outcomes).

---

## 3. Methodological Origin: The SubPop Paper

The project is built on Suh et al., *"Language Model Fine-Tuning on Scaled Survey Data for Predicting Distributions of Public Opinions"* (the **SubPop paper**), source repo: `JosephJeesungSuh/subpop`.

**SubPop's core technical idea:**
- For each (subgroup `g`, question `q`), the prediction target is the empirical human distribution `p_H(A_q | q, g)` over answer options.
- Build a steering prompt (subgroup description) + a multiple-choice survey question (options labeled A, B, C, …).
- Read the LLM's next-token logprobs over option letters at the `Answer:` position.
- Normalize to a probability distribution.
- Train (LoRA) with cross-entropy / KL loss between LLM and human distributions.
- Evaluate with Wasserstein Distance (WD), using ordinal weights when answer options have an order (Likert).

**Why SubPop was the right starting point.** Four candidate papers were surveyed early on; the ranking and reasoning was:

| Rank | Paper | Repo | Why / Why not |
|---|---|---|---|
| 1 | **SubPop** (Suh et al.) — *Language Model Fine-Tuning on Scaled Survey Data for Predicting Distributions of Public Opinions* | `JosephJeesungSuh/subpop` | **Best starting point.** End-to-end: data prep → fine-tune → inference → evaluation. Released LoRA checkpoints. Preprocessed HuggingFace dataset available. Subgroup-conditioned response distributions are the exact target. |
| 2 | SimLLMCultureDist (Cao et al.) — *Specializing LLMs to Simulate Survey Response Distributions for Global Populations* | `yongcaoplus/SimLLMCultureDist` | Useful second template. Fine-tunes on first-token probabilities to match response distributions. Includes `prepare_data.sh`, `train_slurm.sh`, `infer_slurm.sh`. Less integrated than SubPop. |
| 3 | Benchmarking-Distributional-Alignment (Meister et al.) | `nicolemeister/benchmarking-distributional-alignment` | Evaluation-only. Provides datasets, generation code, temperature scaling, eval scripts, reproduction notebooks. Not a fine-tuning base, but useful as a metrics reference. |
| 4 | DSA / Socrates (Stanford HCI) — *Distribution Shift Alignment Helps LLMs Simulate Survey Response Distributions* | `stanfordhci.github.io/socrates` | Site exists but accessible page contents did not confirm a directly reusable code repo at search time. Promising but lower confidence on availability. |

SubPop's scale is what we are trying to compete with: it trained on **~70,000 subpopulation-response pairs** drawn from real Pew Research surveys, covering **22 demographic attributes** and **3,362 questions**. The headline experiments in the paper used **Llama-2-7B** as the primary model. Our CMS-only training data is two orders of magnitude smaller, which is the central reason fine-tuning gains are modest — and the central reason "expand training data" is item #1 on the next-steps list.

**What we kept from SubPop** (must-preserve):
- The task formulation: subgroup prompt + survey question → option distribution.
- Distributional supervision (not one-hot SFT).
- Stable A/B/C/… answer-letter mapping.
- Question-level / topic-level generalization splits (not random row splits).
- Forward-KL training objective via cross-entropy on next-token over option letters (`loss_function_type='ce'` in `train_utils.py`); WD-as-loss is an appendix variant (`loss_function_type='wd'`).
- Sampling weights preserved during weighted subgroup aggregation (CMS `person_weight` plays the role of ATP/Pew weights).
- Ordinal metadata for ordered options so WD evaluation uses Earth-Mover Distance correctly.

**What we changed for NYC:**
- Subgroup ontology (NYC borough instead of US census region; CMS-specific demographic bins).
- Raw-data parser (CMS travel-diary CSV instead of ATP/GSS panel data).
- Question themes (travel behavior + congestion pricing instead of national political opinion).
- Steering metadata files (`cms_demographics.csv`, `cms_steering_prompts.json`).
- Surrounding framework: SubPop is built on the `llama-cookbook` distributed-training scaffolding; we keep the core SubPop method but run on Colab (single L4 GPU, vLLM for zero-shot) rather than a multi-GPU cluster.

### Paper → Code Faithfulness Map

The following table records, for each paper step, where it lives in the SubPop code we depend on. This is the implementation fidelity record.

| Paper step | SubPop code | Alignment |
|---|---|---|
| Build subgroup prompt + question input | `prepare_finetuning_data.py`, `survey_utils.generate_mcq`, `data/subpopulation_metadata/steering_prompts.json` | Faithful |
| Aggregate weighted subgroup response distributions from raw surveys | `generate_distribution.py`, `surveydata_utils.py` | Faithful, but ATP/GSS-specific — replaced by our `step1_cms_adapter.py` for NYC |
| Refine awkward survey questions with GPT-4o | `refine_question.py`, `ActualSurveyData.refine_question_body` | Faithful to appendix |
| Fine-tune on full response distributions with LoRA | `finetuning.py`, `train_utils.py` | Faithful |
| Use forward KL as main training loss | `train_utils.py` with `loss_function_type='ce'` | Faithful |
| Optional Wasserstein training | `train_utils.py` with `loss_function_type='wd'` | Matches appendix comparison |
| Recover model distribution from answer-letter token probabilities | `run_inference.py` | Faithful |
| WD/EMD evaluation | `survey_utils.ordinal_emd`, `run_inference.py` | Mostly faithful; refusal handling is inconsistent in the repo (training targets include refusal in `output_dist`, but `run_inference.py` drops refusal before EMD) — landmine for future surveys with explicit refusal categories |

---

## 4. Phase A — Initial Pipeline (LODES + Mistral-7B)

**Period:** March 2026 – early April 2026
**Outcome:** Working end-to-end pipeline; results not scientifically meaningful; identified the circularity problem and the need to pivot.

### 4.1 Data Source: LODES OD

Used **LODES (LEHD Origin-Destination Employment Statistics)** from the U.S. Census Bureau — block-level commuter flow data. Each row = one (home block, work block) pair with worker counts plus age and earnings marginals (`SA01/02/03`, `SE01/02/03`).

### 4.2 Pipeline Steps (LODES era)

1. **`step1_lodes_od_subgroups_to_prompt_examples.py`**
   - Filtered NY/NJ/CT OD files to NYC-relevant rows (block FIPS prefix matching).
   - Approximated joint counts via independence: `estimated_jobs = SA(age) × SE(earnings) / S000`.
   - Aggregated into 315 cells keyed by (home_region, work_region, age_group, earnings_group).
   - Selected top 20 cells × 5 questions = 100 (group, question) prompts.

2. **`step2_colab_zero_shot_likert_distribution.ipynb` / `step2_vllm_zero_shot_likert_distribution.py`**
   - Ran Mistral-7B-v0.1 on each prompt; extracted top-50 logprobs at the `Answer:` position; normalized over A–E.
   - Output: 100 (group, question) distributions in `lodes_distributions.csv`.

3. **`Step3_lodes_prepare_finetuning_data.py`**
   - Reformatted into SubPop-expected schema (`input_prompt`, `output_token`, `output_dist`).
   - Question-level split: 3 train / 1 val / 1 test (out of 5 questions).

4. **LoRA fine-tuning** (single T4 GPU, 4-bit QLoRA, batch size 1, 50 epochs).

5. **Inference + evaluation** — measured WD between fine-tuned predictions and the placeholder ground truth.

### 4.3 What We Found and Why It Was a Problem

**Reported result:** 76.4% reduction in WD on the held-out test question.

**The fatal circularity:** The step-2 ground-truth distributions were *Mistral's own zero-shot predictions*, not real human data. Fine-tuning then trained Mistral on its own outputs. The 76.4% improvement therefore meant only that the fine-tuned model became a more consistent reproducer of its own prior — it learned nothing about real human opinion.

**Other LODES-era weaknesses:**
- Independence approximation between age and earnings is not justified.
- Cross-tabulated cells produced small effective samples per cell.
- Only 5 questions → trivial generalization split (train on 3, test on 1).
- Only one prompt format (QA), no ablation.
- 4-bit / batch-1 / Mistral training differs significantly from SubPop's published config.

**Faithful vs. forced-by-hardware comparison to the SubPop paper, as of Phase A:**

| Aspect | SubPop paper | Our Phase A | Status |
|---|---|---|---|
| Loss function | Cross-entropy (forward KL) on next-token over option letters | Same | **Faithful** |
| LoRA config | r=8, α=32, dropout=0.05, target Q/V projections | Same | **Faithful** |
| Optimizer | AdamW, weight decay 0 | Same | **Faithful** |
| Scheduler | Cosine decay, linear warmup ratio 0.05 | Same | **Faithful** |
| Learning rate | 2e-4 | Same | **Faithful** |
| Epochs | 50 | Same | **Faithful** |
| Output token augmentation | 100 sampled letters per row | Same | **Faithful** |
| Prompt format | QA (best in SubPop ablations) | QA | **Faithful** |
| Model precision | float16 | 4-bit QLoRA | Forced (T4 16GB cap) — slightly changes numerical behavior |
| Distributed training | FSDP across multi-GPU cluster | Single T4 | Forced |
| Batch size | 128 | 1 | Forced — noisier gradients, but converges similarly given enough epochs |
| Training data | ~70K real Pew rows × 22 demographic attributes × 3,362 questions | 60 placeholder rows, 4 attributes, 3 questions | **Forced + by design** — solved by CMS pivot in Phase C |
| Base model | Llama-2-7B (paper headline) | Mistral-7B-v0.1 | Initial — replaced by Llama-3.1-8B in Phase D |
| Inference engine | vLLM batched offline | HuggingFace transformers forward pass | Different speed, identical logprobs |

The summary for advisors at the end of Phase A: *"Method is a faithful replication of SubPop's core approach. All differences are either forced by Colab T4 hardware (precision, parallelism, batch size) or are placeholders awaiting real data. The 76.4% improvement validates the pipeline mechanics, not the science."*

**Honest framing for advisors:**
> "We used the model's own zero-shot predictions as placeholder ground truth to validate the pipeline infrastructure. The 76.4% improvement shows the pipeline works mechanically — the fine-tuning and evaluation code is correct. The scientifically meaningful version of this experiment requires real survey data as ground truth, which is the next step."

### 4.4 What This Phase Delivered

A working, validated end-to-end pipeline shell. Every downstream piece — data flow, fine-tuning script, evaluation metric, visualization — was confirmed correct. From here, swapping in real ground truth required no code changes, only data changes.

---

## 5. Phase B — Pipeline Upgrade Plan (April 6)

**Document of record:** `Update of the code from initial pipeline set up_4.6.docx`
**Outcome:** Concrete written plan to bring the pipeline up to SubPop-paper quality before real survey arrives.

### 5.1 Identified Gaps vs. SubPop Paper

| Gap | LODES Pipeline | Plan |
|---|---|---|
| Data source | 4-D cross-tab (home × work × age × earnings) | Single-attribute PUMS subgroups, 11 attributes |
| Demographic richness | 4 dims | 11 dims (AGE, SEX, RACE, EDUCATION, INCOME, CREGION, MARITAL, CITIZEN, COMMUTE_MODE, COMMUTE_TIME, WORK_REGION) |
| Subgroup definition | Cross-tabulated cells (small N) | Single-attribute (matches SubPop's 22 / 61 group setups) |
| Joint estimation | Independence approximation | PUMS `PWGTP` direct weighting |
| Questions | 5 | 18 across 5 themes |
| Prompt formats | QA only | QA + BIO + PORTRAY (full ablation) |
| Ground truth | LLM zero-shot self-prediction | 500 real human respondents (planned) |
| Evaluation | Mean WD only | Bootstrap CI + per-attribute breakdown + entropy + inter-group disagreement heatmaps + population-weighted aggregates |
| Statistical bounds | None | Uniform upper + bootstrap lower |
| Generalization tests | Question-split only | Question-split + held-out-subgroup |

### 5.2 Architectural Refactor — Step Map

```
PUMS parquet ──→ step0_data_adapter.py        ──→ subgroup metadata + steering prompts
                          ↓
Hand-designed Q ──→ step1b_expand_and_refine_questions.py ──→ 18 questions + train/val/test split
                          ↓
Real survey CSV ──→ step1c_process_survey_responses.py ──→ subgroup answer distributions
                          ↓
                   step2_vllm_baselines.py    ──→ zero-shot QA/BIO/PORTRAY + uniform/bootstrap bounds
                          ↓
                   step3_prepare_finetuning_data.py ──→ per-attribute steering, multi-format CSVs
                          ↓
                   step4_setup_for_finetune.py ──→ rename to opnqa_*, run torchrun on subpop-main
                          ↓
                   subpop run_inference.py    ──→ results_finetuned_{QA,BIO,PORTRAY}.csv
                          ↓
                   step6_full_evaluation.py   ──→ ablation table, bootstrap CIs, heatmaps, entropy, pop-weighted
```

### 5.3 Critical Architectural Decisions

**Per-attribute steering, not composite profile.** The LODES pipeline used a single `commuter_profile` text describing the whole person at once. SubPop instead conditions on one attribute at a time and computes attribute-marginal distributions. This:
- Keeps cell sizes large (no cross-tab small-cell explosion).
- Matches what real surveys typically report (margin-by-margin).
- Enables the per-attribute breakdown analyses (which dimensions does the model capture well?).

**Question-split, not row-split.** Train on a subset of questions, evaluate on different questions. Tests true generalization, not memorization.

**Three prompt formats run as ablation.** SubPop reports QA tends to be best. The ablation tells us which format an LLM is most steerable in for *our* question set.

---

## 6. Phase C — Pivot to CMS Real Survey Data (April 19)

**Document of record:** `Workflow.docx` (final sections)
**Outcome:** Replaced the synthetic/placeholder ground truth with a real, publicly available NYC survey — eliminating the circularity problem before the custom 500-respondent survey is ready.

### 6.1 Why CMS

The **NYC Citywide Mobility Survey 2022 (CMS)** is publicly available, has 6,886 records, includes a travel diary plus behavioral survey, covers all 5 boroughs, and provides `person_weight` for population-representative estimates. After filtering to diary participants (`person_weight > 0`), 2,966 respondents remained.

This was the right interim dataset because:
- It's real human responses — solves circularity.
- Covers exactly the relevant population (NYC commuters).
- Has the demographic detail SubPop's framework needs.
- Is a stand-in for the 500-respondent custom survey: same downstream pipeline, just different rows in `survey_distributions.csv`.

### 6.2 New Pipeline File: `step1_cms_adapter.py`

Replaces `step1_lodes_od_subgroups_to_prompt_examples.py`. Maps CMS columns to SubPop format:

| CMS Column | Attribute | Groups |
|---|---|---|
| `age` | AGE | 18-24, 25-44, 45-64, 65+ |
| `gender` | GENDER | Male, Female |
| `income_broad` | INCOME | Under $50K, $50K–$100K, $100K–$200K, $200K+ |
| `home_county` | BOROUGH | Bronx, Brooklyn, Manhattan, Queens, Staten Island |
| `r_race` | RACE | White, Non-White |

→ 17 single-attribute subgroups total, each with cell size ≥ 52 (smallest: BOROUGH=Staten Island).

### 6.3 Initial Question Set (5 travel-behavior questions)

| qkey | CMS column | Question | Ordinal? |
|---|---|---|---|
| Q_COMMUTE_MODE | r_work_mode | Typical commute mode (7 options) | Nominal |
| Q_BIKE_FREQ | bike_freq | Frequency of biking | Ordinal |
| Q_TNC_FREQ | tnc_freq | Frequency of Uber/Lyft | Ordinal |
| Q_CITIBIKE_FREQ | citi_bike_freq | Frequency of Citi Bike | Ordinal |
| Q_TELEWORK_FREQ | r_telework_freq | Days/week working from home | Ordinal |

### 6.4 Outputs (CMS pipeline)

All written to `approach2_outputs/cms/`:
- `cms_questions.json` — questions + options + ordinal flags.
- `cms_survey_distributions.csv` — 85 (attribute, group, question) rows in initial 5-question version; later expanded to 391 in the 23-question version. **This is the real ground truth.**
- `cms_subgroup_weights.csv` — CMS person-weighted population shares per subgroup.
- `cms_demographics.csv` — subgroup metadata.
- `cms_steering_prompts.json` — QA / BIO / PORTRAY templates per (attribute, group).
- `cms_question_split.json` — train/val/test split.

### 6.5 Important Conceptual Insight

> Any multiple-choice question works in this pipeline. The inference mechanism reads logprobs of the option letters A, B, C, …; it does not care what the letters semantically mean. So Likert, frequency, binary, and unordered categorical questions all run identically. The only constraint is exhaustive + mutually exclusive options. The `ordinal` field tells the WD computation whether to use ordered Earth-Mover Distance or treat options as nominal.

---

## 7. Phase D — Multi-Format Runs and First Evaluation (April 20)

**Folders:** `Subpop_replication/run1a_apr20_zeroshot_and_QA/` (was `output_colab/`), `run1b_apr20_BIO_finetune/` (was `cms/`).

### 7.1 Run 1a (April 20, morning) — Zero-shot All 3 Formats + QA Fine-tune

Switched from Mistral-7B-v0.1 to **Llama-3.1-8B** to align with SubPop's primary model. Ran on Google Colab with vLLM for zero-shot inference.

Outputs:
- `distributions_QA.csv` (391 rows × 5 demographic attributes)
- `distributions_BIO.csv`
- `distributions_PORTRAY.csv`
- `statistical_bounds.csv` — uniform upper + bootstrap lower per (attribute, group, question)
- `results_finetuned_QA.csv` — QA fine-tuned predictions on 51 test rows (3 test questions × 17 subgroups)
- `checkpoints_QA/` — LoRA adapter

**Evaluation 1 (`evaluation/`)** — first cut, computed bootstrap CI / entropy / per-attribute WD / population-weighted estimates for the 4 methods then available.

**Evaluation 2 (`evaluation_refair/`)** — re-evaluation under a fair coverage protocol: all methods compared on identical (attribute, group, question) grids only (the 51 test rows).

### 7.2 Run 1b (April 20, evening) — BIO Fine-tune Add-on

Same Colab session continuation; added BIO format fine-tuning.
- `checkpoints_BIO/`
- `results_finetuned_BIO.csv` (51 rows)

### 7.3 First Headline Result (April 20, fair evaluation, test-only)

| Method | Test-question WD |
|---|---|
| Zero-shot QA | 0.3087 |
| Zero-shot BIO | 0.3081 |
| Zero-shot PORTRAY | 0.3052 |
| Fine-tuned QA | 0.3040 |

**Initial reading:** Fine-tuning helped only marginally; PORTRAY was unexpectedly the best zero-shot format (counter to SubPop's QA-best finding); fine-tuned BIO not yet downloaded; sample diagnostic — best val epoch 33, val loss 0.2029, train loss 0.0057 → likely overfitting / last checkpoint not best.

This finding (fine-tuning helps but unevenly) drove the next two changes: (a) more questions to get a richer evaluation surface, (b) more careful checkpoint selection / longer training.

---

## 8. Phase E — Question Set Expansion (5 → 23)

**Period:** Late April

> **Important conceptual distinction.** Two parallel question sets exist in this project, and prior internal docs sometimes blurred them:
> - **The 18 designed congestion-pricing Likert questions** (April 6 plan, `step1b_expand_and_refine_questions.py`). These are *opinion* questions across 5 themes — Direct support, Equity & fairness, Transit & alternatives, Economic impact, Environment. They are the questions we plan to put on the **custom 500-respondent survey** when it runs. Examples: Q0 *"Do you support congestion pricing in Manhattan?"*, Q17 *"Reduced traffic from congestion pricing improves the quality of life in my neighborhood."* All on a 5-point Likert scale (Strongly disagree → Strongly agree). These were **never run** as a real evaluation because no real responses exist yet.
> - **The 23 CMS travel-behavior questions** (used for the May 4 results). These are *behavioral and demographic* questions from the existing CMS 2022 instrument (commute mode, bike frequency, telework frequency, EV purchase intent, etc.). Mostly nominal or ordinal-frequency, not Likert opinion. These are the **interim ground truth** — real human responses, but on travel behavior rather than on congestion-pricing opinion.
>
> The pipeline is question-agnostic by design: when the custom survey lands, replacing `cms_survey_distributions.csv` with `survey_distributions.csv` produced by `step1c_process_survey_responses.py` re-runs everything on the 18 designed Likert questions with no other code changes.

### The 23 Interim CMS Questions (used for the May 4 results)

To get a more meaningful evaluation surface than the original 5, the CMS-derived question set was expanded to 23:

`Q_BIKE_CHANGE`, `Q_BIKE_FREQ`, `Q_CITIBIKE_FREQ`, `Q_COMMUTE_MODE`, `Q_EDUCATION`, `Q_EMPLOYMENT`, `Q_EV_PURCHASE`, `Q_HOUSEHOLD_SIZE`, `Q_HOUSING_TENURE`, `Q_INDUSTRY`, `Q_JOB_TYPE`, `Q_NUM_BICYCLES`, `Q_NUM_KIDS`, `Q_NUM_VEHICLES`, `Q_PRIMARY_LANGUAGE`, `Q_RESIDENCE_TYPE`, `Q_STUDENT`, `Q_TELEWORK_FREQ`, `Q_TNC_FREQ`, `Q_TRANSIT_SAFETY`, `Q_VEHICLE_CHANGE`, `Q_WFH_POLICY`, `Q_WORK_ZONE`.

→ 23 questions × 17 subgroups = **391 ground-truth distribution rows** in `cms_survey_distributions.csv`.

**Train/Val/Test Split** (final):
- Test (3 questions, used everywhere as the held-out unseen-question evaluation surface): `Q_BIKE_CHANGE`, `Q_EV_PURCHASE`, `Q_HOUSING_TENURE` → 51 rows.
- Train + val: remaining 20 questions → 340 rows.

**Caveat noted in code review:** All 3 test questions are nominal (`is_ordinal=False`). The metric labeled "WD" in the ablation table is therefore Total Variation Distance for these specific test rows, not the ordinal Earth-Mover Distance. The code is correct; the column header is loose. Future writeups should label this precisely.

### The 18 Designed Congestion-Pricing Likert Questions (planned, not yet run)

Designed in Phase B (April 6) for the eventual 500-respondent custom survey. Drafted in `step1b_expand_and_refine_questions.py`, optionally refinable via GPT-4o (`refine_question_body()` pattern from SubPop's `surveydata_utils.py`), with a stratified 11/3/4 train/val/test split that ensures each theme appears across splits.

| Theme | Approx. Q# range | Example questions |
|---|---|---|
| Direct support | Q0–Q3 | "Do you support congestion pricing in Manhattan?", "The $9 toll is a fair price.", "Congestion pricing should be expanded." |
| Equity & fairness | Q4–Q6 | "Low-income commuters are disproportionately burdened.", "Exemptions for essential workers are sufficient." |
| Transit & alternatives | Q7–Q10 | "Public transit should be prioritized over roads.", "I would switch to transit if service improved." |
| Economic impact | Q11–Q14 | "Commuting costs are a major concern.", "Congestion pricing will hurt local businesses.", "The toll revenue is being well spent." |
| Environment & quality of life | Q15–Q17 | "Traffic congestion is a major problem in NYC.", "Congestion pricing will improve air quality.", "Reduced traffic improves my neighborhood." |

All use the 5-point Likert scale; `ordinal=[1,2,3,4,5]` so WD computation uses ordered EMD. These will become the production target the day the 500-respondent survey responses land.

---

## 9. Phase F — Final Evaluation Run (May 4)

**Folder:** `eval_may04_FINAL/` (was `evaluation_0504/`).
**Outcome:** Complete ablation with 7 method variants, the definitive results to date.

### 9.1 What Changed Between April 20 and May 4

The zero-shot results are byte-for-byte identical (deterministic inference, same model + prompts). The fine-tuned numbers improved because:
- More training epochs (20 with explicit early stopping vs. fewer in the April run).
- Cleaner checkpoint selection logic in the notebook.
- All 3 formats fine-tuned (April 20 had only QA + BIO).
- A "sequential" QA training cell was run, intended as Pew-pretrain → CMS-finetune (matching SubPop's training data scale advantage). However, the Pew pretraining stage was NOT actually executed in May 4; the cell ran the second-stage CMS fine-tune on the base model only. The `Fine-tuned QA (seq)` row in the May 4 ablation reflects whatever checkpoint that cell produced and should be interpreted cautiously.

### 9.2 Pipeline Resumability Redesign (April 30 – May 4)

A separate engineering thread, important for reproducibility and for re-running on Qwen later:
- Added a central config cell with `RUN_QA / RUN_BIO / RUN_PORTRAY / RUN_QA_SEQ / RUN_BIO_SEQ` toggles and `FORCE_*` flags for each stage.
- All compute cells now check existence of their outputs via `should_run(stage, outputs, force)` — runs only what's missing.
- Added `restore_from_drive` for the pretrain-checkpoint handoff file so sequential cells survive Colab session restarts.
- Made skip guards format-aware (e.g., the zero-shot stage's skip check considers QA + BIO + PORTRAY artifacts, not only QA).
- `step6_full_evaluation.py` got a `question_subset_sensitivity()` analysis: evaluates prediction stability across n non-overlapping question subsets. It is **not** cross-validation (the model is fixed); it's a stability check, with a `pct_covered` column and a coverage warning that fires when methods cover different question fractions, plus `common_qkeys_summary.csv` for valid cross-method comparison on the shared support.

> **Note:** The per-format toggle design (`RUN_QA / RUN_BIO / ...`) was replaced in the May 6 notebook rewrite. See Phase G (Section 10) for the new `ACTIVE_MODELS` / `ACTIVE_FORMATS` / `FORCE` dict design.

---

## 10. Phase G — Prompt Format Fix and Notebook Rewrite (May 6)

**Period:** May 6, 2026
**Outcome:** BIO/PORTRAY prompt formats corrected to match SubPop paper; April BIO/PORTRAY results invalidated and flagged for re-run; `colab_run.ipynb` fully rewritten to support dual-model (Llama + Qwen) and all three prompt formats cleanly.

### 10.1 Bug 1 — BIO and PORTRAY Prompt Formats Wrong (Critical)

**File changed:** `approach2_outputs/cms/cms_steering_prompts.json`

A review against the SubPop paper's exact definitions revealed that both BIO and PORTRAY formats had been implemented incorrectly for all 17 (attribute, group) entries across all 5 attributes.

**BIO — what was wrong:** The instructional preamble was missing entirely. The format read only a bare self-description (`"I am 18-24 years old."`), which looks nearly identical to a stripped-down PORTRAY. The SubPop paper's BIO format requires a two-part structure:
```
Below you will be asked to provide a short description of your [attribute]
and then answer some questions.
Description: [first-person statement].
```

**PORTRAY — what was wrong:** The format used `"Portray someone who is 18-24 years old."` — a roleplay command in the wrong voice. The SubPop paper's PORTRAY format is a conditional instruction in the third person:
```
Answer the following question as if you [condition].
```

**Fix applied:** All 17 entries corrected. Examples of the fixed format:
- `AGE / 18-24 / BIO`: `"Below you will be asked to provide a short description of your age group and then answer some questions. Description: I am in the age group 18-24."`
- `AGE / 18-24 / PORTRAY`: `"Answer the following question as if you were in the age group of 18-24."`
- `BOROUGH / Bronx / BIO`: `"Below you will be asked to provide a short description of your NYC borough of residence and then answer some questions. Description: I currently reside in the Bronx, New York City."`
- `BOROUGH / Bronx / PORTRAY`: `"Answer the following question as if you currently resided in the Bronx, New York City."`
- `INCOME / Under $50K / BIO`: `"Below you will be asked to provide a short description of your household income level and then answer some questions. Description: My annual household income is under $50K."`
- `INCOME / Under $50K / PORTRAY`: `"Answer the following question as if your annual household income were under $50K."`

The QA format was already correct and was not changed.

**Downstream consequence:** All existing BIO and PORTRAY results from the April 20 run (`distributions_BIO.csv`, `distributions_PORTRAY.csv`, `results_finetuned_BIO.csv`, and the `cms_BIO_*.csv` / `cms_PORTRAY_*.csv` finetuning data) used the wrong steering prompts and are **invalid**. They must be fully re-run. QA results remain valid.

### 10.2 Notebook Rewrite — Dual-Model Support and Clean Config Design

**File changed:** `colab_run.ipynb` (full rewrite, 25 cells)

The notebook was fully rewritten to:

1. **Support two models** — Llama-3.1-8B and Qwen2.5-7B — using a `MODELS` registry dict defined in Cell 1:
   ```python
   MODELS = {
       'llama': 'meta-llama/Llama-3.1-8B',
       'qwen':  'Qwen/Qwen2.5-7B',
   }
   ```

2. **Replace per-format toggles with list-based config** — the old `RUN_QA / RUN_BIO / RUN_PORTRAY / RUN_QA_SEQ` boolean flags are replaced with:
   ```python
   ACTIVE_MODELS  = ['llama', 'qwen']
   ACTIVE_FORMATS = ['QA', 'BIO', 'PORTRAY']
   FORCE = {
       'pretrain_download': False, 'pretrain_train': False,
       'zeroshot': False, 'bounds': False, 'step3': False,
       'finetune': False, 'sequential': False, 'eval': False,
   }
   ```

3. **Model-aware path helpers** — all output paths route through a `model_tag` param:
   ```python
   def cms_local(rel_path='', model_tag=None):
       base = f'{REPO_DIR}/{OUTPUT_ROOT}/cms'
       if model_tag: base = f'{base}/{model_tag}'
       return base if not rel_path else f'{base}/{rel_path}'
   ```

4. **Drive scaffolding for per-model dirs only** — shared data comes from git clone; Drive scaffolds only per-model output directories.

5. **Sequential training preserved** — Cell 7 reads the per-model pretrain checkpoint path from `pretrain_checkpoint_path.txt` and passes it as `--from_peft_checkpoint` to the second-stage fine-tune.

Qwen has not yet been run on Colab — the notebook supports it but GPU time is needed for the first Qwen zero-shot and fine-tuning runs.

---

## 11. Current Pipeline — Step-by-Step

### Step 0 — `step0_data_adapter.py`
**Input:** `pums_demographics_ny_2024.parquet` (210K records).
**Action:** Filter to adults (174K), bin into single-attribute subgroups (AGE, SEX, RACE, EDUCATION, INCOME, VEHICLES, HOUSEHOLD_SIZE), compute PUMS-weighted population shares (`PWGTP`).
**Outputs (in `approach2_outputs/`):**
- `subgroup_weights.csv` — 29 rows (attribute, group, weight, count, pop_share)
- `demographics_congestion.csv` — subgroup menu for downstream
- `congestion_steering_prompts.json` — QA/BIO/PORTRAY templates per attribute

Optional under the CMS workflow because step 1 also produces weights.

### Step 1 — `step1_cms_adapter.py`
**Input:** `../CMS/CMS_merged.csv` (6,886 raw records).
**Action:** Filter to `person_weight > 0` (2,966 diary participants); map demographic codes to readable labels; for each (attribute, group, question) compute weighted empirical Likert/categorical distribution; build steering prompts.
**Outputs (in `approach2_outputs/cms/`):**
- `cms_questions.json` (23 questions)
- `cms_survey_distributions.csv` (391 rows — **ground truth**)
- `cms_subgroup_weights.csv`
- `cms_demographics.csv`, `cms_steering_prompts.json`, `cms_question_split.json`

### Step 2 — `step2_vllm_baselines.py`
**Requires:** GPU (L4 or better, 24 GB VRAM).
**Action:** For each (attribute, group, question, prompt_format), build the steering prompt + MCQ, run vLLM with `max_tokens=1, logprobs=50`, extract option-letter probabilities, normalize.
**Outputs:** `distributions_{QA,BIO,PORTRAY}.csv`, `statistical_bounds.csv`.
**Modes:** `zeroshot` | `bounds` | `all`.

### Step 3 — `step3_prepare_finetuning_data.py`
**Action:** For each (attribute, group, question) in train/val/test, build the same steering + MCQ prompt as step 2; sample 100 answer tokens from the empirical distribution (soft-label augmentation); each sampled token = one fine-tuning row.
**Outputs (in `approach2_outputs/cms/finetuning_data/`):** `cms_{QA,BIO,PORTRAY,ALL}_{train,val,test}.csv`.
**Optional `--holdout_groups`** for unseen-subpopulation generalization tests.

### Step 4 — LoRA Fine-Tuning (external, GPU)
Three runs, one per format:
```
torchrun --nproc-per-node=2 scripts/experiment/run_finetune.py \
    --dataset_path congestion-pricing --steering_type {QA,BIO,PORTRAY} \
    --peft_method lora --lora_config.r 8 --lora_config.lora_alpha 32 \
    --batch_size_training 128 --num_epochs 50 --lr 2e-4
```
Same hyperparameters as SubPop, applied to Llama-3.1-8B.

### Step 5 — Inference
Reuse `subpop-main/scripts/experiment/run_inference.py`. Produces `results_finetuned_{QA,BIO,PORTRAY}.csv`.

### Step 6 — `step6_full_evaluation.py`
**Action:** Auto-discover all `distributions_*.csv` and `results_finetuned_*.csv`; compute per-(method × attribute × group × question) WD (ordinal Earth-Mover for ordinal questions, TVD for nominal); aggregate.
**Outputs (in `eval_may04_FINAL/`):**
- `ablation_table.csv`, `method_coverage.csv` (top-level)
- `descriptive_all_rows/` and `test_only_fair/` containing, per method:
  - `bootstrap_ci_*.csv` (95% CI per row)
  - `per_attribute_wd_*.csv` (mean WD per demographic dim)
  - `entropy_*.csv` (predicted vs. ground-truth entropy)
  - `population_weighted_*.csv` (CMS-weighted aggregate opinion)
  - `disagreement_heatmaps/` (pairwise WD between groups)
- `question_subset_sensitivity/` — stability analysis across n non-overlapping question subsets

---

## 12. Final Results

> ⚠️ **Validity caveat (May 6, 2026):** The BIO and PORTRAY columns in all tables below — `Zero-shot BIO`, `Zero-shot PORTRAY`, `Fine-tuned BIO`, `Fine-tuned PORTRAY` — were produced using the **wrong** prompt format (Bug 1, Section 10.1). All BIO/PORTRAY results must be treated as invalid and will be replaced after re-running step2, step3, and LoRA fine-tuning with the corrected `cms_steering_prompts.json`. **QA results remain valid.**

### 12.1 Ablation Table (eval_may04_FINAL/ablation_table.csv)

**All 23 questions (descriptive, biased toward zero-shot since fine-tuned only ran on test questions):**

| Attribute | Zero-shot QA | Zero-shot BIO | Zero-shot PORTRAY | Fine-tuned QA | Fine-tuned BIO | Fine-tuned PORTRAY | Fine-tuned QA (seq) | Uniform upper | Bootstrap lower |
|---|---|---|---|---|---|---|---|---|---|
| AGE | 0.261 | 0.248 | 0.250 | 0.290 | 0.299 | 0.296 | 0.283 | 0.327 | 0.067 |
| GENDER | 0.230 | 0.235 | 0.235 | 0.279 | 0.307 | 0.274 | 0.257 | 0.316 | 0.070 |
| INCOME | 0.273 | 0.251 | 0.246 | 0.258 | 0.283 | 0.246 | 0.245 | 0.326 | 0.067 |
| BOROUGH | 0.246 | 0.248 | 0.252 | 0.264 | 0.265 | 0.256 | 0.228 | 0.340 | 0.066 |
| RACE | 0.235 | 0.230 | 0.231 | 0.254 | 0.272 | 0.243 | 0.241 | 0.319 | 0.069 |
| **OVERALL** | **0.253** | **0.245** | **0.246** | **0.269** | **0.283** | **0.264** | **0.250** | **0.328** | **0.067** |

**Test questions only (51 rows, fair comparison — the headline result):**

| Attribute | Zero-shot QA | Zero-shot BIO | Zero-shot PORTRAY | Fine-tuned QA | Fine-tuned BIO | Fine-tuned PORTRAY | Fine-tuned QA (seq) |
|---|---|---|---|---|---|---|---|
| AGE | 0.357 | 0.354 | 0.315 | 0.290 | 0.299 | 0.296 | 0.283 |
| GENDER | 0.236 | 0.283 | 0.315 | 0.279 | 0.307 | 0.274 | 0.257 |
| INCOME | 0.344 | 0.296 | 0.314 | 0.258 | 0.283 | 0.246 | 0.245 |
| BOROUGH | 0.294 | 0.312 | 0.297 | 0.264 | 0.265 | 0.256 | 0.228 |
| RACE | 0.249 | 0.255 | 0.281 | 0.254 | 0.272 | 0.243 | 0.241 |
| **OVERALL** | **0.308** | **0.308** | **0.305** | **0.269** | **0.283** | **0.264** | **0.250** |

### 12.2 Interpretation

1. **Fine-tuning helps, on the right metric.** On the held-out test questions (the meaningful generalization surface), the best fine-tuned model (PORTRAY at 0.264, or `QA (seq)` at 0.250 if interpreted cautiously) is comfortably below the best zero-shot (0.305). The relative WD reduction is ≈12–18%.
2. **The signal is meaningful relative to the bounds.** Uniform baseline ≈ 0.328 (worst plausible predictor). Bootstrap noise floor ≈ 0.067 (best achievable given finite sample sizes per cell). Both zero-shot and fine-tuned sit well below uniform but well above the noise floor → there is real but imperfect demographic signal in the LLM.
3. **Format matters less than expected.** QA/BIO/PORTRAY zero-shot results are within 0.003 of each other overall. After fine-tuning, PORTRAY edges QA, contradicting SubPop's QA-is-best on Pew. Hypothesis: CMS's behavioral / categorical questions (commute mode, employment, residence type) are less amenable to QA's "answer me with one letter" framing than Pew's opinion Likerts.
4. **Per-attribute breakdown.** AGE is the hardest dimension (0.290 fine-tuned QA on test questions); GENDER is the easiest (0.279). BOROUGH consistently improves under fine-tuning, suggesting the model has prior NYC-borough demographic knowledge that fine-tuning sharpens.
5. **The reading on Fine-tuned QA (seq).** The May 4 numbers for this row reflect a partially-implemented sequential setup; treat as a soft estimate, not as evidence that Pew-pretrain → CMS-finetune outperforms direct CMS-finetune.

### 12.3 Consistency Note on Reruns

The headline numbers shifted between April 20 and May 4 only for fine-tuned methods. Zero-shot QA = 0.253 and fine-tuned QA (test-only) went from 0.304 → 0.269 because:
- More epochs in the May 4 training (20 with proper early stopping).
- All 3 formats trained.
- Different effective hyperparameters / checkpoint selection.

This is the expected behavior — zero-shot is deterministic, fine-tuning is sensitive to training configuration.

---

## 13. Code Architecture & File Map

### 13.1 Active Source Files (in `Subpop_replication/`)

| File | Role |
|---|---|
| `step0_data_adapter.py` | PUMS → SubPop-format subgroup metadata |
| `step1_cms_adapter.py` | CMS raw CSV → 391-row ground truth + steering prompts (BIO/PORTRAY fixed May 6 to match SubPop paper format) |
| `step2_vllm_baselines.py` | Zero-shot 3-format inference + statistical bounds |
| `step3_prepare_finetuning_data.py` | Per-attribute steering + soft-label augmentation |
| `step6_full_evaluation.py` | All evaluation analyses + question-subset sensitivity |
| `colab_run.ipynb` | Resumable orchestration for Llama-3.1-8B **and Qwen2.5-7B**; `ACTIVE_MODELS` / `ACTIVE_FORMATS` / `FORCE` dict config; `CMS_SHARED_DIR` / `CMS_SHARED_REL` two-directory design; Drive persistence |
| `subpop-main/` | Vendored SubPop training/inference infrastructure |

### 13.2 Output Folders

| Folder | What it is |
|---|---|
| `Subpop_replication/approach2_outputs/cms/` | Working artifact folder, written by step 1 |
| `Subpop_replication/run1a_apr20_zeroshot_and_QA/` | First Colab run: zero-shot all 3 + QA fine-tune |
| `run1b_apr20_BIO_finetune/` | Same-day continuation: BIO fine-tune |
| `eval_may04_FINAL/` | **Definitive evaluation:** all 7 methods, descriptive + test-only-fair, sensitivity analysis |
| `_duplicate_step1_data_apr19/` | Archived duplicate of approach2_outputs/cms — safe to delete |

### 13.3 Key Data Files (current state)

| File | Rows | Description |
|---|---|---|
| `cms_survey_distributions.csv` | 391 | Ground truth (23 questions × 17 subgroups) |
| `distributions_{QA,BIO,PORTRAY}.csv` | 391 | Zero-shot Llama-3.1-8B predictions |
| `results_finetuned_{QA,BIO,PORTRAY}.csv` | 51 | Fine-tuned predictions on 3 test questions |
| `cms_question_split.json` | — | 20 train / 3 test, all 17 subgroups in each |

---

## 14. Decisions, Reversals, and Why

| Decision | Original Choice | Final Choice | Trigger / Reasoning |
|---|---|---|---|
| Data source | LODES OD commuter flows | CMS 2022 survey | LODES gave only marginal age × earnings; CMS is real human responses |
| Subgroup geometry | Cross-tabulated cells (315) | Single-attribute subgroups (17) | Small-cell instability; SubPop convention; supports per-attribute breakdown |
| Ground truth | LLM zero-shot on its own predictions | CMS empirical distributions | The original was circular — model trained on its own outputs |
| Base model | Mistral-7B-v0.1 | Llama-3.1-8B | SubPop's primary model; better community support; advisor preference |
| Inference precision | 4-bit QLoRA (T4 fit) | float16 / bf16 (L4) | QLoRA changes numerical behavior during training; Colab Pro provides L4 |
| Batch size | 1 | 128 | Single-GPU forced batch=1 originally; multi-GPU enables SubPop's published config |
| Prompt formats | QA only | QA + BIO + PORTRAY ablation | Single-format result is not interpretable without comparison |
| Question count | 5 | 23 | Trivial 3/1/1 split; 20/3 split gives meaningful generalization signal |
| Evaluation | Mean WD only | Bootstrap CI + per-attribute + entropy + heatmaps + pop-weighted + sensitivity | One number is not a research finding |
| Pipeline orchestration | Run-everything-from-scratch script | Resumable notebook with skip-guards + FORCE flags | Colab session restarts; cost of redundant compute |

---

## 15. Known Limitations

1. **CMS is not the target survey.** The 23 CMS questions are about travel behavior, not directly about congestion-pricing opinion. CMS is a stand-in for the upcoming custom 500-respondent congestion-pricing survey. The pipeline is question-agnostic — when the custom survey lands, only the questions and ground truth change.

2. **All 3 test questions are nominal.** `Q_BIKE_CHANGE`, `Q_EV_PURCHASE`, `Q_HOUSING_TENURE` all have `is_ordinal=False`, so the metric reported as "WD" on the test surface is technically Total Variation Distance, not ordinal EMD. This is correct behavior in the code; the column label is imprecise.

3. **Single-attribute steering only.** Each prompt conditions on one demographic dimension (e.g., AGE=30-49). No joint conditioning yet (e.g., AGE=30-49 AND BOROUGH=Brooklyn AND INCOME=$50–100K).

4. **Sequential pretraining incomplete.** The `Fine-tuned QA (seq)` row in the May 4 ablation reflects an intended Pew → CMS sequential training that did not actually run the Pew stage with real Pew data.

5. **Qwen2.5-7B not yet run.** Advisor expressed concern about over-reliance on Llama. The notebook now supports Qwen2.5-7B in parallel with Llama-3.1-8B, but zero-shot and fine-tuning runs for Qwen have not yet been executed (GPU time needed).

6. **Subgroup cell sizes are uneven.** GENDER and RACE have very large cells (n ≈ 400+); BOROUGH=Staten Island is small (n = 52). Bootstrap CIs reflect this — the noise floor is wider for small cells.

7. **No held-out subgroup test yet.** Step 3 supports `--holdout_groups`, but in the current run all 17 subgroups appear in both train and test. So the evaluation tests *unseen questions* but not *unseen demographic groups*.

8. **One model per format.** No ensembling or multi-seed averaging. Fine-tuning variance is real (April 20 vs. May 4 numbers shifted) and is not currently bounded by repeated training.

9. **BIO/PORTRAY results from April run are invalid.** The prompt format for BIO and PORTRAY was incorrect (missing SubPop preamble in BIO; wrong roleplay phrasing in PORTRAY). All BIO and PORTRAY result files from the April 20 run must be re-generated before any BIO/PORTRAY numbers can be reported or compared. QA results remain valid.

---

## 16. Planned Next Steps

### 16.1 Expand Training Data
**a. Combine PUMS and CMS.** Use PUMS for population structure (subgroup population shares) and CMS for response distributions. The current pipeline already supports this — `step0_data_adapter.py` produces PUMS weights, `step1_cms_adapter.py` produces CMS distributions. The integration point is `step3_prepare_finetuning_data.py`'s use of `subgroup_weights.csv`. Awaiting verification that PUMS and CMS bins align consistently (e.g., income brackets, age bins).

**b. Sequential training.** First fine-tune on the Pew Research SubPop-Train data (large-scale, ~70K rows, real opinion distributions), then second-stage fine-tune on CMS. Hypothesis: the model picks up the *mechanism* of subgroup-conditioned distribution prediction from Pew at scale, then specializes to NYC travel from CMS. Risk: catastrophic forgetting if the second stage is too long; mitigation = small LR + few epochs in stage 2.

### 16.2 Try Joint Distribution
Move from single-attribute steering to joint-attribute steering, e.g., conditioning on AGE × BOROUGH or AGE × INCOME × BOROUGH simultaneously. This matches what real surveys ultimately want to predict (a specific person, not a marginal). Risks:
- Cell sizes shrink dramatically (17 → potentially 100+ cells, many with n < 10).
- Need to verify minimum-cell-size threshold and possibly use a hierarchical / Bayesian smoother.

### 16.3 Run Qwen2.5-7B Zero-shot and Fine-tuning
Notebook already supports Qwen2.5-7B. First priority: run zero-shot all 3 formats for Qwen; compare with Llama zero-shot to check model-dependence. Decision on whether to fine-tune Qwen can wait until zero-shot results are available.

### 16.4 Migrate to Real Survey (500 respondents)
When the custom congestion-pricing survey arrives, replace `cms_survey_distributions.csv` with `survey_distributions.csv` from `step1c_process_survey_responses.py`. Steps 2–6 run identically. Only required: the survey instrument has been designed with demographic questions that map to the same bins as PUMS/CMS.

### 16.5 Persona-Form Conversion (advisor request)
Advisor asked whether the pipeline can be re-framed as: give the LLM a persona → it produces an answer. The current logprob-based distribution-extraction is already this, formalized; an open question is whether the eventual end-product should be a persona-prompted single-answer interface for non-technical users to use, with the distribution computed under the hood from many sampled persona instantiations.

---

## 17. Open Questions & Risks

- **Best-checkpoint selection.** The April 20 run showed val loss 0.2029 at epoch 33 but final epoch had val loss 0.2119 (overfitting). The May 4 run added explicit early stopping; need to verify the saved checkpoint is the best-val one, not the last-epoch one.
- **Loss function: KL vs. WD.** SubPop's main objective is forward KL on next-token over option letters. WD-as-loss is an appendix variant. CMS is heavily nominal — KL is the right primary; the WD column on test rows is computed as TVD which is the right move for nominal options.
- **Refusal handling inconsistency** (carried over from SubPop): training targets include refusal in `output_dist`, but `run_inference.py` drops refusal before EMD. CMS does not have explicit refusal categories, so this is currently a non-issue but is a known landmine if survey rows include "Prefer not to answer".
- **Joint-subgroup minimum cell size.** When we move to joint conditioning, what threshold do we accept? SubPop uses ≥10. CMS at AGE × BOROUGH × INCOME × RACE will have many cells < 10.
- **MRP comparison.** Approach 1 (MRP) and Approach 2 (this) target different things — MRP estimates a single mean per cell, this predicts a full distribution per cell. They aren't directly comparable unless metrics are aligned (e.g., compare mean predictions or compare cell-level posterior intervals).

---

## 18. Timeline Summary

| Date | Milestone |
|---|---|
| March 2026 | Started from SubPop repo as base; reviewed 4 candidate papers; selected SubPop |
| Late March – early April | Built LODES-based pipeline; produced 100 (group, question) prompts; ran Mistral-7B zero-shot on Colab |
| April 6 | Identified circularity in LODES + zero-shot self-prediction; wrote pipeline upgrade plan (`Update of the code from initial pipeline set up_4.6.docx`) |
| April 16 | Initial PIPELINE_REPORT.md drafted reflecting CMS-based design |
| April 19 | `step1_cms_adapter.py` written; CMS replaces LODES; 5-question initial CMS run; Workflow.docx records architectural decisions and paper alignment |
| April 20 (AM) | Llama-3.1-8B; first full Colab run: zero-shot QA/BIO/PORTRAY + QA fine-tune (run1a) |
| April 20 (PM) | BIO fine-tune added (run1b) |
| April 24 | `Progress.md` tracker plan written; first fair-evaluation results captured |
| Late April | Question set expanded 5 → 23; train/val/test split locked at 20/3 |
| April 30 – May 4 | Resumable notebook redesign (config cell, FORCE flags, skip guards, Drive persistence); fixed P1 bugs in pretrain handoff and skip-guard scope |
| May 4 | Final evaluation run with all 7 method variants; question-subset sensitivity analysis added; folder cleanup |
| May 5 | Folder rename + duplicate-folder cleanup; consolidated progress report (PROGRESS_FULL_REPORT.md) |
| May 6 | Code review against SubPop paper: BIO/PORTRAY prompts in `cms_steering_prompts.json` and `step1_cms_adapter.py` corrected to match paper's exact format; all April BIO/PORTRAY results invalidated and must be re-run. Notebook fully rewritten to support Llama-3.1-8B + Qwen2.5-7B with `ACTIVE_MODELS`/`ACTIVE_FORMATS`/`FORCE` dict design. README question count corrected (25 → 23). |

---

## Appendix A — Glossary

- **SubPop:** The paper by Suh et al. proposing fine-tuning LLMs on subgroup-conditioned response distributions. Source: `JosephJeesungSuh/subpop`.
- **WD (Wasserstein Distance / EMD):** Earth-Mover's Distance between two distributions. For ordinal options, accounts for ordering. For nominal, reduces to Total Variation.
- **PUMS:** Census Public Use Microdata Sample — individual-level demographic records with replicate weights (`PWGTP`).
- **LODES:** Census LEHD Origin-Destination Employment Statistics — block-level commuter flow data.
- **CMS:** NYC Citywide Mobility Survey 2022 — 6,886 records, 2,966 diary participants, person-weighted.
- **LoRA:** Low-Rank Adaptation; fine-tunes only small added matrices, leaving the base model frozen.
- **Steering prompt:** A short text describing the demographic subgroup, prepended to the survey question to condition the LLM's response.
- **QA / BIO / PORTRAY:** Three SubPop steering formats — Q&A pretext, first-person bio statement, third-person roleplay instruction.

## Appendix B — Source Files Consolidated Into This Report

- `PIPELINE_REPORT.md` (April 16) — Step-by-step technical pipeline reference
- `Progress.md` (April 24) — Project tracker plan and current-state snapshot
- `Update of the code from initial pipeline set up_4.6.docx` (April 6) — Pipeline upgrade plan and architectural rationale
- `Workflow.docx` (April 19) — Original LODES design, paper-method analysis, model/infra comparison vs SubPop, transition reasoning

This consolidated report is the single source of truth going forward; the four originals can stay in place as historical records.
