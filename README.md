# NYC Congestion Pricing — SubPop Replication

Adaptation of the [SubPop](https://arxiv.org/abs/2406.10281) methodology to predict **subgroup-level response distributions** to NYC commuter behavior questions using the CMS 2022 (NYC Citywide Mobility Survey).

---

## What This Does

Given a demographic subgroup (e.g. *Borough = Manhattan*, *Income = <$30k*) and a multiple-choice survey question (e.g. *"What is your primary commute mode?"*), the fine-tuned model predicts the **full probability distribution** over answer options — not just the most likely answer.

Ground truth comes from 2,966 CMS 2022 diary respondents. The model is evaluated using Wasserstein Distance (ordinal questions) and Total Variation Distance (nominal questions).

---

## Dataset

- **Source**: NYC Citywide Mobility Survey (CMS) 2022
- **Questions**: 23 MCQ behavioral questions (17 train / 3 val / 3 test)
- **Demographic attributes**: AGE, GENDER, INCOME, BOROUGH, RACE (17 subgroups total)
- **Distributions**: 391 (question × subgroup) pairs, each a probability vector over 3–7 options

---

## Pipeline Overview

```
step1  →  step3  →  [subpop-main training]  →  step6
(data)   (prep)       (GPU, fine-tune)        (eval)
                          ↑
                      step2 (zero-shot baseline, GPU)
```

### Step 1 — Process CMS data (CPU, ~1 min)
```bash
python step1_cms_adapter.py --cms_csv /path/to/CMS_merged.csv
```
**Outputs** (in `approach2_outputs/cms/`):
- `cms_survey_distributions.csv` — ground truth distributions (391 rows)
- `cms_questions.json` — question bank with ordinal flags
- `cms_steering_prompts.json` — QA / BIO / PORTRAY steering prompts
- `cms_demographics.csv` — 17 (attribute, group) pairs
- `cms_subgroup_weights.csv` — CMS population weights per subgroup
- `cms_question_split.json` — train/val/test question split

### Step 2 — Zero-shot baseline (GPU required, ~5–10 min on L4)
```bash
python step2_vllm_baselines.py \
    --mode zeroshot \
    --model_name meta-llama/Llama-3.1-8B \
    --output_dir approach2_outputs/cms
```
**Outputs**: `distributions_QA.csv`, `distributions_BIO.csv`, `distributions_PORTRAY.csv`

For statistical bounds (requires step 1 ground truth):
```bash
python step2_vllm_baselines.py \
    --mode bounds \
    --ground_truth_csv approach2_outputs/cms/cms_survey_distributions.csv \
    --output_dir approach2_outputs/cms
```
> `--ground_truth_csv` is **required** for bounds mode.

### Step 3 — Prepare fine-tuning data (CPU, ~1 min)
```bash
python step3_prepare_finetuning_data.py \
    --distributions_csv   approach2_outputs/cms/cms_survey_distributions.csv \
    --demographics_csv    approach2_outputs/cms/cms_demographics.csv \
    --steering_json       approach2_outputs/cms/cms_steering_prompts.json \
    --question_split_json approach2_outputs/cms/cms_question_split.json \
    --output_dir          approach2_outputs/cms/finetuning_data
```
**Outputs**: `cms_{QA,BIO,PORTRAY,ALL}_{train,val,test}.csv`

### Step 4 — Fine-tune (GPU required, ~30 min on L4)

Runs SubPop's LoRA fine-tuning from `subpop-main/`. Run from the repo root:
```bash
cd subpop-main
python scripts/experiment/run_finetune.py \
    --model_name              meta-llama/Llama-3.1-8B \
    --dataset                 cms_dataset \
    --steering_type           QA \
    --output_dir              ../approach2_outputs/cms/checkpoints_QA/ \
    --num_epochs              50 \
    --early_stopping_patience 5 \
    --batch_size_training     4 \
    --lr                      2e-4 \
    --use_peft \
    --quantization \
    --loss_function_type      ce \
    --enable_fsdp=False \
    --use_wandb=False
```
Repeat with `--steering_type BIO --output_dir ../approach2_outputs/cms/checkpoints_BIO/` for the BIO variant.

The `cms_dataset` config is registered in `subpop-main/subpop/train/configs/datasets.py` and points to the CSVs generated in step 3.

### Step 5 — Fine-tuned inference

```bash
cd subpop-main
python scripts/experiment/run_inference.py \
    --input_paths             ../approach2_outputs/cms/finetuning_data/cms_QA_test.csv \
    --output_dir              ../approach2_outputs/cms/ \
    --base_model_name_or_path meta-llama/Llama-3.1-8B \
    --lora_path               ../approach2_outputs/cms/checkpoints_QA/<timestamp> \
    --lora_name               cms_QA
cd ..
```

`run_inference.py` writes output as `{output_dir}{input_csv_stem}_{lora_name}.csv`
(i.e. `cms_QA_test_cms_QA.csv`). Rename it to the canonical name step 6 expects:

```bash
mv approach2_outputs/cms/cms_QA_test_cms_QA.csv \
   approach2_outputs/cms/results_finetuned_QA.csv
```

> **Checkpoint path note**: `finetuning.py` appends a timestamp directly to the
> `--output_dir` string, so pass a **trailing slash**:
> `--output_dir ../approach2_outputs/cms/checkpoints_QA/`
> → checkpoint lands in `checkpoints_QA/<timestamp>/`, not `checkpoints_QA<timestamp>`.
> The Colab notebook handles this and the rename automatically.

### Step 6 — Evaluate (CPU, ~1 min)
```bash
python step6_full_evaluation.py \
    --ground_truth_csv    approach2_outputs/cms/cms_survey_distributions.csv \
    --questions_json      approach2_outputs/cms/cms_questions.json \
    --question_split_json approach2_outputs/cms/cms_question_split.json \
    --predictions_dir     approach2_outputs/cms \
    --weights_csv         approach2_outputs/cms/cms_subgroup_weights.csv \
    --output_dir          approach2_outputs/cms/evaluation
```
**Outputs** (under `approach2_outputs/cms/evaluation/`):

| File / Directory | Contents |
|---|---|
| `ablation_table.csv` | SubPop Table 1-style comparison with `(all Qs)` and `(test Qs)` columns |
| `method_coverage.csv` | Audit manifest — n_rows / n_qkeys / n_attributes per method × scope |
| `descriptive_all_rows/` | All 23 questions: bootstrap CIs, per-attribute WD, entropy, disagreement heatmaps, population-weighted opinions |
| `test_only_fair/` | Test questions only (same 3 Qs for all methods — apples-to-apples comparison) |

> The `(test Qs)` ablation columns and `test_only_fair/` outputs are only generated when `--question_split_json` is provided.

---

## Models

| Model | Role | HF access |
|---|---|---|
| `meta-llama/Llama-3.1-8B` | **Primary** — all main results | Gated (approved) |
| `mistralai/Mistral-7B-v0.1` | Optional comparison run (Cell 10 in `colab_run.ipynb`) | Public |

The pipeline auto-detects access via `resolve_model()` in `step2_vllm_baselines.py` — if Llama access is ever lost it falls back to Mistral automatically.

---

## Evaluation Metrics

| Question type | Metric | Questions |
|---|---|---|
| Ordinal (ordered options) | Wasserstein Distance (WD) | Q_TELEWORK_FREQ, Q_WFH_POLICY, Q_BIKE_FREQ, Q_TNC_FREQ, Q_NUM_VEHICLES, Q_NUM_BICYCLES, Q_EDUCATION, Q_HOUSEHOLD_SIZE, Q_NUM_KIDS |
| Nominal (unordered options) | Total Variation Distance (TVD) | Q_COMMUTE_MODE, Q_EMPLOYMENT, Q_STUDENT, Q_JOB_TYPE, Q_RESIDENCE_TYPE, Q_PRIMARY_LANGUAGE, Q_WORK_ZONE, Q_INDUSTRY, Q_CITIBIKE_FREQ, Q_TRANSIT_SAFETY, Q_VEHICLE_CHANGE, Q_BIKE_CHANGE, Q_HOUSING_TENURE, Q_EV_PURCHASE |

---

## Repository Structure

```
Subpop_replication/
├── step1_cms_adapter.py              # CMS data → ground truth distributions
├── step2_vllm_baselines.py           # Zero-shot inference + statistical bounds
├── step3_prepare_finetuning_data.py  # Fine-tuning CSV generation
├── step6_full_evaluation.py          # Evaluation framework
├── subpop-main/                      # SubPop library (training, inference, utils)
│   ├── scripts/experiment/run_finetune.py    # LoRA fine-tuning entry point (step 4)
│   ├── scripts/experiment/run_inference.py   # Inference entry point (step 5)
│   └── subpop/train/configs/datasets.py      # cms_dataset config registered here
├── approach2_outputs/cms/            # Pre-generated CMS outputs (tracked in git)
│   ├── cms_survey_distributions.csv
│   ├── cms_questions.json
│   ├── cms_steering_prompts.json
│   ├── cms_demographics.csv
│   ├── cms_subgroup_weights.csv
│   └── cms_question_split.json
├── requirements.txt                  # CPU (local) dependencies
└── requirements_colab.txt            # GPU (Colab) dependencies
```

Not tracked in git (large or GPU-generated):
- `approach2_outputs/cms/finetuning_data/` — generated by step 3
- `approach2_outputs/cms/checkpoints_*/` — LoRA checkpoints (GB+)
- `approach2_outputs/cms/distributions_*.csv` — step 2 inference results
- `approach2_outputs/cms/results_finetuned_*.csv` — step 5 inference results
- `approach2_outputs/cms_mistral/` — optional Mistral comparison outputs (same layout)

---

## Setup

**Local (CPU — steps 1, 3, 6):**
```bash
pip install -r requirements.txt
cd subpop-main && pip install -e . && cd ..
```

**Colab (GPU — steps 2, 4, 5):**

Open `colab_run.ipynb` directly in Colab:
**File → Open notebook → GitHub → `https://github.com/Twyla123/Subpop_Replication_`**

Then: **Runtime → Change runtime type → L4 GPU**, run cells in order:

| Cells | What happens |
|---|---|
| 1–3 | Mount Drive, install deps, HF login |
| 4 | Zero-shot baselines (QA + BIO + PORTRAY) |
| 5 | Prepare fine-tuning data |
| 6 | Fine-tune QA model (~30 min) |
| 7 | QA inference |
| 7a | Fine-tune BIO model (~30 min) — recommended for comparison |
| 7b | BIO inference |
| 8 | Evaluation (writes `descriptive_all_rows/` + `test_only_fair/`) |
| 9 | Save all results to Google Drive |
| 10 | (Optional) Mistral-7B comparison run |

The notebook handles cloning, dependency install, HF login, all pipeline steps, and saves results back to Google Drive automatically.
