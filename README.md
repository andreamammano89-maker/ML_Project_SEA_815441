# AI Productivity Analysis — Alkemy Project

**Group SEA · Machine Learning Project · A.Y. 2025/2026 · LUISS**

Simone Cavallaro — 815441 · Andrea Mammano — 821311 · Elisa Presciutti — 812891

---

## Repository Structure

```
ML_PROJECT_SEA_815441/
├── data/
│   └── ai_productivity_dataset_final.csv
├── images/
│   ├── lasso_coefficients_s1.png       # Lasso retained features (base)
│   ├── lasso_coefficients_s2.png       # Lasso retained features (log-transformed)
│   ├── log_distribution_comparison.png # Original vs log-transformed distributions
│   ├── lr_coefficients_s1.png          # LR top coefficients (base)
│   ├── lr_coefficients_s2.png          # LR top coefficients (log-transformed)
│   ├── model_comparison_s1.png         # Model comparison — base features
│   ├── model_comparison_s2.png         # Model comparison — base vs log-transformed
│   ├── shap_beeswarm_s1.png            # SHAP beeswarm top 15 features (base)
│   ├── shap_beeswarm_s2.png            # SHAP beeswarm top 15 features (log-transformed)
│   ├── shap_dependence_ai_s1.png       # SHAP dependence: ai_usage_pct (base)
│   └── shap_dependence_ai_s2.png       # SHAP dependence: ai_usage_pct (log-transformed)
├── main.ipynb                          # Full pipeline: EDA + Modelling
├── pyproject.toml                      # uv project configuration
├── requirements.txt                    # pip-compatible dependencies
├── uv.lock                             # Exact dependency versions (uv)
└── README.md
```

---

## How to Recreate the Environment

**With uv (recommended):**
```bash
pip install uv
uv sync
uv run jupyter notebook main.ipynb
```

**With pip:**
```bash
pip install -r requirements.txt
jupyter notebook main.ipynb
```

---

## Section 1 — Introduction

This project analyses how Artificial Intelligence affects business workflows at
Alkemy, a digital services company. The central research question is:

> *Beyond which AI usage threshold does rework destroy profit margin?*

While AI is typically associated with faster execution, the key question is
whether efficiency gains translate into economic value — or are lost to rework,
hidden costs, and structural pricing constraints.

The dataset contains **3,248 task-level observations** and 34 variables covering
AI adoption intensity, operational efficiency, quality metrics, and financial
outcomes. Each row represents a single deliverable task within Alkemy's workflows.

---

## Section 2 — Methods

### 2.1 Data Cleaning

- **Deduplication**: 48 `task_id` duplicates resolved by retaining the most
  recent record based on `updated_at`.
- **Categorical normalisation**: `team` and `task_type` standardised via
  explicit mapping dictionaries. Inconsistent casing and typos reduced to
  4 team categories and 7 task types.
- **`duration_days`**: computed from `created_at` − `delivered_at`;
  14 negative values set to NaN rather than corrected arbitrarily.
- **`legacy_ai_flag`**: `'true'` → 1, `'false'` → 0, `'unknown'` → NaN.
  Correlation with `ai_usage_pct` = −0.014 confirms they measure different constructs.

### 2.2 Missing Value Treatment

Missing values are treated as operational signals — they may reflect incomplete
tracking or process gaps rather than data entry errors.

| Variable | N missing | Strategy | Motivation |
|---|---|---|---|
| `ai_usage_pct` | 143 | Median imputation | Distributions of missing vs full dataset overlap — MCAR confirmed visually |
| `outcome_score`, `brief_quality_score`, `sla_days`, `rework_hours` | 32–72 | Median imputation | Robust to skewed distributions |
| `billable_hours` | 81 | `hours_spent × 0.85` | Industry billing efficiency benchmark, confirmed by company |
| `jira_ticket` | 331 | Dropped | Pure text identifier, no predictive value |
| `legacy_ai_flag` | 337 | Retained as NaN | Forcing imputation would introduce artificial certainty |
| `delivered_at` | 38 | Retained as NaN | Informative missingness — task not yet delivered |

### 2.3 Outlier Treatment

No capping was applied to raw variables. The IQR × 3 rule was used as a
diagnostic only. Statistical outliers correspond to economically meaningful
observations: large `revenue` values are real contracts, high `hours_spent`
are complex tasks, high `rework_hours` are quality failures.

Robustness is ensured methodologically: Spearman correlation (rank-based,
insensitive to extremes) is used alongside Pearson throughout the EDA.

### 2.4 Feature Engineering

| Feature | Formula | Motivation |
|---|---|---|
| `profit_margin` | `profit / revenue × 100` | Normalised margin, comparable across task sizes |
| `rework_rate` | `rework_hours / hours_spent` | Proportion of wasted time |
| `error_rate` | `errors / hours_spent` | Error density per worked hour |
| `billable_ratio` | `billable_hours / hours_spent` | Share of effort recovered as revenue |
| `rework_cost_est` | `rework_hours × median(cost_per_hour)` | Monetary rework cost in € |
| `hidden_cost_ratio` | `rework_cost_est / cost` | Rework as share of total cost |
| `ai_usage_sq` | `ai_usage_pct²` | Quadratic term for non-linear AI effects |
| `ai_bucket` | `pd.cut` into 5 equal-width intervals | Threshold analysis segments |
| `budget_bucket` | Median split of `revenue` | Economic scale segmentation |
| `complexity_bucket` | Tertiles of `task_complexity_score` | Operational complexity segmentation |

All ratio-based features use a stabilised denominator (`hours_spent` clipped
at 0.1) to prevent artefacts on near-zero tasks.

### 2.5 Modelling Pipeline

A unified `sklearn` Pipeline with a `ColumnTransformer` preprocessor is applied
identically across all models:

- **Numeric**: median imputation → `StandardScaler`
- **Categorical**: most-frequent imputation → `OneHotEncoder` (`handle_unknown='ignore'`)

**Target leakage exclusion** is applied explicitly before any model sees the
data. `profit_margin = profit / revenue × 100`, so `profit`, `revenue`, `cost`,
and all algebraically derived variables (`hidden_cost_ratio`, `rework_cost_est`,
`profit_bucket`, `budget_bucket`) are removed. Post-hoc status flags
(`task_status`, `workflow_stage`) are excluded as they are set after delivery.

**Full pipeline:**

```
Raw CSV (3,248 × 34)
        │
        ▼
  Data Cleaning  →  Missing Value Treatment  →  Feature Engineering
        │
        ▼
  Leakage exclusion + Train/Test split (80/20, random_state=42)
        │
        ├──────────────────┬──────────────────┐
        ▼                  ▼                  ▼
  Linear Regression   Lasso (LassoCV)   Random Forest
  (base + log)        (base + log)      (GridSearchCV)
        │                  │                  │
        └──────────────────┴──────────────────┘
                           │
                           ▼
                  SHAP Analysis (TreeExplainer)
                  (base + log-transformed)
```

The full pipeline is run twice:
- **Section 1**: base features
- **Section 2**: `log1p` applied to right-skewed numeric features — to test
  whether feature transformation improves predictive power

---

## Section 3 — Experimental Design

### Experiment 1 — Threshold Detection (EDA)

**Purpose**: identify whether there is an AI usage level beyond which the
marginal gain in profit margin starts to decline, and whether this threshold
varies across pricing models, teams and task types.

**Baseline**: binary AI/No-AI comparison (tasks with `ai_usage_pct > 0` vs
tasks where `ai_usage_pct = 0`). This is the simplest possible comparison
and the most common approach in practice.

**Evaluation metric**: median `profit_margin` per `ai_bucket` (0–20% through
80–100%), with first-differences (`profit_diff`) between adjacent buckets to
identify inflection points. Median is used over mean due to strong right-skew
in `profit_margin`.

### Experiment 2 — Predictive Modelling (Section 1 — Base Features)

**Purpose**: quantify the direct predictive contribution of `ai_usage_pct` to
`profit_margin` in a full multivariate setting, controlling for all other
observable task attributes.

**Baseline**: Linear Regression — interpretable, assumes linearity and
additivity. Any more complex model must outperform this to justify its
added complexity.

**Primary model**: Random Forest with hyperparameters tuned via `GridSearchCV`
(5-fold CV). Tuned parameters: `n_estimators` ∈ {100, 200, 300},
`max_depth` ∈ {4, 6, 8, None}, `min_samples_leaf` ∈ {10, 20, 40}.

**Lasso (LassoCV)**: regularisation strength α selected via 5-fold CV.
Best α = 0.5697. Retained 32 features out of 57 — features zeroed out: 25.

**Evaluation metrics**:
- **R²**: proportion of variance in `profit_margin` explained
- **MAE** (pp): interpretable error in the same units as the target
- **5-fold CV R²**: generalisation estimate, more reliable than test R² alone

### Experiment 3 — Predictive Modelling (Section 2 — Log-Transformed Features)

**Purpose**: test whether applying `log1p` to right-skewed numeric features
improves predictive power, and whether the RF is already robust to skewness
without transformation.

**Baseline**: Section 1 results (base features).

**Evaluation metric**: ΔR² and ΔMAE between Section 1 and Section 2 for
each model. A positive ΔR² for LR and near-zero ΔR² for RF would confirm
that right-skew was a binding constraint for OLS but not for the tree-based model.

---

## Section 4 — Results

### 4.1 Threshold Analysis

The binary AI/No-AI split produces **no statistically significant difference**
in profit margin (Welch's t-test and Mann-Whitney U both p > 0.05). The signal
only becomes visible when AI is treated as a continuous intensity:

| AI Bucket | Median Profit Margin |
|---|---|
| 0–20% | ~21% |
| 20–40% | ~28% |
| 40–60% | ~35% |
| 60–80% | ~43% |
| 80–100% | ~49% |

Tasks operating at 80–100% AI intensity show median margins approximately
28 percentage points higher than those below 20%. However, this aggregate
trend is conditional on pricing model — under hourly contracts, margins peak
at ~26% then decline to ~20% at highest AI intensity, the only segment where
more AI leads to lower profitability.

### 4.2 Model Performance — Section 1 (Base Features)

| Model | Test R² | MAE (pp) | RMSE (pp) | 5-fold CV R² |
|---|---|---|---|---|
| Linear Regression | 0.2048 | 32.06 | 48.55 | 0.0832 |
| Lasso (LassoCV, α=0.57) | ~0.20 | — | — | — |
| **Random Forest (tuned)** | **0.3445** | **27.64** | **44.08** | **0.1844** |

![Model Comparison — Base Features](images/model_comparison_s1.png)

### 4.3 Model Performance — Section 2 (Log-Transformed Features)

| Model | Test R² | ΔR² vs base | MAE (pp) | ΔMAE vs base |
|---|---|---|---|---|
| Linear Regression | 0.2487 | +0.0439 | 30.65 | −1.41 |
| **Random Forest** | **0.3445** | **≈ 0.000** | — | — |

The log transform improves Linear Regression R² by +0.044 — confirming
that right-skewed feature distributions were a binding constraint on OLS.
The Random Forest is unchanged (ΔR² ≈ 0), confirming the tree-based model
already captures skewness without explicit transformation.

![Model Comparison — Base vs Log](images/model_comparison_s2.png)

### 4.4 SHAP Feature Importance

SHAP values are computed using `shap.TreeExplainer` on the best Random Forest.
Results are stable across both base and log-transformed versions.

| Rank | Feature | Direction |
|---|---|---|
| 1 | `pricing_model_hourly` | Negative — hourly pricing compresses margin |
| 2 | `billable_ratio` | Positive — higher billing efficiency → higher margin |
| 3 | `seniority_senior` | Mixed — seniors take complex, costly tasks |
| 4 | `hours_spent` | Negative — more hours = cost overrun on fixed contracts |
| 13 | `ai_usage_pct` | Weak positive — AI is not a direct margin driver |

`ai_usage_pct` ranks **13th out of 57 encoded features**. AI usage has a
statistically detectable but modest direct contribution to margin. Its effect
is structural — mediated by pricing model and billing efficiency.

![SHAP Beeswarm — Base Features](images/shap_beeswarm_s1.png)

![SHAP Dependence: ai_usage_pct](images/shap_dependence_ai_s1.png)

---

## Section 5 — Conclusions

### Summary

The central finding is that AI usage intensity does not directly drive profit
margin — it amplifies existing structural conditions. Margin is primarily
determined by pricing model and billing efficiency, not by how much AI is used.
The only configuration where more AI systematically leads to lower profitability
is high-intensity AI under hourly pricing, where efficiency gains reduce
billable hours and transfer the productivity benefit to the client. The
implication is structural: the intervention point is pricing reform, not AI
restriction. The threshold the research question asks about is not a universal
number — it is conditional on pricing model, with hourly contracts showing
deterioration above ~60% AI intensity and fixed/value-based contracts showing
no such threshold.

### Limitations and Future Work

The best model (Random Forest, Section 1) explains 34.5% of margin variance,
leaving 65.5% unexplained. This reflects the absence of key variables: client
pricing power and tenure, competitive context at time of sale, team capacity
at assignment, and AI tool quality (we observe *how much* AI was used, not
*how effectively*). Collecting these variables is the highest-value next step
for improving predictive power.

A second open question is the SEO team anomaly: margins peak at the 60–80%
AI intensity bucket then decline — a pattern not observed in any other team.
Understanding whether this reflects a workflow integration failure, a task
composition difference, or a small-sample artefact requires targeted follow-up.
Finally, the log-transformation experiment suggests that future work should
explore additional feature transformations (e.g., Box-Cox) or non-linear
preprocessing pipelines to further improve the linear model's performance,
which remains well below the Random Forest ceiling.