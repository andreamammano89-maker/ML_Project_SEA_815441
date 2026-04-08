w# 🧠 ML Project SEA — Full Pipeline (Revised)

**Course:** Machine Learning A.Y. 2025/2026 | **Company:** Alkemy | **Team:** SEA  
**Research Question:** *Beyond which AI usage threshold does rework destroy profit margin?*

---

## Phase 1 — Data Loading & Cleaning

**What:** Load the raw CSV, fix quality issues, and produce a clean dataframe `df`.  
**Why:** The project brief explicitly states the dataset is *"multi-table, incomplete, and imperfect."* Any analysis built on dirty data will produce unreliable conclusions. Evaluators will check whether you handled missing values and inconsistencies correctly.

| Step | Action | Rationale |
|------|--------|-----------|
| 1.1 | Load CSV, print shape/dtypes/head | Understand the data before touching it |
| 1.2 | Deduplication on `task_id` | Duplicate rows inflate AI usage counts and bias every downstream metric |
| 1.3 | Normalize `team` column | Raw data has typos (`Contennt`, `Desgn`, `Paid Media`) that create phantom groups in segmentation |
| 1.4 | Parse date columns | `created_at` and `delivered_at` are strings by default; needed as datetime to compute `duration_days` |
| 1.5 | Map `legacy_ai_flag` | `'unknown'` values should become `NaN`, not a third category, to avoid polluting binary tests |
| 1.6 | Impute numeric missing values | Median imputation for `ai_usage_pct`, `outcome_score`, `rework_hours` — median is robust to outliers unlike mean |
| 1.7 | Impute `billable_hours` | Use `hours_spent × 0.85` as proxy; **the 0.85 factor must be justified in markdown** (industry norm or company-provided benchmark) |
| 1.8 | Cap outliers (IQR × 3) | Extreme values on `profit`, `revenue`, `cost`, `hours_spent`, `rework_hours` would distort all LOWESS curves and regression coefficients |
| 1.9 | Add markdown explanation before each code cell | **Submission rule**: alternating text/code throughout |

---

## Phase 2 — Feature Engineering

**What:** Create derived variables that directly operationalize the research question.  
**Why:** Raw variables (`profit`, `hours_spent`) don't answer "when does AI hurt margin?" We need normalized metrics that make tasks comparable regardless of size, and group labels that enable threshold analysis.

| Feature | Formula | Why it matters |
|---------|---------|----------------|
| `margin_pct` | `profit / revenue × 100` | Normalized profitability — comparable across tasks of different size |
| `ai_flag` | `ai_usage_pct > 0` | Binary split for AI vs No-AI baseline comparison |
| `rework_rate` | `rework_hours / hours_spent` | Proportion of total time wasted on corrections — the key hidden cost driver |
| `billable_ratio` | `billable_hours / hours_spent` | How much work is actually recovered as revenue |
| `cost_per_hour` | `cost / hours_spent` | Hourly cost intensity — needed to estimate rework cost in € |
| `duration_days` | `delivered_at − created_at` | Delivery speed; did AI actually accelerate delivery? |
| `ai_bucket` | 5 bins: 0–20%…80–100% | Enables the core threshold analysis — average metrics per band |
| `is_high_ai` | `ai_usage_pct > 0.6` | Binary flag for regression models — isolates the high-risk zone |
| `rework_cost_est` | `rework_hours × median(cost_per_hour)` | Converts rework into € — makes hidden costs tangible for the business decision |

---

## Phase 3 — Exploratory Data Analysis (EDA)

**What:** Understand distributions, relationships, and data quality visually.  
**Why:** Before testing hypotheses you must know what the data looks like. EDA reveals skewness, bimodality, or unexpected patterns that would invalidate later tests. It also lets you flag interesting findings to investigate deeper.

| # | Visualization | Why |
|---|--------------|-----|
| 3.1 | Histograms + KDE for 6 key variables | Check for skewness (profit often right-skewed), bimodality in `ai_usage_pct` (many zeros?) |
| 3.2 | Missing value matrix (missingno) | Shows *patterns* of missingness — is `outcome_score` missing randomly or for specific task types? |
| 3.3 | Pearson correlation heatmap | First pass at relationships between all numeric variables |
| 3.4 | **Spearman correlation heatmap** | Essential because AI usage → margin is non-linear; Pearson underestimates non-linear associations |

> **Key question:** Is missingness in `outcome_score` or `ai_usage_pct` random (MCAR) or systematic? If it's systematic (e.g., missing for Design tasks), median imputation is biased.

Save all figures to `images/`.

---

## Phase 4 — AI vs No-AI Baseline Comparison

**What:** Statistically test whether tasks with AI involvement differ from tasks without it on every key metric.  
**Why:** Before investigating *how much* AI affects margin, we must confirm *that* it does. This is the baseline test. It also helps evaluators see that the effect is real and significant, not noise. The project rubric explicitly asks: *"Where is value created?"* vs. *"Where are losses incurred?"*

| Step | Detail | Why |
|------|--------|-----|
| 4.1 | Grouped summary stats (mean/median/std) | Descriptive comparison across `ai_flag` |
| 4.2 | Welch t-tests on 6 metrics | Welch (not Student's t) because group variances are unequal — more robust |
| 4.3 | Boxplots for each metric | Visually show distribution overlap (or lack thereof) |
| 4.4 | Interpret results in markdown | Write 2–3 sentences: *"AI tasks have higher revenue but significantly worse rework rate (p<0.001)"* |

---

## Phase 5 — Non-linear Threshold Analysis

**What:** Find the specific AI usage level where margin transitions from positive to negative. This is the core deliverable of the project.  
**Why:** The project brief asks for a *"critical threshold beyond which AI usage harms the operating margin."* This must be a number, not a vague range. A simple average won't find it — we need curve estimation and formal breakpoint detection.

| Step | Method | Why |
|------|--------|-----|
| 5.1 | LOWESS: AI usage vs `hours_spent` | Non-parametric smoother — confirms whether AI actually reduces hours |
| 5.2 | LOWESS: AI usage vs `outcome_score` | Shows at what usage level quality starts degrading |
| 5.3 | LOWESS: AI usage vs `profit` and `margin_pct` | Reveals the inverted-U shape — peak then collapse |
| 5.4 | Bucket table (5 bands) | Exact mean profit, margin, rework, hours, outcome per AI usage band |
| 5.5 | Bar charts with peak ↑ and collapse ↓ annotations | Communicates the threshold visually for the presentation |
| 5.6 | **Piecewise / changepoint detection** (`ruptures` library) | Produces a precise numerical breakpoint (e.g., "0.47") instead of visual inspection of a range |

> **Why `ruptures`?** Piecewise linear regression detects where the slope of margin vs. AI usage changes sign — giving a rigorous, reproducible threshold value that evaluators can't argue with.

---

## Phase 6 — Hidden Cost & Mechanism Analysis

**What:** Quantify rework as a direct margin destroyer in euros and formally test the causal chain.  
**Why:** Project rule #2 asks *"where are losses incurred?"* and #3 asks *"AI → quality or just speed?"* Answering these requires converting rework into money AND testing whether the AI→margin path runs *through* rework (mediation) or independently.

| Step | Detail | Why |
|------|--------|-----|
| 6.1 | Estimate `rework_cost_est` | `rework_hours × median(cost_per_hour)` — converts the hidden cost to € |
| 6.2 | Compute `hidden_cost_factor` | `rework_cost_est / revenue` — shows what % of revenue is destroyed |
| 6.3 | Aggregate by `ai_bucket` | "In the 60–80% band, rework consumes €X of every €100 in revenue" |
| 6.4 | Find profit-negative rework threshold | At what `rework_rate` does mean profit cross zero? |
| 6.5 | **Mediation analysis** | Test: does `rework_rate` mediate the `ai_usage_pct → margin_pct` path? Use Baron-Kenny steps with `pingouin` or manual OLS |
| 6.6 | Pearson + Spearman r | Correlate `ai_usage_pct` with `rework_rate`, `profit`, `outcome_score` |

> **Why mediation?** It's the difference between *"AI and rework both correlate with margin"* (descriptive) and *"AI hurts margin because it causes rework, not for any other reason"* (mechanism). This directly answers project rule question #3 and satisfies the rubric's demand for a mechanism connecting 3+ variables.

---

## Phase 7 — Robustness Checks

**What:** Repeat the threshold analysis across sub-groups to confirm the effect is universal, not a data artifact.  
**Why:** If the effect only appears in one team or one pricing model, the recommendation would be very different. Evaluators expect you to check this. It also helps identify *which* segments are most vulnerable — informing targeted recommendations.

| Dimension | What you look for |
|-----------|-------------------|
| **Team** (Content/Media/Design/SEO) | Is the threshold consistent across all teams, or does Design/SEO behave differently? |
| **Seniority** (junior/mid/senior) | Do seniors maintain margin at higher AI usage? (hypothesis: yes, because they catch errors faster) |
| **Pricing model** (hourly/fixed/value_based) | Does value-based pricing degrade less? (hypothesis: yes, revenue scales with quality not time) |

After the chart, add a markdown cell with explicit interpretation: *"The effect is consistent across all 4 teams (threshold ≈ 50%), but seniors in value-based contracts are ~20% more resilient."*

---

## Phase 8 — Analytical Modeling

**What:** Use regression and interaction analysis to quantify the mechanism, and profiling to identify where AI genuinely adds value.  
**Why:** This is a Machine Learning course — some modeling is expected. More importantly, regression coefficients turn qualitative findings into boardroom-ready numbers. *"A 10-point increase in AI usage reduces margin by 2.3 percentage points, but this effect is 60% weaker for value-based contracts"* is far more actionable than a bar chart.

### 8a. OLS Regression (Mechanism Quantification)
```
margin_pct ~ ai_usage_pct + rework_rate + outcome_score
           + C(pricing_model) + C(seniority) + C(team)
```
- Reports: coefficients, p-values, R²
- Answers: holding everything else constant, what is the isolated effect of AI usage on margin?

### 8b. Interaction Term
```
margin_pct ~ ai_usage_pct * C(pricing_model)
```
- Tests whether hourly pricing *amplifies* the AI damage — the core structural problem
- If the interaction is significant, the business recommendation (switch to value-based) is statistically validated

### 8c. Value-Creation Profiling
- Identify tasks where AI usage > 0 AND margin_pct > 0 AND rework_rate is low
- Profile their characteristics: which team? which seniority? which pricing model? which task complexity?
- This answers project rule question #1: *"Identify the specific tasks where AI produces a measurable and positive impact"*

### 8d. Random Forest (Secondary Confirmation)
- Target: `margin_pct`; features: all engineered variables
- Use SHAP values (not just feature importance) for interpretable output
- Purpose: confirm OLS findings; flag any non-linear importance the OLS missed

---

## Phase 9 — Business Decision

**What:** Translate all findings into a single, concrete, actionable recommendation.  
**Why:** The rubric requires *"1 concrete decision"* and a *"clear business decision"* as an evaluation criterion. Vague advice ("use AI carefully") fails. The decision must be specific enough that a manager can act on it tomorrow.

**Structure:**
1. **Core finding** — one sentence, with a number: *"Beyond 47% AI usage, rework costs eliminate margin for hourly/fixed-price tasks."*
2. **The decision** — one concrete action: *"Cap AI usage at 50% for hourly/fixed tasks; mandate a QA gate at ai_usage > 0.5; pilot value-based pricing for high-AI projects in Q3 2026."*
3. **Decision table** — three rows: Encourage / Monitor / Restrict
4. **Quantified impact** — *"Reducing rework by 30% in the 60–80% AI bucket recovers an estimated ~€X per task across ~900 annual tasks"* (use your computed `rework_cost_est` to fill in the real number)

---

## Phase 10 — AI Reflection ⚠️ MANDATORY

**What:** A dedicated notebook section documenting how AI was used throughout the project.  
**Why:** The project rules state explicitly: *"The final answer is not evaluated; how you used AI to get there is evaluated."* This section IS the evaluation for the company component. Missing it is the single largest risk to your grade.

| Deliverable | What to write |
|-------------|---------------|
| **3 Key Insights** | Non-obvious findings — e.g., seniority as a moderator, pricing model as a structural flaw, the specific threshold value |
| **1 thing discovered thanks to AI** | Be specific: *"The mediation analysis was suggested by the AI co-pilot, which helped us realize we were conflating correlation with mechanism"* |
| **1 mistake made by AI** | Be honest and specific: *"The AI initially suggested using Pearson r for all correlations; this was wrong for non-linear relationships and we corrected it by adding Spearman"* |
| **Full prompt log** | Every significant prompt used, in order, with a short note on why and what changed after each iteration |

---

## Phase 11 — README Finalization

**What:** Update README so it reflects the actual results generated by the code.  
**Why:** Submission rules require figures and tables to be *"generated from the code."* Current README has placeholder qualitative labels instead of real numbers.

| Action | Why |
|--------|-----|
| Replace placeholder result tables with computed values from `threshold_df` | Rules require code-generated results |
| Embed `images/figX.png` figures using `![caption](images/fig.png)` | Rules require an `images/` folder referenced in README |
| Update Methods section with new techniques (mediation, changepoint, OLS) | README must reflect the actual methodology used |

---

## Phase 12 — Submission

**What:** Final run, packaging, and form submission.  
**Why:** Procedural but critical — a technically great project that fails on submission rules gets zero.

| # | Action |
|---|--------|
| 12.1 | Run notebook from scratch (Kernel → Restart & Run All) — verify zero errors |
| 12.2 | Confirm `images/` has all figures and they render on GitHub |
| 12.3 | Create presentation (PDF/PPTX) with the 5 mandatory deliverables from Phase 10 |
| 12.4 | Confirm repo name ends with captain's student ID |
| 12.5 | Submit GitHub URL via designated form **before May 1st at 11:59 PM** |

---

## Summary: What Each Phase Answers

| Phase | Project Rule Question Answered |
|-------|-------------------------------|
| 1–2 | Foundation — enables all downstream analysis |
| 3 | Data quality validation, pattern discovery |
| 4 | "Where is value created?" / "Where are losses incurred?" |
| 5 | "When does it become negative?" → the threshold number |
| 6 | "AI → quality or just speed?" → via mediation analysis |
| 7 | Confirms the effect is real across all segments |
| 8 | Quantifies the mechanism; profiles the "safe" AI usage zone |
| 9 | "1 concrete decision" |
| 10 | The entire company evaluation component |
| 11–12 | Submission compliance |
