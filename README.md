# ML Project SEA — AI Productivity Analysis
**A.Y. 2025/2026 | Alkemy | Team SEA**

---

## Introduction

This project investigates the economic impact of AI-assisted work on task profitability at a marketing agency. The core research question is:

> **Beyond which AI usage threshold does rework destroy profit margin?**

While AI tools are widely adopted to accelerate content and creative workflows, their effect on financial outcomes is non-trivial. Speed gains may be offset by rising error rates, unbillable rework hours, and pricing structures that don't reward efficiency.

---

## Dataset

| Property | Value |
|---|---|
| Source | `data/ai_productivity_dataset_final.csv` |
| Observations | 3,248 tasks |
| Features | 34 variables |
| Period | Jul 2025 – May 2026 |
| Unit of analysis | Individual task |

**Key variables:**

| Variable | Description |
|---|---|
| `ai_usage_pct` | Proportion of task completed with AI assistance (0–0.93) |
| `profit` | Revenue − Cost (€) |
| `rework_hours` | Hours spent fixing AI-generated errors |
| `outcome_score` | Client-rated quality score (0–100) |
| `hours_spent` | Total hours per task |
| `pricing_model` | `hourly` / `fixed` / `value_based` |
| `team` | Content / Media / Design / SEO |
| `seniority` | junior / mid / senior |

---

## Methods

### Data Cleaning
- Deduplicated on `task_id`
- Normalized `team` column (typos and case inconsistencies)
- Parsed date fields (`created_at`, `delivered_at`)
- Imputed missing numeric values with median (< 5% missing per field)
- Capped outliers using IQR × 3 method

### Feature Engineering

| Feature | Formula |
|---|---|
| `margin_pct` | `profit / revenue × 100` |
| `ai_flag` | `ai_usage_pct > 0` |
| `rework_rate` | `rework_hours / hours_spent` |
| `billable_ratio` | `billable_hours / hours_spent` |
| `ai_bucket` | 5 bins: 0–20%, 20–40%, 40–60%, 60–80%, 80–100% |

---

## Experimental Design

The analysis tests whether AI usage has a **monotonic or non-linear** effect on profitability. We:

1. Compare AI vs. non-AI tasks on key metrics (Welch t-tests)
2. Fit LOWESS smoothers to AI usage vs. profit/rework curves
3. Bucket AI usage into 5 bands and compute average margin per band
4. Segment results by team, seniority, and pricing model (robustness)

---

## Results

### AI vs No-AI Comparison

Tasks with AI assistance show higher average revenue but also significantly higher rework rates and lower outcome scores (all differences statistically significant, p < 0.05).

### Non-linear Threshold Effect

The relationship between AI usage and profit margin is **not linear**. Margin improves from 0–40% AI usage, then declines sharply:

| AI Usage Band | Mean Margin | Rework Rate |
|---|---|---|
| 0–20% | Moderate+ | Low |
| 20–40% | **Peak** | Low-Medium |
| 40–60% | Declining | Medium |
| 60–80% | Negative territory | High |
| 80–100% | Worst | Highest |

### Core Mechanism

```
HIGH AI USAGE → ↑ Speed (unbillable under fixed pricing)
              → ↑ Errors → ↑ Rework hours (unbillable cost)
                         → ↓ Outcome score → ↑ Revisions
                                           → ↓ MARGIN
```

### Robustness

- Effect is **consistent across all 4 teams**
- **Senior staff** maintain better margin at high AI usage (lower rework rate)
- **Value-based pricing** shows less margin degradation at high AI usage (revenue scales with output quality, not hours)

---

## Conclusions & Recommendations

> **"Beyond ~40–60% AI usage, rework costs systematically destroy profit margin regardless of speed gains."**

| Scenario | Action |
|---|---|
| AI usage < 40% | ✅ Encourage — net positive margin impact |
| AI usage 40–60% | ⚠️ Monitor — enforce QA checkpoints |
| AI usage > 60% | 🚫 Limit or restructure pricing model |

**Structural recommendations:**
1. **Renegotiate pricing** for high-AI tasks → shift to value-based contracts
2. **Mandatory QA gate** when `ai_usage_pct > 0.5`
3. Track `rework_hours` as a mandatory field for all tasks
4. **Senior oversight** on AI-heavy tasks

---

## Project Structure

```
ML_Project_SEA_815441/
├── data/
│   └── ai_productivity_dataset_final.csv
├── main.ipynb          ← complete pipeline (40 cells)
├── README.md
└── pyproject.toml
```

## Reproducibility

```bash
# Install dependencies
uv sync

# Run full pipeline
uv run jupyter nbconvert --to notebook --execute main.ipynb --output main_executed.ipynb

# Or open interactively
uv run jupyter lab main.ipynb
```

---

*Team SEA — Alkemy AI Productivity Project, 2025/2026*
