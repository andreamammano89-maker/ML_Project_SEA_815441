# ML_Project_SEA_815441
Machine Learning Project (A.Y. 2025/2026) - Alkemy: AI Productivity. Team SEA.


# AI Productivity Paradox: Efficiency vs Profitability

## Overview

This project investigates the relationship between Artificial Intelligence (AI) usage and business performance at the task level. 

While AI is commonly associated with increased efficiency, its impact on profitability is less clear. This analysis explores whether higher AI adoption consistently leads to better financial outcomes, or whether hidden costs offset its benefits.

---

## Research Objective

The main objective of this project is to evaluate the following research question:

> Does increased AI usage improve profitability, or does it introduce inefficiencies that offset its benefits?

The analysis focuses on identifying potential trade-offs between:
- efficiency (time reduction),
- quality (rework and errors),
- and financial performance (profit).

---

## Dataset

The dataset contains **3,248 observations**, each representing a single task performed within a business workflow.

### Key Features:
- **AI Usage**: `ai_usage_pct`
- **Efficiency**: `hours_spent`, `billable_hours`
- **Quality**: `rework_hours`, `errors`, `revisions`
- **Financials**: `revenue`, `cost`, `profit`
- **Contextual Variables**: task complexity, seniority, client tier, etc.

---

## Methodology

The analysis follows a structured Exploratory Data Analysis (EDA) approach:

1. **Data Cleaning**
   - Handling missing values with variable-specific strategies
   - Preserving key variables (e.g., AI usage) without imputation to avoid bias

2. **Univariate Analysis**
   - Distribution of profit, AI usage, time, and rework
   - Identification of skewness and extreme values

3. **Bivariate Analysis**
   - AI vs profit, time, and quality metrics
   - Use of scatter plots and regression lines

4. **Interaction Analysis**
   - Investigation of how AI affects profit indirectly through rework

5. **Threshold Analysis**
   - Identification of non-linear effects using AI usage bins
   - Detection of a critical range where AI impact changes

---

## Key Findings

### 1. AI Improves Efficiency
Higher AI usage is associated with reduced execution time, confirming its role in improving operational efficiency.

---

### 2. AI Increases Variability in Quality
AI usage is associated with increased dispersion in rework and errors, indicating potential instability in output quality.

---

### 3. Profitability is Not Linearly Affected
The relationship between AI usage and profit is weak and highly variable. Increased AI usage does not guarantee higher profitability.

---

### 4. Existence of a Threshold Effect
The analysis indicates a threshold around **40–60% AI usage**:

- Below this range: AI improves efficiency with limited negative effects  
- Above this range: increased rework and variability offset efficiency gains  

---

### 5. The AI Productivity Paradox

The results support the existence of an **AI productivity paradox**:

> Improvements in efficiency do not necessarily translate into improved financial performance.

This occurs because gains in speed are partially offset by hidden costs related to quality and rework.

---

## Business Implications

From a managerial perspective:

- AI should not be treated as a purely productivity-enhancing tool  
- Excessive reliance on AI may introduce hidden operational costs  
- Optimal performance requires balancing:
  - efficiency gains  
  - quality control  

Additionally, AI increases not only expected performance but also **risk**, making outcomes more variable and less predictable.

---

## Conclusion

AI does not uniformly improve performance. Instead, it reshapes the distribution of outcomes by:

- increasing efficiency,
- increasing variability,
- and creating both upside potential and downside risk.

Understanding this trade-off is essential for leveraging AI effectively in business operations.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

---
