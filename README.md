# AI Productivity Analysis — Alkemy Project  

**Group SEA · Machine Learning Project · A.Y. 2025/2026 · LUISS**

Simone Cavallaro - 815441, Andrea Mammano - 821311, Elisa Presciutti - 812891

---

## 1. Introduction  

This project analyzes how Artificial Intelligence affects business workflows, with a specific focus on the relationship between productivity and profitability. While AI is typically associated with faster execution and increased output, the key question is whether this efficiency actually translates into economic value. The objective is therefore not simply to measure performance improvements, but to understand the mechanism through which AI impacts time, quality, and ultimately margins.  

### 1.1 Repository Structure  

The project is organized to ensure clarity, reproducibility, and a clear separation between data processing, feature engineering, and analysis.

```
├── README.md             # Project documentation  
├── data/                 # Raw and cleaned datasets  
├── notebooks/            # Main analysis notebook  
│   └── main.ipynb  
├── images/               # Figures used in the report  
├── src/ (optional)       # Helper functions (if any)  
└── requirements.txt      # Python dependencies  
```

This structure ensures that all results presented in the report can be directly traced back to the corresponding code and data.

### 1.2 Environment and Reproducibility

The analysis was conducted using Python and standard data science libraries.  
All results can be fully reproduced by running the main notebook.

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

## 2. Methods  

### 2.1 Data Understanding 

The dataset contains 3,248 observations and 34 variables, where each row represents a single task within a business workflow.  

The variables can be grouped into three main types:

| Variable Category | Examples | Description |
|------------------|---------|------------|
| Numerical | hours_spent, revenue, profit | Continuous operational and financial measures |
| Categorical | team, task_type, client | Task and organizational attributes |
| Boolean | ai_assisted, sla_breach, scope_change_flag | Binary process indicators |

Overall, the dataset captures four core dimensions of the system: operational efficiency (time and workload), AI adoption (presence and intensity of AI), quality and process performance (errors, revisions, rework), and economic outcomes (profitability).  

Descriptive statistics highlight a highly heterogeneous structure. Financial variables such as revenue and profit are strongly right-skewed: while the average profit is around 349, the distribution spans from large losses (≈ -8510) to very high gains (≈ 14000). This indicates that performance is driven by a limited number of extreme cases rather than typical observations. A similar pattern emerges in operational variables such as hours_spent and rework_hours, where most tasks are relatively contained but a small subset exhibits extremely high values, suggesting the presence of complex or inefficient workflows.  

AI usage (ai_usage_pct) ranges from 0% to 93%, with an average around 36%, showing that AI is not uniformly adopted but varies significantly across tasks. This variability is crucial, as it enables the analysis of non-linear effects and potential threshold dynamics rather than simple binary comparisons. At the same time, quality-related variables (errors, revisions, rework_hours) reveal a concentration of inefficiencies: most tasks show low values, while a minority accounts for a disproportionate share of rework, suggesting that quality issues are not randomly distributed but structurally linked to specific conditions.  

From a data quality perspective, several variables contain missing values (e.g., ai_usage_pct, rework_hours, billable_hours, outcome_score). These are not treated as purely technical issues, but as part of the operational context, potentially reflecting incomplete tracking or process gaps.  

Everything considered, the dataset is characterized by strong asymmetries, extreme values, and heterogeneous behavior across tasks. For this reason, these patterns are preserved rather than removed, as they are central to understanding how AI affects both efficiency and value creation.


### 2.2 Data Preparation  

The data cleaning process was designed to improve consistency and analytical reliability while preserving the structural properties of the dataset, which are essential for understanding the relationship between AI usage and economic outcomes.  

First, duplicate observations were identified at the task level. While no full-row duplicates were present, 48 duplicated task_id entries were detected. Since each task should represent a unique unit of analysis, duplicates were resolved by sorting observations by updated_at and retaining only the most recent record. This choice ensures that the analysis reflects the latest available state of each task, avoiding inconsistencies due to outdated information.  

Text-based categorical variables required normalization to avoid artificial fragmentation of categories. In particular, team and task_type contained inconsistencies due to casing, formatting differences, and typographical errors. These were standardized through string cleaning and explicit mapping rules, reducing noise and consolidating categories into meaningful groups. As a result, team was reduced to four balanced categories (Content, Media, Design, SEO), while task_type was mapped into seven canonical classes (Ad, Article, Design, Development, Release, Report, Ticket). This step is critical for ensuring that downstream grouped analyses and comparisons are statistically meaningful and not distorted by labeling inconsistencies.  

Temporal variables were then processed to extract operational insights. The columns created_at and delivered_at were converted into datetime format, enabling the construction of a new feature, duration_days, representing task completion time. During validation, 14 negative durations were identified, likely caused by data entry errors or misaligned timestamps. Rather than forcing corrections, these values were set to missing (NaN) to avoid introducing artificial bias. The resulting duration feature provides a reliable measure of workflow timing, ranging from 0 to 10 days with an average of 4.5 days.  

The legacy_ai_flag variable required standardization due to inconsistent encoding (mixed casing and presence of an “unknown” category). Values were normalized and mapped into a binary format (1 for True, 0 for False), while “unknown” entries were explicitly treated as missing. This decision preserves information about uncertainty instead of forcing arbitrary classification, which could distort the analysis of AI-related effects.  

To better capture actual AI adoption, a new feature (ai_indicator) was created based on whether ai_usage_pct is greater than zero. This allows distinguishing between tasks with and without measurable AI involvement. A correlation check between this indicator and the legacy_ai_flag revealed a near-zero relationship (-0.014), suggesting that historical AI labels do not align with current measurable usage. This finding highlights a structural inconsistency in how AI adoption is recorded and reinforces the need to rely on quantitative usage metrics rather than legacy flags.  

Overall, the data preparation phase ensures that the dataset is internally consistent, interpretable, and aligned with the analytical objective. Each transformation is designed not only to clean the data, but to preserve and clarify the mechanisms linking AI usage, workflow dynamics, and value creation.


### 2.3 Missing Values and Outlier Treatment  

The handling of missing values and extreme observations was designed to improve data reliability without altering the underlying structure of the system. In this project, cleaning is not treated as a purely technical step, but as a critical phase that directly affects the validity of the conclusions regarding the impact of AI on productivity and profitability.  

The initial analysis showed that missing values are present across several variables, with legacy_ai_flag (~10.5%) and jira_ticket (~10.3%) being the most affected, followed by ai_usage_pct and outcome_score. A missingness matrix was used to verify whether these gaps followed systematic patterns.

![Missing Data Matrix](images/image1.png)

The absence of clear structures (e.g., monotone missingness) suggests that missing values are likely generated by operational factors such as incomplete tracking or inconsistent reporting, rather than by a deterministic process. This distinction is important, as it justifies the use of imputation techniques without introducing structural bias. 

To further validate this assumption, the distributions of key variables were compared between the full dataset and the subset where ai_usage_pct is missing, as shown below:

![Missing AI Distribution Comparison](images/image2.png)

The similarity between these distributions indicates that missingness is approximately random with respect to the main variables of interest. This validation step is critical, as applying imputation without verifying this condition could distort the relationship between AI usage and performance.

The treatment strategy was therefore tailored to the nature of each variable. The jira_ticket column was removed because it acts solely as an identifier and has no explanatory power, while also exhibiting a high level of missingness. For continuous variables (ai_usage_pct, outcome_score, brief_quality_score, sla_days, rework_hours), median imputation was chosen instead of the mean to ensure robustness against skewed distributions and extreme values, which are prominent in this dataset.  

For billable_hours, a domain-informed approach was applied: missing values were replaced with 85% of hours_spent. This choice reflects a realistic billing efficiency assumption and preserves the logical relationship between effort and billed time. Using a purely statistical imputation here would ignore the economic meaning of the variable and risk introducing incoherent values.  

Invalid observations (such as negative billable_hours) were first converted to missing before imputation, ensuring that corrections do not propagate erroneous values. Binary variables (scope_change_flag and sla_breach) were explicitly converted to boolean to prevent them from being treated as continuous features in subsequent analyses, which would lead to incorrect statistical interpretations. Residual missing values in variables such as delivered_at and legacy_ai_flag were intentionally preserved, as forcing imputation would introduce artificial certainty into fields that inherently reflect uncertainty or incomplete information.  

Outlier analysis was conducted using boxplots and the 3×IQR rule as a diagnostic tool. However, no capping or removal was applied. The key reason is that statistical outliers in this dataset correspond to economically meaningful observations. For example, extremely high revenue values represent large contracts, high hours_spent indicate complex tasks, and large rework_hours capture severe quality failures. Removing or capping these values would systematically exclude the most informative cases, introducing selection bias and weakening the analysis of value creation.  

Instead of modifying the data, robustness was ensured through methodological choices. In particular, both Pearson and Spearman correlations were computed. Pearson captures linear relationships, while Spearman, being rank-based, is less sensitive to skewness and extreme values, as illustrated below:

![Spearman Correlation Heatmap](images/image3.png)

This dual approach allows distinguishing between apparent linear effects and more general monotonic relationships, which are more appropriate given the nature of the data.  

The correlation analysis confirms expected structural relationships within the workflow: cost is strongly driven by hours_spent, revenue scales with profit, and quality issues (errors and rework) negatively impact outcome_score. Importantly, ai_usage_pct shows weak linear correlations with profit and quality-related variables. This result is not interpreted as evidence of irrelevance, but as an indication that the effect of AI is not linear. This directly motivates the next phase of the analysis, where threshold effects and segmented behaviors are explored.  

Overall, this phase ensures that the dataset remains both clean and representative. Every transformation is explicitly justified to avoid introducing bias, while preserving the variability and extreme cases that are central to understanding how AI affects both efficiency and economic value.

### 2.4 Feature Engineering 

The feature engineering phase transforms the cleaned dataset into a set of variables specifically designed to capture the economic and operational mechanisms underlying the impact of AI on performance. The objective is not to increase the dimensionality of the dataset arbitrarily, but to construct interpretable indicators that directly relate to the research question: how AI usage affects efficiency, quality, and ultimately profit margins.

All original variables (e.g., profit, revenue, hours_spent) are preserved without modification. This is a deliberate choice: these variables represent real business outcomes, including extreme but meaningful cases. Instead of altering them, new variables are derived to normalize, contextualize, and decompose their relationships.

A first group of features focuses on profitability normalization and comparability. The profit_margin variable expresses profit as a percentage of revenue, allowing comparisons across tasks of very different sizes. Without this transformation, large contracts would dominate the analysis, masking underlying patterns.

A second group captures operational efficiency and quality dynamics. Variables such as rework_rate and error_rate measure inefficiency relative to the time invested, while billable_ratio indicates how much of the effort is actually monetized. At the same time, revenue_per_hour and cost_per_hour translate operational activity into economic productivity metrics, making it possible to observe whether AI affects value creation per unit of effort.

Particular attention is given to rework, which is treated as the primary channel through which inefficiencies translate into economic loss. To make this mechanism explicit, rework_hours is converted into a monetary estimate (rework_cost_est) using the median cost per hour as a robust proxy. This avoids distortions caused by extreme values and ensures that rework is interpreted consistently across tasks. The hidden_cost_ratio further expresses this cost as a share of total cost, providing a direct measure of inefficiency at the economic level.

![Rework Cost → Profit Margin](images/image7.png)

A third group of features is designed to capture the structure and intensity of AI adoption. The ai_flag variable provides a simple binary distinction between tasks with and without AI, useful for baseline comparisons. However, the analysis does not rely on this binary split alone. AI usage is also represented through a quadratic term (ai_usage_sq) to allow for non-linear effects, and through ordered buckets (ai_bucket) that segment tasks into five levels of intensity (from 0–20% up to 80–100%). This structure is essential for identifying threshold effects rather than assuming a continuous linear relationship.

This approach captures the non-linear behavior observed between AI usage and profit outcomes.

![Non-linear relationship: AI → Profit](images/image8.png)

To complement this, the is_high_ai variable isolates high-intensity usage (above 60%), allowing a direct comparison between shallow and deep adoption. This distinction becomes critical in later stages of the analysis.

![AI Mechanism: Speed, Quality, Value](images/image9.png)

Finally, additional segmentation variables are introduced to control for contextual heterogeneity. Tasks are grouped by economic scale (budget_bucket), profitability level (profit_bucket), and operational complexity (complexity_bucket). These dimensions allow testing whether the impact of AI is uniform or concentrated in specific types of tasks.

From a technical perspective, all ratio-based variables are computed using a stabilized denominator. Since a small number of tasks exhibit extremely low recorded hours (likely due to logging errors), hours_spent is clipped to a minimum threshold before calculating ratios. This adjustment affects only a negligible fraction of observations and prevents artificially inflated values, without altering the underlying data distribution.

The resulting dataset includes both the original variables and a set of derived features that explicitly encode efficiency, cost structure, and AI adoption patterns. This transformation is essential: without it, the analysis would remain descriptive. With it, the dataset becomes a structured representation of the mechanisms through which AI may influence productivity and profitability, enabling the causal interpretation developed in the next section.

| Feature | Definition | Purpose |
|--------|-----------|--------|
| profit_margin | profit / revenue | Normalize profitability across tasks |
| rework_rate | rework_hours / hours_spent | Measure operational inefficiency |
| error_rate | errors / hours_spent | Capture quality issues relative to effort |
| billable_ratio | billable_hours / hours_spent | Monetization efficiency |
| revenue_per_hour | revenue / hours_spent | Productivity per unit of work |
| cost_per_hour | cost / hours_spent | Cost intensity |
| rework_cost_est | rework_hours × median cost/hour | Monetary value of inefficiency |
| hidden_cost_ratio | rework_cost_est / cost | Share of cost absorbed by rework |
| ai_flag | ai_usage_pct > 0 | Binary AI usage indicator |
| ai_bucket | categorical bins of ai_usage_pct | Capture threshold effects |
| ai_usage_sq | (ai_usage_pct)^2 | Model non-linear relationships |
| is_high_ai | ai_usage_pct > 60% | Identify high-intensity AI usage |

## 3. Experimental Design  

## 3.1 Setup and Data Preparation

The objective is to ensure that all models are trained and evaluated under consistent conditions, so that differences in performance can be attributed to modelling assumptions rather than to inconsistencies in the data.

The analysis focuses on the relationship between operational variables, AI usage intensity, and economic outcomes. The target variable is defined as profit margin, computed as the ratio between profit and revenue expressed in percentage terms. This transformation normalizes profitability across tasks of different sizes, allowing the models to capture efficiency rather than scale effects. Without this normalization, larger contracts would dominate the analysis and bias the results.

The modelling strategy is structured in two parallel configurations. In the first configuration, models are trained on the original feature space. In the second configuration, the same modelling pipeline is applied after transforming a subset of right-skewed numerical variables using a logarithmic transformation. This design isolates the impact of distributional corrections on model performance, particularly for linear methods that are sensitive to skewness. The pipeline remains identical across the two configurations, ensuring that any observed difference is attributable solely to the transformation.

Binary variables related to process conditions, such as SLA breaches, scope changes, and AI assistance, are converted into numeric format. This ensures that they are treated as quantitative signals rather than categorical labels, avoiding unnecessary encoding and preserving interpretability. Representing these variables as 0/1 indicators allows the models to capture their direct effect on the target without introducing artificial complexity.

Before modelling, a strict feature exclusion process is applied to remove variables that are not admissible. Variables that are algebraically derived from the target, including profit, revenue, cost, and related ratios, are excluded to prevent data leakage. Including these variables would introduce circular relationships and artificially inflate predictive performance.

Identifiers such as task_id, project_id, client, and created_by are removed, as they do not contain predictive information and may introduce spurious correlations. Raw timestamp variables are excluded because their informational content is already captured by derived features such as task duration. Post-hoc variables that are not available at prediction time are also removed to ensure that the modelling setup reflects a realistic decision-making scenario.

After this filtering process, the remaining variables define the set of candidate features. All admissible features are retained and passed to the modelling stage without manual selection, allowing the models to identify relevant patterns directly from the data.

The resulting feature space includes both numerical and categorical variables. Numerical variables capture operational intensity, efficiency, and quality metrics, while categorical variables represent structural characteristics such as team, task type, and pricing model. The dataset includes 30 candidate features, of which 21 are numerical and 9 are categorical.

### Feature Overview  

| Feature | Numeric | Categorical |
|--------|--------|------------|
| task_complexity_score | X | |
| brief_quality_score | X | |
| scope_change_flag | X | |
| sla_days | X | |
| sla_breach | X | |
| hours_spent | X | |
| billable_hours | X | |
| ai_usage_pct | X | |
| ai_assisted | X | |
| revisions | X | |
| errors | X | |
| rework_hours | X | |
| outcome_score | X | |
| legacy_ai_flag | X | |
| duration_days | X | |
| ai_flag | X | |
| rework_rate | X | |
| error_rate | X | |
| billable_ratio | X | |
| ai_usage_sq | X | |
| is_high_ai | X | |
| client_tier | | X |
| team | | X |
| task_type | | X |
| seniority | | X |
| deadline_pressure | | X |
| pricing_model | | X |
| content_version | | X |
| ai_bucket | | X |
| complexity_bucket | | X |


A unified preprocessing pipeline is defined to ensure consistency across all models and experimental configurations. Numerical variables are imputed using the median, a choice motivated by the presence of skewed distributions and extreme values. Median imputation provides robustness and prevents outliers from distorting the imputed values. After imputation, numerical features are standardized to ensure that differences in scale do not affect models that rely on magnitude, such as Linear Regression and Lasso.

Categorical variables are handled through most-frequent imputation followed by one-hot encoding. This transformation allows the model to capture differences across categories without imposing any artificial ordering. The encoding procedure also handles unseen categories, ensuring stability when the model is applied to new data.

The same preprocessing pipeline is reused across all models and across both experimental configurations. This guarantees comparability of results and ensures that differences in performance are driven by modelling choices rather than by inconsistencies in data preparation.

The dataset is split into training and test sets using an 80/20 partition. The training set is used for model estimation and cross-validation, while the test set is reserved for out-of-sample evaluation. A fixed random seed is used to ensure reproducibility.

---

### Train/Test Split  

| Set   | Rows | Share |
|-------|------|-------|
| Train | 2560 | 80%   |
| Test  | 640  | 20%   |


Model performance is evaluated using complementary metrics. R² measures the proportion of variance explained by the model. Mean Absolute Error provides an interpretable measure of average prediction error in percentage points. Root Mean Squared Error penalizes larger deviations more strongly, capturing the impact of extreme errors. Cross-validated R² on the training set provides a more robust estimate of generalization performance and reduces the risk of overfitting.

## 3.2 Lasso Feature Selection  

The first step of the modelling process focuses on identifying which features carry predictive signal for profit margin. Given the relatively high dimensionality of the encoded feature space and the presence of potentially redundant variables, a feature selection procedure is required before fitting interpretable models.

Lasso regression is employed for this purpose. The method introduces an L1 regularization term that penalizes the absolute magnitude of coefficients, forcing less relevant features to shrink exactly to zero. This property makes Lasso particularly suitable for feature selection, as it performs variable selection and regularization simultaneously, reducing model complexity while retaining the most informative predictors.

The regularization strength is selected through cross-validation. Specifically, LassoCV with 5-fold cross-validation is used to identify the value of the penalty parameter that minimizes the mean squared error on held-out data. This ensures a balance between bias and variance, avoiding both overfitting and excessive shrinkage. A sufficiently high number of iterations is used to guarantee convergence over the full candidate feature set.

Lasso is not used as a predictive model in this context, but strictly as a selection mechanism. The objective is to extract a subset of relevant features that can be used in subsequent models. For this reason, Lasso is excluded from the final model comparison and treated as an intermediate step in the modelling pipeline.

The results show that the model retains 32 features out of 57 encoded variables, while the remaining 25 coefficients are shrunk to zero. This confirms the presence of redundancy in the feature space and highlights the importance of regularization in isolating the relevant signal.

#### Lasso Selection 

| Metric | Value |
|--------|------|
| Best alpha (LassoCV) | 0.5697 |
| Features retained | 32 / 57 |
| Features zeroed out | 25 / 57 |

The retained coefficients provide an initial indication of the direction and relative importance of the relationships between predictors and profit margin. Positive coefficients indicate variables associated with higher margins, while negative coefficients capture factors that systematically reduce profitability.

The strongest positive associations are observed for variables related to task structure and monetization. In particular, junior seniority and value-based pricing show large positive coefficients, suggesting that tasks performed by less costly resources and contracts that decouple revenue from time tend to generate higher margins. Task complexity and intermediate levels of AI usage also appear among the positive predictors, indicating that structured tasks and moderate AI adoption contribute positively to profitability.

On the negative side, the most significant coefficients are associated with hourly pricing and senior seniority. The strong negative effect of hourly contracts reflects a structural mechanism: productivity gains reduce billable hours, limiting revenue and compressing margins. Similarly, tasks assigned to senior workers are associated with higher costs, which can reduce profitability unless compensated by higher pricing. Additional negative effects are linked to high time investment and specific task categories, confirming that longer and more resource-intensive tasks tend to erode margins.

Overall, the coefficient structure reveals that profitability is driven by a combination of operational efficiency, pricing strategy, and resource allocation. The results also suggest that the relationship between AI usage and profit margin is not purely linear, as variables capturing AI intensity appear alongside structural and economic drivers.

#### Lasso Coefficient Distribution  

![Lasso Coefficients](images/image10.png)


## 3.3 Linear Regression on Selected Features  

After identifying the relevant predictors through Lasso regularization, the analysis proceeds by estimating a Linear Regression model on the reduced feature set. This step serves a dual purpose. First, it provides a transparent and interpretable baseline model, where each coefficient can be directly associated with a marginal effect on profit margin. Second, it allows evaluating whether the variables selected by Lasso retain predictive power when combined in an unconstrained linear specification.

Since Lasso operates on the encoded feature space, the selected variables are first mapped back to their original columns. This ensures that the preprocessing pipeline can be consistently rebuilt using only the retained features, preserving the distinction between numerical and categorical variables. The model is then estimated using the same preprocessing logic adopted throughout the analysis, guaranteeing comparability across all modelling steps.

Model performance is evaluated using both hold-out and cross-validation metrics. The test set provides an unbiased estimate of predictive accuracy on unseen data, while 5-fold cross-validation on the training set offers a more robust assessment of generalisation.

### Model Performance  

| Metric | Value |
|--------|------|
| Test R² | 0.2048 |
| Test MAE (pp) | 32.06 |
| Test RMSE (pp) | 48.55 |
| 5-fold CV R² | 0.0832 |

The results indicate a modest explanatory power. The model captures approximately 20% of the variance in profit margin on the test set, while the lower cross-validated R² suggests limited generalisation. This gap reflects the complexity of the underlying system: profitability is influenced by multiple interacting factors, many of which are non-linear or context-dependent and cannot be fully captured by a linear specification.

Despite these limitations, the model provides valuable insights into the direction and relative importance of key drivers.

### Coefficient Analysis  

The distribution of coefficients highlights a clear economic structure behind profitability. Positive coefficients are associated with conditions that enhance efficiency or allow firms to retain value, while negative coefficients capture cost pressures and structural constraints.

Among the strongest positive effects, value-based pricing emerges as a dominant driver. This confirms that when revenue is decoupled from time, efficiency gains directly translate into higher margins. Similarly, junior seniority shows a strong positive association, reflecting lower labor costs relative to output. Task-related variables such as complexity and structured task types also contribute positively, suggesting that standardized workflows benefit more from optimization and AI support. Intermediate levels of AI usage appear among the positive coefficients, reinforcing the idea that moderate adoption improves performance.

On the negative side, the most pronounced effect is associated with hourly pricing. This result is structurally consistent: when revenue depends on billable hours, efficiency reduces the number of hours that can be invoiced, limiting profitability. Senior seniority also exhibits a strong negative coefficient, reflecting higher cost structures. Additional negative effects are linked to high time investment, specific task categories, and very high levels of AI usage, indicating that excessive reliance on AI may introduce inefficiencies or coordination costs.

Overall, the coefficient structure confirms that profit margin is not driven by a single factor but by the interaction between pricing mechanisms, cost structure, and operational efficiency. AI plays a role within this system, but its impact depends on how it interacts with these underlying conditions.

### Linear Regression Coefficients  

![Linear Regression Coefficients](images/image11.png)



## 3.4 Random Forest with Hyperparameter Tuning   

To extend the analysis beyond linear assumptions, a Random Forest model is introduced to capture non-linear relationships and interactions among variables. 

The model is trained on the full set of candidate features rather than the subset selected by Lasso. This choice reflects a methodological difference: Lasso performs variable selection under a linear constraint, while Random Forest can extract predictive signal from variables that may appear weak individually but become relevant through interactions. Including all features therefore allows the model to capture a richer representation of the underlying data-generating process.

Hyperparameters are selected through grid search with 5-fold cross-validation, focusing on the number of trees, maximum depth, and minimum number of observations per leaf. The inclusion of a minimum leaf size constraint is particularly important in this context, as it prevents the model from overfitting to small and noisy partitions of the data. The optimal configuration consists of 300 trees, unconstrained depth, and a minimum of 10 observations per leaf, indicating that a relatively flexible structure combined with regularization at the leaf level provides the best trade-off between fit and generalisation.

The evaluation shows a consistent and substantial improvement over the linear model.

| Metric | Linear Regression | Random Forest |
|--------|------------------|--------------|
| Test R² | 0.2048 | 0.3445 |
| Test MAE (pp) | 32.06 | 27.64 |
| Test RMSE (pp) | 48.55 | 44.08 |
| 5-fold CV R² | 0.0832 | 0.1844 |

The increase in R² of approximately 0.14 indicates that a significant portion of the variance in profit margin is explained by patterns that are not captured by a linear specification. The simultaneous reduction in both MAE and RMSE confirms that this improvement is not limited to aggregate fit but also translates into more accurate predictions at the individual task level.

This gap between models provides direct evidence of structural non-linearity in the relationship between operational variables, AI usage, and profitability. Part of the gain can be attributed to interaction effects, where the impact of one variable depends on the level of another. Another component arises from variables that were penalised by Lasso under a linear assumption but retain predictive value in a non-linear framework.

At the same time, the results highlight an important limitation. Even with a flexible model, a large share of variance remains unexplained. This suggests that profitability is influenced by factors not fully captured in the dataset, such as client-specific dynamics, pricing negotiations, or unobserved aspects of task complexity. The modelling results should therefore be interpreted as identifying systematic patterns rather than providing complete predictive control over outcomes.

![Model Comparison](images/image12.png)


## 4. Results
The results highlight three main findings. 
First, AI usage intensity is positively associated with profit margin, but only beyond moderate adoption levels. 
Second, the pricing model is the primary structural driver of profitability and moderates the effect of AI. 
Third, a substantial portion of the predictive signal is non-linear, as shown by the performance gap between Linear Regression and Random Forest.

### 4.1 Segmented Analysis  

The aggregate analysis showed a clear and robust pattern: profit margins increase as AI usage intensity rises. However, this result alone is not sufficient to explain the system, as it implicitly assumes that the effect of AI is homogeneous across all tasks, teams, and economic contexts. This assumption is unlikely to hold in practice. AI does not operate in a vacuum: its impact depends on how work is organized, on the type of task being performed, and on how that work is monetized. For this reason, the analysis is extended by introducing three structural dimensions — team, task type, and pricing model — with the objective of identifying where the aggregate trend is confirmed, where it weakens, and where it reverses, moving from a descriptive pattern to a structural understanding of the system.

At the team level, most units confirm the global pattern, but with important differences in stability. Content, Design, and Media all exhibit a clear monotonic increase in profit margin as AI usage rises. Content tasks move from 18.7% in the 0–20% bucket to 50.6% in the 80–100% bucket, Design increases from 18.6% to 49.2%, and Media from 21.4% to 51.9%. This indicates that in these teams, deeper AI adoption is effectively translated into economic value, with efficiency gains successfully converted into higher margins despite the increase in rework highlighted in earlier analyses. However, this pattern is not universal. The SEO team shows a clear deviation: margins increase up to 41.5% at the 60–80% level but drop sharply to 25.0% at the highest level of AI usage. This reversal is structurally important, as it represents the first case where increasing AI intensity leads to worse outcomes. It suggests that beyond a certain point, AI introduces inefficiencies, coordination issues, or quality degradation that are not compensated by higher revenue. In other words, the effectiveness of AI is mediated by organizational capability: the same level of usage can generate very different results depending on how well it is integrated into the workflow. Importantly, the companion count heatmap confirms that these patterns are supported by sufficient observations in most cells, although the highest AI bucket contains fewer tasks and should be interpreted with caution.

![Team Analysis](images/image4.png)

A stronger heterogeneity emerges when focusing on task types, where the nature of the work itself becomes the dominant factor. Some categories show a consistent and substantial improvement in profitability as AI usage increases. Development provides the clearest example, with margins rising from 19.8% at low AI usage to 70.8% in the highest bucket, followed by Design (22.2% → 56.9%) and Report (19.3% → 60.2%). These tasks are relatively structured and benefit from automation and augmentation, allowing AI to be deeply integrated into the workflow, improving both speed and output quality while translating these gains into economic value. In contrast, other task types display more unstable dynamics. Ad-related tasks peak at 41.2% in the 60–80% range but decline to 25.2% at the highest level, while Release tasks increase up to 50.5% before dropping to 39.6%. This suggests that in more iterative, creative, or coordination-heavy activities, excessive reliance on AI may generate additional rework cycles or reduce output quality, offsetting efficiency gains. At the same time, some categories reveal strong threshold effects: Ticket tasks start with very low margins (11.3%) but rise sharply to 58.7% at high AI usage, indicating that partial adoption is ineffective while deep integration produces substantial benefits. Overall, this confirms that AI does not act uniformly across tasks but amplifies their underlying structure: standardized and repeatable work benefits consistently, while ambiguous or iterative work exhibits diminishing or volatile returns at higher levels of AI intensity.

![Task Analysis](images/image5.png)

The most decisive differences emerge when introducing the pricing model dimension, which defines how productivity gains are translated into economic outcomes. The distribution of pricing models shows that hourly contracts account for the largest share of tasks (48.2%), followed by fixed-price (37.6%) and value-based contracts (14.2%). This composition is critical, as it determines whether efficiency gains are captured as profit or transferred to the client. The results reveal three distinct and structurally different profiles. Under value-based pricing, margins are consistently high and increase steadily with AI usage, from 38.6% to 68.1%. Fixed-price contracts show a similarly strong upward trend, from 23.2% to 70.2%, confirming that when revenue is decoupled from time, reductions in cost directly translate into higher margins. In both cases, AI-driven efficiency gains are fully retained by the firm. The hourly model, however, behaves fundamentally differently. Margins increase only moderately from 13.3% to a peak of 26.3% at the 60–80% level, and then decline to 19.5% at the highest AI intensity. This is the only segment in the entire analysis where more AI leads to lower profitability. The mechanism is direct and entirely economic: when revenue depends on billable hours, efficiency reduces the number of hours that can be billed, and therefore reduces revenue. As a result, the cost savings generated by AI are offset, and eventually dominated, by lost revenue. This finding is fully consistent with the code-based aggregation and confirms that the pricing model is the primary moderator of AI’s economic impact.

![Pricing Model](images/image6.png)

Taken together, these results fundamentally refine the interpretation of the aggregate findings. While the overall data suggested that AI consistently improves profit margins, the segmented analysis shows that this effect is conditional and driven by specific contexts rather than being universal. AI creates value when three conditions are simultaneously satisfied: the task is structured enough to benefit from automation or augmentation, the team is capable of integrating AI effectively into its workflow, and the pricing model allows efficiency gains to be retained rather than transferred. When these conditions are not met, the effect weakens or even reverses. In particular, the combination of high AI usage and hourly pricing represents a structural constraint, where productivity gains translate into revenue loss instead of increased profitability. This section therefore provides the missing explanatory layer of the system: AI does not inherently increase or decrease margins, its impact depends on how it interacts with operational structure, task characteristics, and economic incentives.

## 4.2 SHAP Analysis
### SHAP Analysis: Random Forest (Section 1)

To interpret the behaviour of the Random Forest model beyond aggregate performance metrics, SHAP (SHapley Additive exPlanations) is used to decompose each prediction into the contribution of individual features. This approach provides a consistent and theoretically grounded measure of feature importance, capturing both linear effects and complex non-linear interactions learned by the model.

The beeswarm plot reports the top 15 features ranked by mean absolute SHAP value. Each point represents a single observation, positioned according to the feature’s contribution to the predicted profit margin. The colour gradient reflects the magnitude of the feature value, allowing a simultaneous reading of both direction and intensity of the effect.

![SHAP Beeswarm](images/image13.png)

The ranking highlights a clear hierarchy of drivers. The pricing model emerges as the dominant determinant of profitability, with pricing_model_hourly showing the largest absolute impact. High values of this feature are consistently associated with negative SHAP values, indicating a systematic reduction in profit margin. This result is structurally consistent with the economic mechanism identified earlier: under hourly billing, productivity gains reduce billable hours rather than increasing margins.

The second most important variable, billable_ratio, shows a strong positive association with profit margin. Tasks where a larger share of total hours is billable contribute positively to profitability, confirming that monetisation efficiency is a central driver of economic performance.

Seniority also plays a major role. Senior workers are associated with negative contributions to margin, while junior profiles show positive effects. This pattern reflects cost structure differences: higher labour costs associated with senior roles are not always offset by proportional increases in revenue, particularly in pricing models where efficiency gains are not retained.

Operational variables reinforce this interpretation. Hours_spent has a strong negative contribution, indicating that longer tasks tend to compress margins, while task complexity and rework-related measures contribute with smaller but still relevant effects. These findings confirm that inefficiencies and extended execution times translate directly into economic loss.

AI usage, measured through ai_usage_pct, appears significantly lower in the ranking (14th position). Its mean absolute SHAP value is substantially smaller than the top drivers, indicating that its direct contribution to profit margin is limited when considered in isolation. However, this does not imply irrelevance. Rather, it suggests that the effect of AI is conditional and mediated by other structural variables.

This becomes evident in the dependence plot.

![SHAP Dependence](images/image14.png)

The relationship between ai_usage_pct and the predicted outcome is clearly non-linear. SHAP values increase progressively as AI usage rises, moving from negative contributions at low levels to positive contributions at higher levels. This confirms the threshold effect already observed in the exploratory analysis: AI begins to generate economic value only beyond a certain level of adoption.

The interaction colouring reveals an additional layer of interpretation. The positive effect of AI usage is substantially stronger in non-hourly contexts, while it is attenuated under hourly pricing. This confirms that the economic impact of AI is not intrinsic, but depends on how productivity gains are captured within the pricing structure.

A cross-check between SHAP and Lasso results further strengthens the robustness of these findings. A large majority of the top SHAP features are also retained by Lasso, indicating strong agreement between linear and non-linear methods on the main drivers of profitability. The remaining features identified only by SHAP, including ai_usage_pct, contribute through interaction effects and non-linear patterns that cannot be captured by L1 regularisation.

Overall, the SHAP analysis provides a coherent interpretation of the system. Profitability is primarily driven by structural factors such as pricing model, monetisation efficiency, and cost structure. AI contributes positively, but its effect is indirect, non-linear, and contingent on the surrounding economic context. This explains why its impact appears limited in aggregate metrics, while still playing a meaningful role when interacting with other variables.

### SHAP Analysis: Random Forest (Section 2)

SHAP values are computed on the Random Forest trained on the log-transformed dataset using the same procedure adopted in Section 1. The objective of this step is not to extract new feature importance rankings, but to verify whether the relationships previously identified remain stable after correcting for strong right-skew in key numerical variables. Because Random Forest is invariant to monotonic transformations, applying a log1p transformation should not alter feature importance or the structure of the model. Any deviation would indicate that the Section 1 results were partially driven by scale effects rather than genuine predictive relationships, making this phase a robustness check rather than a new modelling step.

![SHAP beeswarm — log-transformed dataset](images/image15.png)

The beeswarm plot confirms complete stability in the feature importance ranking. The same 15 features appear in identical order as in Section 1, with mean absolute SHAP values matching up to four decimal places. The dominant drivers of profit_margin remain pricing_model_hourly, billable_ratio, and seniority_senior, followed by hours_spent. Pricing model continues to exert the strongest influence, with hourly billing associated with large negative SHAP values, indicating systematic margin compression. Billable_ratio remains the strongest positive operational driver, reflecting the direct link between billable efficiency and profitability, while seniority captures structural differences in task allocation and cost composition. The fact that these relationships remain unchanged after transforming the data confirms that they are not artefacts of skewed distributions, but represent stable and economically meaningful drivers of margin.

| Rank | Feature | Mean |SHAP| |
|------|--------|------------|
| 1 | pricing_model_hourly | 15.23 |
| 2 | billable_ratio | 12.49 |
| 3 | seniority_senior | 10.15 |
| 4 | hours_spent | 7.20 |
| 5 | seniority_junior | 3.60 |
| 6 | task_complexity_score | 3.12 |
| 7 | task_type_Ad | 1.40 |
| 8 | billable_hours | 1.31 |
| 9 | seniority_mid | 1.13 |
| 10 | rework_rate | 1.09 |
| 11 | complexity_bucket_low | 0.94 |
| 12 | task_type_Release | 0.93 |
| 13 | outcome_score | 0.71 |
| 14 | ai_usage_pct | 0.70 |
| 15 | ai_usage_sq | 0.68 |

Within this ranking, ai_usage_pct remains in position 14 out of 57 features, confirming that AI usage intensity has a measurable but relatively modest direct contribution to profit_margin compared to structural and operational factors. This result is consistent with the interpretation developed in Section 1, where AI does not emerge as a dominant standalone driver, but rather as a conditional factor whose effect depends on the surrounding context.

![SHAP dependence plot — ai_usage_pct (log-transformed)](images/image16.png)

The dependence plot for ai_usage_pct shows a clear and stable pattern, with SHAP values increasing as AI usage intensity rises. This indicates that higher AI adoption contributes positively to predicted margins. However, the magnitude of this contribution remains limited relative to the impact of pricing model and billable efficiency. The interaction with pricing_model_hourly is preserved and remains central to the interpretation: observations associated with hourly billing display a weaker positive contribution from AI usage compared to non-hourly tasks. This reflects a structural mechanism already identified in the analysis, whereby productivity gains under hourly pricing translate into reduced billable hours rather than increased margins, effectively shifting the value of AI from the firm to the client.

The invariance of both the beeswarm ranking and the dependence plot across Sections 1 and 2 provides strong validation of the modelling approach. The complete overlap in feature importance, ordering, and SHAP magnitudes confirms that the results are not driven by skewness or scale artefacts, but reflect stable relationships embedded in the data. This consistency strengthens the reliability of the interpretation and supports the use of these findings for decision-making purposes. The positive relationship between AI usage intensity and margin, together with its dependence on pricing structure, can therefore be interpreted as a genuine signal rather than a statistical artefact introduced by the original feature distributions.

## 4.3 Results and Robustness Check

The comparison between the base dataset and the log-transformed dataset provides a direct assessment of whether distributional properties were limiting model performance. The four model configurations are evaluated jointly in order to isolate the effect of the transformation while keeping all other elements of the pipeline unchanged.

| Dataset | Model | Test R² | Test MAE (pp) | Test RMSE (pp) |
|--------|------|--------|----------------|----------------|
| Base | Linear Regression | 0.2048 | 32.06 | 48.55 |
| Base | Random Forest | 0.3445 | 27.64 | 44.08 |
| Log-transformed | Linear Regression | 0.2487 | 30.65 | 47.19 |
| Log-transformed | Random Forest | 0.3445 | 27.64 | 44.08 |

| Model | Δ R² (Log − Base) | Δ MAE (pp) |
|------|------------------|------------|
| Linear Regression | +0.0439 | −1.41 |
| Random Forest | 0.0000 | +0.00 |

![Model comparison — base vs log-transformed](images/image17.png)

The log transformation produces a clear and interpretable effect on Linear Regression. Test R² increases from 0.2048 to 0.2487, corresponding to a gain of +0.0439, while MAE decreases by 1.41 percentage points. This result confirms that right-skewed feature distributions were a binding constraint on the linear model. Extreme values in variables such as hours_spent, rework_hours, revenue, and cost were disproportionately influencing the least-squares solution, distorting the estimated relationships. By compressing the long tail through a log1p transformation, the model is able to better approximate the underlying linear signal and recover part of the lost explanatory power.

In contrast, the Random Forest shows no change in performance across the two datasets. Test R² remains exactly 0.3445, with MAE and RMSE unchanged to two decimal places. This is not incidental, but a direct confirmation of the theoretical property of tree-based models: their predictions depend only on the ordering of feature values, not on their scale. Since log1p is a monotonic transformation, it preserves rank ordering and therefore leads to identical splits and predictions. The zero delta is therefore a critical validation result, confirming that the experimental setup is internally consistent and that no unintended data leakage or transformation artefact has been introduced.

The comparison between the two models after the transformation is equally informative. Even after correcting for skewness, Linear Regression remains substantially below Random Forest in terms of predictive performance, with a residual gap of approximately 0.10 in R². This gap cannot be attributed to distributional issues, as those have already been addressed. Instead, it reflects the presence of non-linear relationships and interaction effects in the data that cannot be captured by a linear model, regardless of feature scaling.

Taken together, these results support three conclusions. First, feature skewness was a real limitation for OLS and its correction leads to measurable improvements in performance. Second, the invariance of Random Forest confirms that the observed improvement is entirely due to the transformation and not to changes in the data or modelling pipeline. Third, a substantial portion of the predictive signal remains inherently non-linear, requiring flexible models to be fully captured.

From a methodological perspective, this section acts as a robustness check of the entire modelling framework. The fact that Linear Regression improves exactly where expected, and Random Forest remains unchanged as predicted by theory, provides strong evidence that the results obtained in previous sections are stable, internally coherent, and not driven by artefacts of feature scale or distribution.


## 5. Conclusions

The analysis provides a consistent and internally validated answer to the research question on the relationship between AI usage and profit margin. The results show that AI does not act as a binary factor but as a continuous driver whose effect becomes visible only at sufficiently high levels of adoption. Low or moderate usage does not generate measurable economic impact, while higher intensity levels are associated with a clear and systematic increase in margins. This pattern is not uniform across the system, but depends critically on structural conditions. In particular, the pricing model emerges as the primary moderator: when revenue is decoupled from time, productivity gains translate directly into higher margins, whereas under hourly billing the same gains reduce billable hours and therefore limit or even reverse profitability improvements. The modelling results confirm that this relationship is real and not an artefact of the data. Linear Regression improves after correcting for skewness, indicating that distributional issues were partially masking the signal, while Random Forest remains unchanged, validating the robustness of the experimental setup. The persistence of a performance gap between linear and non-linear models further shows that part of the relationship between features and margin is inherently non-linear and cannot be captured through linear assumptions alone. Overall, the findings demonstrate that AI adoption can improve profitability, but only when it is sufficiently deep and aligned with the economic structure of the organization.

At the same time, the analysis leaves several questions open. A large share of the variance in profit margin remains unexplained, indicating that important drivers are not captured in the available data. Factors such as client pricing power, negotiation dynamics, project complexity beyond observable proxies, and individual worker skill are likely to play a significant role but are not directly measurable in the dataset. In addition, the results are observational and do not establish causality: higher AI usage is associated with higher margins, but the direction of the relationship cannot be fully isolated from potential confounding factors, such as more capable teams both adopting AI more intensively and achieving better economic outcomes. The segmented analysis also highlights cases where high AI usage leads to worse performance, suggesting that integration quality and workflow design are critical but not explicitly modelled. Future work should therefore focus on incorporating richer operational and contractual data, as well as exploring causal identification strategies, in order to better isolate the mechanisms through which AI affects profitability and to determine under which conditions its adoption produces sustained economic gains.