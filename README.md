## [Section 2] Methods

### Feature Exclusion and Leakage Prevention

Before fitting any model, we excluded variables that cannot legitimately predict `profit_margin`. The exclusion list covers four groups: algebraic components of the target (`profit`, `revenue`, `cost`, and derivatives like `revenue_per_hour` and `hidden_cost_ratio`), which would leak the answer directly; identifier columns (`task_id`, `project_id`, `client`, `created_by`) with no signal; raw timestamps, since elapsed time is already in the engineered `duration_days` column; and post-hoc flags (`task_status`, `workflow_stage`) not available at prediction time. The target is `profit_margin = profit / revenue × 100`.

### Preprocessing

We built a `ColumnTransformer` with two branches, reused across all model runs. Numeric features get median imputation followed by `StandardScaler` (median rather than mean because several variables have heavy right tails we chose not to cap). Categorical features get most-frequent imputation and `OneHotEncoder` with `handle_unknown='ignore'`. Boolean flags (`sla_breach`, `scope_change_flag`, `ai_assisted`, `ai_flag`, `legacy_ai_flag`) are cast to 0/1 integers at load time and treated as numeric. All models use an 80/20 train/test split with `random_state=42`.

### Models

**Lasso (feature selector only).** `LassoCV` with 5-fold CV picks the regularisation strength that minimises held-out MSE. Its only job is to identify which features carry linear signal; the non-zero coefficients feed Linear Regression. Lasso does not appear in the performance table.

![Lasso retained features — base dataset](images/lasso_coefficients_s1.png)

**Linear Regression on Lasso-selected features.** One-hot encoded features with non-zero Lasso coefficients are mapped back to original column names, and a fresh preprocessor is built on the reduced set. Running LR on fewer features than RF is deliberate: the R² gap then reflects both non-linearity and features Lasso discarded, which tells us more than a head-to-head on identical inputs would.

![Linear Regression coefficients — base dataset](images/lr_coefficients_s1.png)

**Random Forest with GridSearchCV.** RF trains on the full candidate set to capture interactions that L1 regularisation suppresses. We searched over `n_estimators` ∈ {100, 200, 300}, `max_depth` ∈ {4, 6, 8, None}, and `min_samples_leaf` ∈ {10, 20, 40} with 5-fold CV. The `min_samples_leaf` range was included specifically to prevent overfitting on ~3,200 observations.

![Model comparison — base dataset](images/model_comparison_s1.png)

**SHAP.** `shap.TreeExplainer` on the tuned RF gives exact Shapley values on the held-out test set. We produced a beeswarm plot (top 15 features by mean |SHAP|) and a dependence plot for `ai_usage_pct`. A cross-check table then compares Lasso-retained features against the SHAP top 15 to flag predictors that are consistent across both methods (those are the ones we trust most).

![SHAP beeswarm — base dataset](images/shap_beeswarm_s1.png)
![SHAP dependence plot — ai_usage_pct (base)](images/shap_dependence_ai_s1.png)

### Log-transformation

`hours_spent` (skew 9.95), `rework_hours` (7.76), `cost` (5.54), and `revenue` (4.72) are heavily right-skewed, and OLS is sensitive to this in a way RF is not. We applied `np.log1p` to seven features where compression preserves interpretability: `hours_spent`, `billable_hours`, `rework_hours`, `revenue`, `cost`, `errors`, `revisions`. The target and everything else are untouched. The full pipeline then reruns on `df_log` with the same split and hyperparameter grid. RF is invariant to monotonic transformations, so its results on `df_log` are a sanity check: if RF performance shifts, something in our setup is wrong.

![Original vs log-transformed distributions](images/log_distribution_comparison.png)

![Lasso retained features — log-transformed](images/lasso_coefficients_s2.png)
![Linear Regression coefficients — log-transformed](images/lr_coefficients_s2.png)

![SHAP beeswarm — log-transformed](images/shap_beeswarm_s2.png)
![SHAP dependence plot — ai_usage_pct (log-transformed)](images/shap_dependence_ai_s2.png)

![Model comparison — base vs log-transformed](images/model_comparison_s2.png)

---

## [Section 3] Experimental Design

### Experiment 1: Base Models

**Purpose:** establish baseline performance and identify which task attributes drive `profit_margin`.

**Baseline:** Linear Regression on Lasso-selected features, chosen for interpretability. Signed coefficients give a first characterisation of the feature-margin relationship.

**Comparison:** tuned Random Forest on the full feature set. The R² gap over LR shows what the linear assumption is costing us.

**Metrics:** Test R², Test MAE (pp), Test RMSE (pp), 5-fold CV R². MAE in percentage points is the most readable: a 30 pp MAE means predictions are off by 30 margin points on average.

### Experiment 2: Log-transformed Features

**Purpose:** test whether right skew is what limits Linear Regression, or whether the ceiling is structural.

**Baseline:** LR and RF results from Experiment 1.

**Comparison:** same models on `df_log`. RF serves as a built-in control — because it depends only on value ordering, `log1p` cannot change its predictions. If LR R² rises while RF R² stays flat, the gain came from fixing the distribution and nothing else. We set +0.03 in LR R² as the threshold for concluding that skew was a material constraint. The delta column (Δ R², Log − Base) makes the comparison explicit.