
# ECON 692 - Problem Set 6
# Part (b): Pooled OLS Estimation

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load the converted dataset
df = pd.read_csv("SexWorkers.csv")

# Create the natural logarithm of price
df["log_price"] = np.log(df["price"])

# Estimate the pooled OLS model
model_b = smf.ols(
    formula="log_price ~ nocondom",
    data=df
).fit(cov_type="cluster", cov_kwds={"groups": df["id"]})

# Display regression results
print("\n========================================")
print("PART (b): POOLED OLS REGRESSION")
print("========================================")
print(model_b.summary())

# Extract the estimated coefficients
beta0 = model_b.params["Intercept"]
beta1 = model_b.params["nocondom"]

# Calculate exact percentage effect
percentage_effect = 100 * (np.exp(beta1) - 1)

print("\n========================================")
print("INTERPRETATION")
print("========================================")
print(f"Intercept (beta0) = {beta0:.6f}")
print(f"Nocondom coefficient (beta1) = {beta1:.6f}")
print(f"Cluster-robust SE(beta1) = {model_b.bse['nocondom']:.6f}")
print(f"P-value = {model_b.pvalues['nocondom']:.6f}")
print(f"Exact percentage difference = {percentage_effect:.2f}%")
print(f"Number of observations = {int(model_b.nobs)}")

##answer d)##

# =====================================================
# PART (d): EXPANDED POOLED OLS REGRESSION
# =====================================================

# Add the explanatory variables selected in part (c)
formula_d = (
    "log_price ~ nocondom + bar + street + attractive "
    "+ school + age + rich + regular + alcohol"
)

# Estimate expanded Pooled OLS
# Cluster standard errors by individual worker (id)
model_d = smf.ols(
    formula=formula_d,
    data=df
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["id"]}
)

# Display the full regression results
print("\n========================================")
print("PART (d): EXPANDED POOLED OLS REGRESSION")
print("========================================")
print(model_d.summary())

# Extract the nocondom coefficient and its statistics
beta1_d = model_d.params["nocondom"]
se_d = model_d.bse["nocondom"]
pvalue_d = model_d.pvalues["nocondom"]

# Calculate the exact percentage price difference
percentage_d = 100 * (np.exp(beta1_d) - 1)

# Compare with the original Pooled OLS model from part (b)
beta1_b = model_b.params["nocondom"]
percentage_b = 100 * (np.exp(beta1_b) - 1)

print("\n========================================")
print("COMPARISON: PART (b) vs PART (d)")
print("========================================")

print(f"Part (b) coefficient = {beta1_b:.6f}")
print(f"Part (d) coefficient = {beta1_d:.6f}")

print(f"\nPart (b) exact percentage effect = {percentage_b:.2f}%")
print(f"Part (d) exact percentage effect = {percentage_d:.2f}%")

print(f"\nChange in coefficient = {beta1_d - beta1_b:.6f}")

print(f"\nPart (d) cluster-robust SE = {se_d:.6f}")
print(f"Part (d) p-value = {pvalue_d:.6f}")

print(f"\nPart (b) R-squared = {model_b.rsquared:.4f}")
print(f"Part (d) R-squared = {model_d.rsquared:.4f}")

print("\nNumber of observations:", int(model_d.nobs))

##answer e)##

# =====================================================
# PART (e): RANDOM EFFECTS (RE) ESTIMATION
# =====================================================

from linearmodels.panel import RandomEffects
import statsmodels.api as sm

# Create a panel-data structure using individual ID
# and transaction number as the two indices
panel_df = df.set_index(["id", "trans"]).sort_index()

# Dependent variable
y_re = panel_df["log_price"]

# Include the same explanatory variables as part (d)
re_variables = [
    "nocondom",
    "bar",
    "street",
    "attractive",
    "school",
    "age",
    "rich",
    "regular",
    "alcohol"
]

# Independent variables, including intercept
X_re = sm.add_constant(panel_df[re_variables])

# Estimate the Random Effects model
re_model = RandomEffects(y_re, X_re)

# Cluster standard errors at the individual level
re_results = re_model.fit(
    cov_type="clustered",
    cluster_entity=True
)

# Display the RE regression results
print("\n========================================")
print("PART (e): RANDOM EFFECTS REGRESSION")
print("========================================")

print(re_results.summary)

# Extract coefficient and inference statistics
beta1_re = re_results.params["nocondom"]
se_re = re_results.std_errors["nocondom"]
pvalue_re = re_results.pvalues["nocondom"]

# Exact percentage price difference
percentage_re = 100 * (np.exp(beta1_re) - 1)

print("\n========================================")
print("RANDOM EFFECTS INTERPRETATION")
print("========================================")

print(f"RE nocondom coefficient = {beta1_re:.6f}")
print(f"RE cluster-robust SE = {se_re:.6f}")
print(f"RE p-value = {pvalue_re:.6f}")
print(f"RE exact percentage difference = {percentage_re:.2f}%")

# =====================================================
# COMPARE POOLED OLS WITH RANDOM EFFECTS
# =====================================================

beta1_ols = model_d.params["nocondom"]
se_ols = model_d.bse["nocondom"]
pvalue_ols = model_d.pvalues["nocondom"]

percentage_ols = 100 * (np.exp(beta1_ols) - 1)

print("\n========================================")
print("PART (e): POOLED OLS VS RANDOM EFFECTS")
print("========================================")

print(f"{'Statistic':<24} {'Pooled OLS':>15} {'Random Effects':>17}")

print("-" * 60)

print(f"{'Coefficient':<24} {beta1_ols:>15.6f} {beta1_re:>17.6f}")

print(f"{'Standard Error':<24} {se_ols:>15.6f} {se_re:>17.6f}")

print(f"{'P-value':<24} {pvalue_ols:>15.6f} {pvalue_re:>17.6f}")

print(f"{'Percentage Effect':<24} {percentage_ols:>14.2f}% {percentage_re:>16.2f}%")

print("\nChange in coefficient:",
      round(beta1_re - beta1_ols, 6))
## answer g)##


# =====================================================
# PART (g): FIXED EFFECTS (FE) ESTIMATION
# =====================================================

from linearmodels.panel import PanelOLS

# Use the panel dataset created in part (e)
# panel_df has a MultiIndex: ["id", "trans"]

print("\n========================================")
print("PART (g): CHECK WITHIN-INDIVIDUAL VARIATION")
print("========================================")

# Check how many individuals exhibit variation in
# each explanatory variable across transactions
all_variables = [
    "nocondom", "bar", "street", "attractive",
    "school", "age", "rich", "regular", "alcohol"
]

for variable in all_variables:
    n_changing = (
        panel_df.groupby(level="id")[variable]
        .nunique()
        .gt(1)
        .sum()
    )
    print(f"{variable:<12}: {n_changing} workers with within variation")


# =====================================================
# Identify time-varying regressors
# =====================================================

# FE cannot separately identify variables that have
# no variation within individuals
fe_variables = [
    variable for variable in all_variables
    if (
        panel_df.groupby(level="id")[variable]
        .nunique()
        .gt(1)
        .any()
    )
]

print("\nVariables included in FE model:")
print(fe_variables)

# Dependent variable: log transaction price
y_fe = panel_df["log_price"]

# Independent variables with within-individual variation
X_fe = panel_df[fe_variables]

# =====================================================
# Estimate individual Fixed Effects model
# =====================================================

fe_model = PanelOLS(
    dependent=y_fe,
    exog=X_fe,
    entity_effects=True,
    drop_absorbed=True
)

# Cluster standard errors at the individual level
fe_results = fe_model.fit(
    cov_type="clustered",
    cluster_entity=True
)

print("\n========================================")
print("PART (g): FIXED EFFECTS REGRESSION")
print("========================================")

print(fe_results.summary)


# =====================================================
# Interpret nocondom coefficient
# =====================================================

beta1_fe = fe_results.params["nocondom"]
se_fe = fe_results.std_errors["nocondom"]
pvalue_fe = fe_results.pvalues["nocondom"]

percentage_fe = 100 * (np.exp(beta1_fe) - 1)

print("\n========================================")
print("FIXED EFFECTS INTERPRETATION")
print("========================================")

print(f"FE nocondom coefficient = {beta1_fe:.6f}")
print(f"FE cluster-robust SE = {se_fe:.6f}")
print(f"FE p-value = {pvalue_fe:.6f}")
print(f"FE exact percentage difference = {percentage_fe:.2f}%")


# =====================================================
# Compare Pooled OLS, RE, and FE
# =====================================================

print("\n========================================")
print("COMPARISON: POOLED OLS, RE, AND FE")
print("========================================")

print(f"{'Model':<20} {'Coefficient':>14} {'SE':>12} {'P-value':>12} {'Effect (%)':>14}")
print("-" * 76)

models_comparison = [
    ("Pooled OLS", beta1_ols, se_ols, pvalue_ols),
    ("Random Effects", beta1_re, se_re, pvalue_re),
    ("Fixed Effects", beta1_fe, se_fe, pvalue_fe)
]

for name, beta, se, pvalue in models_comparison:
    percentage = 100 * (np.exp(beta) - 1)
    print(
        f"{name:<20} {beta:>14.6f} {se:>12.6f} "
        f"{pvalue:>12.6f} {percentage:>13.2f}%"
    )
##answer i)##


# =====================================================
# PART (i): HAUSMAN SPECIFICATION TEST (RE VS FE)
# =====================================================

import numpy as np
from scipy.stats import chi2

# Estimate conventional covariance matrices for
# the classical Hausman test
re_hausman = re_model.fit(cov_type="unadjusted")
fe_hausman = fe_model.fit(cov_type="unadjusted")

# Compare only coefficients available in both models
common_vars = [
    "nocondom",
    "bar",
    "street",
    "rich",
    "regular",
    "alcohol"
]

# Coefficient vectors
b_fe = fe_hausman.params.loc[common_vars].to_numpy()
b_re = re_hausman.params.loc[common_vars].to_numpy()

# Coefficient difference
b_diff = b_fe - b_re

# Covariance matrices for the common coefficients
V_fe = fe_hausman.cov.loc[
    common_vars, common_vars
].to_numpy()

V_re = re_hausman.cov.loc[
    common_vars, common_vars
].to_numpy()

# Covariance difference
V_diff = V_fe - V_re
V_diff = (V_diff + V_diff.T) / 2

print("\n========================================")
print("PART (i): HAUSMAN SPECIFICATION TEST")
print("========================================")

print("Comparison of FE and RE coefficients:")

for name, fe_coef, re_coef in zip(
    common_vars, b_fe, b_re
):
    print(
        f"{name:<12} FE = {fe_coef:>10.6f} "
        f"RE = {re_coef:>10.6f}"
    )

# Check whether the covariance difference is
# positive definite for the conventional test
eigenvalues = np.linalg.eigvalsh(V_diff)

print("\nMinimum covariance eigenvalue:",
      f"{eigenvalues.min():.10f}")

if eigenvalues.min() > 1e-10:

    # Hausman test statistic
    H = float(b_diff.T @ np.linalg.solve(V_diff, b_diff))

    # Degrees of freedom
    df_hausman = len(common_vars)

    # Chi-square p-value
    p_value = chi2.sf(H, df_hausman)

    print("\nHausman statistic =", round(H, 6))
    print("Degrees of freedom =", df_hausman)
    print("P-value =", round(p_value, 6))

    if p_value < 0.05:
        print("\nReject H0 at the 5% level.")
        print("The FE specification is preferred.")
    else:
        print("\nFail to reject H0 at the 5% level.")
        print("RE is not rejected in favor of FE.")

else:
    print(
        "\nThe covariance difference is not positive definite."
    )
    print(
        "The conventional Hausman statistic cannot "
        "be reliably computed using this matrix."
    )
    print(
        "A correlated random effects (Mundlak) test "
        "is recommended instead."
    )
