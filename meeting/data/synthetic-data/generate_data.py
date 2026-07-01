"""
Generate an artificial dataset for a binary outcome (e.g., loan default) from a
logistic regression model.

Model:
    x_i ~ Uniform(0, 1)
    p_i = 1 / (1 + exp(-(beta0 + beta1 * x_i)))
    y_i ~ Bernoulli(p_i),   y in {0, 1}   (1 = default)

We choose (beta0, beta1) so the marginal default rate is roughly 20%.
Because x ~ U(0,1), the marginal P(Y=1) = integral_0^1 Lambda(beta0 + beta1*x) dx.
With beta0 = -2.5 and beta1 = 1.5, this integral is about 0.20.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- 1. Set parameters ----
np.random.seed(42)          # reproducibility
n = 100
beta0, beta1 = -2.5, 1.5    # tuned so E[P(Y=1)] ≈ 0.20

# ---- 2. Generate x ~ U(0,1) ----
x = np.random.uniform(0.0, 1.0, size=n)

# ---- 3. Compute default probability and draw y ~ Bernoulli(p) ----
linear_index = beta0 + beta1 * x
p = 1.0 / (1.0 + np.exp(-linear_index))
y = np.random.binomial(1, p)

# Report the realized default rate in this sample
print(f"beta0 = {beta0}, beta1 = {beta1}")
print(f"Theoretical marginal default rate (grid) ≈ "
      f"{np.mean(1/(1+np.exp(-(beta0 + beta1*np.linspace(0,1,10001))))):.4f}")
print(f"Realized default rate in sample of n={n}: {y.mean():.4f}")

# ---- 4. Save to CSV ----
df = pd.DataFrame({"x": x, "y": y})
csv_path = "/mnt/user-data/outputs/credit_data.csv"
df.to_csv(csv_path, index=False)
print(f"CSV saved to: {csv_path}")

# ---- 5. Scatter plot of x vs y (with true probability curve overlaid) ----
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(x, y, alpha=0.6, edgecolor="k", s=45,
           label=f"observations (n={n})")

# Overlay true P(Y=1 | x) for context
x_grid = np.linspace(0, 1, 400)
p_grid = 1.0 / (1.0 + np.exp(-(beta0 + beta1 * x_grid)))
ax.plot(x_grid, p_grid, color="crimson", lw=2,
        label=r"true $P(Y=1\mid x)$")

ax.set_xlabel("x  (explanatory variable)")
ax.set_ylabel("y  (0 = non-default, 1 = default)")
ax.set_title("Scatter plot of x and y  |  default rate ≈ 20%")
ax.set_yticks([0, 1])
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.15, 1.15)
ax.grid(alpha=0.3)
ax.legend(loc="center right")
fig.tight_layout()

png_path = "/mnt/user-data/outputs/scatter_x_y.png"
fig.savefig(png_path, dpi=150)
print(f"Scatter plot saved to: {png_path}")
