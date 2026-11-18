"""
G12 extra experiments (run after g12_dp_experiment.py has created data.csv):
  A) Clamping bound vs error for DP AVG (bias-variance trade-off)
  B) Dataset size vs error at fixed epsilon
Outputs: clamp_results.csv, size_results.csv, fig4_clamping.png, fig5_dataset_size.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
TRIALS = 1000
df = pd.read_csv("data.csv")
sal_all = df["salary"].to_numpy()          # already bounded to [0, 200000]

def dp_avg(x, lo, hi, eps, trials):
    """Noisy AVG = noisy SUM / noisy COUNT, eps/2 each. Values clamped to [lo,hi]."""
    xc = np.clip(x, lo, hi)
    sens = max(abs(lo), abs(hi))
    ns = xc.sum() + rng.laplace(0, sens / (eps / 2), trials)
    nc = np.maximum(len(x) + rng.laplace(0, 1 / (eps / 2), trials), 1)
    return np.clip(ns / nc, lo, hi)

# ---------------- A) clamping bound vs error (n=1000 and n=100000, eps=0.1)
rows = []
bounds = [20_000, 40_000, 60_000, 80_000, 100_000, 150_000, 200_000, 400_000]
for n in [1_000, 100_000]:
    x = rng.choice(sal_all, n, replace=False)
    true_avg = x.mean()                       # true, unclamped mean
    for eps in [0.1, 1.0]:
        for hi in bounds:
            est = dp_avg(x, 0, hi, eps, TRIALS)
            err = np.abs(est - true_avg) / true_avg
            bias = (np.clip(x, 0, hi).mean() - true_avg) / true_avg
            rows.append((n, eps, hi, err.mean(), bias))
clamp = pd.DataFrame(rows, columns=["n", "epsilon", "upper_bound",
                                    "mean_rel_error", "clamp_bias_rel"])
clamp.to_csv("clamp_results.csv", index=False)
print(clamp.to_string(index=False))

plt.figure(figsize=(5, 3.6))
for (n, eps), g in clamp.groupby(["n", "epsilon"]):
    plt.plot(g["upper_bound"] / 1000, g["mean_rel_error"], marker="o",
             label=f"n={n:,}, $\\epsilon$={eps}")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Clamping upper bound (thousand)"); plt.ylabel("Mean relative error (AVG)")
plt.title("Clamping bound vs DP AVG error"); plt.legend(fontsize=6); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("fig4_clamping.png", dpi=200); plt.close()

# ---------------- B) dataset size vs error (eps = 1)
rows = []
for n in [1_000, 3_000, 10_000, 30_000, 100_000]:
    x = rng.choice(sal_all, n, replace=False)
    ages = rng.choice(df["age"].to_numpy(), n, replace=False)
    tc = int((ages > 40).sum()); ts = x.sum(); ta = x.mean()
    eps = 1.0
    c = tc + rng.laplace(0, 1 / eps, TRIALS)
    s = ts + rng.laplace(0, 200_000 / eps, TRIALS)
    a = dp_avg(x, 0, 200_000, eps, TRIALS)
    rows += [(n, "COUNT", (np.abs(c - tc) / tc).mean()),
             (n, "SUM", (np.abs(s - ts) / ts).mean()),
             (n, "AVG", (np.abs(a - ta) / ta).mean())]
size = pd.DataFrame(rows, columns=["n", "query", "mean_rel_error"])
size.to_csv("size_results.csv", index=False)
print(size.to_string(index=False))

plt.figure(figsize=(5, 3.6))
for q, g in size.groupby("query"):
    plt.plot(g["n"], g["mean_rel_error"], marker="o", label=q)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Number of rows"); plt.ylabel("Mean relative error")
plt.title("Error vs dataset size ($\\epsilon$=1)"); plt.legend(fontsize=7)
plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("fig5_dataset_size.png", dpi=200); plt.close()
print("Done.")
