"""
G12: Accuracy vs privacy budget (epsilon) for differentially private SQL-style
aggregate queries, using the Laplace mechanism implemented in numpy.

Run:  python g12_dp_experiment.py
Outputs: data.csv, results.csv, group_results.csv, timing.csv, fig1/2/3 PNGs
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N_ROWS = 100_000
TRIALS = 1000
EPSILONS = [0.01, 0.1, 0.5, 1, 2, 5, 10]
SAL_LO, SAL_HI = 0, 200_000          # clamping bounds for salary
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- 1. data
def make_data():
    cities = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata",
              "Pune", "Ropar", "Leh"]
    # skewed on purpose, so some groups are small
    p = np.array([0.30, 0.25, 0.20, 0.12, 0.07, 0.04, 0.015, 0.005])
    df = pd.DataFrame({
        "age": np.clip(rng.normal(38, 12, N_ROWS), 18, 70).astype(int),
        "salary": np.clip(rng.lognormal(10.8, 0.6, N_ROWS), SAL_LO, SAL_HI),
        "city": rng.choice(cities, N_ROWS, p=p / p.sum()),
        "department": rng.choice(["Eng", "Sales", "HR", "Ops", "Fin", "Legal"],
                                 N_ROWS, p=[.35, .25, .1, .15, .1, .05]),
    })
    return df

df = make_data()
df.to_csv("data.csv", index=False)
sal = df["salary"].clip(SAL_LO, SAL_HI)

# ---------------------------------------------------- 2. true (exact) answers
true_count = int((df["age"] > 40).sum())
true_sum = float(sal.sum())
true_avg = float(sal.mean())
group_true = df.groupby("city").size()

# ---------------------------------------------- 3. Laplace mechanism, 1000x
def lap(scale, size):
    return rng.laplace(0.0, scale, size)

rows = []
group_rows = []
for eps in EPSILONS:
    # COUNT: sensitivity 1
    noisy = true_count + lap(1 / eps, TRIALS)
    rows += [("COUNT", eps, t, true_count, v) for t, v in enumerate(noisy)]

    # SUM: sensitivity = max(|lo|,|hi|) after clamping
    sens_sum = max(abs(SAL_LO), abs(SAL_HI))
    noisy = true_sum + lap(sens_sum / eps, TRIALS)
    rows += [("SUM", eps, t, true_sum, v) for t, v in enumerate(noisy)]

    # AVG = noisy SUM / noisy COUNT, budget split eps/2 each (sequential comp.)
    ns = true_sum + lap(sens_sum / (eps / 2), TRIALS)
    nc = np.maximum(len(df) + lap(1 / (eps / 2), TRIALS), 1)
    noisy = np.clip(ns / nc, SAL_LO, SAL_HI)
    rows += [("AVG", eps, t, true_avg, v) for t, v in enumerate(noisy)]

    # GROUP BY city COUNT: groups are disjoint -> parallel composition,
    # each group gets the full eps, sensitivity 1
    for city, tc in group_true.items():
        noisy = tc + lap(1 / eps, TRIALS)
        rows += [("GROUPBY_COUNT", eps, t, tc, v) for t, v in enumerate(noisy)]
        group_rows += [(city, int(tc), eps, t, v) for t, v in enumerate(noisy)]

res = pd.DataFrame(rows, columns=["query", "epsilon", "trial", "true", "noisy"])
res["rel_error"] = (res["noisy"] - res["true"]).abs() / res["true"].abs()
res.to_csv("results.csv", index=False)

gres = pd.DataFrame(group_rows, columns=["city", "group_size", "epsilon",
                                         "trial", "noisy"])
gres["rel_error"] = (gres["noisy"] - gres["group_size"]).abs() / gres["group_size"]
gres.to_csv("group_results.csv", index=False)

summary = (res.groupby(["query", "epsilon"])["rel_error"]
           .agg(mean="mean", p95=lambda x: np.percentile(x, 95)).reset_index())
summary.to_csv("summary.csv", index=False)
print(summary.to_string(index=False))

# ----------------------------------------------------------- 4. runtime
def timeit(fn, reps=200):
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000  # ms per query

age = df["age"].to_numpy()
salv = sal.to_numpy()
plain = {
    "COUNT": lambda: int((age > 40).sum()),
    "SUM": lambda: float(salv.sum()),
    "AVG": lambda: float(salv.mean()),
    "GROUPBY_COUNT": lambda: df.groupby("city").size(),
}
dp = {
    "COUNT": lambda: int((age > 40).sum()) + rng.laplace(0, 1 / 1.0),
    "SUM": lambda: float(salv.sum()) + rng.laplace(0, SAL_HI / 1.0),
    "AVG": lambda: (float(salv.sum()) + rng.laplace(0, SAL_HI / 0.5)) /
                   max(len(salv) + rng.laplace(0, 1 / 0.5), 1),
    "GROUPBY_COUNT": lambda: df.groupby("city").size() +
                             rng.laplace(0, 1.0, df["city"].nunique()),
}
trows = []
for q in plain:
    p_ms, d_ms = timeit(plain[q]), timeit(dp[q])
    trows.append((q, p_ms, d_ms, (d_ms - p_ms) / p_ms * 100))
tdf = pd.DataFrame(trows, columns=["query", "plain_ms", "dp_ms", "overhead_pct"])
tdf.to_csv("timing.csv", index=False)
print(tdf.to_string(index=False))

# ------------------------------------------------------------ 5. graphs
# Fig 1: mean relative error vs epsilon, per query
plt.figure(figsize=(5, 3.6))
for q, g in summary.groupby("query"):
    plt.plot(g["epsilon"], g["mean"], marker="o", label=q)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Privacy budget $\\epsilon$"); plt.ylabel("Mean relative error")
plt.title("Error vs privacy budget"); plt.legend(fontsize=7); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig("fig1_error_vs_epsilon.png", dpi=200); plt.close()

# Fig 2: error vs group size for GROUP BY at fixed epsilons
gm = gres.groupby(["city", "group_size", "epsilon"])["rel_error"].mean().reset_index()
plt.figure(figsize=(5, 3.6))
for eps in [0.1, 1, 10]:
    g = gm[gm["epsilon"] == eps].sort_values("group_size")
    plt.plot(g["group_size"], g["rel_error"], marker="o", label=f"$\\epsilon$={eps}")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Group size (rows)"); plt.ylabel("Mean relative error")
plt.title("GROUP BY count: error vs group size"); plt.legend(fontsize=7)
plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("fig2_error_vs_group_size.png", dpi=200); plt.close()

# Fig 3: runtime plain vs DP
x = np.arange(len(tdf)); w = 0.38
plt.figure(figsize=(5, 3.6))
plt.bar(x - w / 2, tdf["plain_ms"], w, label="Plain")
plt.bar(x + w / 2, tdf["dp_ms"], w, label="DP (Laplace)")
plt.xticks(x, tdf["query"], fontsize=7, rotation=15)
plt.ylabel("Time per query (ms)"); plt.title("Runtime: plain vs DP")
plt.legend(fontsize=7); plt.tight_layout()
plt.savefig("fig3_runtime.png", dpi=200); plt.close()
print("Done.")
