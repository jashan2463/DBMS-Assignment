"""Cross-check: numpy Laplace baseline vs OpenDP Laplace mechanism for COUNT."""
import numpy as np, pandas as pd
import opendp.prelude as dp
dp.enable_features("contrib")

df = pd.read_csv("data.csv")
true_count = int((df["age"] > 40).sum())
TRIALS = 1000
rng = np.random.default_rng(1)

rows = []
for eps in [0.1, 1, 10]:
    # OpenDP: Laplace mechanism on an integer with sensitivity 1 (scale = 1/eps)
    space = (dp.atom_domain(T=int), dp.absolute_distance(T=int))
    meas = space >> dp.m.then_laplace(scale=1.0 / eps)
    odp = np.array([meas(true_count) for _ in range(TRIALS)])
    base = true_count + rng.laplace(0, 1 / eps, TRIALS)
    rows.append((eps, np.abs(odp - true_count).mean() / true_count,
                 np.abs(base - true_count).mean() / true_count,
                 odp.std(), base.std(), np.sqrt(2) / eps))
out = pd.DataFrame(rows, columns=["epsilon", "opendp_mean_rel_err", "numpy_mean_rel_err",
                                  "opendp_std", "numpy_std", "theory_std"])
out.to_csv("opendp_check.csv", index=False)
print(out.to_string(index=False))
