"""
bnn_tmdb_movies_hmc.py
======================
 
BNN-HMC on the TMDB 5000 Movies dataset, following the preprocessing of
 
    Durham MATH3431, Practical 3
    https://www.maths.dur.ac.uk/users/viet.c.nguyen/MATH3431/Practicals/Practical_3.html
 
Task: predict `vote_average` from 5 features (budget, popularity, revenue,
runtime, vote_count).  Preprocessing:
  * keep 6 columns: budget, popularity, revenue, runtime, vote_average,
    vote_count
  * drop rows with NA runtime
  * drop rows with budget == 0 or revenue == 0 (missing-data sentinel)
  * 80 / 20 train / test split (deterministic with --seed)
  * keep the original units for both inputs and target; MSE is reported
    on the raw vote_average scale.
 
Architecture: a shallow BNN [5, hidden, 1] with Gaussian prior and Gaussian
likelihood (noise_scale fixed).  Hidden defaults to 50 (lit. standard).
 
The script reproduces the two curves from the notebook --
 
    * Acceptance rate   vs   step size eps
    * Efficiency = eps * accept  vs  step size eps
 
for every activation in --activations.  Run:
 
    python bnn_tmdb_movies_hmc.py --sweep \\
        --activations tanh relu alpha_tanh alpha_relu alpha_sigmoid \\
        --step_sizes 5e-4 1e-3 1.5e-3 2e-3 2.5e-3 3e-3 3.5e-3 4e-3 \\
        -T 0.1
 
and it will write ./results_tmdb/tmdb_sweep_summary.csv plus two PNG plots
(acceptance_rate.png, efficiency.png) that mirror the notebook.
"""
 
import argparse
import os
import time
import urllib.request
from functools import partial
 
import numpy as np
import pandas as pd
from tqdm import tqdm
 
import jax
import jax.numpy as jnp
from jax import grad, jit, random
from jax.scipy.stats import norm
 
np.set_printoptions(suppress=True, precision=6)
 
# Durham practical's hosted copy of the TMDB 5000 Movies CSV
TMDB_URL = ("https://www.maths.dur.ac.uk/users/viet.c.nguyen/"
            "MATH3431/Practicals/tmdb_5000_movies.csv")
 
# The 6 columns the practical keeps
KEEP_COLS = ["budget", "popularity", "revenue", "runtime",
             "vote_average", "vote_count"]
 
# Target: vote_average
TARGET_COL = "vote_average"
 
 
# =============================================================================
#  Activation zoo  (matches the notebook + UCI script)
# =============================================================================
ACTIVATIONS = {
    "tanh":          lambda x: jax.nn.tanh(x),
    "sigmoid":       lambda x: jax.nn.sigmoid(x),
    "relu":          lambda x: jax.nn.relu(x),
    "gelu":          lambda x: jax.nn.gelu(x),
    "swish":         lambda x: x * jax.nn.sigmoid(x),
    "mish":          lambda x: x * jax.nn.tanh(jax.nn.softplus(x)),
    "softplus":      lambda x: jax.nn.softplus(x),
    "leaky_relu":    lambda x: jax.nn.leaky_relu(x),
}
 
 
# =============================================================================
#  Shared HMC machinery  (identical to the UCI script)
# =============================================================================
def net2vec(net):
    return jnp.concatenate([jnp.ravel(x) for layer in net for x in layer])
 
 
@partial(jax.jit, static_argnums=(1, 2, 3))
def log_pdf_params(params, mean, variance, bias):
    log_pdf = 0.0
    if bias:
        for W, b in params:
            log_pdf += jnp.sum(norm.logpdf(W, mean, variance))
            log_pdf += jnp.sum(norm.logpdf(b, mean, variance))
    else:
        for W in params:
            log_pdf += jnp.sum(norm.logpdf(W, mean, variance))
    return log_pdf
 
 
@partial(jit, static_argnums=(2, 3))
def leapfrog(q, p, potential, num_steps, step_size):
    grad_init = grad(potential)(q)
    p = jax.tree_util.tree_map(lambda p, g: p - 0.5 * step_size * g, p, grad_init)
    for step in range(1, num_steps + 1):
        q = jax.tree_util.tree_map(lambda q, p: q + step_size * p, q, p)
        if step != num_steps:
            p = jax.tree_util.tree_map(
                lambda p, g: p - step_size * g, p, grad(potential)(q)
            )
    grad_end = grad(potential)(q)
    p = jax.tree_util.tree_map(lambda p, g: p - 0.5 * step_size * g, p, grad_end)
    p = jax.tree_util.tree_map(lambda p: -p, p)
    return q, p, grad_end, grad_init
 
 
def hmc_nn(n_samples, burn_in, potential, initial_params, layer_sizes,
           num_steps, step_size, bnn, progress=True):
    samples, accept_samples = [], []
    deltaH_hist, term1_hist = [], []
    accepted = 0
    q_curr = initial_params.copy()
    bias = bnn.bias
 
    iterator = range(n_samples + burn_in)
    if progress:
        iterator = tqdm(iterator)
 
    for i in iterator:
        p_curr = bnn.init_network_params(random.PRNGKey(i), scale=1.0)
        q_new, p_new, grad_end, grad_init = leapfrog(
            q_curr, p_curr, potential, num_steps, step_size
        )
        deltaH = ((potential(q_curr) - log_pdf_params(p_curr, 0, 1, bias))
                  - (potential(q_new) - log_pdf_params(p_new, 0, 1, bias)))
 
        if i >= burn_in:
            term_1 = (1.0 / 8.0) * (step_size ** 2) * (
                jnp.linalg.norm(net2vec(grad_end)) ** 2
                - jnp.linalg.norm(net2vec(grad_init)) ** 2
            )
            deltaH_hist.append(float(-deltaH))
            term1_hist.append(float(term_1))
 
        if jnp.log(np.random.rand()) < deltaH:
            samples.append(q_new)
            q_curr = q_new
            if i >= burn_in:
                accepted += 1
                accept_samples.append(True)
        else:
            samples.append(q_curr)
            if i >= burn_in:
                accept_samples.append(False)
 
    return (samples[burn_in + 1:], accepted / n_samples,
            deltaH_hist, term1_hist, accept_samples)
 
 
class BNN:
    """Gaussian-prior, fixed-noise BNN regressor."""
 
    def __init__(self, layer_sizes, prior_variance, noise_scale,
                 nonlinearity, bias=True):
        self.layer_sizes    = layer_sizes
        self.prior_variance = prior_variance
        self.noise_scale    = noise_scale
        self.nonlinearity   = nonlinearity
        self.bias           = bias
 
    def random_layer_params(self, m, n, key, scale=0.1):
        if self.bias:
            w_key, b_key = random.split(key)
            return (scale * random.normal(w_key, (m, n)),
                    scale * random.normal(b_key, (n,)))
        w_key, _ = random.split(key)
        return scale * random.normal(w_key, (m, n))
 
    def init_network_params(self, key, scale):
        keys = random.split(key, len(self.layer_sizes))
        return [self.random_layer_params(m, n, k, scale)
                for m, n, k in zip(self.layer_sizes[:-1],
                                   self.layer_sizes[1:], keys)]
 
    def nn_predict(self, params, inputs):
        if self.bias:
            for W, b in params:
                outputs = jnp.dot(inputs, W) + b
                inputs = self.nonlinearity(outputs)
            return outputs
        for W in params:
            outputs = jnp.dot(inputs, W)
            inputs = self.nonlinearity(outputs)
        return outputs
 
    @partial(jit, static_argnums=(0,))
    def neg_log_posterior(self, params, inputs, targets):
        return (- self.logprob(params, inputs, targets)
                - log_pdf_params(params, 0, self.prior_variance, self.bias))
 
    def logprob(self, params, inputs, targets):
        preds = self.nn_predict(params, inputs)
        return jnp.sum(norm.logpdf(preds, targets, self.noise_scale))
 
    def hmc_sampling(self, inputs, targets, n_samples, burn_in,
                     num_steps, step_size, seed=0, progress=True):
        neg_log_p = lambda q: self.neg_log_posterior(q, inputs, targets)
        init_key  = random.PRNGKey(seed)
        init_pars = self.init_network_params(init_key, scale=self.noise_scale)
        return hmc_nn(n_samples, burn_in, neg_log_p, init_pars,
                      self.layer_sizes, num_steps, step_size, bnn=self,
                      progress=progress)
 
    def predict(self, samples, inputs):
        out = [self.nn_predict(s, inputs) for s in samples]
        return jnp.stack(out, axis=0)          # [S, N, 1]
 
 
# =============================================================================
#  TMDB loading + preprocessing  (follows Practical 3 exactly)
# =============================================================================
def download_tmdb(cache_path):
    """Download the Durham practical's CSV.  Uses a browser User-Agent
    because the default `python-urllib/...` UA is sometimes 403'd."""
    if not os.path.exists(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"[data] downloading {TMDB_URL}")
        req = urllib.request.Request(
            TMDB_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; bnn-hmc-script/1.0)"},
        )
        with urllib.request.urlopen(req) as r, open(cache_path, "wb") as f:
            f.write(r.read())
    return cache_path
 
 
def load_tmdb(data_dir="./data", seed=0):
    """Return z-normalised (X_tr, y_tr, X_te, y_te, scaler, features).

    The pipeline:
      1. read CSV
      2. keep the 6 columns
      3. drop NA runtime rows
      4. drop rows with budget == 0 or revenue == 0 (missing-data sentinel)
      5. 80/20 train/test split with `seed`
      6. z-normalise each feature and the target using training statistics
         (mean and std fit on train, applied to test)
    """
    path = download_tmdb(os.path.join(data_dir, "tmdb_5000_movies.csv"))
    df = pd.read_csv(path)
    df = df[KEEP_COLS].dropna(subset=["runtime"]).reset_index(drop=True)
    df = df[(df["budget"] != 0) & (df["revenue"] != 0)].reset_index(drop=True)

    # 80/20 split
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(df))
    n      = len(df)
    n_tr   = int(0.8 * n)
    tr_idx = idx[:n_tr]
    te_idx = idx[n_tr:]
    df_tr, df_te = df.iloc[tr_idx], df.iloc[te_idx]

    tgt_idx = KEEP_COLS.index(TARGET_COL)
    feat_idx = [i for i in range(len(KEEP_COLS)) if i != tgt_idx]
    features = [KEEP_COLS[i] for i in feat_idx]

    arr_tr = df_tr[KEEP_COLS].values.astype(np.float32)
    arr_te = df_te[KEEP_COLS].values.astype(np.float32)

    X_tr, y_tr = arr_tr[:, feat_idx], arr_tr[:, tgt_idx].reshape(-1, 1)
    X_te, y_te = arr_te[:, feat_idx], arr_te[:, tgt_idx].reshape(-1, 1)

    # Z-normalisation — fit on train only
    X_mu    = X_tr.mean(axis=0)
    X_sigma = X_tr.std(axis=0)
    X_sigma = np.where(X_sigma == 0, 1.0, X_sigma)
    y_mu    = float(y_tr.mean())
    y_sigma = float(y_tr.std()) or 1.0

    X_tr = (X_tr - X_mu) / X_sigma
    X_te = (X_te - X_mu) / X_sigma
    y_tr = (y_tr - y_mu) / y_sigma
    y_te = (y_te - y_mu) / y_sigma

    scaler = {"y_mu": y_mu, "y_sigma": y_sigma}

    print(f"[data] total={n}  train={len(X_tr)}  test={len(X_te)}  "
          f"features={features}  target={TARGET_COL}")
    return X_tr, y_tr, X_te, y_te, scaler, features
 
 
# =============================================================================
#  Experiment runners
# =============================================================================
def build_bnn(input_dim, hidden, activation, prior_variance, noise_scale):
    return BNN([input_dim, hidden, 1], prior_variance, noise_scale,
               ACTIVATIONS[activation], bias=True)
 
 
def num_steps_from_T(T, eps):
    L = max(1, int(round(T / eps)))
    return L, eps * L
 
 
def evaluate(bnn, samples, X, y, scaler):
    """Return (mse_normalised, mse_original) on a dataset."""
    if len(samples) == 0:
        return float("nan"), float("nan")
    preds = bnn.predict(samples, jnp.asarray(X))
    preds  = np.asarray(preds).squeeze(-1).mean(axis=0)  # [N]
    y_flat = y.squeeze(-1)
    mse_norm = float(np.mean((preds - y_flat) ** 2))
    preds_orig = preds  * scaler["y_sigma"] + scaler["y_mu"]
    y_orig     = y_flat * scaler["y_sigma"] + scaler["y_mu"]
    mse_orig   = float(np.mean((preds_orig - y_orig) ** 2))
    return mse_norm, mse_orig


def one_run(bnn, X_tr, y_tr, X_te, y_te, scaler, *,
            n_samples, burn_in, num_steps, step_size, seed):
    t0 = time.time()
    samples, acc_rate, deltaHs, term1s, acc_flags = bnn.hmc_sampling(
        jnp.asarray(X_tr), jnp.asarray(y_tr),
        n_samples=n_samples, burn_in=burn_in,
        num_steps=num_steps, step_size=step_size, seed=seed,
    )
    wall = time.time() - t0
    deltaHs = np.asarray(deltaHs)
    term1s  = np.asarray(term1s)
    term2s  = term1s - deltaHs
    mse_norm, mse_orig = evaluate(bnn, samples, X_te, y_te, scaler)
    return {
        "samples":    samples,
        "accepted":   acc_flags,
        "accept_rate": float(acc_rate),
        "deltaH":     deltaHs,
        "term_1":     term1s,
        "term_2":     term2s,
        "mse_norm":   mse_norm,
        "mse_orig":   mse_orig,
        "wall_s":     wall,
    }
 
 
def run_sweep(X_tr, y_tr, X_te, y_te, scaler, *, hidden, activations, step_sizes,
              trajectory_length, n_samples, burn_in, prior_variance,
              noise_scale, seed, out_dir):
    """Sweep (activation x step_size); evaluate on the test set."""
    input_dim = X_tr.shape[1]
    print(f"[sweep] input_dim={input_dim}  hidden={hidden}  "
          f"n_train={len(X_tr)}  n_test={len(X_te)}")

    rows = []
    for act in activations:
        bnn = build_bnn(input_dim, hidden, act, prior_variance, noise_scale)
        for eps in step_sizes:
            L, T_actual = num_steps_from_T(trajectory_length, eps)
            print(f"[sweep] activation={act}  eps={eps:g}  L={L}  "
                  f"T_target={trajectory_length:g}  T_actual={T_actual:g}")
            try:
                res = one_run(bnn, X_tr, y_tr, X_te, y_te, scaler,
                              n_samples=n_samples, burn_in=burn_in,
                              num_steps=L, step_size=eps, seed=seed)
            except Exception as e:
                print(f"  -> run failed: {e}")
                continue
            row = {
                "activation":    act,
                "step_size":     eps,
                "num_steps":     L,
                "T_target":      trajectory_length,
                "T_actual":      T_actual,
                "accept_rate":   res["accept_rate"],
                "efficiency":    eps * res["accept_rate"],
                "mean_deltaH":   float(np.mean(res["deltaH"])),
                "mean_term_1":   float(np.mean(res["term_1"])),
                "mean_term_2":   float(np.mean(res["term_2"])),
                "std_deltaH":    float(np.std(res["deltaH"])),
                "std_term_1":    float(np.std(res["term_1"])),
                "std_term_2":    float(np.std(res["term_2"])),
                "test_mse_norm": res["mse_norm"],
                "test_mse_orig": res["mse_orig"],
                "wall_s":        res["wall_s"],
            }
            print(f"  accept={row['accept_rate']:.3f}  "
                  f"eff={row['efficiency']:.4g}  "
                  f"test_mse_orig={row['test_mse_orig']:.4g}")
            rows.append(row)
 
            hist = pd.DataFrame({
                "deltaH":   res["deltaH"],
                "term_1":   res["term_1"],
                "term_2":   res["term_2"],
                "accepted": res["accepted"],
            })
            hist.to_csv(os.path.join(out_dir,
                        f"tmdb_hist_{act}_eps{eps:g}.csv"), index=False)
 
    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, f"tmdb_sweep_summary_{seed}.csv")
    df.to_csv(out_path, index=False)
    print(f"[sweep] summary saved to {out_path}")
    return df
 
 
# =============================================================================
#  Plotting  (reproduces the two notebook curves)
# =============================================================================
def plot_curves(df, out_dir):
    """Plot acceptance_rate vs eps and efficiency vs eps, one line per
    activation.  Saves acceptance_rate.png and efficiency.png."""
    import matplotlib
    matplotlib.use("Agg")              # headless servers / no display
    import matplotlib.pyplot as plt
 
    df = df.sort_values(["activation", "step_size"]).reset_index(drop=True)
 
    # --- acceptance rate curve ----------------------------------------------
    plt.figure(figsize=(8, 5))
    for act, g in df.groupby("activation"):
        plt.plot(g["step_size"], g["accept_rate"], marker="o", label=act)
    plt.xlabel(r"Step Size $\epsilon$")
    plt.ylabel("Acceptance Rate")
    plt.title("HMC Acceptance Rate across Activation Functions\n"
              "(TMDB 5000 Movies, BNN regression on vote_average)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Activation")
    plt.tight_layout()
    acc_path = os.path.join(out_dir, "acceptance_rate.png")
    plt.savefig(acc_path, dpi=150)
    plt.close()
 
    # --- efficiency curve ---------------------------------------------------
    plt.figure(figsize=(8, 5))
    for act, g in df.groupby("activation"):
        g = g.copy()
        if "efficiency" not in g.columns:
            g["efficiency"] = g["step_size"] * g["accept_rate"]
        best = g.loc[g["efficiency"].idxmax()]
        plt.plot(g["step_size"], g["efficiency"], marker="o", label=act)
        plt.scatter([best["step_size"]], [best["efficiency"]],
                    s=80, facecolors="none", edgecolors="k", zorder=5)
    plt.xlabel(r"Step Size $\epsilon$")
    plt.ylabel(r"Efficiency  $\epsilon \times$ Acceptance Rate")
    plt.title("HMC Efficiency across Activation Functions\n"
              "(TMDB 5000 Movies, BNN regression on vote_average)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Activation")
    plt.tight_layout()
    eff_path = os.path.join(out_dir, "efficiency.png")
    plt.savefig(eff_path, dpi=150)
    plt.close()
 
    # Report the best epsilon per activation
    print("\n--- Best epsilon (by efficiency) per activation ---")
    for act, g in df.groupby("activation"):
        g = g.copy()
        if "efficiency" not in g.columns:
            g["efficiency"] = g["step_size"] * g["accept_rate"]
        best = g.loc[g["efficiency"].idxmax()]
        print(f"  {act:14s}  eps*={best['step_size']:.4g}  "
              f"accept={best['accept_rate']:.3f}  "
              f"eff={best['efficiency']:.4g}")
    print(f"\n[plot] saved {acc_path}")
    print(f"[plot] saved {eff_path}")
 
 
# =============================================================================
#  Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--out_dir",  default="./results_tmdb")
    p.add_argument("--seed",     type=int, default=234,
                   help="Practical 3 uses seed 234.")
 
    # model
    p.add_argument("--hidden",         type=int,   default=50,
                   help="Hidden units in the single hidden layer.")
    p.add_argument("--prior_variance", type=float, default=1.0)
    p.add_argument("--noise_scale",    type=float, default=0.1)
    p.add_argument("--activation",     default="tanh", choices=list(ACTIVATIONS))
 
    # sampler
    p.add_argument("--n_samples", type=int,   default=1500)
    p.add_argument("--burn_in",   type=int,   default=500)
    p.add_argument("--step_size", type=float, default=1e-3)
    p.add_argument("--num_steps", type=int,   default=None,
                   help="Leapfrog steps L.  If omitted and -T is given, "
                        "L = round(T / step_size).")
    p.add_argument("-T", "--trajectory_length", type=float, default=0.1)
 
    # sweep
    p.add_argument("--sweep", action="store_true",
                   help="Sweep over activations x step sizes and produce "
                        "acceptance_rate.png + efficiency.png.")
    p.add_argument("--activations", nargs="+",
                   default=["tanh", "relu", "alpha_tanh",
                            "alpha_relu", "alpha_sigmoid"])
    p.add_argument("--step_sizes", nargs="+", type=float,
                   default=[5e-4, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 3.5e-3,
                            4e-3])
    p.add_argument("--no_plot", action="store_true",
                   help="Skip matplotlib plotting at the end of --sweep.")
    p.add_argument("--plot_only", default=None,
                   help="Path to an existing sweep-summary CSV; skip sampling "
                        "and just redraw the two curves from it.")
 
    # device
    p.add_argument("--cpu_only", action="store_true")
    return p.parse_args()
 
 
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
 
    # Shortcut: just redraw curves from a saved summary CSV
    if args.plot_only is not None:
        df = pd.read_csv(args.plot_only)
        plot_curves(df, args.out_dir)
        return
 
    if args.cpu_only:
        jax.config.update("jax_platform_name", "cpu")
 
    print(f"[jax] version: {jax.__version__}  "
          f"backend: {jax.default_backend()}  devices: {jax.devices()}")
 
    X_tr, y_tr, X_te, y_te, scaler, features = load_tmdb(
        data_dir=args.data_dir, seed=args.seed)

    if args.sweep:
        df = run_sweep(X_tr, y_tr, X_te, y_te, scaler,
                       hidden=args.hidden,
                       activations=args.activations,
                       step_sizes=args.step_sizes,
                       trajectory_length=args.trajectory_length,
                       n_samples=args.n_samples,
                       burn_in=args.burn_in,
                       prior_variance=args.prior_variance,
                       noise_scale=args.noise_scale,
                       seed=args.seed,
                       out_dir=args.out_dir)
        if not args.no_plot:
            plot_curves(df, args.out_dir)
        return
 
    # Single run --------------------------------------------------------------
    bnn = build_bnn(X_tr.shape[1], args.hidden, args.activation,
                    args.prior_variance, args.noise_scale)
 
    if args.num_steps is not None:
        L = args.num_steps
        T_actual = args.step_size * L
    else:
        L, T_actual = num_steps_from_T(args.trajectory_length, args.step_size)
    print(f"[hmc] arch={bnn.layer_sizes}  act={args.activation}  "
          f"eps={args.step_size:g}  L={L}  T_actual={T_actual:g}")
 
    # evaluate on the *test* set for single-run mode
    res = one_run(bnn, X_tr, y_tr, X_te, y_te, scaler,
                  n_samples=args.n_samples, burn_in=args.burn_in,
                  num_steps=L, step_size=args.step_size, seed=args.seed)

    print("\n================= Results =================")
    print(f" activation:        {args.activation}")
    print(f" architecture:      {bnn.layer_sizes}")
    print(f" step_size (eps):   {args.step_size}")
    print(f" num_steps (L):     {L}")
    print(f" T = eps * L:       {T_actual:g}")
    print(f" acceptance rate:   {res['accept_rate']:.4f}")
    print(f" mean deltaH:       {np.mean(res['deltaH']):+.4f}")
    print(f" mean term_1:       {np.mean(res['term_1']):+.4f}")
    print(f" mean term_2:       {np.mean(res['term_2']):+.4f}")
    print(f" test MSE (norm):   {res['mse_norm']:.4g}")
    print(f" test MSE (orig):   {res['mse_orig']:.4g}")
    print(f" wall time:         {res['wall_s']:.1f}s")
 
    pd.DataFrame({
        "deltaH":   res["deltaH"],
        "term_1":   res["term_1"],
        "term_2":   res["term_2"],
        "accepted": res["accepted"],
    }).to_csv(os.path.join(args.out_dir,
             f"tmdb_history_{args.activation}.csv"), index=False)
 
 
if __name__ == "__main__":
    main()
