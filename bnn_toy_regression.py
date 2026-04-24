"""
bnn_toy_sweep.py
================
 
Sweep BNN-HMC on the user's toy 1-D regression dataset over
 
    seeds  x  step-sizes  x  activations
 
with the trajectory length T = eps * L held fixed (so L is auto-computed
per step-size, just as in the UCI / TMDB scripts).
 
Inputs
------
A pickle containing (X_train, Y_train, X_test, Y_test) -- the same file
referenced in the user's notebook:
    /content/drive/MyDrive/bnn_regression_results/data/toy_dataset.pkl
 
You can supply the path with --pickle or, if you don't have it, generate
a fresh toy dataset with --generate (matches the typical 1-D Snelson-style
problem used in BNN demos).
 
Outputs (in --out_dir, default ./results_toy)
---------------------------------------------
    toy_sweep_summary.csv                       -- one row per (seed, eps, act)
    toy_hist_seed{S}_{act}_eps{eps}.csv         -- per-iteration histories
    toy_efficiency.png  (optional, if matplotlib available)
 
Architecture defaults follow the notebook:  [1, 20, 20, 20, 20, 1]  with
softplus activation, prior_variance=1, noise_scale=0.1.
 
Usage
-----
    # Single-seed sanity check
    python bnn_toy_sweep.py --pickle toy_dataset.pkl \\
        --activations softplus tanh relu \\
        --step_sizes 5e-4 1e-3 2e-3 4e-3 \\
        --seeds 0 -T 0.1
 
    # Many seeds for confidence bands on the efficiency plot
    python bnn_toy_sweep.py --pickle toy_dataset.pkl \\
        --activations softplus tanh relu sigmoid gelu \\
        --step_sizes 5e-4 1e-3 1.5e-3 2e-3 2.5e-3 3e-3 3.5e-3 4e-3 \\
        --seeds 0 1 2 3 4 -T 0.1
"""
 
import argparse
import os
import pickle
import time
from functools import partial
 
import numpy as np
import pandas as pd
from tqdm import tqdm
 
import jax
import jax.numpy as jnp
from jax import grad, jit, random
from jax.scipy.stats import norm
 
np.set_printoptions(suppress=True, precision=6)
 
 
# =============================================================================
#  Activation zoo (matches the previous scripts)
# =============================================================================
ACTIVATIONS = {
    "tanh":          lambda x: jax.nn.tanh(x),
    "sigmoid":       lambda x: jax.nn.sigmoid(x),
    "relu":          lambda x: jax.nn.relu(x),
    "leaky_relu":    lambda x: jax.nn.leaky_relu(x, negative_slope=0.01),
    "softplus":      lambda x: jax.nn.softplus(x),
    "gelu":          lambda x: jax.nn.gelu(x),
    "swish":         lambda x: x * jax.nn.sigmoid(x),
    "mish":          lambda x: x * jax.nn.tanh(jax.nn.softplus(x)),
}
 
 
# =============================================================================
#  HMC machinery (same as the notebook / UCI / TMDB scripts)
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
           num_steps, step_size, bnn, mh_rng=None, progress=True):
    samples, accept_samples = [], []
    deltaH_hist, term1_hist = [], []
    accepted = 0
    q_curr = initial_params.copy()
    bias = bnn.bias
 
    iterator = range(n_samples + burn_in)
    if progress:
        iterator = tqdm(iterator)
 
    if mh_rng is None:
        mh_rng = np.random.RandomState(0)
 
    for i in iterator:
        # NOTE: keep the notebook's convention of seeding momenta with the
        # iteration index, but XOR in the run seed via the BNN's seed so
        # multiple seeds give different momentum streams.
        p_curr = bnn.init_network_params(random.PRNGKey(i ^ bnn._mom_seed),
                                         scale=1.0)
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
 
        if jnp.log(mh_rng.rand()) < deltaH:
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
    """Same BNN as the notebook (Gaussian prior, fixed-noise Gaussian
    likelihood)."""
 
    def __init__(self, layer_sizes, prior_variance, noise_scale,
                 nonlinearity, bias=True):
        self.layer_sizes    = layer_sizes
        self.prior_variance = prior_variance
        self.noise_scale    = noise_scale
        self.nonlinearity   = nonlinearity
        self.bias           = bias
        self._mom_seed      = 0     # set per-run by hmc_sampling
 
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
        self._mom_seed = int(seed) * 7919   # large prime for momentum decorr
        neg_log_p = lambda q: self.neg_log_posterior(q, inputs, targets)
        init_key  = random.PRNGKey(seed)
        init_pars = self.init_network_params(init_key, scale=self.noise_scale)
        mh_rng    = np.random.RandomState(seed + 12345)
        return hmc_nn(n_samples, burn_in, neg_log_p, init_pars,
                      self.layer_sizes, num_steps, step_size, bnn=self,
                      mh_rng=mh_rng, progress=progress)
 
    def predict(self, samples, inputs):
        out = [self.nn_predict(s, inputs) for s in samples]
        return jnp.stack(out, axis=0)        # [S, N, D_out]
 
 
# =============================================================================
#  Toy data:  load from pickle (notebook convention) or generate
# =============================================================================
def load_toy_pickle(path):
    with open(path, "rb") as f:
        X_tr, Y_tr, X_te, Y_te = pickle.load(f)
    X_tr = np.asarray(X_tr, dtype=np.float32)
    Y_tr = np.asarray(Y_tr, dtype=np.float32)
    X_te = np.asarray(X_te, dtype=np.float32)
    Y_te = np.asarray(Y_te, dtype=np.float32)
    if X_tr.ndim == 1: X_tr = X_tr.reshape(-1, 1)
    if X_te.ndim == 1: X_te = X_te.reshape(-1, 1)
    if Y_tr.ndim == 1: Y_tr = Y_tr.reshape(-1, 1)
    if Y_te.ndim == 1: Y_te = Y_te.reshape(-1, 1)
    return X_tr, Y_tr, X_te, Y_te
 
 
def generate_toy(n_train=20, n_test=200, noise=0.1, seed=0):
    """Snelson-style 1-D regression problem, used in many BNN demos."""
    rng = np.random.RandomState(seed)
    X_tr = rng.uniform(-3, 3, size=(n_train, 1)).astype(np.float32)
    Y_tr = (np.sin(X_tr) + 0.3 * np.cos(2 * X_tr)
            + noise * rng.randn(*X_tr.shape)).astype(np.float32)
    X_te = np.linspace(-4, 4, n_test).reshape(-1, 1).astype(np.float32)
    Y_te = (np.sin(X_te) + 0.3 * np.cos(2 * X_te)).astype(np.float32)
    return X_tr, Y_tr, X_te, Y_te
 
 
# =============================================================================
#  Sweep
# =============================================================================
def num_steps_from_T(T, eps):
    L = max(1, int(round(T / eps)))
    return L, eps * L
 
 
def evaluate(bnn, samples, X_te, Y_te):
    if len(samples) == 0:
        return float("nan")
    preds = bnn.predict(samples, jnp.asarray(X_te))      # [S, N, 1]
    pred_mean = np.asarray(preds).mean(axis=0)            # [N, 1]
    return float(np.mean((pred_mean - Y_te) ** 2))
 
 
def one_run(bnn, X_tr, Y_tr, X_te, Y_te, *,
            n_samples, burn_in, num_steps, step_size, seed):
    t0 = time.time()
    samples, acc_rate, deltaHs, term1s, acc_flags = bnn.hmc_sampling(
        jnp.asarray(X_tr), jnp.asarray(Y_tr),
        n_samples=n_samples, burn_in=burn_in,
        num_steps=num_steps, step_size=step_size, seed=seed,
        progress=False,
    )
    deltaHs = np.asarray(deltaHs)
    term1s  = np.asarray(term1s)
    return {
        "samples":       samples,
        "accepted":      acc_flags,
        "accept_rate":   float(acc_rate),
        "deltaH":        deltaHs,
        "term_1":        term1s,
        "term_2":        term1s - deltaHs,
        "test_mse":      evaluate(bnn, samples, X_te, Y_te),
        "wall_s":        time.time() - t0,
    }
 
 
def run_sweep(X_tr, Y_tr, X_te, Y_te, *,
              layer_sizes, activations, step_sizes, seeds,
              trajectory_length, n_samples, burn_in,
              prior_variance, noise_scale, out_dir):
    rows = []
    total = len(activations) * len(step_sizes) * len(seeds)
    print(f"[sweep] {total} runs:  {len(activations)} activations  x  "
          f"{len(step_sizes)} step-sizes  x  {len(seeds)} seeds")
 
    bar = tqdm(total=total, desc="sweep")
    for act in activations:
        bnn = BNN(layer_sizes, prior_variance, noise_scale,
                  ACTIVATIONS[act], bias=True)
        for eps in step_sizes:
            L, T_actual = num_steps_from_T(trajectory_length, eps)
            for seed in seeds:
                try:
                    res = one_run(bnn, X_tr, Y_tr, X_te, Y_te,
                                  n_samples=n_samples, burn_in=burn_in,
                                  num_steps=L, step_size=eps, seed=seed)
                except Exception as e:
                    print(f"\n  [fail] act={act} eps={eps} seed={seed}: {e}")
                    bar.update(1); continue
                row = {
                    "activation":    act,
                    "step_size":     eps,
                    "num_steps":     L,
                    "T_target":      trajectory_length,
                    "T_actual":      T_actual,
                    "seed":          seed,
                    "accept_rate":   res["accept_rate"],
                    "efficiency":    eps * res["accept_rate"],
                    "mean_deltaH":   float(np.mean(res["deltaH"])),
                    "mean_term_1":   float(np.mean(res["term_1"])),
                    "mean_term_2":   float(np.mean(res["term_2"])),
                    "std_deltaH":    float(np.std(res["deltaH"])),
                    "std_term_1":    float(np.std(res["term_1"])),
                    "std_term_2":    float(np.std(res["term_2"])),
                    "test_mse":      res["test_mse"],
                    "wall_s":        res["wall_s"],
                }
                rows.append(row)
 
                # per-iteration history
                hist = pd.DataFrame({
                    "deltaH":   res["deltaH"],
                    "term_1":   res["term_1"],
                    "term_2":   res["term_2"],
                    "accepted": res["accepted"],
                })
                hist.to_csv(os.path.join(
                    out_dir,
                    f"toy_hist_seed{seed}_{act}_eps{eps:g}.csv"
                ), index=False)
                bar.set_postfix(act=act, eps=f"{eps:g}",
                                seed=seed,
                                acc=f"{res['accept_rate']:.2f}")
                bar.update(1)
    bar.close()
 
    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "toy_sweep_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"[sweep] summary saved to {out_path}")
    return df
 
 
# =============================================================================
#  Plotting (with confidence bands across seeds)
# =============================================================================
def plot_efficiency_with_seeds(df, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed; skipping.")
        return
 
    # Aggregate: mean / std over seeds for each (activation, step_size)
    grp = df.groupby(["activation", "step_size"]).agg(
        accept_mean=("accept_rate", "mean"),
        accept_std =("accept_rate", "std"),
        eff_mean   =("efficiency",  "mean"),
        eff_std    =("efficiency",  "std"),
        mse_mean   =("test_mse",    "mean"),
    ).reset_index().fillna(0.0)
 
    # ---- efficiency vs eps ------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for act, g in grp.groupby("activation"):
        g = g.sort_values("step_size")
        ax.plot(g["step_size"], g["eff_mean"], marker="o", label=act)
        ax.fill_between(g["step_size"],
                        g["eff_mean"] - g["eff_std"],
                        g["eff_mean"] + g["eff_std"],
                        alpha=0.2)
    ax.set_xlabel(r"Step Size $\epsilon$")
    ax.set_ylabel(r"Efficiency $= \epsilon \cdot$ Acceptance Rate")
    ax.set_title("HMC efficiency on toy data (mean ± std across seeds)")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(title="Activation")
    plt.tight_layout()
    eff_path = os.path.join(out_dir, "toy_efficiency.png")
    plt.savefig(eff_path, dpi=150, bbox_inches="tight")
    plt.close()
 
    # ---- acceptance vs eps ------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for act, g in grp.groupby("activation"):
        g = g.sort_values("step_size")
        ax.plot(g["step_size"], g["accept_mean"], marker="o", label=act)
        ax.fill_between(g["step_size"],
                        g["accept_mean"] - g["accept_std"],
                        g["accept_mean"] + g["accept_std"],
                        alpha=0.2)
    ax.set_xlabel(r"Step Size $\epsilon$")
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("HMC acceptance on toy data (mean ± std across seeds)")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(title="Activation")
    plt.tight_layout()
    acc_path = os.path.join(out_dir, "toy_acceptance.png")
    plt.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close()
 
    print(f"[plot] saved {eff_path}")
    print(f"[plot] saved {acc_path}")
 
 
# =============================================================================
#  Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    # data
    g = p.add_mutually_exclusive_group()
    g.add_argument("--pickle", default=None,
                   help="Path to a pickle file containing "
                        "(X_train, Y_train, X_test, Y_test).")
    g.add_argument("--generate", action="store_true",
                   help="Generate a synthetic 1-D toy dataset instead of "
                        "loading from pickle.")
    p.add_argument("--out_dir", default="./results_toy")
 
    # model
    p.add_argument("--layer_sizes", nargs="+", type=int,
                   default=[1, 50, 1],
                   help="Architecture (incl. input + output dims).")
    p.add_argument("--prior_variance", type=float, default=1.0)
    p.add_argument("--noise_scale",    type=float, default=0.1)
 
    # sampler
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--burn_in",   type=int, default=1000)
    p.add_argument("-T", "--trajectory_length", type=float, default=0.1,
                   help="Fixed trajectory length T = eps * L.")
 
    # sweep grid
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--step_sizes", nargs="+", type=float,
                   default=[5e-4, 1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 3.5e-3,
                            4e-3])
    p.add_argument("--activations", nargs="+",
                   default=["softplus", "tanh", "relu",
                            "leaky_relu", "sigmoid"])
 
    # plotting / device
    p.add_argument("--no_plot",   action="store_true")
    p.add_argument("--cpu_only",  action="store_true")
    return p.parse_args()
 
 
def main():
    args = parse_args()
    if args.cpu_only:
        jax.config.update("jax_platform_name", "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
 
    print(f"[jax] version: {jax.__version__}  "
          f"backend: {jax.default_backend()}  devices: {jax.devices()}")
 
    # ---- data ------------------------------------------------------------
    if args.pickle is not None:
        X_tr, Y_tr, X_te, Y_te = load_toy_pickle(args.pickle)
        src = f"pickle:{args.pickle}"
    elif args.generate:
        X_tr, Y_tr, X_te, Y_te = generate_toy()
        src = "synthetic Snelson-style 1-D"
    else:
        # default: try the notebook's path, else generate
        default_pkl = ("/content/drive/MyDrive/bnn_regression_results/"
                       "data/toy_dataset.pkl")
        if os.path.exists(default_pkl):
            X_tr, Y_tr, X_te, Y_te = load_toy_pickle(default_pkl)
            src = f"pickle:{default_pkl}"
        else:
            X_tr, Y_tr, X_te, Y_te = generate_toy()
            src = "synthetic Snelson-style 1-D (no pickle found)"
 
    print(f"[data] source: {src}")
    print(f"[data] X_tr {X_tr.shape}  Y_tr {Y_tr.shape}  "
          f"X_te {X_te.shape}  Y_te {Y_te.shape}")
    if args.layer_sizes[0] != X_tr.shape[1]:
        raise ValueError(f"layer_sizes[0]={args.layer_sizes[0]} but "
                         f"X has {X_tr.shape[1]} features.")
    if args.layer_sizes[-1] != Y_tr.shape[1]:
        raise ValueError(f"layer_sizes[-1]={args.layer_sizes[-1]} but "
                         f"Y has {Y_tr.shape[1]} outputs.")
 
    # ---- run sweep -------------------------------------------------------
    df = run_sweep(X_tr, Y_tr, X_te, Y_te,
                   layer_sizes=args.layer_sizes,
                   activations=args.activations,
                   step_sizes=args.step_sizes,
                   seeds=args.seeds,
                   trajectory_length=args.trajectory_length,
                   n_samples=args.n_samples,
                   burn_in=args.burn_in,
                   prior_variance=args.prior_variance,
                   noise_scale=args.noise_scale,
                   out_dir=args.out_dir)
 
    # ---- summary table to console ---------------------------------------
    if not df.empty:
        agg = (df.groupby(["activation", "step_size"])
                 .agg(accept_mean=("accept_rate", "mean"),
                      accept_std =("accept_rate", "std"),
                      eff_mean   =("efficiency",  "mean"),
                      mse_mean   =("test_mse",    "mean"))
                 .reset_index())
        print("\n[summary] aggregated across seeds:")
        print(agg.to_string(index=False))
 
    if not args.no_plot:
        plot_efficiency_with_seeds(df, args.out_dir)
 
 
if __name__ == "__main__":
    main()
