"""
bnn_moons_sweep.py
==================
 
Sweep BNN-HMC classification on the make_moons dataset over
 
    seeds  x  trajectory_lengths (T)  x  step_sizes (eps)  x  activations
 
with L = round(T / eps) auto-computed per (T, eps), so every run respects
the fixed-trajectory-length policy.  tqdm tracks the sampling progress of
*each* HMC run individually (one bar per run, posted as it goes).
 
Outputs (in --out_dir, default ./results_moons)
-----------------------------------------------
    moons_sweep_summary.csv
        one row per (seed, T, eps, act):
        accept_rate, efficiency=eps*accept, mean/std deltaH / term_1 / term_2,
        test_accuracy, mean_test_loglik, num_steps, T_actual, wall_s
    moons_hist_seed{S}_T{T}_{act}_eps{eps}.csv
        per-iteration history (deltaH, term_1, term_2, accepted) for every run
 
Usage
-----
    # Quick check
    python bnn_moons_sweep.py \\
        --activations tanh relu sigmoid \\
        --trajectory_lengths 0.1 0.2 \\
        --step_sizes 0.01 0.02 0.05 \\
        --seeds 0 1
 
    # Full sweep matching the original script's defaults but multi-seed
    python bnn_moons_sweep.py \\
        --activations sigmoid tanh softplus mish gelu swish relu leaky_relu \\
        --trajectory_lengths 0.1 0.2 0.3 \\
        --step_sizes 0.01 0.019 0.028 0.037 0.046 0.055 0.064 0.073 0.082 0.091 0.1 \\
        --seeds 0 1 2 3 4
"""
 
import argparse
import os
import time
from functools import partial
 
import numpy as np
import pandas as pd
from tqdm import tqdm
 
import jax
import jax.numpy as jnp
from jax import grad, jit, tree_util, random
from jax.scipy.stats import norm
 
from sklearn.datasets import make_moons
 
np.set_printoptions(suppress=True, precision=6)
 
 
# =============================================================================
#  Activations
# =============================================================================
def get_nonlinearity(name):
    """Return a JAX activation function by name (matches utils.get_nonlinearity
    in the original script)."""
    table = {
        "tanh":       lambda x: jax.nn.tanh(x),
        "sigmoid":    lambda x: jax.nn.sigmoid(x),
        "relu":       lambda x: jax.nn.relu(x),
        "lrelu":      lambda x: jax.nn.leaky_relu(x, negative_slope=0.01),
        "leaky_relu": lambda x: jax.nn.leaky_relu(x, negative_slope=0.01),
        "softplus":   lambda x: jax.nn.softplus(x),
        "gelu":       lambda x: jax.nn.gelu(x),
        "swish":      lambda x: x * jax.nn.sigmoid(x),
        "mish":       lambda x: x * jax.nn.tanh(jax.nn.softplus(x)),
        "elu":        lambda x: jax.nn.elu(x),
    }
    if name not in table:
        raise ValueError(f"Unknown activation '{name}'. "
                         f"Choices: {list(table)}")
    return table[name]
 
 
# =============================================================================
#  Helpers
# =============================================================================
def net2vec(net):
    return jnp.concatenate([jnp.ravel(x) for layer in net for x in layer])
 
 
def num_steps_from_T(T, eps):
    """L = round(T/eps), clamped to >= 1.  Returns (L, T_actual)."""
    L = max(1, int(round(T / eps)))
    return L, eps * L
 
 
@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def log_pdf_params(params, mean_w, var_w, mean_b, var_b):
    """Gaussian log-prior over weights (ndim>1) and biases (ndim==1)."""
    log_pdf = 0.0
    for leaf in tree_util.tree_leaves(params):
        if leaf.ndim > 1:
            log_pdf += jnp.sum(norm.logpdf(leaf, mean_w, jnp.sqrt(var_w)))
        else:
            log_pdf += jnp.sum(norm.logpdf(leaf, mean_b, jnp.sqrt(var_b)))
    return log_pdf
 
 
# =============================================================================
#  Leapfrog (returns end + start gradients so we can compute term_1)
# =============================================================================
@partial(jit, static_argnums=(2, 3))
def leapfrog(q, p, potential, num_steps, step_size):
    grad_init = grad(potential)(q)
    p = jax.tree_util.tree_map(lambda p, g: p - 0.5 * step_size * g,
                                p, grad_init)
    for step in range(1, num_steps + 1):
        q = jax.tree_util.tree_map(lambda q, p: q + step_size * p, q, p)
        if step != num_steps:
            p = jax.tree_util.tree_map(
                lambda p, g: p - step_size * g, p, grad(potential)(q)
            )
    grad_end = grad(potential)(q)
    p = jax.tree_util.tree_map(lambda p, g: p - 0.5 * step_size * g,
                                p, grad_end)
    p = jax.tree_util.tree_map(lambda p: -p, p)
    return q, p, grad_end, grad_init
 
 
# =============================================================================
#  HMC driver -- one run, with its own tqdm bar
# =============================================================================
def hmc_nn(n_samples, burn_in, potential, initial_params, layer_sizes,
           num_steps, step_size, bnn, mh_rng, mom_seed, progress_desc):
    samples, accept_flags = [], []
    deltaH_hist, term1_hist = [], []
    accepted = 0
    q_curr = initial_params
 
    mean_w, var_w = bnn.prior_mean_w, bnn.prior_var_w
    mean_b, var_b = bnn.prior_mean_b, bnn.prior_var_b
 
    bar = tqdm(range(n_samples + burn_in),
               desc=progress_desc, leave=False, dynamic_ncols=True)
    for i in bar:
        p_curr = bnn.init_network_params(random.PRNGKey(i ^ mom_seed),
                                         scale=1.0)
        q_new, p_new, grad_end, grad_init = leapfrog(
            q_curr, p_curr, potential, num_steps, step_size
        )
        deltaH = (
            (potential(q_curr) - log_pdf_params(p_curr, mean_w, var_w,
                                                mean_b, var_b))
            - (potential(q_new) - log_pdf_params(p_new, mean_w, var_w,
                                                  mean_b, var_b))
        )
 
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
                accept_flags.append(True)
        else:
            samples.append(q_curr)
            if i >= burn_in:
                accept_flags.append(False)
 
        # Live acceptance update on the bar (only after burn-in is finished)
        if i >= burn_in:
            done = i - burn_in + 1
            bar.set_postfix(acc=f"{accepted/done:.3f}")
    bar.close()
 
    return (samples[burn_in + 1:], accepted / max(1, n_samples),
            deltaH_hist, term1_hist, accept_flags)
 
 
# =============================================================================
#  BNN  (binary classification, sigmoid output, BCE likelihood)
# =============================================================================
class BNN:
    def __init__(self, layer_sizes, prior_mean_w, prior_var_w,
                 prior_mean_b, prior_var_b, init_scale, nonlinearity):
        self.layer_sizes  = layer_sizes
        self.prior_mean_w = prior_mean_w
        self.prior_var_w  = prior_var_w
        self.prior_mean_b = prior_mean_b
        self.prior_var_b  = prior_var_b
        self.init_scale   = init_scale
        self.nonlinearity = nonlinearity
 
    def random_layer_params(self, m, n, key, scale):
        w_key, b_key = random.split(key)
        return (scale * random.normal(w_key, (m, n)),
                scale * random.normal(b_key, (n,)))
 
    def init_network_params(self, key, scale):
        keys = random.split(key, len(self.layer_sizes))
        return [self.random_layer_params(m, n, k, scale)
                for m, n, k in zip(self.layer_sizes[:-1],
                                   self.layer_sizes[1:], keys)]
 
    def nn_predict(self, params, inputs):
        """Forward pass; returns sigmoid probabilities for a binary task."""
        activations = inputs
        for W, b in params[:-1]:
            outputs = jnp.dot(activations, W) + b
            activations = self.nonlinearity(outputs)
        final_w, final_b = params[-1]
        logits = jnp.dot(activations, final_w) + final_b
        return jax.nn.sigmoid(logits)
 
    @partial(jit, static_argnums=(0,))
    def neg_log_posterior(self, params, inputs, targets):
        probs = self.nn_predict(params, inputs)
        # BCE (matches the original code's "logprob" sign convention)
        eps = 1e-7
        probs = jnp.clip(probs, eps, 1 - eps)
        nll = - jnp.sum(
            targets * jnp.log(probs) + (1 - targets) * jnp.log(1 - probs)
        )
        log_prior = log_pdf_params(params, self.prior_mean_w, self.prior_var_w,
                                   self.prior_mean_b, self.prior_var_b)
        return nll - log_prior
 
    def hmc_sampling(self, inputs, targets, *, n_samples, burn_in,
                     num_steps, step_size, seed, progress_desc):
        neg_log_p = lambda q: self.neg_log_posterior(q, inputs, targets)
        init_key  = random.PRNGKey(seed)
        init_pars = self.init_network_params(init_key, scale=self.init_scale)
        mh_rng    = np.random.RandomState(seed + 12345)
        mom_seed  = int(seed) * 7919
        return hmc_nn(n_samples, burn_in, neg_log_p, init_pars,
                      self.layer_sizes, num_steps, step_size, bnn=self,
                      mh_rng=mh_rng, mom_seed=mom_seed,
                      progress_desc=progress_desc)
 
 
# =============================================================================
#  Evaluation
# =============================================================================
def eval_predictive(bnn, samples, X, y):
    """Posterior-predictive: average sigmoid probabilities over samples,
    then threshold at 0.5.  Returns (accuracy, mean_log_lik)."""
    if len(samples) == 0:
        return float("nan"), float("nan")
    X = jnp.asarray(X)
    probs_sum = jnp.zeros_like(jnp.asarray(y, dtype=jnp.float32))
    for s in samples:
        probs_sum = probs_sum + bnn.nn_predict(s, X)
    mean_probs = np.asarray(probs_sum) / len(samples)
    y_arr = np.asarray(y).reshape(mean_probs.shape)
    preds = (mean_probs > 0.5).astype(np.int32)
    acc = float((preds == y_arr.astype(np.int32)).mean())
    eps = 1e-7
    mean_probs = np.clip(mean_probs, eps, 1 - eps)
    ll = float(np.mean(
        y_arr * np.log(mean_probs) + (1 - y_arr) * np.log(1 - mean_probs)
    ))
    return acc, ll
 
 
# =============================================================================
#  Sweep
# =============================================================================
def run_sweep(X_tr, y_tr, X_te, y_te, *,
              layer_sizes, activations, trajectory_lengths, step_sizes, seeds,
              n_samples, burn_in,
              prior_mean_w, prior_var_w, prior_mean_b, prior_var_b,
              init_scale, out_dir):
 
    # --- skip combinations where T < eps so we never schedule L=0 -----
    grid = []
    for act in activations:
        for T in trajectory_lengths:
            for eps in step_sizes:
                if eps > T:           # would round to L=0; skip
                    continue
                for seed in seeds:
                    grid.append((act, T, eps, seed))
 
    print(f"[sweep] {len(grid)} runs total  "
          f"({len(activations)} activations  x  "
          f"{len(trajectory_lengths)} trajectory lengths  x  "
          f"{len(step_sizes)} step-sizes  x  {len(seeds)} seeds, "
          f"after dropping eps > T)")
 
    rows = []
    for run_idx, (act, T, eps, seed) in enumerate(grid, start=1):
        nonlinearity = get_nonlinearity(act)
        bnn = BNN(layer_sizes,
                  prior_mean_w, prior_var_w, prior_mean_b, prior_var_b,
                  init_scale, nonlinearity)
        L, T_actual = num_steps_from_T(T, eps)
 
        desc = (f"[{run_idx}/{len(grid)}] {act:9s} "
                f"T={T:g} eps={eps:g} L={L} seed={seed}")
        t0 = time.time()
        try:
            samples, acc_rate, deltaHs, term1s, acc_flags = bnn.hmc_sampling(
                jnp.asarray(X_tr), jnp.asarray(y_tr),
                n_samples=n_samples, burn_in=burn_in,
                num_steps=L, step_size=eps, seed=seed,
                progress_desc=desc,
            )
        except Exception as e:
            print(f"  -> FAILED ({type(e).__name__}: {e})")
            continue
        wall = time.time() - t0
 
        deltaHs = np.asarray(deltaHs)
        term1s  = np.asarray(term1s)
        term2s  = term1s - deltaHs
        test_acc, test_ll = eval_predictive(bnn, samples, X_te, y_te)
 
        row = {
            "activation":         act,
            "T_target":           T,
            "T_actual":           T_actual,
            "step_size":          eps,
            "num_steps":          L,
            "seed":               seed,
            "accept_rate":        acc_rate,
            "efficiency":         eps * acc_rate,
            "mean_deltaH":        float(np.mean(deltaHs)) if len(deltaHs) else np.nan,
            "mean_term_1":        float(np.mean(term1s))  if len(term1s)  else np.nan,
            "mean_term_2":        float(np.mean(term2s))  if len(term2s)  else np.nan,
            "std_deltaH":         float(np.std(deltaHs))  if len(deltaHs) else np.nan,
            "std_term_1":         float(np.std(term1s))   if len(term1s)  else np.nan,
            "std_term_2":         float(np.std(term2s))   if len(term2s)  else np.nan,
            "test_accuracy":      test_acc,
            "test_loglik":        test_ll,
            "wall_s":             wall,
        }
        print(f"  done: accept={acc_rate:.3f}  eff={row['efficiency']:.4g}  "
              f"acc={test_acc:.3f}  ll={test_ll:+.3f}  wall={wall:.1f}s")
        rows.append(row)
 
        # per-iteration history
        pd.DataFrame({
            "deltaH":   deltaHs,
            "term_1":   term1s,
            "term_2":   term2s,
            "accepted": acc_flags,
        }).to_csv(os.path.join(
            out_dir,
            f"moons_hist_seed{seed}_T{T:g}_{act}_eps{eps:g}.csv"
        ), index=False)
 
    df = pd.DataFrame(rows)
    out_path = os.path.join(out_dir, "moons_sweep_summary.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[sweep] summary saved to {out_path}")
    return df
 
 
# =============================================================================
#  Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
 
    # data
    p.add_argument("--n_samples_data",    type=int, default=400)
    p.add_argument("--moons_noise",       type=float, default=0.1)
    p.add_argument("--moons_random_state", type=int, default=0)
    p.add_argument("--n_train",            type=int, default=300)
    p.add_argument("--out_dir",            default="./results_moons")
 
    # model
    p.add_argument("--hidden",         type=int, default=50,
                   help="Hidden units in the single hidden layer "
                        "([2, hidden, 1]).")
    p.add_argument("--init_scale",     type=float, default=0.1)
    p.add_argument("--prior_mean_w",   type=float, default=0.0)
    p.add_argument("--prior_var_w",    type=float, default=1.0)
    p.add_argument("--prior_mean_b",   type=float, default=0.0)
    p.add_argument("--prior_var_b",    type=float, default=1.0)
 
    # sampler
    p.add_argument("--n_samples", type=int, default=2000,
                   help="Post burn-in HMC samples per run.")
    p.add_argument("--burn_in",   type=int, default=1000)
 
    # sweep grid
    p.add_argument("--seeds",               nargs="+", type=int,
                   default=[0, 1, 2, 3, 4])
    p.add_argument("--trajectory_lengths",  nargs="+", type=float,
                   default=[0.1, 0.2, 0.3])
    p.add_argument("--step_sizes",          nargs="+", type=float,
                   default=list(np.linspace(0.01, 0.1, 11)))
    p.add_argument("--activations",         nargs="+",
                   default=["sigmoid", "tanh", "softplus", "mish",
                            "gelu", "swish", "relu", "leaky_relu"])
 
    # device
    p.add_argument("--cpu_only", action="store_true")
    return p.parse_args()
 
 
def main():
    args = parse_args()
    if args.cpu_only:
        jax.config.update("jax_platform_name", "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
 
    print(f"[jax] version: {jax.__version__}  "
          f"backend: {jax.default_backend()}  devices: {jax.devices()}")
 
    # ---- data ------------------------------------------------------------
    X, y = make_moons(n_samples=args.n_samples_data,
                      noise=args.moons_noise,
                      random_state=args.moons_random_state)
    y = y.reshape(-1, 1).astype(np.float32)
    X = X.astype(np.float32)
    X_tr, y_tr = X[:args.n_train], y[:args.n_train]
    X_te, y_te = X[args.n_train:], y[args.n_train:]
    print(f"[data] X_tr {X_tr.shape}  y_tr {y_tr.shape}  "
          f"X_te {X_te.shape}  y_te {y_te.shape}")
 
    layer_sizes = [X_tr.shape[1], args.hidden, 1]
    print(f"[model] layer_sizes={layer_sizes}")
 
    # ---- sweep -----------------------------------------------------------
    df = run_sweep(X_tr, y_tr, X_te, y_te,
                   layer_sizes=layer_sizes,
                   activations=args.activations,
                   trajectory_lengths=args.trajectory_lengths,
                   step_sizes=args.step_sizes,
                   seeds=args.seeds,
                   n_samples=args.n_samples,
                   burn_in=args.burn_in,
                   prior_mean_w=args.prior_mean_w,
                   prior_var_w=args.prior_var_w,
                   prior_mean_b=args.prior_mean_b,
                   prior_var_b=args.prior_var_b,
                   init_scale=args.init_scale,
                   out_dir=args.out_dir)
 
    # ---- summary -----------------------------------------------------------
    if not df.empty:
        agg = (df.groupby(["activation", "T_target", "step_size"])
                 .agg(accept_mean=("accept_rate", "mean"),
                      accept_std =("accept_rate", "std"),
                      eff_mean   =("efficiency",  "mean"),
                      acc_mean   =("test_accuracy", "mean"))
                 .reset_index())
        print("\n[summary] aggregated across seeds:")
        print(agg.to_string(index=False))
 
 
if __name__ == "__main__":
    main()
