"""
bnn_moons_sweep.py
==================

Sweep a Bayesian-Neural-Network classifier on `sklearn.datasets.make_moons`
over

    seeds  x  trajectory_lengths (T)  x  step_sizes (eps)
        with  L = round(T / eps)  (so eps * L stays close to T)

Built directly on top of the user-supplied `bnn_classification_moon.py`:
the BNN class, leapfrog, HMC driver, and prior structure are unchanged --
only the experiment loop is rewritten as a 3-axis sweep.

Each individual HMC run shows a tqdm progress bar (n_samples + burn_in
iterations).  The outer sweep does NOT have its own bar (so the per-run
bars stay readable).

Outputs (in --out_dir, default ./results_moon)
----------------------------------------------
    moons_sweep_summary.csv
        one row per (seed, T, eps, activation):
        accept_rate, mean test accuracy, deltaH/term1/term2 means, ...
    moons_hist_seed{s}_T{T}_eps{eps}_{act}.csv
        per-iteration deltaH / term_1 / term_2 / accepted (post burn-in)
    samples_moon/sample_seed{s}_T{T}_eps{eps}_n{L}_{act}.pkl
        raw sample chain (only if --save_samples)

Usage
-----
    python bnn_moons_sweep.py \\
        --activations sigmoid tanh relu softplus gelu swish mish lrelu \\
        --seeds 0 1 2 \\
        --trajectory_lengths 0.1 0.2 0.5 \\
        --step_sizes 0.01 0.02 0.04 0.06 0.08 0.1 \\
        --n_samples 2000 --burn_in 1000

If `utils.get_nonlinearity` is importable (i.e. you have the user's
`utils.py` next to this file), it will be used.  Otherwise we fall back to
a built-in dictionary covering the same activation names.
"""

import argparse
import os
import pickle as pkl
import time
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import grad, jit, random, tree_util
from jax.scipy.stats import norm

from sklearn.datasets import make_moons

np.set_printoptions(suppress=True, precision=6)


# =============================================================================
#  Activation handling -- prefer the user's utils.get_nonlinearity if present
# =============================================================================
try:
    from utils import get_nonlinearity as _user_get_nonlinearity   # noqa
    HAS_USER_UTILS = True
except Exception:
    HAS_USER_UTILS = False


def _act_lrelu(x):
    return jax.nn.leaky_relu(x, negative_slope=0.01)


def _act_mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


_FALLBACK_ACTIVATIONS = {
    "sigmoid":  jax.nn.sigmoid,
    "tanh":     jnp.tanh,
    "relu":     jax.nn.relu,
    "lrelu":    _act_lrelu,
    "softplus": jax.nn.softplus,
    "gelu":     jax.nn.gelu,
    "swish":    lambda x: x * jax.nn.sigmoid(x),
    "mish":     _act_mish,
}


def get_nonlinearity(name):
    if HAS_USER_UTILS:
        try:
            return _user_get_nonlinearity(name)
        except Exception:
            pass
    if name not in _FALLBACK_ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. "
                         f"Known: {list(_FALLBACK_ACTIVATIONS)}")
    return _FALLBACK_ACTIVATIONS[name]


# =============================================================================
#  Helpers (preserved from the original script + small additions)
# =============================================================================
def net2vec(net):
    vec = []
    for W, b in net:
        vec.extend(jnp.ravel(W))
        vec.extend(jnp.ravel(b))
    return jnp.array(vec)


def num_params(layer_sizes):
    n = 0
    for m, k in zip(layer_sizes[:-1], layer_sizes[1:]):
        n += m * k + k
    return n


@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def log_pdf_params(params, mean_w, var_w, mean_b, var_b):
    """Gaussian-prior log-pdf -- same semantics as the original script,
    but written without nonlocal-side-effects so it composes with jit."""
    log_pdf = 0.0
    for W, b in params:
        log_pdf += jnp.sum(norm.logpdf(W, mean_w, jnp.sqrt(var_w)))
        log_pdf += jnp.sum(norm.logpdf(b, mean_b, jnp.sqrt(var_b)))
    return log_pdf


def num_steps_from_T(T, eps):
    """L = round(T/eps), clamped to >=1.  Returns (L, eps*L)."""
    L = max(1, int(round(T / eps)))
    return L, eps * L


# =============================================================================
#  Leapfrog -- modified to also return start/end gradients (for term_1)
# =============================================================================
@partial(jit, static_argnums=(2, 3))
def leapfrog(q, p, potential, num_steps, step_size):
    """Leapfrog integrator.  Returns:
        q_new, p_new, grad_end, grad_init
    where grad_init = grad(potential)(q_input)  and
          grad_end  = grad(potential)(q_output) -- both needed to compute
    the term_1 = (eps^2/8)(||grad_end||^2 - ||grad_init||^2) diagnostic.
    """
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


# =============================================================================
#  HMC driver  (per-run tqdm, returns diagnostic histories)
# =============================================================================
def hmc_nn(n_samples, burn_in, potential, initial_params, layer_sizes,
           num_steps, step_size, bnn, mh_rng, mom_seed, tqdm_desc=None):
    samples       = []
    accept_flags  = []
    deltaH_hist   = []
    term1_hist    = []
    accepted      = 0
    q_curr        = initial_params

    mean_w, var_w = bnn.prior_mean_w, bnn.prior_var_w
    mean_b, var_b = bnn.prior_mean_b, bnn.prior_var_b

    pbar = tqdm(range(n_samples + burn_in), desc=tqdm_desc,
                leave=False, dynamic_ncols=True)

    for i in pbar:
        # Momentum stream depends on iteration AND run-specific mom_seed,
        # so different seeds give different momentum sequences.
        p_curr = bnn.init_network_params(
            random.PRNGKey(int(i) ^ int(mom_seed)), scale=1.0
        )

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
            deltaH_hist.append(float(-deltaH))   # store -deltaH
            term1_hist.append(float(term_1))

        if jnp.log(mh_rng.rand()) < deltaH:
            samples.append(q_new)
            q_curr = q_new
            if i >= burn_in:
                accepted += 1
                accept_flags.append(True)
                pbar.set_postfix(acc=f"{accepted/(i-burn_in+1):.2f}")
        else:
            samples.append(q_curr)
            if i >= burn_in:
                accept_flags.append(False)

    pbar.close()
    return (samples[burn_in + 1:], accepted / n_samples,
            deltaH_hist, term1_hist, accept_flags)


# =============================================================================
#  BNN classifier (faithful to the original)
# =============================================================================
class BNN:
    """Simple BNN classifier with Gaussian priors on W and b, sigmoid output."""

    def __init__(self, layer_sizes, prior_mean_w, prior_var_w,
                 prior_mean_b, prior_var_b, init_scale, nonlinearity):
        self.layer_sizes  = layer_sizes
        self.prior_mean_w = prior_mean_w
        self.prior_var_w  = prior_var_w
        self.prior_mean_b = prior_mean_b
        self.prior_var_b  = prior_var_b
        self.init_scale   = init_scale
        self.nonlinearity = nonlinearity

    def random_layer_params(self, m, n, key, scale=0.1):
        w_key, b_key = random.split(key)
        return (scale * random.normal(w_key, (m, n)),
                scale * random.normal(b_key, (n,)))

    def init_network_params(self, key, scale):
        keys = random.split(key, len(self.layer_sizes))
        return [self.random_layer_params(m, n, k, scale)
                for m, n, k in zip(self.layer_sizes[:-1],
                                   self.layer_sizes[1:], keys)]

    def nn_predict(self, params, inputs):
        activations = inputs
        for W, b in params[:-1]:
            outputs = jnp.dot(activations, W) + b
            activations = self.nonlinearity(outputs)
        final_W, final_b = params[-1]
        logits = jnp.dot(activations, final_W) + final_b
        return jax.nn.sigmoid(logits)

    @partial(jit, static_argnums=(0,))
    def neg_log_posterior(self, params, inputs, targets):
        probs = self.nn_predict(params, inputs)
        return self.logprob(probs, targets) - log_pdf_params(
            params, self.prior_mean_w, self.prior_var_w,
            self.prior_mean_b, self.prior_var_b)

    def logprob(self, probs, targets, eps=1e-7):
        probs = jnp.clip(probs, eps, 1 - eps)
        loss = - targets * jnp.log(probs) - (1 - targets) * jnp.log(1 - probs)
        return jnp.sum(loss)

    def hmc_sampling(self, inputs, targets, *, n_samples, burn_in,
                     num_steps, step_size, seed, tqdm_desc):
        neg_log_p = lambda params: self.neg_log_posterior(params, inputs, targets)
        init_key  = random.PRNGKey(seed)
        init_params = self.init_network_params(init_key, scale=self.init_scale)
        mh_rng    = np.random.RandomState(seed + 12345)
        return hmc_nn(n_samples, burn_in, neg_log_p, init_params,
                      self.layer_sizes, num_steps, step_size,
                      bnn=self, mh_rng=mh_rng,
                      mom_seed=seed * 7919, tqdm_desc=tqdm_desc)


def accuracy(params, inputs, targets, bnn, threshold=0.5):
    probs = bnn.nn_predict(params, inputs)
    pred  = (probs > threshold).astype(jnp.int32)
    return float(jnp.mean(pred == targets.astype(jnp.int32)))


def mean_test_accuracy(samples, X_te, y_te, bnn, max_samples=200):
    """Average per-sample accuracy on the test set.  Sub-samples to keep
    the eval cheap when the chain is long."""
    if len(samples) == 0:
        return float("nan")
    if len(samples) > max_samples:
        idx = np.linspace(0, len(samples) - 1, max_samples).astype(int)
        chain = [samples[i] for i in idx]
    else:
        chain = samples
    acc = 0.0
    for s in chain:
        acc += accuracy(s, X_te, y_te, bnn)
    return acc / len(chain)


# =============================================================================
#  Sweep
# =============================================================================
def run_sweep(*, layer_sizes, prior_mean_w, prior_var_w, prior_mean_b,
              prior_var_b, init_scale, activations, seeds,
              trajectory_lengths, step_sizes, n_samples, burn_in,
              data_seed, n_data, noise, n_train, save_samples, out_dir):
    sample_dir = os.path.join(out_dir, "samples_moon")
    if save_samples:
        os.makedirs(sample_dir, exist_ok=True)

    results = []
    total = (len(activations) * len(seeds) *
             len(trajectory_lengths) * len(step_sizes))
    print(f"[sweep] {total} runs:  {len(activations)} acts  x  "
          f"{len(seeds)} seeds  x  {len(trajectory_lengths)} T  x  "
          f"{len(step_sizes)} eps")

    run_idx = 0
    for act in activations:
        nonlin = get_nonlinearity(act)
        for seed in seeds:
            bnn = BNN(layer_sizes=layer_sizes,
                      prior_mean_w=prior_mean_w, prior_var_w=prior_var_w,
                      prior_mean_b=prior_mean_b, prior_var_b=prior_var_b,
                      init_scale=init_scale, nonlinearity=nonlin)

            # Per-seed data so the data is also varied across seeds.
            X, y = make_moons(n_samples=n_data, noise=noise,
                              random_state=data_seed + seed)
            y = y.reshape(-1, 1).astype(np.float32)
            X = X.astype(np.float32)
            X_tr, y_tr = X[:n_train],  y[:n_train]
            X_te, y_te = X[n_train:],  y[n_train:]

            for T in trajectory_lengths:
                for eps in step_sizes:
                    L, T_actual = num_steps_from_T(T, eps)
                    run_idx += 1
                    desc = (f"[{run_idx:>3d}/{total}] {act:>8s} "
                            f"seed={seed} T={T:g} eps={eps:.4g} L={L}")
                    t0 = time.time()
                    try:
                        (samples, acc_rate,
                         deltaHs, term1s, acc_flags) = bnn.hmc_sampling(
                            jnp.asarray(X_tr), jnp.asarray(y_tr),
                            n_samples=n_samples, burn_in=burn_in,
                            num_steps=L, step_size=float(eps),
                            seed=int(seed), tqdm_desc=desc,
                        )
                    except Exception as e:
                        tqdm.write(f"  -> failed: {e}")
                        continue

                    deltaHs = np.asarray(deltaHs)
                    term1s  = np.asarray(term1s)
                    term2s  = term1s - deltaHs
                    test_acc = mean_test_accuracy(samples, X_te, y_te, bnn)

                    row = {
                        "activation":   act,
                        "seed":         int(seed),
                        "T_target":     float(T),
                        "T_actual":     float(T_actual),
                        "step_size":    float(eps),
                        "num_steps":    int(L),
                        "accept_rate":  float(acc_rate),
                        "efficiency":   float(eps) * float(acc_rate),
                        "test_accuracy": test_acc,
                        "mean_deltaH":  float(np.mean(deltaHs)),
                        "mean_term_1":  float(np.mean(term1s)),
                        "mean_term_2":  float(np.mean(term2s)),
                        "std_deltaH":   float(np.std(deltaHs)),
                        "std_term_1":   float(np.std(term1s)),
                        "std_term_2":   float(np.std(term2s)),
                        "wall_s":       time.time() - t0,
                    }
                    results.append(row)

                    tqdm.write(
                        f"  acc_rate={acc_rate:.3f}  "
                        f"test_acc={test_acc:.3f}  "
                        f"<dH>={row['mean_deltaH']:+.3f}  "
                        f"<t1>={row['mean_term_1']:+.3f}  "
                        f"({row['wall_s']:.1f}s)"
                    )

                    # save per-iteration history
                    hist_path = os.path.join(
                        out_dir,
                        f"moons_hist_seed{seed}_T{T:g}_"
                        f"eps{eps:.4g}_{act}.csv")
                    pd.DataFrame({
                        "deltaH":   deltaHs,
                        "term_1":   term1s,
                        "term_2":   term2s,
                        "accepted": acc_flags,
                    }).to_csv(hist_path, index=False)

                    if save_samples:
                        pkl_name = (f"sample_seed{seed}_T{T:g}_"
                                    f"eps{eps:.4g}_n{L}_{act}.pkl")
                        with open(os.path.join(sample_dir, pkl_name), "wb") as f:
                            pkl.dump(samples, f)

    df = pd.DataFrame(results)
    out_csv = os.path.join(out_dir, "moons_sweep_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[sweep] summary saved to {out_csv}")
    return df


# =============================================================================
#  Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # data
    p.add_argument("--n_data",    type=int,   default=400)
    p.add_argument("--n_train",   type=int,   default=300)
    p.add_argument("--noise",     type=float, default=0.1)
    p.add_argument("--data_seed", type=int,   default=0,
                   help="Base seed for make_moons; per-run data uses "
                        "data_seed + seed.")

    # model / prior
    p.add_argument("--layer_sizes",  nargs="+", type=int,
                   default=[2, 50, 1])
    p.add_argument("--init_scale",   type=float, default=0.1)
    p.add_argument("--prior_mean_w", type=float, default=0.0)
    p.add_argument("--prior_var_w",  type=float, default=1.0)
    p.add_argument("--prior_mean_b", type=float, default=0.0)
    p.add_argument("--prior_var_b",  type=float, default=1.0)

    # sampler
    p.add_argument("--n_samples",  type=int, default=2000)
    p.add_argument("--burn_in",    type=int, default=1000)

    # sweep grid
    p.add_argument("--seeds", nargs="+", type=int,
                   default=[0, 1, 2])
    p.add_argument("--trajectory_lengths", "-T", nargs="+", type=float,
                   default=[0.1, 0.2, 0.5])
    p.add_argument("--step_sizes", nargs="+", type=float,
                   default=[0.01, 0.02, 0.04, 0.06, 0.08, 0.1])
    p.add_argument("--activations", nargs="+",
                   default=["sigmoid", "tanh", "relu", "softplus",
                            "gelu", "swish", "mish", "lrelu"])

    # i/o
    p.add_argument("--out_dir",      default="./results_moon")
    p.add_argument("--save_samples", action="store_true",
                   help="Also pickle full sample chains (large).")

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
    print(f"[utils] using "
          f"{'user utils.get_nonlinearity' if HAS_USER_UTILS else 'fallback activations'}")

    df = run_sweep(
        layer_sizes=args.layer_sizes,
        prior_mean_w=args.prior_mean_w, prior_var_w=args.prior_var_w,
        prior_mean_b=args.prior_mean_b, prior_var_b=args.prior_var_b,
        init_scale=args.init_scale,
        activations=args.activations,
        seeds=args.seeds,
        trajectory_lengths=args.trajectory_lengths,
        step_sizes=args.step_sizes,
        n_samples=args.n_samples, burn_in=args.burn_in,
        data_seed=args.data_seed, n_data=args.n_data,
        noise=args.noise, n_train=args.n_train,
        save_samples=args.save_samples,
        out_dir=args.out_dir,
    )

    if not df.empty:
        agg = (df.groupby(["activation", "T_target", "step_size"])
                 .agg(accept_mean=("accept_rate", "mean"),
                      accept_std =("accept_rate", "std"),
                      eff_mean   =("efficiency",  "mean"),
                      acc_mean   =("test_accuracy", "mean"))
                 .reset_index())
        print("\n[summary] aggregated across seeds (head):")
        print(agg.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
