"""6-model comparison with all known issues fixed.

Fixes vs notebooks:
  - β upper bound: 15 (was 10, saturating at 9.8)
  - obs_precision: 20.0 everywhere (notebook 03 used 1.0)
  - initial_sigma1: 0.25 in NLL eval (notebook 03 used 1.0)
  - decay prior: constrained to [0, 0.15] (was [0, 1], fitting 0.22)
  - sticky model: now included in comparison

Divergence/stability fixes (this revision):
  - NLL via log-mean-exp over draws (posterior-predictive) across all six
    nll_* functions; was a per-draw mean of -log p, which let a few outlier
    high-β draws blow up to 1e6+. (This is the fix that makes the comparison
    trustworthy.)
  - target_accept=0.99 for HGF (was 0.95) — smaller steps for its tricky geometry.
  - sampler diagnostics (divergences, max R-hat) saved per model.
  - floatX kept at float32: a float64 trial broke the RW family (4000/4000
    divergences) under numpyro default initvals, so it was reverted.

Usage:
  cd bayesd_misfits
  .venv/bin/python scripts/run_comparison.py          # quick (20 sub, 2 chains)
  FULL_RUN=1 .venv/bin/python scripts/run_comparison.py  # production
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import logsumexp

import hssm
from ssms.rl import ModelConfig
from ssms.rl.env import Bandit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bayesd_misfits.data import ensure_data_downloaded, load_challenge_data
from bayesd_misfits.model import (
    NArmRescorlaWagner, NArmDualAlphaRW, NArmRWDualAlphaSticky,
)
from bayesd_misfits.hgf import NArmHGF

warnings.filterwarnings("ignore")
logging.getLogger("jax._src.xla_bridge").setLevel("ERROR")
# float32 (NOT float64): an earlier attempt used float64 to stabilize HGF, but it
# catastrophically broke the RW-family geometry (4000/4000 divergences, R-hat 3.66)
# under numpyro's default initvals. float32 was the setting that produced sensible
# RW/DualAlpha fits, so we keep it. HGF stability is handled instead via its tighter
# target_accept (0.99) and the outlier-robust log-mean-exp NLL.
hssm.set_floatX("float32", update_jax=True)

SEED = 20260719
FULL_RUN = os.environ.get("FULL_RUN", "0") == "1"
N_SUB = 50 if FULL_RUN else 30
N_TRIALS = 120
N_CHAINS = 4 if FULL_RUN else 2
N_TUNE = 1000 if FULL_RUN else 500
N_DRAWS = 1000 if FULL_RUN else 500
N_NLL_DRAWS = 200 if FULL_RUN else 100

# HGF constants — MUST be consistent between fitting and evaluation
OBS_PRECISION = 20.0
INITIAL_SIGMA1 = 0.25
INITIAL_SIGMA2 = 1.0
INITIAL_MU1 = 0.5
INITIAL_MU2 = -1.0
THETA_VAR = 0.01

PE_PRIOR = {"name": "Normal", "mu": 0,
            "sigma": {"name": "HalfNormal", "sigma": 0.5}}


def hp(name, lo, hi, mu, sig, re_sig=0.05):
    """Hierarchical param: group intercept + per-participant random effect."""
    return hssm.Param(
        name, formula=f"{name} ~ 1 + (1|participant_id)",
        prior={
            "Intercept": hssm.Prior("TruncatedNormal", lower=lo, upper=hi,
                                    mu=mu, sigma=sig),
            "1|participant_id": {"name": "Normal", "mu": 0,
                                 "sigma": {"name": "HalfNormal", "sigma": re_sig}},
        },
    )


def load_data():
    ensure_data_downloaded()
    df = load_challenge_data(
        feedback_transform="normalize", rt_placeholder=-1.0,
        group_by="trajectory",
    )
    tc = df.groupby("participant_id").size()
    df = df[df["participant_id"].isin(tc[tc == N_TRIALS].index)].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    pids = sorted(df["participant_id"].unique())
    sel = rng.choice(pids, size=min(N_SUB, len(pids)), replace=False)
    df = df[df["participant_id"].isin(sel)].sort_values(
        ["participant_id", "trial_id"]).reset_index(drop=True)
    pid_map = {p: i for i, p in enumerate(sorted(df["participant_id"].unique()))}
    df["participant_id"] = df["participant_id"].map(pid_map)
    data = df[["participant_id", "trial_id", "response", "feedback"]].copy()
    print(f"Data: {data['participant_id'].nunique()} sub × {N_TRIALS} trials"
          f" = {len(data)} rows")
    return data


def draw_theta(idata, params, n_sub, idx):
    post = idata.posterior
    if hasattr(post, "to_dataset"):
        post = post.to_dataset()
    post = post.stack(sample=("chain", "draw"))
    theta = {}
    for name in params:
        re = post[f"{name}_1|participant_id"]
        dim = [d for d in re.dims if d != "sample"][0]
        vals = (post[f"{name}_Intercept"] + re).isel(sample=idx)
        ids = [int(v) for v in re[dim].values]
        theta[name] = (pd.Series(np.asarray(vals.values), index=ids)
                       .sort_index().reindex(range(n_sub)).to_numpy())
    return theta


# ── NLL functions ──


def nll_rw(idata, data, n_sub):
    """One-step-ahead posterior-predictive NLL for RW (single alpha).

    Returns log-mean-exp NLL across posterior draws: NLL_t = -log(mean_d p_d).
    Averaging probabilities before the log prevents a single high-β draw that
    mispredicts from inflating the mean (the cause of earlier 1e6+ blow-ups).
    """
    post = idata.posterior
    if hasattr(post, "to_dataset"):
        post = post.to_dataset()
    post = post.stack(sample=("chain", "draw"))
    didx = np.random.default_rng(42).choice(
        post.sizes["sample"], min(N_NLL_DRAWS, post.sizes["sample"]),
        replace=False)
    logps = []
    for d in didx:
        th = draw_theta(idata, ["rl_alpha", "beta"], n_sub, int(d))
        for pid in range(n_sub):
            a, b = th["rl_alpha"][pid], th["beta"][pid]
            Q = np.full(4, 0.5)
            for _, r in data[data["participant_id"] == pid].sort_values("trial_id").iterrows():
                act, rew = int(r["response"]), float(r["feedback"])
                lg = b * Q
                logps.append(lg[act] - logsumexp(lg))      # log p(chosen)
                Q[act] += a * (rew - Q[act])
    lp = np.array(logps).reshape(len(didx), -1)            # (n_draws, n_trials)
    return -logsumexp(lp, axis=0) + np.log(lp.shape[0])    # log-mean-exp → NLL


def nll_rw_decay(idata, data, n_sub):
    """One-step-ahead posterior-predictive NLL for RW + decay (log-mean-exp)."""
    post = idata.posterior
    if hasattr(post, "to_dataset"):
        post = post.to_dataset()
    post = post.stack(sample=("chain", "draw"))
    didx = np.random.default_rng(42).choice(
        post.sizes["sample"], min(N_NLL_DRAWS, post.sizes["sample"]),
        replace=False)
    logps = []
    for d in didx:
        th = draw_theta(idata, ["rl_alpha", "rl_decay", "beta"], n_sub, int(d))
        for pid in range(n_sub):
            a, dc, b = th["rl_alpha"][pid], th["rl_decay"][pid], th["beta"][pid]
            Q = np.full(4, 0.5)
            for _, r in data[data["participant_id"] == pid].sort_values("trial_id").iterrows():
                act, rew = int(r["response"]), float(r["feedback"])
                lg = b * Q
                logps.append(lg[act] - logsumexp(lg))
                Q = (1.0 - dc) * Q + dc * 0.5
                Q[act] += a * (rew - Q[act])
    lp = np.array(logps).reshape(len(didx), -1)
    return -logsumexp(lp, axis=0) + np.log(lp.shape[0])


def nll_dual_alpha(idata, data, n_sub):
    """One-step-ahead posterior-predictive NLL for dual-alpha RW (log-mean-exp)."""
    post = idata.posterior
    if hasattr(post, "to_dataset"):
        post = post.to_dataset()
    post = post.stack(sample=("chain", "draw"))
    didx = np.random.default_rng(42).choice(
        post.sizes["sample"], min(N_NLL_DRAWS, post.sizes["sample"]),
        replace=False)
    logps = []
    for d in didx:
        th = draw_theta(idata, ["rl_alpha_pos", "rl_alpha_neg", "beta"],
                        n_sub, int(d))
        for pid in range(n_sub):
            ap, an, b = (th["rl_alpha_pos"][pid], th["rl_alpha_neg"][pid],
                         th["beta"][pid])
            Q = np.full(4, 0.5)
            for _, r in data[data["participant_id"] == pid].sort_values("trial_id").iterrows():
                act, rew = int(r["response"]), float(r["feedback"])
                lg = b * Q
                logps.append(lg[act] - logsumexp(lg))
                pe = rew - Q[act]
                Q[act] += (ap if pe >= 0 else an) * pe
    lp = np.array(logps).reshape(len(didx), -1)
    return -logsumexp(lp, axis=0) + np.log(lp.shape[0])


def nll_dual_alpha_decay(idata, data, n_sub):
    """One-step-ahead posterior-predictive NLL for dual-alpha RW + decay (log-mean-exp)."""
    post = idata.posterior
    if hasattr(post, "to_dataset"):
        post = post.to_dataset()
    post = post.stack(sample=("chain", "draw"))
    didx = np.random.default_rng(42).choice(
        post.sizes["sample"], min(N_NLL_DRAWS, post.sizes["sample"]),
        replace=False)
    logps = []
    for d in didx:
        th = draw_theta(idata,
                        ["rl_alpha_pos", "rl_alpha_neg", "rl_decay", "beta"],
                        n_sub, int(d))
        for pid in range(n_sub):
            ap, an = th["rl_alpha_pos"][pid], th["rl_alpha_neg"][pid]
            dc, b = th["rl_decay"][pid], th["beta"][pid]
            Q = np.full(4, 0.5)
            for _, r in data[data["participant_id"] == pid].sort_values("trial_id").iterrows():
                act, rew = int(r["response"]), float(r["feedback"])
                lg = b * Q
                logps.append(lg[act] - logsumexp(lg))
                Q = (1.0 - dc) * Q + dc * 0.5
                pe = rew - Q[act]
                Q[act] += (ap if pe >= 0 else an) * pe
    lp = np.array(logps).reshape(len(didx), -1)
    return -logsumexp(lp, axis=0) + np.log(lp.shape[0])


def nll_sticky(idata, data, n_sub):
    """One-step-ahead posterior-predictive NLL for dual-alpha + sticky (log-mean-exp)."""
    post = idata.posterior
    if hasattr(post, "to_dataset"):
        post = post.to_dataset()
    post = post.stack(sample=("chain", "draw"))
    didx = np.random.default_rng(42).choice(
        post.sizes["sample"], min(N_NLL_DRAWS, post.sizes["sample"]),
        replace=False)
    logps = []
    for d in didx:
        th = draw_theta(idata,
                        ["rl_alpha_pos", "rl_alpha_neg", "sticky", "beta"],
                        n_sub, int(d))
        for pid in range(n_sub):
            ap, an = th["rl_alpha_pos"][pid], th["rl_alpha_neg"][pid]
            st, b = th["sticky"][pid], th["beta"][pid]
            Q = np.full(4, 0.5)
            last = -1
            for _, r in data[data["participant_id"] == pid].sort_values("trial_id").iterrows():
                act, rew = int(r["response"]), float(r["feedback"])
                q_eff = Q.copy()
                if last >= 0:
                    q_eff[last] += st
                lg = b * q_eff
                logps.append(lg[act] - logsumexp(lg))
                pe = rew - Q[act]
                Q[act] += (ap if pe >= 0 else an) * pe
                last = act
    lp = np.array(logps).reshape(len(didx), -1)
    return -logsumexp(lp, axis=0) + np.log(lp.shape[0])


def nll_hgf(idata, data, n_sub):
    """One-step-ahead posterior-predictive NLL for HGF (log-mean-exp).

    Uses correct obs_precision & initial_sigma1.
    """
    post = idata.posterior
    if hasattr(post, "to_dataset"):
        post = post.to_dataset()
    post = post.stack(sample=("chain", "draw"))
    didx = np.random.default_rng(42).choice(
        post.sizes["sample"], min(N_NLL_DRAWS, post.sizes["sample"]),
        replace=False)
    logps = []
    for d in didx:
        th = draw_theta(idata, ["omega", "kappa", "beta"], n_sub, int(d))
        for pid in range(n_sub):
            om, ka, be = th["omega"][pid], th["kappa"][pid], th["beta"][pid]
            mu1 = np.full(4, INITIAL_MU1)
            s1 = np.full(4, INITIAL_SIGMA1)
            mu2 = np.full(4, INITIAL_MU2)
            s2 = np.full(4, INITIAL_SIGMA2)
            for _, r in data[data["participant_id"] == pid].sort_values("trial_id").iterrows():
                act, rew = int(r["response"]), float(r["feedback"])
                # Prediction (all arms)
                mu1h = mu1.copy()
                s1h = s1 + np.exp(ka * mu2 + om)
                s2h = s2 + THETA_VAR
                # Choice prediction
                lg = be * mu1h
                logps.append(lg[act] - logsumexp(lg))
                # Level 1 update (chosen arm)
                pi1 = 1.0 / s1h[act] + OBS_PRECISION
                psi1 = OBS_PRECISION / pi1
                mu1[act] = mu1h[act] + psi1 * (rew - mu1h[act])
                s1[act] = 1.0 / pi1
                # Level 2 update (chosen arm)
                pi1h_val = 1.0 / s1h[act]
                pe = mu1[act] - mu1h[act]
                d1 = (pi1h_val / pi1) + pi1h_val * pe ** 2 - 1.0
                pi2 = 1.0 / s2h[act] + 0.5 * (ka * pi1h_val) ** 2
                psi2 = 0.5 * ka * pi1h_val / pi2
                mu2[act] = mu2[act] + psi2 * d1
                s2[act] = 1.0 / pi2
                # Unchosen: carry forward predicted uncertainty
                for a2 in range(4):
                    if a2 != act:
                        s1[a2] = s1h[a2]
                        s2[a2] = s2h[a2]
    lp = np.array(logps).reshape(len(didx), -1)
    return -logsumexp(lp, axis=0) + np.log(lp.shape[0])


# ── Model definitions ──


def build_models(data):
    env = Bandit.bernoulli(probabilities=[0.25] * 4,
                           response_labels=[0, 1, 2, 3])

    # β prior: upper=15 (was 10, saturating at 9.8). re_sig=0.5 (beta values can vary more)
    beta_param = hp("beta", 0, 15, 5, 2, re_sig=0.5)

    # Decay prior: upper=0.15 (was 1.0). re_sig=0.01 (tight to prevent going below 0 or above 0.15)
    decay_param = hp("rl_decay", 0, 0.15, 0.03, 0.03, re_sig=0.01)

    models = {}

    # 1. RW
    models["RW"] = {
        "config": ModelConfig("4AB_RW", "RW", "inv_temp_softmax_4",
                              NArmRescorlaWagner(4), env, response=["response"]),
        "include": [hp("rl_alpha", 0, 1, 0.3, 0.15, re_sig=0.05), beta_param],
        "nll_fn": nll_rw,
    }

    # 2. RW + Decay
    models["RW+Decay"] = {
        "config": ModelConfig("4AB_RW_D", "RW+decay", "inv_temp_softmax_4",
                              NArmRescorlaWagner(4, use_decay=True), env,
                              response=["response"]),
        "include": [hp("rl_alpha", 0, 1, 0.3, 0.15, re_sig=0.05), decay_param, beta_param],
        "nll_fn": nll_rw_decay,
    }

    # 3. DualAlpha
    models["DualAlpha"] = {
        "config": ModelConfig("4AB_DA", "DualAlpha", "inv_temp_softmax_4",
                              NArmDualAlphaRW(4), env, response=["response"]),
        "include": [hp("rl_alpha_pos", 0, 1, 0.3, 0.15, re_sig=0.05),
                    hp("rl_alpha_neg", 0, 1, 0.3, 0.15, re_sig=0.05), beta_param],
        "nll_fn": nll_dual_alpha,
    }

    # 4. DualAlpha + Decay
    models["DualAlpha+Decay"] = {
        "config": ModelConfig("4AB_DA_D", "DualAlpha+decay",
                              "inv_temp_softmax_4",
                              NArmDualAlphaRW(4, use_decay=True), env,
                              response=["response"]),
        "include": [hp("rl_alpha_pos", 0, 1, 0.3, 0.15, re_sig=0.05),
                    hp("rl_alpha_neg", 0, 1, 0.3, 0.15, re_sig=0.05), decay_param,
                    beta_param],
        "nll_fn": nll_dual_alpha_decay,
    }

    # 5. DualAlpha + Sticky
    models["Sticky"] = {
        "config": ModelConfig("4AB_Sticky", "DualAlpha+sticky",
                              "inv_temp_softmax_4",
                              NArmRWDualAlphaSticky(4), env,
                              response=["response"]),
        "include": [hp("rl_alpha_pos", 0, 1, 0.3, 0.15, re_sig=0.05),
                    hp("rl_alpha_neg", 0, 1, 0.3, 0.15, re_sig=0.05),
                    hp("sticky", -3, 3, 0, 0.5, re_sig=0.1), beta_param],
        "nll_fn": nll_sticky,
    }

    # 6. HGF (obs_precision=20.0, NOT 1.0)
    models["HGF"] = {
        "config": ModelConfig("4AB_HGF", "HGF", "inv_temp_softmax_4",
                              NArmHGF(4, obs_precision=OBS_PRECISION), env,
                              response=["response"]),
        "include": [hp("omega", -8, 2, -2, 1, re_sig=0.5),
                    hp("kappa", 0, 4, 1, 0.5, re_sig=0.1), beta_param],
        "nll_fn": nll_hgf,
    }

    for m in models.values():
        m["config"].validate()

    return models


def main():
    print(f"{'='*60}")
    print(f"6-Model Comparison (FULL_RUN={FULL_RUN})")
    print(f"  sub={N_SUB} trials={N_TRIALS} chains={N_CHAINS}"
          f" tune={N_TUNE} draws={N_DRAWS}")
    print(f"  β prior: TruncatedNormal(0, 15, mu=5, sigma=2)")
    print(f"  decay prior: TruncatedNormal(0, 0.15, mu=0.03, sigma=0.03)")
    print(f"  HGF obs_precision: {OBS_PRECISION}")
    print(f"{'='*60}\n")

    data = load_data()
    n_sub = data["participant_id"].nunique()
    models = build_models(data)

    idatas = {}
    diagnostics = {}  # per-model sampler diagnostics (divergences, R-hat)
    for name, spec in models.items():
        print(f"\n{'='*60}")
        print(f"Fitting {name}...")
        mc = hssm.rl.RLSSMConfig.from_ssms_model(spec["config"])
        model = hssm.RLSSM(
            data=data, model_config=mc,
            p_outlier=0, lapse=None, process_initvals=False,
            include=spec["include"],
        )
        # HGF has the trickiest geometry (precision/exp updates): take smaller
        # steps (target_accept=0.99) to avoid divergences.
        ta = 0.99 if name == "HGF" else 0.95
        idata = model.sample(
            sampler="numpyro", draws=N_DRAWS, tune=N_TUNE,
            chains=N_CHAINS, cores=1, target_accept=ta,
            random_seed=SEED,
            idata_kwargs={"log_likelihood": False},
        )
        idatas[name] = idata
        div = int(idata.sample_stats["diverging"].sum())
        rh = max(float(az.rhat(idata)[v].max())
                 for v in az.rhat(idata).data_vars)
        diagnostics[name] = {"divergences": div, "max_rhat": rh}
        print(f"  divergences={div}, max R-hat={rh:.3f}")

    # NLL evaluation
    print(f"\n{'='*60}")
    print("One-step-ahead NLL per trial (lower = better)")
    print(f"{'='*60}")
    results = {}
    for name, spec in models.items():
        nll = spec["nll_fn"](idatas[name], data, n_sub)
        results[name] = {
            "mean": float(nll.mean()),
            "sd": float(nll.std()),
            "hdi3": float(np.quantile(nll, 0.03)),
            "hdi97": float(np.quantile(nll, 0.97)),
            "divergences": diagnostics[name]["divergences"],
            "max_rhat": diagnostics[name]["max_rhat"],
        }

    print(f"\n{'Model':<20} {'Mean':>8} {'SD':>8} {'94% HDI':>22}")
    print(f"{'─'*20} {'─'*8} {'─'*8} {'─'*22}")
    for name in models:
        r = results[name]
        print(f"{name:<20} {r['mean']:>8.4f} {r['sd']:>8.4f} "
              f"[{r['hdi3']:.4f}, {r['hdi97']:.4f}]")
    print(f"{'Uniform random':<20} {np.log(4):>8.4f}")

    best = min(results, key=lambda k: results[k]["mean"])
    print(f"\nBest model: {best} (NLL = {results[best]['mean']:.4f})")

    # Parameter estimates
    for name in models:
        idata = idatas[name]
        mc = hssm.rl.RLSSMConfig.from_ssms_model(models[name]["config"])
        var_names = [f"{p}_Intercept" for p in mc.list_params]
        print(f"\n{'='*40}")
        print(f"{name}:")
        print(az.summary(idata, var_names=var_names, kind="stats",
                         round_to=3))

    # Save results
    out = {
        "n_sub": n_sub, "n_trials": N_TRIALS,
        "chains": N_CHAINS, "tune": N_TUNE, "draws": N_DRAWS,
        "obs_precision": OBS_PRECISION, "beta_upper": 15,
        "decay_upper": 0.15,
        "floatX": "float64",
        "nll_method": "log_mean_exp (posterior-predictive)",
        "diagnostics": diagnostics,
        "results": results,
    }
    out_path = Path(__file__).resolve().parent.parent / "model_comparison_results_fixed.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
