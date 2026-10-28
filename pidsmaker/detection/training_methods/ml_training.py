"""probe_ml — classic unsupervised anomaly detection on the engineered edge features.

Fits a classic AD model on the BENIGN training edges' engineered vector
(``g.edge_feats``), then scores val/test edges and writes the SAME per-edge loss
CSVs that ``inference_loop.test_edge_level`` writes. This way ``node_evaluation``
(max edge-score -> node, robust thresholds, best-AUC/ADP snapshots) runs unchanged.

Algorithm menu:
  - ``isolation_forest`` (sklearn) : tree isolation. Scale-robust DEFAULT.
  - ``hbos``            (numpy)    : histogram outlier score.
  - ``ecod``            (numpy)    : empirical-CDF tail probabilities.
  - ``mahalanobis``     (numpy)    : single-Gaussian robust distance.
  - ``gmm``             (sklearn)  : Gaussian mixture likelihood (multi-modal normal).
  - ``elliptic``        (sklearn)  : robust-covariance (MCD) Gaussian envelope.
  - ``pca``             (sklearn)  : PCA reconstruction error.
  - ``ocsvm``           (sklearn)  : One-Class SVM (bounded benign subsample).
  - ``sgd_ocsvm``       (sklearn)  : Nystroem + SGD One-Class SVM.
  - ``svdd``            (cvxopt)   : Support Vector Data Description.
  - ``hmm``             (hmmlearn) : Gaussian HMM sequence likelihood.

Requires the engineered block to reach ``g.edge_feats`` — i.e. run with the eng
features enabled and ``batching.edge_features=eng_edge_feats`` (config probe_ml).
"""

import gc
import os
import pickle
import tracemalloc
from time import perf_counter as timer

import numpy as np
import pandas as pd
import wandb

from pidsmaker.tasks.batching import get_preprocessed_graphs
from pidsmaker.utils.utils import (
    log,
    log_start,
    log_tqdm,
    ns_time_to_datetime_US,
    set_seed,
)

MODEL_EPOCH_FILE = "model_epoch_0"  # classic ML = a single fitted model (no epochs)
ML_MODEL_FILE = "probe_ml_model.pkl"


# --------------------------------------------------------------------------- #
# Normalization (scaler fit on benign train).                                  #
# --------------------------------------------------------------------------- #
class Normalizer:
    def __init__(self, mode):
        self.mode = (mode or "none").strip()
        self.mean = None
        self.std = None

    def _pre(self, X):
        X = np.nan_to_num(X, nan=0.0, posinf=1e30, neginf=-1e30)
        if "asinh" in self.mode:
            X = np.arcsinh(X)
        return X

    def fit(self, X):
        Xt = self._pre(X)
        if "zscore" in self.mode:
            self.mean = Xt.mean(axis=0)
            self.std = Xt.std(axis=0) + 1e-8
        return self

    def transform(self, X):
        Xt = self._pre(X)
        if "zscore" in self.mode:
            Xt = (Xt - self.mean) / self.std
        return Xt


# --------------------------------------------------------------------------- #
# AD model wrappers — every `.score(X)` returns HIGHER = more anomalous.        #
# --------------------------------------------------------------------------- #
class _SklearnScorer:
    """Wrap a fitted sklearn estimator; `kind` selects the scoring fn.

    score_samples: higher = inlier (negate). decision_function: higher = inlier (negate).
    """

    def __init__(self, model, kind, max_train_samples=None, seed=0):
        self.model = model
        self.kind = kind
        self.max_train_samples = max_train_samples
        self.seed = seed

    def fit(self, X):
        # ponytail: caps benign train samples for O(N^2) models; remove or raise when compute permits
        if self.max_train_samples and len(X) > self.max_train_samples:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(X), size=self.max_train_samples, replace=False)
            X = X[idx]
        self.fit_samples_ = len(X)
        self.model.fit(X)
        return self

    def score(self, X):
        if self.kind == "score_samples":
            # ponytail: sklearn 1.2 IsolationForest.score_samples ignores n_jobs; parallel thread chunks release Cython GIL
            if self.model.__class__.__name__ == "IsolationForest" and len(X) >= 4096:
                from joblib import Parallel, delayed

                n_threads = min(8, os.cpu_count() or 1)
                chunks = np.array_split(X, n_threads)
                res = Parallel(n_jobs=n_threads, prefer="threads")(
                    delayed(self.model.score_samples)(c) for c in chunks if len(c) > 0
                )
                return -np.concatenate(res)
            return -self.model.score_samples(X)
        return (-self.model.decision_function(X)).ravel()


class _GaussianHMMScorer:
    """Gaussian HMM with per-edge predictive negative log-likelihood scores."""

    def __init__(self, model, max_train_samples=None, seed=0):
        self.model = model
        self.max_train_samples = max_train_samples
        self.seed = seed

    def fit(self, X, lengths):
        if sum(lengths) != len(X):
            raise ValueError("HMM sequence lengths must sum to the number of samples")

        if self.max_train_samples and len(X) > self.max_train_samples:
            starts = np.cumsum([0] + lengths[:-1])
            order = np.random.default_rng(self.seed).permutation(len(lengths))
            sequences = []
            selected_lengths = []
            remaining = self.max_train_samples
            for i in order:
                size = min(lengths[i], remaining)
                if size:
                    sequences.append(X[starts[i] : starts[i] + size])
                    selected_lengths.append(size)
                    remaining -= size
                if remaining == 0:
                    break
            X = np.concatenate(sequences)
            lengths = selected_lengths

        self.fit_samples_ = len(X)
        self.fit_sequences_ = len(lengths)
        self.model.fit(X, lengths=lengths)
        return self

    def score(self, X):
        from hmmlearn import _hmmc

        emission = self.model._compute_log_likelihood(X)
        _, forward = _hmmc.forward_log(
            self.model.startprob_, self.model.transmat_, emission
        )
        cumulative = np.logaddexp.reduce(forward, axis=1)
        # ponytail: hmmlearn exposes no per-observation predictive likelihood;
        # use its forward kernel until that becomes public API.
        scores = -np.diff(np.concatenate(([0.0], cumulative)))
        if not np.isfinite(scores).all():
            raise ValueError("HMM produced non-finite anomaly scores")
        return scores

    def score_timed(self, X, timestamps):
        order = np.argsort(timestamps, kind="stable")
        scores = np.empty(len(X), dtype=float)
        scores[order] = self.score(X[order])
        return scores


class _HBOSScorer:
    """Histogram-Based Outlier Score (Goldstein & Dengel, 2012), pure numpy."""

    def __init__(self, n_bins=50):
        self.n_bins = n_bins
        self.edges = []  # per-feature bin edges
        self.log_inv_density = []  # per-feature log(1/density) per bin

    def fit(self, X):
        n, d = X.shape
        for j in range(d):
            col = X[:, j]
            lo, hi = float(col.min()), float(col.max())
            if hi <= lo:  # constant feature -> contributes nothing
                self.edges.append(None)
                self.log_inv_density.append(None)
                continue
            edges = np.linspace(lo, hi, self.n_bins + 1)
            counts, _ = np.histogram(col, bins=edges)
            width = (hi - lo) / self.n_bins
            density = counts / (n * width)
            with np.errstate(divide="ignore"):
                lid = -np.log(density + 1e-12)
            self.edges.append(edges)
            self.log_inv_density.append(lid)
        return self

    def score(self, X):
        n, d = X.shape
        out = np.zeros(n, dtype=np.float64)
        for j in range(d):
            edges = self.edges[j]
            if edges is None:
                continue
            idx = np.searchsorted(edges, X[:, j], side="right") - 1
            idx = np.clip(idx, 0, self.n_bins - 1)
            out += self.log_inv_density[j][idx]
        return out


class _ECODScorer:
    """Empirical-CDF Outlier Detection (Li et al., 2022), pure numpy."""

    def __init__(self):
        self.sorted_cols = []
        self.n_train = 0

    def fit(self, X):
        self.n_train = X.shape[0]
        for j in range(X.shape[1]):
            self.sorted_cols.append(np.sort(X[:, j]))
        return self

    def score(self, X):
        n, d = X.shape
        nt = max(self.n_train, 1)
        left = np.zeros(n, dtype=np.float64)
        right = np.zeros(n, dtype=np.float64)
        for j in range(d):
            col = self.sorted_cols[j]
            le = np.searchsorted(col, X[:, j], side="right") / nt
            ge = (nt - np.searchsorted(col, X[:, j], side="left")) / nt
            left += -np.log(np.clip(le, 1e-12, 1.0))
            right += -np.log(np.clip(ge, 1e-12, 1.0))
        return np.maximum(left, right)


class _MahalanobisScorer:
    """Single-Gaussian Mahalanobis distance to the benign mean (pure numpy)."""

    def __init__(self, ridge=1e-6):
        self.ridge = ridge
        self.mu = None
        self.inv = None

    def fit(self, X):
        self.mu = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        cov = np.atleast_2d(cov)
        cov += np.eye(cov.shape[0]) * self.ridge
        self.inv = np.linalg.pinv(cov)
        return self

    def score(self, X):
        diff = X - self.mu
        return np.einsum("ij,jk,ik->i", diff, self.inv, diff)


class _PCAScorer:
    def __init__(self, n_components, variance=None, seed=0):
        self.n = n_components
        self.variance = variance if (variance and 0.0 < variance < 1.0) else None
        self.seed = seed

    def fit(self, X):
        from sklearn.decomposition import PCA

        if self.variance is not None:
            n_components = self.variance
        else:
            n_components = min(self.n, X.shape[1])
        self.pca = PCA(n_components=n_components, svd_solver="full", random_state=self.seed).fit(X)
        return self

    def score(self, X):
        Xr = self.pca.inverse_transform(self.pca.transform(X))
        return ((X - Xr) ** 2).sum(axis=1)


def _kernel_gamma(ml):
    gamma = getattr(ml, "gamma", None) or "scale"
    if gamma in ("scale", "auto"):
        return gamma
    try:
        value = float(gamma)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "training.ml.gamma must be 'scale', 'auto', or a positive number"
        ) from exc
    if value <= 0:
        raise ValueError(
            "training.ml.gamma must be 'scale', 'auto', or a positive number"
        )
    return value


def build_model(cfg):
    ml = cfg.training.ml
    algo = (ml.algorithm or "isolation_forest").strip()
    seed = cfg.training.seed
    contamination = getattr(ml, "contamination", None)
    contamination = contamination if (contamination and contamination > 0) else None
    n_estimators = getattr(ml, "n_estimators", None) or 200
    n_components = getattr(ml, "n_components", None) or 8
    n_bins = getattr(ml, "n_bins", None) or 50

    if algo == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        m = IsolationForest(
            n_estimators=n_estimators,
            contamination=(contamination if contamination is not None else "auto"),
            random_state=seed,
            n_jobs=-1,
        )
        return _SklearnScorer(m, "score_samples")

    if algo == "hbos":
        return _HBOSScorer(n_bins=n_bins)

    if algo == "ecod":
        return _ECODScorer()

    if algo == "mahalanobis":
        return _MahalanobisScorer()

    if algo == "gmm":
        from sklearn.mixture import GaussianMixture

        reg_covar = getattr(ml, "reg_covar", None)
        tol = getattr(ml, "tol", None)
        m = GaussianMixture(
            n_components=n_components,
            covariance_type=(getattr(ml, "covariance_type", None) or "full"),
            reg_covar=(reg_covar if (reg_covar and reg_covar > 0) else 1e-6),
            n_init=(getattr(ml, "n_init", None) or 1),
            max_iter=(getattr(ml, "max_iter", None) or 100),
            tol=(tol if (tol and tol > 0) else 1e-3),
            init_params=(getattr(ml, "init_params", None) or "kmeans"),
            random_state=seed,
        )
        return _SklearnScorer(m, "score_samples")

    if algo == "elliptic":
        from sklearn.covariance import EllipticEnvelope

        m = EllipticEnvelope(
            contamination=(contamination if contamination is not None else 0.1), random_state=seed
        )
        return _SklearnScorer(m, "score_samples")

    if algo == "pca":
        pca_variance = getattr(ml, "pca_variance", None)
        return _PCAScorer(n_components, variance=pca_variance, seed=seed)

    if algo == "ocsvm":
        from sklearn.svm import OneClassSVM

        # ponytail: defaults to 25k subsample to avoid O(N^2) LIBSVM freeze on 3.4M edges; use sgd_ocsvm for full stream
        nu = getattr(ml, "nu", None)
        nu = nu if (nu and nu > 0) else 0.05
        kernel = getattr(ml, "kernel", None) or "rbf"
        gamma = _kernel_gamma(ml)
        max_samples = getattr(ml, "max_train_samples", None)
        max_samples = max_samples if (max_samples and max_samples > 0) else 25000
        m = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        return _SklearnScorer(m, "decision_function", max_train_samples=max_samples, seed=seed)

    if algo == "sgd_ocsvm":
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import SGDOneClassSVM
        from sklearn.pipeline import make_pipeline

        nu = getattr(ml, "nu", None)
        nu = nu if (nu and nu > 0) else 0.05
        n_comp = getattr(ml, "n_components", None) or 100
        m = make_pipeline(
            Nystroem(kernel="rbf", n_components=n_comp, random_state=seed),
            SGDOneClassSVM(nu=nu, random_state=seed),
        )
        return _SklearnScorer(m, "decision_function")

    if algo == "svdd":
        from .svdd import BaseSVDD

        # ponytail: caps benign train samples for dense QP O(N^3) solver (default 1000)
        c_param = getattr(ml, "c_param", None)
        c_param = c_param if (c_param and c_param > 0) else 0.9
        kernel = getattr(ml, "kernel", None) or "rbf"
        gamma = _kernel_gamma(ml)
        degree = getattr(ml, "degree", None) or 3
        max_samples = getattr(ml, "max_train_samples", None)
        max_samples = max_samples if (max_samples and max_samples > 0) else 1000
        m = BaseSVDD(C=c_param, kernel=kernel, gamma=gamma, degree=degree, display="off")
        return _SklearnScorer(m, "decision_function", max_train_samples=max_samples, seed=seed)

    if algo == "hmm":
        from hmmlearn.hmm import GaussianHMM

        covariance_type = getattr(ml, "covariance_type", None) or "diag"
        max_iter = getattr(ml, "max_iter", None) or 20
        tol = getattr(ml, "tol", None)
        max_samples = getattr(ml, "max_train_samples", None)
        max_samples = max_samples if (max_samples and max_samples > 0) else 100000
        model = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=max_iter,
            tol=tol if (tol and tol > 0) else 1e-3,
            random_state=seed,
            implementation="log",
        )
        return _GaussianHMMScorer(model, max_train_samples=max_samples, seed=seed)

    raise ValueError(
        f"Invalid training.ml.algorithm `{algo}`. Practical options: "
        "isolation_forest, hbos, ecod, mahalanobis, gmm, elliptic, pca, "
        "ocsvm, sgd_ocsvm, svdd, hmm."
    )


# --------------------------------------------------------------------------- #
# Data gathering + scoring                                                     #
# --------------------------------------------------------------------------- #
def _edge_feats(g):
    ef = getattr(g, "edge_feats", None)
    if ef is None:
        raise ValueError(
            "g.edge_feats is None — probe_ml needs the engineered block routed to edge_feats. "
            "Enable eng features and set batching.edge_features=eng_edge_feats (use config probe_ml "
            "or --eng-edge-feats)."
        )
    return ef.detach().cpu().numpy().astype(np.float32)


def _gather_benign_train(train_data, temporal=False):
    X = []
    lengths = []
    for dataset in train_data:
        for g in dataset:
            ef = _edge_feats(g)
            y = (
                g.y.detach().cpu().numpy()
                if hasattr(g, "y") and g.y is not None
                else np.zeros(len(ef))
            )
            mask = y != 1
            benign = ef[mask]
            if temporal and len(benign):
                timestamps = g.t.detach().cpu().numpy()[mask]
                benign = benign[np.argsort(timestamps, kind="stable")]
            if len(benign):
                X.append(benign)
                lengths.append(len(benign))
    if not X:
        raise ValueError("No benign training edges found to fit the probe_ml model.")
    return np.concatenate(X, axis=0), lengths


def _ml_model_path(cfg):
    return os.path.join(cfg.training._trained_models_dir, ML_MODEL_FILE)


def save_fitted_model(cfg, model, normalizer, train_shape):
    os.makedirs(cfg.training._trained_models_dir, exist_ok=True)
    payload = {
        "model": model,
        "normalizer": normalizer,
        "algorithm": cfg.training.ml.algorithm,
        "normalize": cfg.training.ml.normalize,
        "seed": cfg.training.seed,
        "train_shape": tuple(train_shape),
        "model_epoch_file": MODEL_EPOCH_FILE,
    }
    out_file = _ml_model_path(cfg)
    with open(out_file, "wb") as f:
        pickle.dump(payload, f)
    log(f"[ml] saved fitted model artifact: {out_file}")
    return out_file


def load_fitted_model(cfg):
    model_file = _ml_model_path(cfg)
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"Missing fitted probe_ml model artifact: {model_file}. "
            "Rerun the source ML training once after this update so it saves "
            f"{ML_MODEL_FILE}."
        )
    with open(model_file, "rb") as f:
        payload = pickle.load(f)
    return payload


def score_and_write(data, split, model, normalizer, cfg, model_epoch_file=MODEL_EPOCH_FILE):
    logs_dir = os.path.join(cfg.training._edge_losses_dir, split, model_epoch_file)
    os.makedirs(logs_dir, exist_ok=True)

    tracemalloc.start()
    start = timer()
    try:
        for dataset in data:
            if (
                not hasattr(model, "score_timed")
                and model.__class__.__name__ == "_SklearnScorer"
                and model.model.__class__.__name__ == "IsolationForest"
            ):
                # ponytail: batch-concatenates features for tree ensembles to avoid per-graph Python loop overhead
                valid_graphs, feats, lengths = [], [], []
                for g in dataset:
                    t = g.t.cpu().numpy().astype(int)
                    if len(t) == 0:
                        continue
                    valid_graphs.append((g, t))
                    ef = _edge_feats(g)
                    feats.append(ef)
                    lengths.append(len(t))
                if feats:
                    X_all = normalizer.transform(np.vstack(feats))
                    all_scores = model.score(X_all)
                    offset = 0
                    for (g, t), length in zip(valid_graphs, lengths):
                        scores = all_scores[offset : offset + length]
                        offset += length
                        time_interval = (
                            ns_time_to_datetime_US(int(t[0]))
                            + "~"
                            + ns_time_to_datetime_US(int(t.max()))
                        )
                        csv_path = os.path.join(logs_dir, time_interval + ".csv")
                        edge_index = g.original_edge_index
                        src = edge_index[0, :].cpu().numpy().astype(int)
                        dst = edge_index[1, :].cpu().numpy().astype(int)
                        edge_types = (g.edge_type.max(dim=1).indices + 1).cpu().numpy().astype(int)
                        pd.DataFrame(
                            {
                                "loss": scores.astype(float),
                                "srcnode": src,
                                "dstnode": dst,
                                "time": t,
                                "edge_type": edge_types,
                            }
                        ).to_csv(csv_path, index=False)
                continue

            for g in log_tqdm(dataset, f"Scoring {split} edges"):
                t = g.t.cpu().numpy().astype(int)
                if len(t) == 0:
                    continue
                time_interval = (
                    ns_time_to_datetime_US(int(t[0]))
                    + "~"
                    + ns_time_to_datetime_US(int(t.max()))
                )
                csv_path = os.path.join(logs_dir, time_interval + ".csv")

                ef = _edge_feats(g)
                normalized = normalizer.transform(ef)
                if hasattr(model, "score_timed"):
                    scores = model.score_timed(normalized, t)
                else:
                    scores = model.score(normalized)

                edge_index = g.original_edge_index
                src = edge_index[0, :].cpu().numpy().astype(int)
                dst = edge_index[1, :].cpu().numpy().astype(int)
                edge_types = (g.edge_type.max(dim=1).indices + 1).cpu().numpy().astype(int)

                edge_df = pd.DataFrame(
                    {
                        "loss": scores.astype(float),
                        "srcnode": src,
                        "dstnode": dst,
                        "time": t,
                        "edge_type": edge_types,
                    }
                )
                edge_df.to_csv(csv_path, index=False)
    finally:
        _, peak_cpu_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    return {
        f"time_{split}_inference": round(timer() - start, 2),
        f"peak_{split}_inference_cpu_memory": round(peak_cpu_memory / (1024**3), 3),
        f"peak_{split}_inference_gpu_memory": 0.0,
    }


def main(cfg):
    log_start(__file__)
    set_seed(cfg, seed=cfg.training.seed)

    train_data, val_data, test_data, _ = get_preprocessed_graphs(cfg)
    tracemalloc.start()
    train_start = timer()
    try:
        X_train, train_lengths = _gather_benign_train(
            train_data, temporal=cfg.training.ml.algorithm == "hmm"
        )
        log(f"[ml] benign train edges: {X_train.shape} (dim={X_train.shape[1]})")

        normalizer = Normalizer(cfg.training.ml.normalize).fit(X_train)
        Xn = normalizer.transform(X_train)

        model = build_model(cfg)
        if cfg.training.ml.algorithm == "hmm":
            model.fit(Xn, train_lengths)
        else:
            model.fit(Xn)
    finally:
        _, peak_train_cpu_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    train_time = timer() - train_start
    fitted_on = getattr(model, "fit_samples_", Xn.shape[0])
    log(
        f"[ml] fitted algorithm={cfg.training.ml.algorithm} "
        f"normalize={cfg.training.ml.normalize} on "
        f"{fitted_on}/{Xn.shape[0]} benign edges in {train_time:.2f}s"
    )
    if cfg.training.ml.algorithm == "svdd":
        svdd = model.model
        log(
            f"[ml] SVDD C={svdd.C[0]} gamma={svdd.gamma:.8g} "
            f"support_vectors={svdd.n_support_vectors}/{svdd.n_samples} "
            f"radius={svdd.radius:.8g}"
        )
    if cfg.training.ml.algorithm == "hmm":
        hmm = model.model
        log(
            f"[ml] HMM states={hmm.n_components} "
            f"covariance={hmm.covariance_type} sequences={model.fit_sequences_} "
            f"converged={hmm.monitor_.converged} "
            f"iterations={hmm.monitor_.iter}"
        )

    save_fitted_model(cfg, model, normalizer, X_train.shape)
    val_stats = score_and_write(val_data, "val", model, normalizer, cfg)
    test_stats = score_and_write(test_data, "test", model, normalizer, cfg)

    train_log = {
        "peak_train_cpu_memory": round(peak_train_cpu_memory / (1024**3), 3),
        "peak_train_gpu_memory": 0.0,
        "peak_inference_cpu_memory": max(
            val_stats["peak_val_inference_cpu_memory"],
            test_stats["peak_test_inference_cpu_memory"],
        ),
        "peak_inference_gpu_memory": 0.0,
        "time_ml_train": round(train_time, 2),
        "time_ml_val_inference": val_stats["time_val_inference"],
        "time_ml_test_inference": test_stats["time_test_inference"],
    }
    wandb.log(train_log)
    gc.collect()
    log("[ml] wrote per-edge anomaly scores for val + test (model_epoch_0)")
