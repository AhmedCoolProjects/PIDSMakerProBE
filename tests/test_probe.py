import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

"""Unit and functional tests for the ProBE (probe_ml) pipeline.

Tests:
1. 61-feature extraction across all 14 categories.
2. Producer vs consumer category slicing (zero-rebuild category ablations).
3. Normalizers: asinh_zscore, zscore, asinh, none.
4. Classical ML detectors: isolation_forest, gmm, pca, mahalanobis, hbos, ecod.
5. End-to-end config parsing with --eng-edge-feats.
"""

import numpy as np
import pytest

from pidsmaker.config import get_runtime_required_args, get_yml_cfg
from pidsmaker.detection.training_methods import ml_training
from pidsmaker.preprocessing.build_graph_methods import edge_feat_engineering


def test_schema_dimensions():
    """Verify all 14 categories and 61 engineered features are registered."""
    cats = edge_feat_engineering.ALL_CATEGORIES
    names = edge_feat_engineering.feature_names_from_categories(cats)
    assert len(cats) == 14
    assert len(names) == 61
    assert len(names) == len(set(names)), "Feature names must be unique"


def test_feature_computation_synthetic_window():
    """Verify compute() runs over window events and produces valid float32 features."""
    surviving = [
        {
            "src": "proc_1",
            "dst": "file_1",
            "time": 1_500_000_000,
            "label": "EVENT_READ",
            "event_uuid": "u1",
            "abs_ts": [1_000_000_000, 1_500_000_000],
            "src_type": "subject",
            "dst_type": "file",
            "src_path": "/bin/cat",
            "dst_path": "/etc/shadow",
            "nf_remote_ip": None,
            "nf_remote_port": None,
            "nf_local_port": None,
            "src_label": "subject /bin/cat cat",
            "dst_label": "file /etc/shadow",
        },
        {
            "src": "proc_1",
            "dst": "net_1",
            "time": 2_000_000_000,
            "label": "EVENT_CONNECT",
            "event_uuid": "u2",
            "abs_ts": [2_000_000_000],
            "src_type": "subject",
            "dst_type": "netflow",
            "src_path": "/bin/cat",
            "dst_path": None,
            "nf_remote_ip": "192.168.1.100",
            "nf_remote_port": "443",
            "nf_local_port": "54321",
            "src_label": "subject /bin/cat cat",
            "dst_label": "netflow 192.168.1.100 443",
        },
    ]
    raw_events = [
        ("proc_1", "file_1", "EVENT_READ", 1_000_000_000),
        ("proc_1", "file_1", "EVENT_READ", 1_500_000_000),
        ("proc_1", "net_1", "EVENT_CONNECT", 2_000_000_000),
    ]

    feats = edge_feat_engineering.compute(
        surviving,
        raw_events,
        window_start_ns=0,
        window_end_ns=3_000_000_000,
        categories=edge_feat_engineering.ALL_CATEGORIES,
    )
    assert feats.shape == (2, 61)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()


def test_producer_consumer_category_slicing():
    """Verify consumer can slice arbitrary category subsets from stored producer."""
    producer_cats = edge_feat_engineering.ALL_CATEGORIES
    consumer_cats = ["pair", "time", "structural", "ema"]
    expected_dim = 5 + 5 + 4 + 4  # 18

    indices = edge_feat_engineering.feature_indices_from_categories(
        producer_cats, consumer_cats
    )
    assert len(indices) == expected_dim

    rng = np.random.default_rng(42)
    producer_matrix = rng.standard_normal((10, 61)).astype(np.float32)
    consumer_matrix = producer_matrix[:, indices]
    assert consumer_matrix.shape == (10, expected_dim)


@pytest.mark.parametrize("norm_mode", ["asinh_zscore", "zscore", "asinh", "none"])
def test_normalizer(norm_mode):
    rng = np.random.default_rng(0)
    X = np.abs(rng.standard_normal((100, 61)).astype(np.float32)) * 100.0
    norm = ml_training.Normalizer(norm_mode).fit(X)
    Xn = norm.transform(X)
    assert Xn.shape == X.shape
    assert np.isfinite(Xn).all()


@pytest.mark.parametrize(
    "algo",
    ["isolation_forest", "gmm", "pca", "mahalanobis", "hbos", "ecod"],
)
def test_ml_detectors_fit_and_score(algo):
    rng = np.random.default_rng(123)
    X_train = rng.standard_normal((80, 61)).astype(np.float32)
    X_test = rng.standard_normal((20, 61)).astype(np.float32)

    args = get_runtime_required_args(
        args=[
            "probe_ml",
            "CADETS_E3",
            "--cpu",
            f"--training.ml.algorithm={algo}",
            "--eng-edge-feats",
            "--eng-edge-feats-categories=all",
            "--artifact_dir=/tmp/probe_test_art",
        ]
    )
    cfg = get_yml_cfg(args)
    norm = ml_training.Normalizer(cfg.training.ml.normalize).fit(X_train)
    model = ml_training.build_model(cfg).fit(norm.transform(X_train))
    scores = model.score(norm.transform(X_test))

    assert scores.shape == (20,)
    assert np.isfinite(scores).all()


def test_probe_config_integration():
    """Verify probe_ml configuration loads and flips eng_edge_feats consumer & producer."""
    args = get_runtime_required_args(
        args=[
            "probe_ml",
            "CADETS_E3",
            "--cpu",
            "--eng-edge-feats",
            "--eng-edge-feats-categories=pair,time",
            "--artifact_dir=/tmp/probe_test_art",
        ]
    )
    cfg = get_yml_cfg(args)
    assert cfg.construction.compute_eng_edge_feats is True
    assert "eng_edge_feats" in cfg.batching.edge_features
    assert edge_feat_engineering.get_active_dim(cfg) == 10  # 5 pair + 5 time
