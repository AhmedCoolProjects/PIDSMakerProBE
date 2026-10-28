#!/usr/bin/env python3
"""ProBE Quickstart & Reproduction Demo.

Demonstrates the 61-feature extraction and runs all 4 primary classical ML
detectors (Isolation Forest, GMM, PCA, Mahalanobis) on synthetic provenance data
in ~1 second, with zero database or GPU dependencies.

Usage:
    python scripts/demo_probe_quickstart.py
"""

import sys
from pathlib import Path
import numpy as np

# Add repo root to import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pidsmaker.config import get_runtime_required_args, get_yml_cfg
from pidsmaker.detection.training_methods import ml_training
from pidsmaker.preprocessing.build_graph_methods import edge_feat_engineering


def main():
    print("=" * 70)
    print("  ProBE: Provenance Behavioral Edge Feature & ML Pipeline Demo")
    print("=" * 70)

    # 1. Inspect Category Schema
    cats = edge_feat_engineering.CATEGORY_SCHEMA
    total_dim = sum(len(f) for f in cats.values())
    print(f"\n[1] Registered Feature Schema: {len(cats)} categories -> {total_dim} features")
    for cat_name, feats in cats.items():
        print(f"    - {cat_name:18s} ({len(feats):2d} dims): {', '.join(feats[:3])}{'...' if len(feats) > 3 else ''}")

    # 2. Extract features from a synthetic provenance window
    print("\n[2] Computing 61-d features on synthetic provenance edges...")
    surviving = [
        {
            "src": "proc_bash", "dst": "file_passwd", "time": 1000,
            "label": "EVENT_READ", "event_uuid": "u1",
            "abs_ts": [900, 1000], "src_type": "subject", "dst_type": "file",
            "src_path": "/bin/bash", "dst_path": "/etc/passwd",
            "nf_remote_ip": None, "nf_remote_port": None, "nf_local_port": None,
            "src_label": "subject /bin/bash bash", "dst_label": "file /etc/passwd",
        },
        {
            "src": "proc_bash", "dst": "net_c2", "time": 2000,
            "label": "EVENT_CONNECT", "event_uuid": "u2",
            "abs_ts": [2000], "src_type": "subject", "dst_type": "netflow",
            "src_path": "/bin/bash", "dst_path": None,
            "nf_remote_ip": "198.51.100.23", "nf_remote_port": "443", "nf_local_port": "48922",
            "src_label": "subject /bin/bash bash", "dst_label": "netflow 198.51.100.23 443",
        },
    ]
    raw_events = [
        ("proc_bash", "file_passwd", "EVENT_READ", 900),
        ("proc_bash", "file_passwd", "EVENT_READ", 1000),
        ("proc_bash", "net_c2", "EVENT_CONNECT", 2000),
    ]
    extracted = edge_feat_engineering.compute(
        surviving, raw_events, window_start_ns=0, window_end_ns=5000,
        categories=list(cats.keys())
    )
    print(f"    Output feature matrix: shape={extracted.shape}, dtype={extracted.dtype}")

    # 3. Fit & evaluate ML detectors on benign edges vs an anomalous edge
    print("\n[3] Fitting & scoring classical ML detectors (train on benign, score test)...")
    rng = np.random.default_rng(42)
    # 500 benign edges + 5 anomalous burst edges
    X_benign = rng.standard_normal((500, 61)).astype(np.float32)
    X_anomalous = (rng.standard_normal((5, 61)) * 4.0 + 8.0).astype(np.float32)
    X_test = np.vstack([X_benign[:50], X_anomalous])
    y_test = np.array([0] * 50 + [1] * 5)

    models_to_test = [
        ("isolation_forest", {}),
        ("gmm", {"training.ml.n_components": 4, "training.ml.covariance_type": "full"}),
        ("pca", {"training.ml.pca_variance": 0.95}),
        ("mahalanobis", {}),
    ]

    print(f"    {'Algorithm':<20s} {'Mean Benign Score':<20s} {'Mean Attack Score':<20s} {'Separation':<12s}")
    print("    " + "-" * 72)

    for algo_name, extra_overrides in models_to_test:
        cli_args = [
            "probe_ml", "CADETS_E3", "--cpu",
            f"--training.ml.algorithm={algo_name}",
            "--eng-edge-feats",
            "--artifact_dir=/tmp/probe_demo_art",
        ]
        for k, v in extra_overrides.items():
            cli_args.append(f"--{k}={v}")

        cfg = get_yml_cfg(get_runtime_required_args(args=cli_args))
        norm = ml_training.Normalizer(cfg.training.ml.normalize).fit(X_benign)
        model = ml_training.build_model(cfg).fit(norm.transform(X_benign))
        scores = model.score(norm.transform(X_test))

        benign_sc = float(scores[:50].mean())
        attack_sc = float(scores[50:].mean())
        sep = "PASS (attack > benign)" if attack_sc > benign_sc else "WARN"
        print(f"    {algo_name:<20s} {benign_sc:<20.4f} {attack_sc:<20.4f} {sep:<12s}")

    print("\n" + "=" * 70)
    print("  All ProBE models initialized, trained, and evaluated successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
