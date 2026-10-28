<p align="center">
  <img width="80%" src="./.github/img/pidsmaker_title.png" alt="PIDSMAKER logo"/>
</p>

<div align="center">

[![Docs](https://img.shields.io/badge/Docs-Online-ed6a2f?style=flat&labelColor=gray)](https://ubc-provenance.github.io/PIDSMaker/)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.19813646-ed6a2f?style=flat&labelColor=gray)](https://doi.org/10.5281/zenodo.19813646)
[![License](https://img.shields.io/github/license/ubc-provenance/PIDSMaker?style=flat&color=ed6a2f&labelColor=gray)](LICENSE)
[![Release](https://img.shields.io/github/v/release/ubc-provenance/PIDSMaker?style=flat&color=ed6a2f&labelColor=gray)](https://github.com/ubc-provenance/PIDSMaker/releases)
[![Stars](https://img.shields.io/github/stars/ubc-provenance/PIDSMaker?style=flat&color=ed6a2f&labelColor=white&logo=github&logoColor=black)](https://github.com/ubc-provenance/PIDSMaker/stargazers)

</div>

<p align="center">
  <strong>
    <a href="https://www.usenix.org/conference/usenixsecurity25/presentation/bilot">📄 Experiments Paper</a>
    &nbsp;|&nbsp;
	<a href="https://www.arxiv.org/abs/2601.22983">📄 Framework Paper</a>
    &nbsp;|&nbsp;
    <a href="https://tristanbilot.github.io/public_content/slides_pidsmaker.pdf">🎙️ NDSS PRISM'26 Slides</a>
	  &nbsp;|&nbsp;
    <a href="https://ubc-provenance.github.io/PIDSMaker/">📘 Documentation</a>
    &nbsp;|&nbsp;
    <a href="https://ubc-provenance.github.io/PIDSMaker/ten-minute-install/">⚙️ Installation</a>
  </strong>
</p>

---

The first framework designed to build and experiment with provenance-based intrusion detection systems (PIDSs) using deep learning and lightweight statistical/ML architectures.
This repository (**PIDSMakerProBE**) integrates **ProBE** (`probe_ml`), a fast, GPU-free provenance-based intrusion detection pipeline built on **61 engineered edge features** across 14 behavioral, structural, temporal, and semantic categories, paired with classical unsupervised anomaly detectors (`gmm`, `isolation_forest`, `pca`, `mahalanobis`, etc.).

### Supported Systems

The framework currently integrates the following PIDSs.

| PIDS       | Venue               | Paper |
|------------|---------------------|-------|
| **ProBE (`probe_ml`)** | **This Repo** | Lightweight ML anomaly detection (`gmm`, `isolation_forest`, `pca`, `mahalanobis`) on 61-d engineered provenance edge features |
| Velox      | USENIX Security 2025 | [Link](https://tfjmp.org/publications/2025-usenixsec-2.pdf) |
| Orthrus    | USENIX Security 2025 | [Link](https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-103-jiang-baoxiang.pdf) |
| R-Caid     | IEEE S&P 2024        | [Link](https://gangw.web.illinois.edu/rcaid-sp24.pdf) |
| Flash      | IEEE S&P 2024        | [Link](https://dartlab.org/assets/pdf/flash.pdf) |
| Kairos     | IEEE S&P 2024        | [Link](https://arxiv.org/pdf/2308.05034) |
| Magic      | USENIX Security 2024 | [Link](https://www.usenix.org/system/files/usenixsecurity24-jia-zian.pdf) |
| NodLink    | NDSS 2024           | [Link](https://arxiv.org/pdf/2311.02331) |
| ThreaTrace | IEEE TIFS 2022      | [Link](https://arxiv.org/pdf/2111.04333) |

### Supported Datasets

It also includes several easy-to-install provenance datasets for APT detection.

| Dataset | OS | Attacks | Size (GB) |
|---------|------|---------|-----------|
| CADETS_E3 | FreeBSD | 3 | 10 |
| THEIA_E3 | Linux | 2 | 12 |
| CLEARSCOPE_E3 | Android | 1 | 4.8 |
| FIVEDIRECTIONS_E3 | Windows | 2 | 22 |
| TRACE_E3 | Linux | 3 | 100 |
| CADETS_E5 | FreeBSD | 2 | 276 |
| THEIA_E5 | Linux | 1 | 36 |
| CLEARSCOPE_E5 | Android | 2 | 49 |
| FIVEDIRECTIONS_E5 | Windows | 4 | 280 |
| TRACE_E5 | Linux | 1 | 710 |
| optc_h201 | Windows | 1 | 9 |
| optc_h501 | Windows | 1 | 6.7 |
| optc_h051 | Windows | 1 | 7.7 |
| ATLASV2_EDR | Windows | 10 | 1 |
| CARBANAKV2_EDR | Windows + Linux | 1 | 6.6 |

## 📄 Documentation

A [comprehensive documentation](https://ubc-provenance.github.io/PIDSMaker/) is available, explaining all possible arguments and providing examples on how integrating new systems.

### Pipeline

The framework integrates a [pipeline](https://ubc-provenance.github.io/PIDSMaker/pipeline) composed of seven stages, each parameterizable via configurable arguments, enabling flexible customization of new systems.

<img src="docs/docs/img/pipeline.svg" style="width: 100%"/>


## 🚀 ProBE (`probe_ml`) Architecture & Pipeline

**ProBE** replaces expensive text-embedding training (Word2Vec/Doc2Vec/FastText) and heavy GNN/TGN message passing with a deterministic, within-window **61-dimensional engineered edge feature extractor** (`pidsmaker/preprocessing/build_graph_methods/edge_feat_engineering.py`) and a fast unsupervised anomaly scoring stage (`pidsmaker/detection/training_methods/ml_training.py`).

### 1. Pipeline Stages
1. **Graph Construction & 61-d Feature Extraction (`construction`)**:
   - Streams provenance events from PostgreSQL in configurable time windows (`time_window_size`).
   - Performs chronological same-operation run fusion per `(src, dst)` pair while retaining pre-fusion timestamps and raw endpoint attributes (`path`, `cmd_line`, `netflow` 5-tuple).
   - Computes the 61 engineered features (`eng`) across all 14 categories (`--eng-edge-feats`) and attaches the `float32` vector to each surviving edge.
2. **Feature Routing (`feat_inference` & `batching`)**:
   - Uses `featurization.used_method: only_type` (skips neural text embeddings).
   - Stores all producer categories once in `feat_inference` (`edge_feat_schema.json`) and slices the active consumer categories into `g.edge_feats` during `batching` (`--eng-edge-feats-categories`), allowing zero-rebuild feature-family ablations.
3. **Unsupervised ML Fitting & Scoring (`training.used_method: ml`)**:
   - Fits a feature normalizer (`asinh_zscore`, `zscore`, `asinh`, or `none`) and a classical anomaly detector (`gmm`, `isolation_forest`, `pca`, `mahalanobis`, `hbos`, `ecod`, `elliptic`, `ocsvm`, `sgd_ocsvm`, `svdd`, `hmm`) on **benign training edges** only.
   - Saves the fitted artifact to `trained_models/probe_ml_model.pkl` and writes per-edge anomaly scores for `val` and `test` splits in standard PIDSMaker loss CSV format.
4. **Node-Level Evaluation (`evaluation`)**:
   - Aggregates edge anomaly scores into node scores and evaluates with robust validation thresholds (`percentile_999_val_loss`, `percentile_99_val_loss`, `mean_std_val_loss`, `max_val_loss`) and fixed false-positive budget metrics (`precision_at_fp_<B>`).

### 2. The 61 Engineered Edge Features (14 Categories)

| Category | Dim | Features |
|---|---:|---|
| `pair` | 5 | `pair_new`, `triplet_new`, `src_new`, `dst_new`, `op_new_for_src` |
| `time` | 5 | `t_norm`, `dt_since_last_pair_norm`, `run_length_log`, `run_duration_norm`, `iat_cv` |
| `structural` | 4 | `src_out_degree_log`, `dst_in_degree_log`, `src_dsttype_diversity_log`, `src_op_diversity_log` |
| `freq` | 4 | `op_freq`, `op_rarity`, `pair_raw_count_log`, `pair_run_count_log` |
| `sequence` | 4 | `pair_run_index_norm`, `pair_num_runs_log`, `is_first_run`, `is_last_run` |
| `semantic` | 4 | `is_cross_type`, `typepair_new`, `typepair_freq`, `typepair_rarity` |
| `entropy` | 2 | `src_op_entropy`, `src_dst_entropy` |
| `ema` | 4 | `ema_pair_intensity_log`, `ema_op_intensity_log`, `ema_dst_inflow_log`, `ema_iat_z` |
| `path` | 5 | `file_in_tmp`, `exec_from_tmp`, `write_then_exec`, `file_in_sensitive`, `file_is_hidden` |
| `netflow` | 5 | `is_outbound`, `remote_is_external`, `listener_makes_outbound`, `outbound_fanout_log`, `remote_port_wellknown` |
| `global_novelty` | 6 | `dst_ip_unseen_global`, `exe_name_unseen_global`, `file_path_unseen_global`, `dir_unseen_global`, `triple_unseen_global`, `src_exe_unseen_does_outbound` |
| `lineage` | 5 | `src_was_written_this_window`, `dst_then_executed`, `proc_spawn_fanout_log`, `parent_makes_outbound`, `chain_depth_norm` |
| `beaconing` | 4 | `dst_ip_portfan_log`, `beacon_regularity`, `single_dst_dominance`, `beacon_period_stable_global` |
| `host_os_profile` | 4 | `win_temp_exe`, `has_ads_or_zoneid`, `lolbin_unusual_child`, `proc_label_deleted_or_so_inject` |
| **Total (`all`)** | **61** | All 14 categories enabled via `--eng-edge-feats --eng-edge-feats-categories=all` |

### 3. Running ProBE (`probe_ml`)

```bash
# Isolation Forest (default)
python pidsmaker/main.py probe_ml CADETS_E3 --cpu \
  --eng-edge-feats --eng-edge-feats-categories=all \
  --training.ml.algorithm=isolation_forest

# Gaussian Mixture Model (GMM)
python pidsmaker/main.py probe_ml CADETS_E3 --cpu \
  --eng-edge-feats --eng-edge-feats-categories=all \
  --training.ml.algorithm=gmm \
  --training.ml.n_components=8 \
  --training.ml.covariance_type=full

# Principal Component Analysis (PCA reconstruction error)
python pidsmaker/main.py probe_ml CADETS_E3 --cpu \
  --eng-edge-feats --eng-edge-feats-categories=all \
  --training.ml.algorithm=pca \
  --training.ml.pca_variance=0.95

# Mahalanobis Distance
python pidsmaker/main.py probe_ml CADETS_E3 --cpu \
  --eng-edge-feats --eng-edge-feats-categories=all \
  --training.ml.algorithm=mahalanobis

# Category ablation (reuse stored 61-d producer artifacts, consume a subset)
python pidsmaker/main.py probe_ml CADETS_E3 --cpu \
  --eng-edge-feats \
  --eng-edge-feats-producer-categories=all \
  --eng-edge-feats-categories=pair,time,structural,ema \
  --training.ml.algorithm=gmm
```

## Setup

### ⬇️ Anonymous Repository (Double-Blind Review)
```
https://anonymous.4open.science/r/PIDSMakerProBE-2778/
```

### 💻 Installation with Docker

We have made the installation of PIDSMaker inclusing pre-processed databases for DARPA TC and OpTC datasets easy and fast. Simply follow [these guidelines](https://ubc-provenance.github.io/PIDSMaker/ten-minute-install/).

## 🧪 Basic usage of the framework

Once you have a followed the installation guidelines, you can open a shell in the `pids container` and experiment in multiple ways.
Replace `SYSTEM` by `probe_ml`, `velox`, `orthrus`, `nodlink`, `threatrace`, `kairos`, `rcaid`, `flash`, `magic`.

1. Run in the shell:
    ```shell
    python pidsmaker/main.py SYSTEM DATASET
    ```

2. Run in the shell, monitored to weights & biases (W&B):
    ```shell
    python pidsmaker/main.py SYSTEM DATASET --wandb
    ```

3. Run in background, monitored to W&B (recommended for multiple parallel runs and for research):
    ```shell
    ./run.sh SYSTEM DATASET
    ```

You can still watch the logs in your shell using `tail -f nohup.out`.

We generally using using W&B for experiment monitoring and historization (see installation guidelines). 

**Warning:** Before performing evaluations, you should tune all systems (see docs [here](https://ubc-provenance.github.io/PIDSMaker/features/tuning/)).

## Reproducing results

PIDSs exhibit significant instability—that is, high sensitivity to training perturbations—due to their self-supervised training nature. 
Running the same configuration with different random seeds or minor hyperparameter changes often yields substantially different results. 
Consequently, reproducing results as the framework evolves presents a real challenge.

Based on our experiments, we provide [tuned hyperparameters](https://ubc-provenance.github.io/PIDSMaker/tuned_systems) for the main systems.
However, we can't guarantee that these hyperparameters will lead to satisfactory results due to instability.

We recommend [running each system multiple times](https://ubc-provenance.github.io/PIDSMaker/features/instability/) to increase the likelihood of obtaining a run with good metrics. Alternatively, you can perform [hyperparameter tuning](https://ubc-provenance.github.io/PIDSMaker/features/tuning/) for each system.

## Customize existing systems

The default configuration files in `config/*.yml` represent the architecture of existing PIDSs in YAML format. They contain the original hyperparameters used by each system. 

The main strength of PIDSMaker is the customization of existing systems for easy experimentation. 
A few examples below.

### From CLI

<i>Running Kairos with embedding size of 128 instead of 100, and last neighbor sampling set to last 10 neighbors instead of 20.</i>

```shell
python pidsmaker/main.py kairos CADETS_E3 \
  --training.node_hid_dim=128 \
  --batching.intra_graph_batching.tgn_last_neighbor.tgn_neighbor_size=10
```

<i>Running Orthrus with Doc2vec instead of word2vec, and 3 GraphSAGE layers instead of 2 attention layers.</i>

```shell
python pidsmaker/main.py orthrus CADETS_E3 \
  --featurization.used_method=doc2vec \
  --featurization.emb_dim=128 \
  --training.encoder.used_methods=tgn,sage \
  --training.encoder.sage.num_layers=3
```

### From a new YAML config file

Want to create a new PIDS? Create a new config under `config/your_system.yml`, inherit from existing PIDSs and tune it as you want.

<i>Magic with node type prediction instead of its hybrid masked feature reconstruction and structure prediction objective function, and use a 2-layer MLP with ReLU as decoder, and use NodLink's thresholding method.</i>

``` yaml
_include_yml: magic

training:
  decoder:
    used_methods: predict_node_type
    predict_node_type:
      node_mlp:
        architecture_str: linear(0.5) | relu

evaluation:
  node_evaluation:
    threshold_method: nodlink
```

### Interactive embedding viewer

PIDSMaker also ships an interactive 3D web viewer for exploring a run's node
embeddings — temporal playback, per-node inspection, attack-graph overlays, and
switching between the featurization and GNN-encoder spaces. Train a run with
`--save_for_viz` to persist the artifacts it needs, then launch it (see the
[web viewer README](pidsmaker/vizgen/web/README.md)):

```bash
python -m pidsmaker.vizgen.web.viz_server
```

![PIDSMaker web embedding viewer](.github/img/web_ui.png)

## Hyperparameter tuning

PIDSMaker supports easy hyperparameter tuning for existing or new models. 
Follow the [instructions](https://ubc-provenance.github.io/PIDSMaker/features/tuning/) available in our documentation.

You can specify the range of hyperparameters to search in a yaml config.

```yaml
method: grid 

parameters:
  training.lr:
    values: [0.001, 0.0001]
  training.node_hid_dim:
    values: [32, 64, 128, 256]
  featurization.used_method:
    values: [fasttext, word2vec]
```

Then run the framework in tuning mode.

```sh
./run.sh my_system CADETS_E3 --tuning_mode=hyperparameters
```

Once you find the best hyperparameters, store them in a yaml file and run your tuned model.

```sh
./run.sh my_system CADETS_E3 --tuned
```

## Citation

If you use this work, please cite the two following papers:
```
@article{bilot2026pidsmaker,
  title={PIDSMaker: Building and Evaluating Provenance-based Intrusion Detection Systems},
  author={Bilot, Tristan and Jiang, Baoxiang and Pasquier, Thomas},
  journal={arXiv preprint arXiv:2601.22983},
  year={2026}
}
@inproceedings{bilot2025simpler,
	title={{Sometimes Simpler is Better: A Comprehensive Analysis of State-of-the-Art Provenance-Based Intrusion Detection Systems}},
	author={Bilot, Tristan and Jiang, Baoxiang and  Li, Zefeng and  El Madhoun, Nour and Al Agha, Khaldoun and Zouaoui, Anis and Pasquier, Thomas},
	booktitle={Security Symposium (USENIX Sec'25)},
	year={2025},
	organization={USENIX}
}
```

## Contributing

Pull requests are welcome! Please follow the [contribution guidelines](https://ubc-provenance.github.io/PIDSMaker/contributing/).

## License

See [licence](LICENSE).
