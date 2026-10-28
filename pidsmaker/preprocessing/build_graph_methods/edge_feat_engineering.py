"""Feature-engineered edge features (v2).

Pure, dependency-light module (numpy + stdlib) so it can be unit-tested in
isolation from torch / networkx / the database.

Design contract (see FEAT_ENG_PLAN.md, FEAT_ENG_BRAINSTORM.md):
  - Features are computed in `construction`, over ALL pre-fusion events of a
    time window, but attached ONLY to the surviving (post-fusion) edges.
  - Features are grouped into CATEGORIES (`CATEGORY_SCHEMA`). `compute()` always
    DERIVES every category, but EMITS only the columns of the SELECTED
    categories, in canonical category order. This is the "derive all, feed only
    chosen ones" contract: which categories reach the model is a config choice
    (`construction.eng_edge_feats_categories`), defaulting to all.
  - `compute()` returns a row-aligned `np.ndarray [n, active_dim]` (float32),
    one row per surviving edge in the order given.

Resolved design decisions (FEAT_ENG_BRAINSTORM.md §8):
  - Count features are `log1p`-compressed for window-size robustness.
  - The engineered block is ALL-NUMERIC (no raw categorical ids): sequence
    information is exposed as numeric summaries.
  - EMA features use a time-aware decay with `tau = window_span * ema_tau_frac`
    and reset per window (honours the within-window constraint C3).

Adding a feature later: append its name to the relevant list in CATEGORY_SCHEMA
and add its computation in the matching `_cat_*` function. Adding a whole new
category: add an entry to CATEGORY_SCHEMA + its `_cat_*` function + a dispatch
line in `_compute_all`. No caller signature changes are required.
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownLambdaType=false, reportExplicitAny=false, reportAny=false
import math
import re
from bisect import bisect_left
from collections import OrderedDict, defaultdict
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Schema: ordered categories -> ordered feature names. The width of the         #
# engineered block for a given selection is the sum of the selected categories' #
# feature counts. Canonical order is the iteration order of this dict.          #
# --------------------------------------------------------------------------- #
CATEGORY_SCHEMA: OrderedDict[str, list[str]] = OrderedDict(
    [
        # A. Pair-based / novelty — is this (src,dst[,op]) relationship new?
        (
            "pair",
            ["pair_new", "triplet_new", "src_new", "dst_new", "op_new_for_src"],
        ),
        # B. Time-based / recency / burstiness — when, and how bursty?
        (
            "time",
            ["t_norm", "dt_since_last_pair_norm", "run_length_log", "run_duration_norm", "iat_cv"],
        ),
        # C. Structural / topological — local fan-out / fan-in / diversity.
        (
            "structural",
            [
                "src_out_degree_log",
                "dst_in_degree_log",
                "src_dsttype_diversity_log",
                "src_op_diversity_log",
            ],
        ),
        # D. Frequency / volume / rarity — how common / heavy is this op/pair?
        (
            "freq",
            ["op_freq", "op_rarity", "pair_raw_count_log", "pair_run_count_log"],
        ),
        # E. Sequence / transition — position of this run in the pair's chain.
        (
            "sequence",
            ["pair_run_index_norm", "pair_num_runs_log", "is_first_run", "is_last_run"],
        ),
        # F. Semantic / type-flow priors — distribution over (src_type,dst_type).
        (
            "semantic",
            ["is_cross_type", "typepair_new", "typepair_freq", "typepair_rarity"],
        ),
        # G. Entropy / diversity — dispersion of the source's behaviour.
        (
            "entropy",
            ["src_op_entropy", "src_dst_entropy"],
        ),
        # H. EMA / time-decayed recency — decayed intensity + beaconing surprise.
        (
            "ema",
            ["ema_pair_intensity_log", "ema_op_intensity_log", "ema_dst_inflow_log", "ema_iat_z"],
        ),
        # --- V2: CADETS_E3-targeted, label-derived (see FEAT_ENG_BRAINSTORM / -----
        #     CLAUDE_FEATS_ENG §3b). Use RAW node attrs (path / netflow 5-tuple),
        #     not the display label. Degrade to 0 when attrs are absent.
        # J. Path / filesystem semantics — payload drop & execute-from-/tmp.
        (
            "path",
            [
                "file_in_tmp",
                "exec_from_tmp",
                "write_then_exec",
                "file_in_sensitive",
                "file_is_hidden",
            ],
        ),
        # K. Netflow direction / external-ness — outbound C2 / beaconing fan-out.
        (
            "netflow",
            [
                "is_outbound",
                "remote_is_external",
                "listener_makes_outbound",
                "outbound_fanout_log",
                "remote_port_wellknown",
            ],
        ),
        # --- V3: targeted at the weak datasets (optc_501 / theia_e3 / cadets_e3).
        #     See 03_FEATURES_ENGINEERING.md Part B. M/N/O are WITHIN-WINDOW (no new
        #     state); L (global_novelty) and N.`beacon_period_stable_global` need a
        #     training-scope prior store, so they degrade to 0 when no prior is given.
        # L. Global novelty — "never seen in benign training" identity priors.
        (
            "global_novelty",
            [
                "dst_ip_unseen_global",
                "exe_name_unseen_global",
                "file_path_unseen_global",
                "dir_unseen_global",
                "triple_unseen_global",
                "src_exe_unseen_does_outbound",
            ],
        ),
        # M. Lineage — kill-chain / parent-child structure (within-window).
        (
            "lineage",
            [
                "src_was_written_this_window",
                "dst_then_executed",
                "proc_spawn_fanout_log",
                "parent_makes_outbound",
                "chain_depth_norm",
            ],
        ),
        # N. Beaconing — periodic C2 to a single remote endpoint.
        (
            "beaconing",
            [
                "dst_ip_portfan_log",
                "beacon_regularity",
                "single_dst_dominance",
                "beacon_period_stable_global",
            ],
        ),
        # O. Host/OS profile — platform-aware TTP flags (Windows/Linux/Android).
        (
            "host_os_profile",
            [
                "win_temp_exe",
                "has_ads_or_zoneid",
                "lolbin_unusual_child",
                "proc_label_deleted_or_so_inject",
            ],
        ),
    ]
)

# --- V2 lexicons (module constants; TODO make config-driven). Op names follow
#     DARPA TC (rel2id_darpa_tc); a few non-DARPA synonyms are included defensively.
EXEC_OPS = {"EVENT_EXECUTE", "EXECUTE", "EVENT_MMAP", "EVENT_LOADLIBRARY", "EVENT_MPROTECT"}
WRITE_OPS = {"EVENT_WRITE", "WRITE", "ACTION_FILE_WRITE", "MODIFY"}
# Process spawn ops (for lineage spawn-fanout). DARPA TC: EVENT_CLONE/EXECUTE;
# OpTC: START/CREATE; atlas: ACTION_CREATE_PROCESS. EXEC_OPS are also spawns.
SPAWN_OPS = {
    "EVENT_CLONE",
    "EVENT_FORK",
    "CLONE",
    "FORK",
    "START",
    "CREATE",
    "ACTION_CREATE_PROCESS",
}
TMP_DIRS = ("/tmp/", "/var/tmp/", "/dev/shm/", "/private/tmp/", "/data/local/tmp")
SENSITIVE_DIRS = ("/etc/", "/var/log/", "/var/spool/", "/root/", "/boot/", "/.ssh/", "/usr/home/")
# Remote ports that indicate a service the host is dialing OUT to (C2/download).
SERVICE_PORTS = {80, 443, 8080, 8000, 8443, 53, 21, 22, 25, 3128, 1080}
EPHEMERAL_MIN = 1024  # ports >= this (and not a known service) are treated as client-side

# --- V3 lexicons (host_os_profile). Plain substring/lexicon checks on raw paths
#     and labels; all degrade to 0 when the pattern/OS is absent.
# Windows temp-exec dropper dirs (case-insensitive match on the path).
WIN_TEMP_MARKERS = (
    "\\windows\\temp\\",
    "/windows/temp/",
    "\\appdata\\local\\temp\\",
    "/appdata/local/temp/",
)
# NTFS alternate-data-stream / mark-of-the-web markers.
ADS_MARKERS = (":zone.identifier", "::$data")
# Living-off-the-land binaries commonly abused as malicious parents (Windows).
LOLBINS = {
    "lsass.exe",
    "svchost.exe",
    "cmd.exe",
    "powershell.exe",
    "gpupdate.exe",
    "rundll32.exe",
    "cscript.exe",
    "wscript.exe",
    "schtasks.exe",
    "mshta.exe",
    "regsvr32.exe",
    "sc.exe",
    "conhost.exe",
}
# Sensitive individual files (credential / data-exfil) across OSes.
SENSITIVE_FILES = (
    "/etc/passwd",
    "/etc/shadow",
    "passwordfile.dat",
    "pass_mgr",
    "calllog.db",
    "external.db",
    "/.ssh/",
)
# Volatile path tokens stripped before any GLOBAL identity lookup so the random
# component of a path doesn't defeat the "seen in training" check.
_VOLATILE_DIR_RE = None  # compiled lazily in _normalize_for_prior

ALL_CATEGORIES: list[str] = list(CATEGORY_SCHEMA.keys())

# Flattened, fully-ordered list of every engineered feature across all
# categories. Kept for tooling/debugging; the model input uses the SELECTED
# subset via `feature_names_from_categories`.
FEATURE_SCHEMA: list[str] = [name for cat in ALL_CATEGORIES for name in CATEGORY_SCHEMA[cat]]

DEFAULT_EMA_TAU_FRAC = 0.25
_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Selection helpers (single source of truth for the engineered block width).    #
# --------------------------------------------------------------------------- #
def parse_categories(raw: Any) -> list[str]:
    """Normalise a category selection into an ordered list of valid tokens.

    Accepts None or "all" (-> every category), a comma-separated string, or an
    iterable of tokens. None means "unset -> default to all"; an explicit EMPTY
    selection ([] or "") means "no categories" (so the disabled path can report
    zero width). Unknown tokens are dropped; the result is always in canonical
    CATEGORY_SCHEMA order so producer and consumers agree on layout.
    """
    if raw is None:
        return list(ALL_CATEGORIES)
    if isinstance(raw, str):
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
    else:
        tokens = [str(t).strip() for t in raw if str(t).strip()]
    if "all" in tokens:
        return list(ALL_CATEGORIES)
    selected = set(tokens)
    return [cat for cat in ALL_CATEGORIES if cat in selected]


def feature_names_from_categories(categories: Any) -> list[str]:
    """Flatten the selected categories into their ordered feature-name list."""
    cats = categories if isinstance(categories, list) else parse_categories(categories)
    return [name for cat in cats if cat in CATEGORY_SCHEMA for name in CATEGORY_SCHEMA[cat]]


def get_producer_categories(cfg: Any) -> list[str]:
    """Categories stored during construction/feat-inference."""
    if not bool(cfg.construction.compute_eng_edge_feats):
        return []
    raw = getattr(cfg.construction, "eng_edge_feats_producer_categories", None)
    if raw is None:
        # Backward-compatible default: store everything once; consumer slices later.
        raw = "all"
    return parse_categories(raw)


def get_consumer_categories(cfg: Any) -> list[str]:
    """Categories exposed to the model as g.edge_feats."""
    if not bool(cfg.construction.compute_eng_edge_feats):
        return []
    raw = getattr(cfg.batching, "eng_edge_feats_consumer_categories", None)
    if raw is None:
        raw = getattr(cfg.construction, "eng_edge_feats_categories", None)
    return parse_categories(raw)


def get_active_categories(cfg: Any) -> list[str]:
    """Backward-compatible alias for model-consumed engineered categories."""
    return get_consumer_categories(cfg)


def get_producer_feature_names(cfg: Any) -> list[str]:
    """Ordered engineered columns stored in graph/msg artifacts."""
    return feature_names_from_categories(get_producer_categories(cfg))


def get_active_feature_names(cfg: Any) -> list[str]:
    """Ordered names of the engineered columns actually fed to the model."""
    return feature_names_from_categories(get_consumer_categories(cfg))


def get_producer_dim(cfg: Any) -> int:
    """Width of the engineered block stored in graph/msg artifacts."""
    return len(get_producer_feature_names(cfg))


def get_active_dim(cfg: Any) -> int:
    """Width of the engineered block fed to the model (0 when disabled)."""
    return len(get_active_feature_names(cfg))


def feature_indices_from_categories(source_categories: Any, selected_categories: Any) -> list[int]:
    """Column indices selecting `selected_categories` from a source category layout."""
    source = parse_categories(source_categories)
    selected = set(parse_categories(selected_categories))
    idx = []
    offset = 0
    for cat in source:
        width = len(CATEGORY_SCHEMA[cat])
        if cat in selected:
            idx.extend(range(offset, offset + width))
        offset += width
    return idx


def get_consumer_indices_from_producer(cfg: Any) -> list[int]:
    """Indices that slice producer eng vectors down to consumer-selected vectors."""
    return feature_indices_from_categories(
        get_producer_categories(cfg), get_consumer_categories(cfg)
    )


def get_ema_tau_frac(cfg: Any) -> float:
    """EMA decay horizon as a fraction of the window span (default 0.25)."""
    val = getattr(cfg.construction, "eng_edge_feats_ema_tau_frac", None)
    return float(val) if val is not None else DEFAULT_EMA_TAU_FRAC


# --------------------------------------------------------------------------- #
# Public entry point.                                                           #
# --------------------------------------------------------------------------- #
def compute(
    surviving_edges: list[dict[str, Any]],
    raw_events: list[tuple[Any, Any, Any, Any]] | None = None,
    window_start_ns: int | None = None,
    window_end_ns: int | None = None,
    categories: Any = None,
    ema_tau_frac: float = DEFAULT_EMA_TAU_FRAC,
    prior: dict[str, Any] | None = None,
):
    """Compute engineered features for the surviving edges of one time window.

    Args:
        surviving_edges: list of dicts in emission order (chronological by
            ``t_last`` with a stable tiebreaker). Recognised keys per edge:
              - ``src``, ``dst``     : node ids (required)
              - ``time``             : t_last, the run's LAST absorbed event (required)
              - ``label``            : the operation/edge type (for triplet/op feats)
              - ``abs_ts``           : list of the run's absorbed raw timestamps
                                       (for run-length / inter-arrival feats);
                                       falls back to ``[time]`` if absent
              - ``src_type``, ``dst_type`` : node types (for semantic feats);
                                       semantic feats are 0 when types are absent
        raw_events: full pre-fusion event stream of the window as tuples
            ``(src, dst, op, timestamp)``. Used by recency / EMA features.
        window_start_ns / window_end_ns: window bounds (ns) used to normalise
            times and to size the EMA decay horizon.
        categories: which categories to EMIT (None / "all" -> every category).
            All categories are still DERIVED; only selected ones are returned.
        ema_tau_frac: EMA decay horizon as a fraction of the window span.
        prior: optional training-scope prior store for the ``global_novelty``
            category and ``beacon_period_stable_global``. Recognised keys (all
            optional; missing ones make the corresponding feature degrade to 0):
              - ``ips``        : set of remote IPs seen in benign training
              - ``exe_names``  : set of executable basenames seen in training
              - ``file_paths`` : set of (normalised) file paths seen in training
              - ``dirs``       : set of file directories seen in training
              - ``triples``    : set of (src_type, op, dst_type) seen in training
              - ``beacon_periods`` : {(proc_label?, remote_ip): median_period_ns}
            When ``prior`` is None the whole ``global_novelty`` category and
            ``beacon_period_stable_global`` emit 0 (so the within-window M/N/O
            categories remain usable with no prior store).

    Returns:
        np.ndarray of shape [len(surviving_edges), active_dim], dtype float32,
        row-aligned to ``surviving_edges``; columns ordered by
        ``feature_names_from_categories(categories)``.
    """
    selected = parse_categories(categories)
    out_names = [
        name for cat in selected if cat in CATEGORY_SCHEMA for name in CATEGORY_SCHEMA[cat]
    ]

    n = len(surviving_edges)
    if n == 0:
        return np.zeros((0, len(out_names)), dtype=np.float32)

    columns = _compute_all(
        surviving_edges, raw_events, window_start_ns, window_end_ns, ema_tau_frac, prior
    )
    if not out_names:
        return np.zeros((n, 0), dtype=np.float32)
    return np.column_stack([columns[name] for name in out_names]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Internal: derive every requested category into {feature_name: np.ndarray[n]}. #
# --------------------------------------------------------------------------- #
def _span(surviving_edges, window_start_ns, window_end_ns):
    if window_start_ns is not None and window_end_ns is not None:
        return max(int(window_end_ns) - int(window_start_ns), 1), window_start_ns
    times = [int(e["time"]) for e in surviving_edges]
    start = min(times) if times else 0
    return max((max(times) if times else 0) - start, 1), start


def _compute_all(
    surviving_edges, raw_events, window_start_ns, window_end_ns, ema_tau_frac, prior=None
):
    n = len(surviving_edges)
    span, start = _span(surviving_edges, window_start_ns, window_end_ns)

    srcs = [e["src"] for e in surviving_edges]
    dsts = [e["dst"] for e in surviving_edges]
    ops = [e.get("label") for e in surviving_edges]
    times = [int(e["time"]) for e in surviving_edges]
    abs_ts = [list(e.get("abs_ts") or [e["time"]]) for e in surviving_edges]

    cols = {}
    # Each category is independent; we derive all of them (cheap, one window) so
    # the selection is purely about what we EMIT, not what we compute.
    cols.update(_cat_pair(n, srcs, dsts, ops))
    cols.update(_cat_time(n, srcs, dsts, times, abs_ts, span, start, raw_events))
    cols.update(_cat_structural(n, surviving_edges, srcs, dsts, ops))
    cols.update(_cat_freq(n, srcs, dsts, ops, abs_ts))
    cols.update(_cat_sequence(n, srcs, dsts))
    cols.update(_cat_semantic(n, surviving_edges))
    cols.update(_cat_entropy(n, srcs, dsts, ops))
    cols.update(_cat_ema(n, srcs, dsts, ops, times, span, raw_events, ema_tau_frac))
    cols.update(_cat_path(n, surviving_edges, ops))
    cols.update(_cat_netflow(n, surviving_edges))
    # --- V3 categories (M/N/O within-window; L needs the prior, degrades to 0).
    cols.update(_cat_global_novelty(n, surviving_edges, ops, prior))
    cols.update(_cat_lineage(n, surviving_edges, srcs, dsts, ops, span, start))
    cols.update(_cat_beaconing(n, surviving_edges, times, prior))
    cols.update(_cat_host_os_profile(n, surviving_edges, ops))
    return cols


def _zeros(n):
    return np.zeros(n, dtype=np.float64)


# --- A. Pair-based / novelty ------------------------------------------------ #
def _cat_pair(n, srcs, dsts, ops):
    pair_new = _zeros(n)
    triplet_new = _zeros(n)
    src_new = _zeros(n)
    dst_new = _zeros(n)
    op_new_for_src = _zeros(n)

    seen_pair, seen_triplet, seen_src, seen_dst, seen_src_op = set(), set(), set(), set(), set()
    for i in range(n):
        p, t = (srcs[i], dsts[i]), (srcs[i], dsts[i], ops[i])
        so = (srcs[i], ops[i])
        if p not in seen_pair:
            pair_new[i] = 1.0
            seen_pair.add(p)
        if t not in seen_triplet:
            triplet_new[i] = 1.0
            seen_triplet.add(t)
        if srcs[i] not in seen_src:
            src_new[i] = 1.0
            seen_src.add(srcs[i])
        if dsts[i] not in seen_dst:
            dst_new[i] = 1.0
            seen_dst.add(dsts[i])
        if so not in seen_src_op:
            op_new_for_src[i] = 1.0
            seen_src_op.add(so)
    return {
        "pair_new": pair_new,
        "triplet_new": triplet_new,
        "src_new": src_new,
        "dst_new": dst_new,
        "op_new_for_src": op_new_for_src,
    }


# --- B. Time-based / recency / burstiness ----------------------------------- #
def _cat_time(n, srcs, dsts, times, abs_ts, span, start, raw_events):
    t_norm = _zeros(n)
    dt_since_last_pair_norm = _zeros(n)
    run_length_log = _zeros(n)
    run_duration_norm = _zeros(n)
    iat_cv = _zeros(n)

    # Per-pair sorted raw timestamps, for "previous raw event of this pair".
    pair_raw_times = defaultdict(list)
    if raw_events:
        for s, d, _op, ts in raw_events:
            pair_raw_times[(s, d)].append(int(ts))
        for k in pair_raw_times:
            pair_raw_times[k].sort()

    for i in range(n):
        t_norm[i] = (times[i] - start) / span

        ts_list = abs_ts[i]
        run_length_log[i] = math.log1p(len(ts_list))
        run_duration_norm[i] = (ts_list[-1] - ts_list[0]) / span if len(ts_list) > 1 else 0.0
        if len(ts_list) > 2:
            gaps = np.diff(np.sort(np.asarray(ts_list, dtype=np.float64)))
            mean = gaps.mean()
            iat_cv[i] = (gaps.std() / mean) if mean > _EPS else 0.0

        raw_t = pair_raw_times.get((srcs[i], dsts[i]))
        if raw_t:
            pos = bisect_left(raw_t, times[i])
            if pos > 0:
                dt_since_last_pair_norm[i] = min((times[i] - raw_t[pos - 1]) / span, 1.0)
            else:
                dt_since_last_pair_norm[i] = 1.0  # first event of the pair (novel)
        else:
            dt_since_last_pair_norm[i] = 1.0
    return {
        "t_norm": t_norm,
        "dt_since_last_pair_norm": dt_since_last_pair_norm,
        "run_length_log": run_length_log,
        "run_duration_norm": run_duration_norm,
        "iat_cv": iat_cv,
    }


# --- C. Structural / topological -------------------------------------------- #
def _cat_structural(n, surviving_edges, srcs, dsts, ops):
    src_out, dst_in = defaultdict(set), defaultdict(set)
    src_dsttypes, src_ops = defaultdict(set), defaultdict(set)
    for i in range(n):
        src_out[srcs[i]].add(dsts[i])
        dst_in[dsts[i]].add(srcs[i])
        src_ops[srcs[i]].add(ops[i])
        dt = surviving_edges[i].get("dst_type")
        src_dsttypes[srcs[i]].add(dt if dt is not None else dsts[i])

    src_out_degree_log = _zeros(n)
    dst_in_degree_log = _zeros(n)
    src_dsttype_diversity_log = _zeros(n)
    src_op_diversity_log = _zeros(n)
    for i in range(n):
        src_out_degree_log[i] = math.log1p(len(src_out[srcs[i]]))
        dst_in_degree_log[i] = math.log1p(len(dst_in[dsts[i]]))
        src_dsttype_diversity_log[i] = math.log1p(len(src_dsttypes[srcs[i]]))
        src_op_diversity_log[i] = math.log1p(len(src_ops[srcs[i]]))
    return {
        "src_out_degree_log": src_out_degree_log,
        "dst_in_degree_log": dst_in_degree_log,
        "src_dsttype_diversity_log": src_dsttype_diversity_log,
        "src_op_diversity_log": src_op_diversity_log,
    }


# --- D. Frequency / volume / rarity ----------------------------------------- #
def _cat_freq(n, srcs, dsts, ops, abs_ts):
    op_counts = defaultdict(int)
    pair_raw_count = defaultdict(int)
    pair_run_count = defaultdict(int)
    for i in range(n):
        op_counts[ops[i]] += 1
        pair_raw_count[(srcs[i], dsts[i])] += len(abs_ts[i])
        pair_run_count[(srcs[i], dsts[i])] += 1

    op_freq = _zeros(n)
    op_rarity = _zeros(n)
    pair_raw_count_log = _zeros(n)
    pair_run_count_log = _zeros(n)
    for i in range(n):
        f = op_counts[ops[i]] / n
        op_freq[i] = f
        op_rarity[i] = -math.log(f + _EPS)
        pair_raw_count_log[i] = math.log1p(pair_raw_count[(srcs[i], dsts[i])])
        pair_run_count_log[i] = math.log1p(pair_run_count[(srcs[i], dsts[i])])
    return {
        "op_freq": op_freq,
        "op_rarity": op_rarity,
        "pair_raw_count_log": pair_raw_count_log,
        "pair_run_count_log": pair_run_count_log,
    }


# --- E. Sequence / transition ----------------------------------------------- #
def _cat_sequence(n, srcs, dsts):
    # Surviving edges arrive in emission order; group indices per pair to get
    # each run's ordinal position within its pair's chain of runs.
    pair_indices = defaultdict(list)
    for i in range(n):
        pair_indices[(srcs[i], dsts[i])].append(i)

    pair_run_index_norm = _zeros(n)
    pair_num_runs_log = _zeros(n)
    is_first_run = _zeros(n)
    is_last_run = _zeros(n)
    for idxs in pair_indices.values():
        m = len(idxs)
        for rank, i in enumerate(idxs):
            pair_run_index_norm[i] = rank / (m - 1) if m > 1 else 0.0
            pair_num_runs_log[i] = math.log1p(m)
            is_first_run[i] = 1.0 if rank == 0 else 0.0
            is_last_run[i] = 1.0 if rank == m - 1 else 0.0
    return {
        "pair_run_index_norm": pair_run_index_norm,
        "pair_num_runs_log": pair_num_runs_log,
        "is_first_run": is_first_run,
        "is_last_run": is_last_run,
    }


# --- F. Semantic / type-flow priors ----------------------------------------- #
def _cat_semantic(n, surviving_edges):
    src_types = [e.get("src_type") for e in surviving_edges]
    dst_types = [e.get("dst_type") for e in surviving_edges]
    has_types = all(st is not None and dt is not None for st, dt in zip(src_types, dst_types))

    is_cross_type = _zeros(n)
    typepair_new = _zeros(n)
    typepair_freq = _zeros(n)
    typepair_rarity = _zeros(n)
    if has_types:
        tp_counts = defaultdict(int)
        for st, dt in zip(src_types, dst_types):
            tp_counts[(st, dt)] += 1
        seen_tp = set()
        for i in range(n):
            tp = (src_types[i], dst_types[i])
            is_cross_type[i] = 1.0 if src_types[i] != dst_types[i] else 0.0
            if tp not in seen_tp:
                typepair_new[i] = 1.0
                seen_tp.add(tp)
            f = tp_counts[tp] / n
            typepair_freq[i] = f
            typepair_rarity[i] = -math.log(f + _EPS)
    return {
        "is_cross_type": is_cross_type,
        "typepair_new": typepair_new,
        "typepair_freq": typepair_freq,
        "typepair_rarity": typepair_rarity,
    }


# --- G. Entropy / diversity ------------------------------------------------- #
def _entropy(counter):
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        p = c / total
        if p > 0:
            h -= p * math.log(p)
    return h


def _cat_entropy(n, srcs, dsts, ops):
    src_op_counts = defaultdict(lambda: defaultdict(int))
    src_dst_counts = defaultdict(lambda: defaultdict(int))
    for i in range(n):
        src_op_counts[srcs[i]][ops[i]] += 1
        src_dst_counts[srcs[i]][dsts[i]] += 1

    src_op_entropy = _zeros(n)
    src_dst_entropy = _zeros(n)
    op_cache, dst_cache = {}, {}
    for i in range(n):
        s = srcs[i]
        if s not in op_cache:
            op_cache[s] = _entropy(src_op_counts[s])
            dst_cache[s] = _entropy(src_dst_counts[s])
        src_op_entropy[i] = op_cache[s]
        src_dst_entropy[i] = dst_cache[s]
    return {"src_op_entropy": src_op_entropy, "src_dst_entropy": src_dst_entropy}


# --- H. EMA / time-decayed recency ------------------------------------------ #
def _decayed_counts(events_by_key, tau):
    """Time-aware decaying counter per key: lambda_k = 1 + exp(-dt/tau)*lambda_{k-1}.

    Returns {(key, timestamp): lambda_after_processing_that_timestamp}.
    """
    out = {}
    for key, ts_list in events_by_key.items():
        acc, prev_t = 0.0, None
        for t in ts_list:  # already sorted
            acc = 1.0 if prev_t is None else 1.0 + math.exp(-(t - prev_t) / tau) * acc
            prev_t = t
            out[(key, t)] = acc
    return out


def _iat_zscores(times_by_pair, tau):
    """Time-aware EMA mean/var of inter-arrival gaps -> standardised surprise z.

    z_k = (gap_k - mu_{k-1}) / sqrt(var_{k-1} + eps), with EMA weight
    w = 1 - exp(-gap/tau). Returns {(pair, timestamp): z}.
    """
    out = {}
    for key, ts_list in times_by_pair.items():
        mu, var, prev_t, have_baseline = 0.0, 0.0, None, False
        for t in ts_list:
            if prev_t is None:
                z = 0.0
            else:
                gap = t - prev_t
                if not have_baseline:
                    z, mu, var, have_baseline = 0.0, float(gap), 0.0, True
                else:
                    z = (gap - mu) / math.sqrt(var + _EPS)
                    w = 1.0 - math.exp(-gap / tau)
                    new_mu = (1.0 - w) * mu + w * gap
                    var = (1.0 - w) * var + w * (gap - new_mu) ** 2
                    mu = new_mu
            prev_t = t
            out[(key, t)] = z
    return out


def _cat_ema(n, srcs, dsts, ops, times, span, raw_events, ema_tau_frac):
    ema_pair_intensity_log = _zeros(n)
    ema_op_intensity_log = _zeros(n)
    ema_dst_inflow_log = _zeros(n)
    ema_iat_z = _zeros(n)

    if not raw_events:
        return {
            "ema_pair_intensity_log": ema_pair_intensity_log,
            "ema_op_intensity_log": ema_op_intensity_log,
            "ema_dst_inflow_log": ema_dst_inflow_log,
            "ema_iat_z": ema_iat_z,
        }

    tau = max(span * ema_tau_frac, 1.0)
    pair_times = defaultdict(list)
    srcop_times = defaultdict(list)
    dst_times = defaultdict(list)
    for s, d, op, ts in sorted(raw_events, key=lambda r: int(r[3])):
        ts = int(ts)
        pair_times[(s, d)].append(ts)
        srcop_times[(s, op)].append(ts)
        dst_times[d].append(ts)

    pair_lambda = _decayed_counts(pair_times, tau)
    srcop_lambda = _decayed_counts(srcop_times, tau)
    dst_lambda = _decayed_counts(dst_times, tau)
    pair_z = _iat_zscores(pair_times, tau)

    for i in range(n):
        p, so, d, t = (srcs[i], dsts[i]), (srcs[i], ops[i]), dsts[i], times[i]
        ema_pair_intensity_log[i] = math.log1p(pair_lambda.get((p, t), 0.0))
        ema_op_intensity_log[i] = math.log1p(srcop_lambda.get((so, t), 0.0))
        ema_dst_inflow_log[i] = math.log1p(dst_lambda.get((d, t), 0.0))
        ema_iat_z[i] = pair_z.get((p, t), 0.0)
    return {
        "ema_pair_intensity_log": ema_pair_intensity_log,
        "ema_op_intensity_log": ema_op_intensity_log,
        "ema_dst_inflow_log": ema_dst_inflow_log,
        "ema_iat_z": ema_iat_z,
    }


# --- V2 helpers ------------------------------------------------------------- #
def _clean_str(v):
    """Normalise a raw attr to a usable string, or None for empty / 'None'."""
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "none":
        return None
    return s


def _file_path(edge):
    """Path of the FILE endpoint of an edge (dst preferred, else src)."""
    if edge.get("dst_type") == "file":
        return _clean_str(edge.get("dst_path"))
    if edge.get("src_type") == "file":
        return _clean_str(edge.get("src_path"))
    # Fallback: any path-bearing dst then src (covers subjects with exe paths).
    return _clean_str(edge.get("dst_path")) or _clean_str(edge.get("src_path"))


def _starts_with_any(path, prefixes):
    return any(path.startswith(p) for p in prefixes)


def _to_port(v):
    s = _clean_str(v)
    if s is None:
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _is_service_port(p):
    return p is not None and (p in SERVICE_PORTS or p < EPHEMERAL_MIN)


def _is_ephemeral_port(p):
    return p is not None and p >= EPHEMERAL_MIN and p not in SERVICE_PORTS


def _is_private_ip(ip):
    """RFC1918 / loopback / link-local / unspecified (host-agnostic)."""
    if ip is None:
        return True  # absent -> not "external"
    parts = ip.split(".")
    if len(parts) != 4:
        return True  # not a parseable IPv4 (e.g. hostname) -> don't call it external
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError:
        return True
    if a == 10 or a == 127 or a == 0:
        return True
    if a == 192 and b == 168:
        return True
    if a == 172 and 16 <= b <= 31:
        return True
    if a == 169 and b == 254:
        return True
    return False


# --- J. Path / filesystem semantics ----------------------------------------- #
def _cat_path(n, surviving_edges, ops):
    file_in_tmp = _zeros(n)
    exec_from_tmp = _zeros(n)
    write_then_exec = _zeros(n)
    file_in_sensitive = _zeros(n)
    file_is_hidden = _zeros(n)

    paths = [_file_path(e) for e in surviving_edges]
    # Files written somewhere in this window (for the drop->execute fingerprint).
    written = {paths[i] for i in range(n) if paths[i] and ops[i] in WRITE_OPS}

    for i in range(n):
        p = paths[i]
        if not p:
            continue
        in_tmp = _starts_with_any(p, TMP_DIRS)
        file_in_tmp[i] = 1.0 if in_tmp else 0.0
        file_in_sensitive[i] = 1.0 if _starts_with_any(p, SENSITIVE_DIRS) else 0.0
        file_is_hidden[i] = 1.0 if p.rsplit("/", 1)[-1].startswith(".") else 0.0
        is_exec = ops[i] in EXEC_OPS
        exec_from_tmp[i] = 1.0 if (is_exec and in_tmp) else 0.0
        write_then_exec[i] = 1.0 if (is_exec and p in written) else 0.0
    return {
        "file_in_tmp": file_in_tmp,
        "exec_from_tmp": exec_from_tmp,
        "write_then_exec": write_then_exec,
        "file_in_sensitive": file_in_sensitive,
        "file_is_hidden": file_is_hidden,
    }


# --- K. Netflow direction / external-ness ----------------------------------- #
def _proc_endpoint(edge):
    """The PROCESS (subject) node id of a netflow edge, whichever side it is on.

    Outbound flows are process->netflow (process=src); inbound flows are
    netflow->process (process=dst). Aggregates must key on the process, not the
    edge src, or the "listener makes outbound" flip is split across both sides.
    """
    if edge.get("src_type") == "subject":
        return edge.get("src")
    if edge.get("dst_type") == "subject":
        return edge.get("dst")
    return None


def _cat_netflow(n, surviving_edges):
    is_outbound = _zeros(n)
    remote_is_external = _zeros(n)
    listener_makes_outbound = _zeros(n)
    outbound_fanout_log = _zeros(n)
    remote_port_wellknown = _zeros(n)

    # Per-edge netflow decode. Direction is intrinsic to the netflow 5-tuple
    # (local vs remote port), independent of the edge's orientation.
    has_nf = [False] * n
    outbound = [False] * n
    inbound_service = [False] * n
    external = [False] * n
    remote_ip = [None] * n
    proc = [None] * n
    for i, e in enumerate(surviving_edges):
        if (
            e.get("nf_remote_ip") is None
            and e.get("nf_remote_port") is None
            and e.get("nf_local_port") is None
        ):
            continue
        r_ip = _clean_str(e.get("nf_remote_ip"))
        r_port = _to_port(e.get("nf_remote_port"))
        l_port = _to_port(e.get("nf_local_port"))
        has_nf[i] = True
        remote_ip[i] = r_ip
        proc[i] = _proc_endpoint(e)
        external[i] = not _is_private_ip(r_ip)
        # Outbound (host as client): local ephemeral, remote a service.
        outbound[i] = _is_ephemeral_port(l_port) and _is_service_port(r_port)
        # Inbound service (host as server): local well-known, remote ephemeral.
        inbound_service[i] = _is_service_port(l_port) and _is_ephemeral_port(r_port)

        is_outbound[i] = 1.0 if outbound[i] else 0.0
        remote_is_external[i] = 1.0 if external[i] else 0.0
        remote_port_wellknown[i] = 1.0 if _is_service_port(r_port) else 0.0

    # Per-process aggregates: the "listener acting as client" flip + fan-out.
    proc_has_inbound = defaultdict(bool)
    proc_has_outbound = defaultdict(bool)
    proc_ext_dests = defaultdict(set)
    for i in range(n):
        if not has_nf[i] or proc[i] is None:
            continue
        if inbound_service[i]:
            proc_has_inbound[proc[i]] = True
        if outbound[i]:
            proc_has_outbound[proc[i]] = True
            if external[i] and remote_ip[i] is not None:
                proc_ext_dests[proc[i]].add(remote_ip[i])

    for i in range(n):
        if not has_nf[i] or proc[i] is None:
            continue
        if proc_has_inbound[proc[i]] and proc_has_outbound[proc[i]]:
            listener_makes_outbound[i] = 1.0
        outbound_fanout_log[i] = math.log1p(len(proc_ext_dests[proc[i]]))
    return {
        "is_outbound": is_outbound,
        "remote_is_external": remote_is_external,
        "listener_makes_outbound": listener_makes_outbound,
        "outbound_fanout_log": outbound_fanout_log,
        "remote_port_wellknown": remote_port_wellknown,
    }


# --------------------------------------------------------------------------- #
# V3 helpers + categories (L global_novelty, M lineage, N beaconing,           #
# O host_os_profile). See 03_FEATURES_ENGINEERING.md Part B.                    #
# --------------------------------------------------------------------------- #
def _basename(path):
    """Last path component of a unix- or windows-style path (or None)."""
    p = _clean_str(path)
    if p is None:
        return None
    p = p.replace("\\", "/").rstrip("/")
    return p.rsplit("/", 1)[-1] if "/" in p else p


def _dirname(path):
    """Directory of a unix- or windows-style path (or None)."""
    p = _clean_str(path)
    if p is None:
        return None
    p = p.replace("\\", "/")
    return p.rsplit("/", 1)[0] if "/" in p else None


def _exe_name(edge):
    """Executable basename of the subject endpoint of an edge (src preferred).

    Subject labels in this corpus are exe paths (e.g. '/usr/sbin/sshd ...',
    'None main', '.../biGuWCmNsuCIG.exe ...'); we take the first whitespace
    token's basename so args don't pollute the identity.
    """
    for side in ("src", "dst"):
        if edge.get(f"{side}_type") == "subject":
            lbl = _clean_str(edge.get(f"{side}_path")) or _clean_str(edge.get(f"{side}_label"))
            if lbl is None:
                continue
            first = lbl.split()[0] if lbl.split() else lbl
            # 'None main' style: label is 'None <name>' -> take the name.
            if first.lower() == "none" and len(lbl.split()) > 1:
                first = lbl.split()[1]
            return _basename(first)
    return None


def _normalize_for_prior(path):
    """Strip volatile tokens (random temp dirs, cache hashes, ephemeral parts)
    so a path matches its training-time form for the 'seen in training' check."""
    p = _clean_str(path)
    if p is None:
        return None
    pl = p.replace("\\", "/")
    # Windows temp dropper: '/Windows/Temp/rad<HEX>.tmp/<exe>' -> '/Windows/Temp/<exe>'.
    pl = re.sub(r"(?i)(/windows/temp)/rad[0-9a-f]+\.tmp/", r"\1/", pl)
    pl = re.sub(r"(?i)(/temp)/[0-9a-f]{6,}/", r"\1/", pl)
    # Firefox cache: '.../cache2/entries/<HASH>' -> '.../cache2/entries/<HASH>'.
    pl = re.sub(r"(/cache2/entries)/[0-9A-Fa-f]{16,}", r"\1/<HASH>", pl)
    return pl


def _remote_ip(edge):
    if (
        edge.get("nf_remote_ip") is None
        and edge.get("nf_remote_port") is None
        and edge.get("nf_local_port") is None
    ):
        return None
    return _clean_str(edge.get("nf_remote_ip"))


def _is_outbound_edge(edge):
    """True if this netflow edge is the host acting as client (outbound)."""
    l_port = _to_port(edge.get("nf_local_port"))
    r_port = _to_port(edge.get("nf_remote_port"))
    return _is_ephemeral_port(l_port) and _is_service_port(r_port)


# --- L. Global novelty (training-scope priors) ------------------------------ #
def _cat_global_novelty(n, surviving_edges, ops, prior):
    dst_ip_unseen_global = _zeros(n)
    exe_name_unseen_global = _zeros(n)
    file_path_unseen_global = _zeros(n)
    dir_unseen_global = _zeros(n)
    triple_unseen_global = _zeros(n)
    src_exe_unseen_does_outbound = _zeros(n)

    if not prior:
        return {
            "dst_ip_unseen_global": dst_ip_unseen_global,
            "exe_name_unseen_global": exe_name_unseen_global,
            "file_path_unseen_global": file_path_unseen_global,
            "dir_unseen_global": dir_unseen_global,
            "triple_unseen_global": triple_unseen_global,
            "src_exe_unseen_does_outbound": src_exe_unseen_does_outbound,
        }

    ips = prior.get("ips") or set()
    exe_names = prior.get("exe_names") or set()
    file_paths = prior.get("file_paths") or set()
    dirs = prior.get("dirs") or set()
    triples = prior.get("triples") or set()

    for i, e in enumerate(surviving_edges):
        r_ip = _remote_ip(e)
        if r_ip is not None and not _is_private_ip(r_ip):
            if r_ip not in ips:
                dst_ip_unseen_global[i] = 1.0

        exe = _exe_name(e)
        exe_unseen = exe is not None and exe not in exe_names
        if exe_unseen:
            exe_name_unseen_global[i] = 1.0

        fp = _normalize_for_prior(_file_path(e))
        if fp is not None and fp not in file_paths:
            file_path_unseen_global[i] = 1.0
        d = _dirname(fp)
        if d is not None and d not in dirs:
            dir_unseen_global[i] = 1.0

        st, dt, op = e.get("src_type"), e.get("dst_type"), ops[i]
        if st is not None and dt is not None:
            if (st, op, dt) not in triples:
                triple_unseen_global[i] = 1.0

        if exe_unseen and r_ip is not None and not _is_private_ip(r_ip) and _is_outbound_edge(e):
            src_exe_unseen_does_outbound[i] = 1.0

    return {
        "dst_ip_unseen_global": dst_ip_unseen_global,
        "exe_name_unseen_global": exe_name_unseen_global,
        "file_path_unseen_global": file_path_unseen_global,
        "dir_unseen_global": dir_unseen_global,
        "triple_unseen_global": triple_unseen_global,
        "src_exe_unseen_does_outbound": src_exe_unseen_does_outbound,
    }


# --- M. Lineage / kill-chain (within-window) -------------------------------- #
def _cat_lineage(n, surviving_edges, srcs, dsts, ops, span, start):
    src_was_written_this_window = _zeros(n)
    dst_then_executed = _zeros(n)
    proc_spawn_fanout_log = _zeros(n)
    parent_makes_outbound = _zeros(n)
    chain_depth_norm = _zeros(n)

    paths = [_file_path(e) for e in surviving_edges]
    # Files written this window (path -> earliest write time) and executed paths.
    written_paths = {}
    executed_paths = set()
    for i in range(n):
        p = paths[i]
        if p and ops[i] in WRITE_OPS:
            written_paths.setdefault(p, surviving_edges[i].get("time"))
        if p and ops[i] in EXEC_OPS:
            executed_paths.add(p)

    # exe-basename of the src subject (the process running this edge).
    src_exe = [_exe_name(e) for e in surviving_edges]
    # written file basenames this window (to detect 'ran a just-dropped binary').
    written_basenames = {_basename(p) for p in written_paths}

    # Process spawn fan-out: parent subject -> set(child node ids) on spawn ops.
    spawn_children = defaultdict(set)
    for i in range(n):
        if ops[i] in SPAWN_OPS or ops[i] in EXEC_OPS:
            spawn_children[srcs[i]].add(dsts[i])

    # Processes that make an outbound external flow this window.
    proc_outbound = set()
    for e in surviving_edges:
        if _is_outbound_edge(e):
            r_ip = _remote_ip(e)
            if r_ip is not None and not _is_private_ip(r_ip):
                pe = _proc_endpoint(e)
                if pe is not None:
                    proc_outbound.add(pe)

    for i in range(n):
        p = paths[i]
        # src ran a binary that was dropped (written) this window.
        if src_exe[i] is not None and src_exe[i] in written_basenames:
            src_was_written_this_window[i] = 1.0
        # this edge writes a file that is later executed this window (drop half).
        if p and ops[i] in WRITE_OPS and p in executed_paths:
            dst_then_executed[i] = 1.0
        proc_spawn_fanout_log[i] = math.log1p(len(spawn_children.get(srcs[i], ())))
        if srcs[i] in proc_outbound or dsts[i] in proc_outbound:
            parent_makes_outbound[i] = 1.0
        # chain stage: 0 = file write, 0.5 = exec, 1 = network.
        st, dt = surviving_edges[i].get("src_type"), surviving_edges[i].get("dst_type")
        if dt == "netflow" or st == "netflow":
            chain_depth_norm[i] = 1.0
        elif ops[i] in EXEC_OPS:
            chain_depth_norm[i] = 0.5
        elif ops[i] in WRITE_OPS:
            chain_depth_norm[i] = 0.0
        else:
            chain_depth_norm[i] = 0.25

    return {
        "src_was_written_this_window": src_was_written_this_window,
        "dst_then_executed": dst_then_executed,
        "proc_spawn_fanout_log": proc_spawn_fanout_log,
        "parent_makes_outbound": parent_makes_outbound,
        "chain_depth_norm": chain_depth_norm,
    }


# --- N. Beaconing (periodic C2 to a single endpoint) ------------------------ #
def _cat_beaconing(n, surviving_edges, times, prior):
    dst_ip_portfan_log = _zeros(n)
    beacon_regularity = _zeros(n)
    single_dst_dominance = _zeros(n)
    beacon_period_stable_global = _zeros(n)

    # Per process: local ports used to each remote IP, outbound dst counts,
    # and per-(proc,remote_ip) timestamps for regularity.
    proc_ip_localports = defaultdict(lambda: defaultdict(set))
    proc_outbound_counts = defaultdict(lambda: defaultdict(int))
    proc_ip_times = defaultdict(list)
    edge_proc = [None] * n
    edge_ip = [None] * n
    for i, e in enumerate(surviving_edges):
        r_ip = _remote_ip(e)
        if r_ip is None:
            continue
        pe = _proc_endpoint(e)
        if pe is None:
            continue
        edge_proc[i], edge_ip[i] = pe, r_ip
        l_port = _to_port(e.get("nf_local_port"))
        if _is_outbound_edge(e):
            if l_port is not None:
                proc_ip_localports[pe][r_ip].add(l_port)
            proc_outbound_counts[pe][r_ip] += 1
            proc_ip_times[(pe, r_ip)].append(times[i])

    # regularity per (proc, ip): 1 - CV of inter-arrival gaps (>=3 events).
    regularity = {}
    for key, ts in proc_ip_times.items():
        if len(ts) >= 3:
            g = np.diff(np.sort(np.asarray(ts, dtype=np.float64)))
            m = g.mean()
            cv = (g.std() / m) if m > _EPS else 0.0
            regularity[key] = max(0.0, 1.0 - cv)

    beacon_periods = (prior or {}).get("beacon_periods") or {}

    for i in range(n):
        pe, r_ip = edge_proc[i], edge_ip[i]
        if pe is None or r_ip is None:
            continue
        dst_ip_portfan_log[i] = math.log1p(len(proc_ip_localports[pe].get(r_ip, ())))
        beacon_regularity[i] = regularity.get((pe, r_ip), 0.0)
        counts = proc_outbound_counts[pe]
        tot = sum(counts.values())
        if tot > 0:
            single_dst_dominance[i] = max(counts.values()) / tot
        # global: does the observed period match this endpoint's historical one?
        if beacon_periods and (pe, r_ip) in proc_ip_times:
            ts = sorted(proc_ip_times[(pe, r_ip)])
            if len(ts) >= 2:
                obs = float(np.median(np.diff(np.asarray(ts, dtype=np.float64))))
                hist = beacon_periods.get(r_ip) or beacon_periods.get((pe, r_ip))
                if hist and hist > 0:
                    ratio = obs / hist
                    if 0.5 <= ratio <= 2.0:
                        beacon_period_stable_global[i] = 1.0

    return {
        "dst_ip_portfan_log": dst_ip_portfan_log,
        "beacon_regularity": beacon_regularity,
        "single_dst_dominance": single_dst_dominance,
        "beacon_period_stable_global": beacon_period_stable_global,
    }


# --- O. Host/OS profile (platform-aware TTP flags) -------------------------- #
def _cat_host_os_profile(n, surviving_edges, ops):
    win_temp_exe = _zeros(n)
    has_ads_or_zoneid = _zeros(n)
    lolbin_unusual_child = _zeros(n)
    proc_label_deleted_or_so_inject = _zeros(n)

    def _all_path_strs(e):
        out = []
        for k in ("src_path", "dst_path", "src_label", "dst_label"):
            v = _clean_str(e.get(k))
            if v is not None:
                out.append(v)
        return out

    for i, e in enumerate(surviving_edges):
        strs = _all_path_strs(e)
        low = [s.lower() for s in strs]

        # win_temp_exe: a path in a Windows temp dir ending in .exe.
        for s in low:
            if any(m in s for m in WIN_TEMP_MARKERS) and ".exe" in s:
                win_temp_exe[i] = 1.0
                break
        # ADS / mark-of-the-web.
        for s in low:
            if any(m in s for m in ADS_MARKERS):
                has_ads_or_zoneid[i] = 1.0
                break
        # LOLBin parent doing a spawn/exec (unusual living-off-the-land child).
        src_exe = _exe_name(e)
        if src_exe is not None and src_exe.lower() in LOLBINS:
            if ops[i] in SPAWN_OPS or ops[i] in EXEC_OPS:
                lolbin_unusual_child[i] = 1.0
        # anti-forensics: '(deleted)' label, or a .so/.dll loaded from tmp.
        for s in low:
            if "(deleted)" in s:
                proc_label_deleted_or_so_inject[i] = 1.0
                break
            if (s.endswith(".so") or s.endswith(".dll")) and any(
                t in s for t in (d.lower() for d in TMP_DIRS)
            ):
                proc_label_deleted_or_so_inject[i] = 1.0
                break

    return {
        "win_temp_exe": win_temp_exe,
        "has_ads_or_zoneid": has_ads_or_zoneid,
        "lolbin_unusual_child": lolbin_unusual_child,
        "proc_label_deleted_or_so_inject": proc_label_deleted_or_so_inject,
    }
