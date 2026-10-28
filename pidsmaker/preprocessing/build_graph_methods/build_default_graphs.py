"""Default provenance graph construction from PostgreSQL database.

Builds provenance graphs from DARPA TC/OpTC datasets stored in PostgreSQL.
Creates time-windowed graph snapshots with node features, edge types, and timestamps.
Supports attack mimicry generation for data augmentation and 61-d engineered edge features.
"""

import os
from collections import defaultdict
from datetime import datetime, timedelta

import networkx as nx
import numpy as np
import torch

import pidsmaker.mimicry as mimicry
from pidsmaker.config import get_darpa_tc_node_feats_from_cfg, get_dates_from_cfg
from pidsmaker.preprocessing.build_graph_methods import edge_feat_engineering
from pidsmaker.utils.dataset_utils import get_rel2id
from pidsmaker.utils.utils import (
    datetime_to_ns_time_US,
    get_split_to_files,
    init_database_connection,
    log,
    log_start,
    log_tqdm,
    ns_time_to_datetime_US,
    stream_query,
    stringtomd5,
)


def compute_indexid2msg(cfg):
    """Compute mapping from node index IDs to node types and feature labels."""
    cur, connect = init_database_connection(cfg)

    use_hashed_label = cfg.construction.use_hashed_label
    node_label_features = get_darpa_tc_node_feats_from_cfg(cfg)
    indexid2msg = {}
    build_attrs = bool(cfg.construction.compute_eng_edge_feats)
    indexid2attrs = {}

    def get_label_str_from_features(attrs, node_type):
        node_features = node_label_features[node_type]
        label_str = " ".join([attrs[d] for d in node_features])
        if use_hashed_label:
            label_str = stringtomd5(label_str)
        return label_str

    # netflow
    n_netflow = 0
    for i in stream_query(connect, "select * from netflow_node_table;", name="pids_netflow"):
        attrs = {
            "type": "netflow",
            "local_ip": str(i[2]),
            "local_port": str(i[3]),
            "remote_ip": str(i[4]),
            "remote_port": str(i[5]),
        }
        index_id = str(i[-1])
        node_type = attrs["type"]
        label_str = get_label_str_from_features(attrs, node_type)

        indexid2msg[index_id] = [node_type, label_str]
        if build_attrs:
            indexid2attrs[index_id] = {
                "node_type": node_type,
                "local_ip": attrs["local_ip"],
                "local_port": attrs["local_port"],
                "remote_ip": attrs["remote_ip"],
                "remote_port": attrs["remote_port"],
            }
        n_netflow += 1
    log(f"Number of netflow nodes: {n_netflow}")

    # subject
    n_subject = 0
    for i in stream_query(connect, "select * from subject_node_table;", name="pids_subject"):
        attrs = {"type": "subject", "path": str(i[2]), "cmd_line": str(i[3])}
        index_id = str(i[-1])
        node_type = attrs["type"]
        label_str = get_label_str_from_features(attrs, node_type)

        indexid2msg[index_id] = [node_type, label_str]
        if build_attrs:
            indexid2attrs[index_id] = {
                "node_type": node_type,
                "path": attrs["path"],
                "cmd_line": attrs["cmd_line"],
            }
        n_subject += 1
    log(f"Number of process nodes: {n_subject}")

    # file
    n_file = 0
    for i in stream_query(connect, "select * from file_node_table;", name="pids_file"):
        attrs = {"type": "file", "path": str(i[2])}
        index_id = str(i[-1])
        node_type = attrs["type"]
        label_str = get_label_str_from_features(attrs, node_type)

        indexid2msg[index_id] = [node_type, label_str]
        if build_attrs:
            indexid2attrs[index_id] = {"node_type": node_type, "path": attrs["path"]}
        n_file += 1
    log(f"Number of file nodes: {n_file}")

    return indexid2msg, indexid2attrs


def save_indexid2msg(indexid2msg, split2nodes, cfg):
    """Filter and save node index-to-message mapping for valid nodes only."""
    valid_nodes = set().union(*split2nodes.values())
    filtered_indexid2msg = {node: label for node, label in indexid2msg.items() if node in valid_nodes}

    out_dir = cfg.construction._dicts_dir
    os.makedirs(out_dir, exist_ok=True)
    log("Saving indexid2msg to disk...")
    torch.save(filtered_indexid2msg, os.path.join(out_dir, "indexid2msg.pkl"))


def compute_and_save_split2nodes(cfg):
    """Compute and save mapping of dataset splits to their node sets."""
    split_to_files = get_split_to_files(cfg, cfg.construction._graphs_dir)
    split2nodes = defaultdict(set)

    for split, files in split_to_files.items():
        for path in log_tqdm(files, desc=f"Check nodes in {split} set"):
            G = torch.load(path)
            split2nodes[split].update(G.nodes())
            del G
    split2nodes = dict(split2nodes)

    out_dir = cfg.construction._dicts_dir
    os.makedirs(out_dir, exist_ok=True)
    log("Saving split2nodes to disk...")
    torch.save(split2nodes, os.path.join(out_dir, "split2nodes.pkl"))

    return split2nodes


def gen_edge_fused_tw(indexid2msg, cfg, indexid2attrs=None):
    """Generate time-windowed provenance graphs from database events."""
    indexid2attrs = indexid2attrs or {}
    cur, connect = init_database_connection(cfg)
    rel2id = get_rel2id(cfg)
    include_edge_type = rel2id

    mimicry_edge_num = cfg.construction.mimicry_edge_num
    if mimicry_edge_num is not None and mimicry_edge_num > 0:
        attack_mimicry_events = mimicry.gen_mimicry_edges(cfg)
    else:
        attack_mimicry_events = defaultdict(list)

    compute_eng = bool(cfg.construction.compute_eng_edge_feats)
    eng_categories = edge_feat_engineering.get_producer_categories(cfg) if compute_eng else []
    eng_tau_frac = edge_feat_engineering.get_ema_tau_frac(cfg)

    eng_stats = {"windows": 0, "raw_edges": 0, "surviving_edges": 0}

    def _build_and_save_graph(
        temp_list, start_time, end_event_ts, date, cfg, indexid2msg, indexid2attrs
    ):
        time_interval = (
            ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(end_event_ts)
        )

        node_info = {}
        edge_list = []
        if compute_eng:
            edge_info = {}
            raw_events = []
            for (
                _src_node,
                src_index_id,
                operation,
                _dst_node,
                dst_index_id,
                event_uuid,
                timestamp_rec,
                _id,
            ) in temp_list:
                if src_index_id not in node_info:
                    node_type, label = indexid2msg[src_index_id]
                    node_info[src_index_id] = {"label": label, "node_type": node_type}
                if dst_index_id not in node_info:
                    node_type, label = indexid2msg[dst_index_id]
                    node_info[dst_index_id] = {"label": label, "node_type": node_type}

                edge_info.setdefault((src_index_id, dst_index_id), []).append(
                    (timestamp_rec, operation, event_uuid)
                )
                raw_events.append((src_index_id, dst_index_id, operation, timestamp_rec))

            surviving = []
            for (src, dst), data in edge_info.items():
                sorted_data = sorted(data, key=lambda x: x[0])
                cur_op = None
                cur_ts = None
                cur_uuids = None
                runs = []
                for ts, op, uuid in sorted_data:
                    if op != cur_op:
                        if cur_op is not None:
                            runs.append((cur_op, cur_ts, cur_uuids))
                        cur_op = op
                        cur_ts = [ts]
                        cur_uuids = [uuid]
                    else:
                        cur_ts.append(ts)
                        cur_uuids.append(uuid)
                if cur_op is not None:
                    runs.append((cur_op, cur_ts, cur_uuids))

                src_a = indexid2attrs.get(src, {})
                dst_a = indexid2attrs.get(dst, {})
                src_path = src_a.get("path")
                dst_path = dst_a.get("path")
                if src_a.get("node_type") == "netflow":
                    nf_a = src_a
                elif dst_a.get("node_type") == "netflow":
                    nf_a = dst_a
                else:
                    nf_a = {}

                for op, ts_list, uuid_list in runs:
                    surviving.append(
                        {
                            "src": src,
                            "dst": dst,
                            "time": ts_list[-1],
                            "label": op,
                            "event_uuid": uuid_list[-1],
                            "abs_ts": ts_list,
                            "src_type": node_info[src]["node_type"],
                            "dst_type": node_info[dst]["node_type"],
                            "src_path": src_path,
                            "dst_path": dst_path,
                            "nf_remote_ip": nf_a.get("remote_ip"),
                            "nf_remote_port": nf_a.get("remote_port"),
                            "nf_local_port": nf_a.get("local_port"),
                            "src_label": node_info[src]["label"],
                            "dst_label": node_info[dst]["label"],
                        }
                    )

            surviving.sort(key=lambda e: (e["time"], e["src"], e["dst"]))

            feats = edge_feat_engineering.compute(
                surviving,
                raw_events=raw_events,
                window_start_ns=int(start_time),
                window_end_ns=int(end_event_ts),
                categories=eng_categories,
                ema_tau_frac=eng_tau_frac,
            )

            eng_stats["windows"] += 1
            eng_stats["raw_edges"] += len(raw_events)
            eng_stats["surviving_edges"] += len(surviving)

            for i, e in enumerate(surviving):
                edge_list.append(
                    {
                        "src": e["src"],
                        "dst": e["dst"],
                        "time": e["time"],
                        "label": e["label"],
                        "event_uuid": e["event_uuid"],
                        "eng": feats[i].astype(np.float32),
                    }
                )
        elif cfg.construction.fuse_edge:
            edge_info = {}
            for (
                _src_node,
                src_index_id,
                operation,
                _dst_node,
                dst_index_id,
                event_uuid,
                timestamp_rec,
                _id,
            ) in temp_list:
                if src_index_id not in node_info:
                    node_type, label = indexid2msg[src_index_id]
                    node_info[src_index_id] = {"label": label, "node_type": node_type}
                if dst_index_id not in node_info:
                    node_type, label = indexid2msg[dst_index_id]
                    node_info[dst_index_id] = {"label": label, "node_type": node_type}

                if (src_index_id, dst_index_id) not in edge_info:
                    edge_info[(src_index_id, dst_index_id)] = []

                edge_info[(src_index_id, dst_index_id)].append(
                    (timestamp_rec, operation, event_uuid)
                )

            for (src, dst), data in edge_info.items():
                sorted_data = sorted(data, key=lambda x: x[0])
                operation_list = [entry[1] for entry in sorted_data]

                indices = []
                current_type = None
                current_start_index = None

                for idx, item in enumerate(operation_list):
                    if item == current_type:
                        continue
                    else:
                        if current_type is not None and current_start_index is not None:
                            indices.append(current_start_index)
                        current_type = item
                        current_start_index = idx

                if current_type is not None and current_start_index is not None:
                    indices.append(current_start_index)

                for k in indices:
                    edge_list.append(
                        {
                            "src": src,
                            "dst": dst,
                            "time": sorted_data[k][0],
                            "label": sorted_data[k][1],
                            "event_uuid": sorted_data[k][2],
                        }
                    )
        else:
            for (
                _src_node,
                src_index_id,
                operation,
                _dst_node,
                dst_index_id,
                event_uuid,
                timestamp_rec,
                _id,
            ) in temp_list:
                if src_index_id not in node_info:
                    node_type, label = indexid2msg[src_index_id]
                    node_info[src_index_id] = {"label": label, "node_type": node_type}
                if dst_index_id not in node_info:
                    node_type, label = indexid2msg[dst_index_id]
                    node_info[dst_index_id] = {"label": label, "node_type": node_type}

                edge_list.append(
                    {
                        "src": src_index_id,
                        "dst": dst_index_id,
                        "time": timestamp_rec,
                        "label": operation,
                        "event_uuid": event_uuid,
                    }
                )

        graph = nx.MultiDiGraph()

        for node, info in node_info.items():
            graph.add_node(node, node_type=info["node_type"], label=info["label"])

        for i, edge in enumerate(edge_list):
            attrs = {
                "event_uuid": edge["event_uuid"],
                "time": edge["time"],
                "label": edge["label"],
                "y": 0,
            }
            if "eng" in edge:
                attrs["eng"] = edge["eng"]
            graph.add_edge(edge["src"], edge["dst"], **attrs)

            NUM_TEST_EDGES = 2000
            if cfg._test_mode and i >= NUM_TEST_EDGES:
                break

        date_dir = f"{cfg.construction._graphs_dir}/graph_{date}/"
        os.makedirs(date_dir, exist_ok=True)
        graph_name = f"{date_dir}/{time_interval}"

        torch.save(graph, graph_name)

    dates = get_dates_from_cfg(cfg)

    log("Building graphs...")
    for date in dates:
        date_start = f"{date} 00:00:00"
        date_stop = f"{(datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')} 00:00:00"

        timestamps = [date_start, date_stop]

        for i in range(0, len(timestamps) - 1):
            start = timestamps[i]
            stop = timestamps[i + 1]
            start_ns_timestamp = datetime_to_ns_time_US(start)
            end_ns_timestamp = datetime_to_ns_time_US(stop)

            attack_index = 0
            mimicry_events = []
            for attack_tuple in cfg.dataset.attack_to_time_window:
                attack_start_time = datetime_to_ns_time_US(attack_tuple[1])
                attack_end_time = datetime_to_ns_time_US(attack_tuple[2])

                if mimicry_edge_num > 0 and (
                    attack_start_time >= start_ns_timestamp and attack_end_time <= end_ns_timestamp
                ):
                    log(
                        f"Insert mimicry events into attack {attack_index} when building graphs from {date_start} to {date_stop}"
                    )
                    mimicry_events.extend(attack_mimicry_events[attack_index])
                attack_index += 1

            sql = (
                "select * from event_table "
                f"where timestamp_rec>'{start_ns_timestamp}' and timestamp_rec<'{end_ns_timestamp}' "
                "ORDER BY timestamp_rec, event_uuid;"
            )

            def filtered_event_stream(_sql=sql, _mimicry=mimicry_events):
                for row in stream_query(connect, _sql, name="pids_events"):
                    if row[2] in include_edge_type:
                        yield row
                for row in _mimicry:
                    if row[2] in include_edge_type:
                        yield row

            BATCH = 1024
            window_size_in_ns = cfg.construction.time_window_size * 60_000_000_000

            stream = filtered_event_stream()
            temp_list = []
            start_time = None
            stream_exhausted = False

            while not stream_exhausted:
                batch_buf = []
                for _ in range(BATCH):
                    try:
                        batch_buf.append(next(stream))
                    except StopIteration:
                        stream_exhausted = True
                        break

                if not batch_buf:
                    break
                if start_time is None:
                    start_time = batch_buf[0][-2]

                temp_list.extend(batch_buf)
                last_event_ts = batch_buf[-1][-2]
                last_batch = stream_exhausted

                if (last_event_ts > start_time + window_size_in_ns) or last_batch:
                    _build_and_save_graph(
                        temp_list=temp_list,
                        start_time=start_time,
                        end_event_ts=last_event_ts,
                        date=date,
                        cfg=cfg,
                        indexid2msg=indexid2msg,
                        indexid2attrs=indexid2attrs,
                    )

                    start_time = last_event_ts
                    temp_list.clear()

                    if cfg._test_mode:
                        break

    if compute_eng:
        total_raw = eng_stats["raw_edges"]
        total_surv = eng_stats["surviving_edges"]
        reduction = (1 - total_surv / total_raw) * 100 if total_raw else 0.0
        log(
            f"[eng] TOTAL across {eng_stats['windows']} windows: "
            f"{total_raw} edges before fusion -> {total_surv} surviving after fusion "
            f"({reduction:.1f}% fused). Engineered features (categories="
            f"{eng_categories}, emitted width="
            f"{len(edge_feat_engineering.feature_names_from_categories(eng_categories))}) "
            f"attached to all surviving edges."
        )


def main(cfg):
    """Main construction pipeline: build graphs from database and save metadata."""
    log_start(__file__)

    indexid2msg, indexid2attrs = compute_indexid2msg(cfg=cfg)

    gen_edge_fused_tw(indexid2msg=indexid2msg, indexid2attrs=indexid2attrs, cfg=cfg)

    split2nodes = compute_and_save_split2nodes(cfg)
    save_indexid2msg(indexid2msg, split2nodes, cfg)
