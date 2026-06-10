from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


BASE_DIR = Path(__file__).resolve().parent


def _timestamp(value: Optional[str]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    return pd.Timestamp(value)


def _load_location_lookup(lookup_path: Path) -> Dict[int, str]:
    lookup = pd.read_csv(lookup_path)
    required = {"LocationID", "Zone"}
    missing = required - set(lookup.columns)
    if missing:
        raise ValueError(f"Lookup file is missing columns: {sorted(missing)}")

    lookup = lookup.dropna(subset=["LocationID", "Zone"]).copy()
    lookup["LocationID"] = lookup["LocationID"].astype(int)
    return dict(zip(lookup["LocationID"], lookup["Zone"].astype(str)))


def _read_square_matrix(matrix_path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(matrix_path, index_col=0)
    matrix.index = [str(idx).lstrip("\ufeff") for idx in matrix.index]
    matrix.columns = [str(col).lstrip("\ufeff") for col in matrix.columns]
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Template matrix must be square: {matrix_path}")
    if list(matrix.index) != list(matrix.columns):
        raise ValueError("Template matrix index and columns must have the same order.")
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _load_zone_names(lookup_path: Path, template_path: Optional[Path]) -> List[str]:
    if template_path is not None:
        template = _read_square_matrix(template_path)
        return [str(zone) for zone in template.index]

    lookup = pd.read_csv(lookup_path)
    zones = lookup["Zone"].dropna().astype(str).drop_duplicates().tolist()
    return sorted(zones)


def _iter_od_batches(
    parquet_path: Path,
    batch_size: int,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
) -> Iterable[pd.DataFrame]:
    columns = ["PULocationID", "DOLocationID"]
    if start_date is not None or end_date is not None:
        columns.append("pickup_datetime")

    parquet_file = pq.ParquetFile(parquet_path)
    missing = set(columns) - set(parquet_file.schema.names)
    if missing:
        raise ValueError(f"Parquet file is missing columns: {sorted(missing)}")

    for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
        frame = batch.to_pandas()
        if "pickup_datetime" in frame.columns:
            pickup_time = pd.to_datetime(frame["pickup_datetime"])
            mask = pd.Series(True, index=frame.index)
            if start_date is not None:
                mask &= pickup_time >= start_date
            if end_date is not None:
                mask &= pickup_time < end_date
            frame = frame.loc[mask]
        if not frame.empty:
            yield frame[["PULocationID", "DOLocationID"]]


def _aggregate_flows(
    parquet_path: Path,
    location_to_zone: Dict[int, str],
    batch_size: int,
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    excluded_locations: set[int],
    include_self: bool,
) -> Tuple[Counter[Tuple[str, str]], Counter[str], int]:
    flows: Counter[Tuple[str, str]] = Counter()
    origin_totals: Counter[str] = Counter()
    kept_rows = 0

    for frame in _iter_od_batches(parquet_path, batch_size, start_date, end_date):
        frame = frame.dropna(subset=["PULocationID", "DOLocationID"]).copy()
        if frame.empty:
            continue

        frame["PULocationID"] = frame["PULocationID"].astype(int)
        frame["DOLocationID"] = frame["DOLocationID"].astype(int)

        if excluded_locations:
            frame = frame[
                ~frame["PULocationID"].isin(excluded_locations)
                & ~frame["DOLocationID"].isin(excluded_locations)
            ]
        if frame.empty:
            continue

        frame["origin_zone"] = frame["PULocationID"].map(location_to_zone)
        frame["dest_zone"] = frame["DOLocationID"].map(location_to_zone)
        frame = frame.dropna(subset=["origin_zone", "dest_zone"])
        if not include_self:
            frame = frame[frame["origin_zone"] != frame["dest_zone"]]
        if frame.empty:
            continue

        kept_rows += len(frame)
        grouped = frame.groupby(["origin_zone", "dest_zone"], sort=False).size()
        for (origin, dest), count in grouped.items():
            count_int = int(count)
            flows[(str(origin), str(dest))] += count_int
            origin_totals[str(origin)] += count_int

    return flows, origin_totals, kept_rows


def _retain_top_k(
    flows: Counter[Tuple[str, str]],
    top_k: int,
    min_flow: int,
) -> Dict[str, List[Tuple[str, int]]]:
    by_origin: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for (origin, dest), count in flows.items():
        if count >= min_flow:
            by_origin[origin].append((dest, int(count)))

    retained: Dict[str, List[Tuple[str, int]]] = {}
    for origin, edges in by_origin.items():
        edges.sort(key=lambda item: (-item[1], item[0]))
        retained[origin] = edges if top_k <= 0 else edges[:top_k]
    return retained


def _edge_weight(
    count: int,
    origin_total: int,
    retained_total: int,
    weight_mode: str,
) -> float:
    if weight_mode == "count":
        return float(count)
    if weight_mode == "binary":
        return 1.0
    if weight_mode == "retained_share":
        return float(count / retained_total) if retained_total > 0 else 0.0
    if weight_mode == "log_count":
        return float(np.log1p(count))
    return float(count / origin_total) if origin_total > 0 else 0.0


def _apply_symmetrize(values: np.ndarray, symmetrize: str) -> np.ndarray:
    if symmetrize == "max":
        return np.maximum(values, values.T)
    if symmetrize == "sum":
        return values + values.T
    if symmetrize == "mean":
        return (values + values.T) / 2.0
    return values


def _build_matrix(
    zone_names: List[str],
    retained: Dict[str, List[Tuple[str, int]]],
    origin_totals: Counter[str],
    weight_mode: str,
    symmetrize: str,
) -> pd.DataFrame:
    zone_set = set(zone_names)
    matrix = pd.DataFrame(0.0, index=zone_names, columns=zone_names)

    for origin, edges in retained.items():
        if origin not in zone_set:
            continue
        retained_total = sum(count for dest, count in edges if dest in zone_set)
        for dest, count in edges:
            if dest not in zone_set:
                continue
            matrix.at[origin, dest] = _edge_weight(
                count=count,
                origin_total=origin_totals[origin],
                retained_total=retained_total,
                weight_mode=weight_mode,
            )

    values = _apply_symmetrize(matrix.to_numpy(dtype=float), symmetrize)

    if symmetrize != "none":
        matrix = pd.DataFrame(values, index=zone_names, columns=zone_names)

    return matrix


def _align_reference_matrix(reference_path: Path, zone_names: List[str]) -> pd.DataFrame:
    reference = _read_square_matrix(reference_path)
    missing = set(zone_names) - set(reference.index)
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"Random reference matrix is missing {len(missing)} zones, e.g. {sample}"
        )
    return reference.reindex(index=zone_names, columns=zone_names).fillna(0.0)


def _candidate_mask(node_count: int, include_self: bool) -> np.ndarray:
    mask = np.ones((node_count, node_count), dtype=bool)
    if not include_self:
        np.fill_diagonal(mask, False)
    return mask


def _reference_edge_count(reference: pd.DataFrame, include_self: bool) -> int:
    values = reference.to_numpy(dtype=float)
    mask = _candidate_mask(values.shape[0], include_self)
    return int(((values > 0.0) & mask).sum())


def _sample_random_weights(
    rng: np.random.Generator,
    edge_count: int,
    weight_mode: str,
    reference: Optional[pd.DataFrame],
    include_self: bool,
) -> np.ndarray:
    if weight_mode == "binary":
        return np.ones(edge_count, dtype=float)
    if weight_mode == "uniform":
        return rng.uniform(0.0, 1.0, size=edge_count)

    if reference is None:
        raise ValueError("--random-weight-mode reference requires a reference matrix.")

    values = reference.to_numpy(dtype=float)
    mask = _candidate_mask(values.shape[0], include_self)
    positive_weights = values[(values > 0.0) & mask]
    if positive_weights.size == 0:
        raise ValueError("Reference matrix has no positive edge weights to sample.")
    return rng.choice(positive_weights, size=edge_count, replace=True).astype(float)


def _build_random_matrix_by_edge_count(
    zone_names: List[str],
    edge_count: int,
    include_self: bool,
    random_seed: int,
    random_weight_mode: str,
    reference: Optional[pd.DataFrame],
) -> pd.DataFrame:
    node_count = len(zone_names)
    mask = _candidate_mask(node_count, include_self)
    rows, cols = np.where(mask)
    possible_edges = len(rows)
    if edge_count > possible_edges:
        raise ValueError(
            f"Requested {edge_count} random edges, but only {possible_edges} are possible."
        )

    rng = np.random.default_rng(random_seed)
    matrix = np.zeros((node_count, node_count), dtype=float)
    if edge_count == 0:
        return pd.DataFrame(matrix, index=zone_names, columns=zone_names)

    selected = rng.choice(possible_edges, size=edge_count, replace=False)
    weights = _sample_random_weights(
        rng=rng,
        edge_count=edge_count,
        weight_mode=random_weight_mode,
        reference=reference,
        include_self=include_self,
    )
    matrix[rows[selected], cols[selected]] = weights
    return pd.DataFrame(matrix, index=zone_names, columns=zone_names)


def _build_random_matrix_by_top_k(
    zone_names: List[str],
    top_k: int,
    include_self: bool,
    random_seed: int,
    random_weight_mode: str,
    reference: Optional[pd.DataFrame],
) -> pd.DataFrame:
    node_count = len(zone_names)
    rng = np.random.default_rng(random_seed)
    matrix = np.zeros((node_count, node_count), dtype=float)

    for row_idx in range(node_count):
        candidates = np.arange(node_count)
        if not include_self:
            candidates = candidates[candidates != row_idx]

        edge_count = len(candidates) if top_k == 0 else min(top_k, len(candidates))
        if edge_count == 0:
            continue

        selected_cols = rng.choice(candidates, size=edge_count, replace=False)
        weights = _sample_random_weights(
            rng=rng,
            edge_count=edge_count,
            weight_mode=random_weight_mode,
            reference=reference,
            include_self=include_self,
        )
        matrix[row_idx, selected_cols] = weights

    return pd.DataFrame(matrix, index=zone_names, columns=zone_names)


def _build_random_matrix(
    zone_names: List[str],
    random_mode: str,
    random_edge_count: Optional[int],
    top_k: int,
    include_self: bool,
    random_seed: int,
    random_weight_mode: str,
    symmetrize: str,
    reference: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if random_mode == "edge_count":
        edge_count = (
            _reference_edge_count(reference, include_self)
            if random_edge_count is None and reference is not None
            else random_edge_count
        )
        if edge_count is None:
            candidate_count = len(zone_names) if include_self else max(len(zone_names) - 1, 0)
            per_origin_count = candidate_count if top_k == 0 else min(top_k, candidate_count)
            edge_count = len(zone_names) * per_origin_count
        matrix = _build_random_matrix_by_edge_count(
            zone_names=zone_names,
            edge_count=edge_count,
            include_self=include_self,
            random_seed=random_seed,
            random_weight_mode=random_weight_mode,
            reference=reference,
        )
    else:
        if random_edge_count is not None:
            raise ValueError("--random-edge-count is only valid with --random-mode edge_count.")
        matrix = _build_random_matrix_by_top_k(
            zone_names=zone_names,
            top_k=top_k,
            include_self=include_self,
            random_seed=random_seed,
            random_weight_mode=random_weight_mode,
            reference=reference,
        )

    values = _apply_symmetrize(matrix.to_numpy(dtype=float), symmetrize)
    if symmetrize != "none":
        matrix = pd.DataFrame(values, index=zone_names, columns=zone_names)
    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sparse OD or random graph matrices for taxi-zone GNN experiments."
    )
    parser.add_argument(
        "--graph-type",
        choices=["od", "random"],
        default="od",
        help="Build an OD-flow graph from trip data or a random baseline graph.",
    )
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data.parquet")
    parser.add_argument("--lookup", type=Path, default=BASE_DIR / "taxi-zone-lookup.csv")
    parser.add_argument(
        "--template",
        type=Path,
        default=BASE_DIR / "edge_weight_matrix_with_flow.csv",
        help="Use this existing matrix to preserve row/column zone order.",
    )
    parser.add_argument(
        "--use-lookup-zones",
        action="store_true",
        help="Ignore --template and build rows/columns from taxi-zone-lookup.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Defaults to edge_weight_matrix_od.csv or "
            "edge_weight_matrix_random.csv."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Keep the top K destination zones by flow for each origin zone. Use 0 to keep all.",
    )
    parser.add_argument(
        "--min-flow",
        type=int,
        default=1,
        help="Drop OD edges with fewer than this many trips before top-k filtering.",
    )
    parser.add_argument(
        "--weight-mode",
        choices=["row_share", "retained_share", "count", "binary", "log_count"],
        default="row_share",
        help="How to write retained edge weights.",
    )
    parser.add_argument(
        "--symmetrize",
        choices=["none", "max", "sum", "mean"],
        default="none",
        help="Optionally convert directed edges to an undirected matrix.",
    )
    parser.add_argument(
        "--random-mode",
        choices=["edge_count", "per_origin_top_k"],
        default="edge_count",
        help=(
            "For random graphs, either match a total edge count or sample top-k "
            "outgoing destinations per origin."
        ),
    )
    parser.add_argument(
        "--random-edge-count",
        type=int,
        default=None,
        help=(
            "Total random directed edges for --random-mode edge_count. If omitted, "
            "uses --random-reference, then --template, then --top-k."
        ),
    )
    parser.add_argument(
        "--random-reference",
        type=Path,
        default=None,
        help=(
            "Matrix whose positive edge count or weights define the random baseline. "
            "Defaults to --template when available."
        ),
    )
    parser.add_argument(
        "--random-weight-mode",
        choices=["binary", "uniform", "reference"],
        default="binary",
        help="How to assign weights to random edges.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--start-date", type=_timestamp, default=None)
    parser.add_argument("--end-date", type=_timestamp, default=None)
    parser.add_argument(
        "--exclude-location-ids",
        type=int,
        nargs="*",
        default=[],
        help="LocationIDs to remove from origins and destinations.",
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="Keep trips where pickup and dropoff map to the same zone.",
    )
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is None:
        args.output = BASE_DIR / (
            "edge_weight_matrix_random.csv"
            if args.graph_type == "random"
            else "edge_weight_matrix_od.csv"
        )
    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {args.output}\n"
            "Pass --overwrite if you want to replace it, or pass --output to write "
            "a new file.\n"
            "For a random graph, use: --graph-type random "
            "--output edge_weight_matrix_random.csv --overwrite"
        )
    if args.top_k < 0:
        raise ValueError("--top-k must be >= 0")
    if args.min_flow < 1:
        raise ValueError("--min-flow must be >= 1")
    if args.random_edge_count is not None and args.random_edge_count < 0:
        raise ValueError("--random-edge-count must be >= 0")

    template_path = None if args.use_lookup_zones else args.template
    zone_names = _load_zone_names(args.lookup, template_path)

    kept_rows = None
    if args.graph_type == "random":
        reference_path = (
            args.random_reference
            if args.random_reference is not None
            else template_path
        )
        reference = (
            _align_reference_matrix(reference_path, zone_names)
            if reference_path is not None
            else None
        )
        matrix = _build_random_matrix(
            zone_names=zone_names,
            random_mode=args.random_mode,
            random_edge_count=args.random_edge_count,
            top_k=args.top_k,
            include_self=args.include_self,
            random_seed=args.random_seed,
            random_weight_mode=args.random_weight_mode,
            symmetrize=args.symmetrize,
            reference=reference,
        )
    else:
        location_to_zone = _load_location_lookup(args.lookup)
        flows, origin_totals, kept_rows = _aggregate_flows(
            parquet_path=args.data,
            location_to_zone=location_to_zone,
            batch_size=args.batch_size,
            start_date=args.start_date,
            end_date=args.end_date,
            excluded_locations=set(args.exclude_location_ids),
            include_self=args.include_self,
        )
        retained = _retain_top_k(flows, top_k=args.top_k, min_flow=args.min_flow)
        matrix = _build_matrix(
            zone_names=zone_names,
            retained=retained,
            origin_totals=origin_totals,
            weight_mode=args.weight_mode,
            symmetrize=args.symmetrize,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.output, encoding="utf-8")

    edge_count = int((matrix.to_numpy() > 0).sum())
    density = edge_count / float(matrix.shape[0] * matrix.shape[1])
    print(f"Saved {args.graph_type} graph: {args.output}")
    print(f"Zones: {matrix.shape[0]}")
    if kept_rows is not None:
        print(f"Trips retained for counting: {kept_rows}")
    print(f"Nonzero edges: {edge_count} ({density:.4%} density)")
    print(f"Top-k per origin: {'all' if args.top_k == 0 else args.top_k}")
    if args.graph_type == "random":
        print(f"Random mode: {args.random_mode}")
        print(f"Random seed: {args.random_seed}")
        print(f"Random weight mode: {args.random_weight_mode}")
    else:
        print(f"Weight mode: {args.weight_mode}")
    print(f"Symmetrize: {args.symmetrize}")


if __name__ == "__main__":
    main()
