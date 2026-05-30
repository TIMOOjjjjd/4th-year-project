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


def _load_zone_names(lookup_path: Path, template_path: Optional[Path]) -> List[str]:
    if template_path is not None:
        template = pd.read_csv(template_path, index_col=0)
        if template.shape[0] != template.shape[1]:
            raise ValueError(f"Template matrix must be square: {template_path}")
        if list(template.index) != list(template.columns):
            raise ValueError("Template matrix index and columns must have the same order.")
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

    values = matrix.to_numpy(dtype=float)
    if symmetrize == "max":
        values = np.maximum(values, values.T)
    elif symmetrize == "sum":
        values = values + values.T
    elif symmetrize == "mean":
        values = (values + values.T) / 2.0

    if symmetrize != "none":
        matrix = pd.DataFrame(values, index=zone_names, columns=zone_names)

    return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a sparse OD graph matrix from taxi trip parquet data."
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
        default=BASE_DIR / "edge_weight_matrix_od.csv",
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
        help="Optionally convert directed OD edges to an undirected matrix.",
    )
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
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace it: {args.output}")
    if args.top_k < 0:
        raise ValueError("--top-k must be >= 0")
    if args.min_flow < 1:
        raise ValueError("--min-flow must be >= 1")

    template_path = None if args.use_lookup_zones else args.template
    location_to_zone = _load_location_lookup(args.lookup)
    zone_names = _load_zone_names(args.lookup, template_path)

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
    print(f"Saved OD graph: {args.output}")
    print(f"Zones: {matrix.shape[0]}")
    print(f"Trips retained for counting: {kept_rows}")
    print(f"Nonzero edges: {edge_count} ({density:.4%} density)")
    print(f"Top-k per origin: {'all' if args.top_k == 0 else args.top_k}")
    print(f"Weight mode: {args.weight_mode}")
    print(f"Symmetrize: {args.symmetrize}")


if __name__ == "__main__":
    main()
