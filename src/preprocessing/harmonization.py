"""Utilities for harmonizing NSL-KDD and CIC-IDS-2017 feature spaces."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Canonical column definitions

COMMON_COLUMNS: List[str] = [
    "duration_ms",
    "protocol",
    "fwd_bytes",
    "bwd_bytes",
    "fwd_packets",
    "bwd_packets",
    "connection_state",
    "urgent_count",
    "connection_rate",
    "service_rate",
    "error_rate",
    "land",
    "flow_bytes_per_s",
    "label_binary",
    "label_multiclass",
]

AVERAGE_TCP_PACKET_SIZE = 576.0

# Connection state identifiers harmonized between datasets.
NSL_CONNECTION_STATE_MAP: Dict[str, int] = {
    "SF": 0,
    "S0": 1,
    "REJ": 2,
    "RSTO": 2,
    "RSTR": 2,
    "SH": 4,
    "S1": 5,
    "S2": 5,
    "S3": 5,
    "OTH": 6,
}

NSL_CANONICAL_COLUMNS: List[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

CIC_CANONICAL_NAME_BY_NORMALIZED: Dict[str, str] = {
    "flow_id": "Flow_ID",
    "source_ip": "Source_IP",
    "source_port": "Source_Port",
    "destination_ip": "Destination_IP",
    "destination_port": "Destination_Port",
    "protocol": "Protocol",
    "timestamp": "Timestamp",
    "flow_duration": "Flow_Duration",
    "total_fwd_packets": "Total_Fwd_Packets",
    "total_backward_packets": "Total_Backward_Packets",
    "total_length_of_fwd_packets": "Total_Length_of_Fwd_Packets",
    "total_length_of_bwd_packets": "Total_Length_of_Bwd_Packets",
    "total_fwd_bytes": "Total_Fwd_Bytes",
    "total_backward_bytes": "Total_Backward_Bytes",
    "fwd_packet_length_max": "Fwd_Packet_Length_Max",
    "fwd_packet_length_min": "Fwd_Packet_Length_Min",
    "fwd_packet_length_mean": "Fwd_Packet_Length_Mean",
    "fwd_packet_length_std": "Fwd_Packet_Length_Std",
    "bwd_packet_length_max": "Bwd_Packet_Length_Max",
    "bwd_packet_length_min": "Bwd_Packet_Length_Min",
    "bwd_packet_length_mean": "Bwd_Packet_Length_Mean",
    "bwd_packet_length_std": "Bwd_Packet_Length_Std",
    "flow_bytes_s": "Flow_Bytes/s",
    "flow_packets_s": "Flow_Packets/s",
    "flow_iat_mean": "Flow_IAT_Mean",
    "flow_iat_std": "Flow_IAT_Std",
    "flow_iat_max": "Flow_IAT_Max",
    "flow_iat_min": "Flow_IAT_Min",
    "fwd_iat_total": "Fwd_IAT_Total",
    "fwd_iat_mean": "Fwd_IAT_Mean",
    "fwd_iat_std": "Fwd_IAT_Std",
    "fwd_iat_max": "Fwd_IAT_Max",
    "fwd_iat_min": "Fwd_IAT_Min",
    "bwd_iat_total": "Bwd_IAT_Total",
    "bwd_iat_mean": "Bwd_IAT_Mean",
    "bwd_iat_std": "Bwd_IAT_Std",
    "bwd_iat_max": "Bwd_IAT_Max",
    "bwd_iat_min": "Bwd_IAT_Min",
    "fwd_psh_flags": "Fwd_PSH_Flags",
    "bwd_psh_flags": "Bwd_PSH_Flags",
    "fwd_urg_flags": "Fwd_URG_Flags",
    "bwd_urg_flags": "Bwd_URG_Flags",
    "fwd_header_length": "Fwd_Header_Length",
    "bwd_header_length": "Bwd_Header_Length",
    "fwd_packets_s": "Fwd_Packets/s",
    "bwd_packets_s": "Bwd_Packets/s",
    "min_packet_length": "Min_Packet_Length",
    "max_packet_length": "Max_Packet_Length",
    "packet_length_mean": "Packet_Length_Mean",
    "packet_length_std": "Packet_Length_Std",
    "packet_length_variance": "Packet_Length_Variance",
    "fin_flag_count": "FIN_Flag_Count",
    "syn_flag_count": "SYN_Flag_Count",
    "rst_flag_count": "RST_Flag_Count",
    "psh_flag_count": "PSH_Flag_Count",
    "ack_flag_count": "ACK_Flag_Count",
    "urg_flag_count": "URG_Flag_Count",
    "cwe_flag_count": "CWE_Flag_Count",
    "ece_flag_count": "ECE_Flag_Count",
    "down_up_ratio": "Down/Up_Ratio",
    "average_packet_size": "Average_Packet_Size",
    "avg_fwd_segment_size": "Avg_Fwd_Segment_Size",
    "avg_bwd_segment_size": "Avg_Bwd_Segment_Size",
    "fwd_header_length_1": "Fwd_Header_Length.1",
    "fwd_avg_bytes_bulk": "Fwd_Avg_Bytes/Bulk",
    "fwd_avg_packets_bulk": "Fwd_Avg_Packets/Bulk",
    "fwd_avg_bulk_rate": "Fwd_Avg_Bulk_Rate",
    "bwd_avg_bytes_bulk": "Bwd_Avg_Bytes/Bulk",
    "bwd_avg_packets_bulk": "Bwd_Avg_Packets/Bulk",
    "bwd_avg_bulk_rate": "Bwd_Avg_Bulk_Rate",
    "subflow_fwd_packets": "Subflow_Fwd_Packets",
    "subflow_fwd_bytes": "Subflow_Fwd_Bytes",
    "subflow_bwd_packets": "Subflow_Bwd_Packets",
    "subflow_bwd_bytes": "Subflow_Bwd_Bytes",
    "init_win_bytes_forward": "Init_Win_bytes_forward",
    "init_win_bytes_backward": "Init_Win_bytes_backward",
    "act_data_pkt_fwd": "act_data_pkt_fwd",
    "min_seg_size_forward": "min_seg_size_forward",
    "active_mean": "Active_Mean",
    "active_std": "Active_Std",
    "active_max": "Active_Max",
    "active_min": "Active_Min",
    "idle_mean": "Idle_Mean",
    "idle_std": "Idle_Std",
    "idle_max": "Idle_Max",
    "idle_min": "Idle_Min",
    "label": "Label",
}

CIC_CANONICAL_COLUMNS: List[str] = list(CIC_CANONICAL_NAME_BY_NORMALIZED.values())

UNION_COLUMNS: List[str] = (
    NSL_CANONICAL_COLUMNS
    + ["label_kdd", "difficulty"]
    + [col for col in CIC_CANONICAL_COLUMNS if col != "Label"]
    + ["Label", "label_cic", "label_binary", "label_multiclass"]
)

STRING_COLUMNS_UNION: Tuple[str, ...] = (
    "protocol_type",
    "service",
    "flag",
    "label_kdd",
    "Flow_ID",
    "Source_IP",
    "Destination_IP",
    "Timestamp",
    "Label",
    "label_cic",
    "label_multiclass",
)

PROTOCOL_MAP: Dict[int, str] = {6: "TCP", 17: "UDP", 1: "ICMP"}

NSL_FLAG_MAPPING: Dict[str, Dict[str, int]] = {
    "SF": {"syn": 1, "ack": 1, "fin": 0, "rst": 0},
    "S0": {"syn": 1, "ack": 0, "fin": 0, "rst": 0},
    "REJ": {"syn": 0, "ack": 0, "fin": 0, "rst": 1},
    "RSTO": {"syn": 0, "ack": 0, "fin": 0, "rst": 1},
    "RSTR": {"syn": 0, "ack": 0, "fin": 0, "rst": 1},
    "SH": {"syn": 1, "ack": 0, "fin": 1, "rst": 0},
    "S1": {"syn": 1, "ack": 1, "fin": 0, "rst": 0},
    "S2": {"syn": 1, "ack": 1, "fin": 0, "rst": 0},
    "S3": {"syn": 1, "ack": 1, "fin": 0, "rst": 0},
    "OTH": {"syn": 0, "ack": 0, "fin": 0, "rst": 0},
}

NSL_RAW_COLUMNS: List[str] = NSL_CANONICAL_COLUMNS + ["label", "difficulty"]


# ---------------------------------------------------------------------------
# Pydantic schemas (row-level validation)


def _validate_non_negative(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if value < 0:
        raise ValueError("value must be non-negative")
    return value


class CommonSchema(BaseModel):
    duration_ms: float = Field(..., ge=0)
    protocol: str
    fwd_bytes: int = Field(..., ge=0)
    bwd_bytes: int = Field(..., ge=0)
    fwd_packets: int = Field(..., ge=0)
    bwd_packets: int = Field(..., ge=0)
    connection_state: int = Field(..., ge=0)
    urgent_count: float = Field(..., ge=0)
    connection_rate: float = Field(..., ge=0)
    service_rate: float = Field(..., ge=0)
    error_rate: float = Field(..., ge=0)
    land: int = Field(..., ge=0)
    flow_bytes_per_s: float = Field(..., ge=0)
    label_binary: int = Field(..., ge=0, le=1)
    label_multiclass: str


class NSLSchema(BaseModel):
    duration: float = Field(..., ge=0)
    protocol_type: str
    service: str
    flag: str
    src_bytes: float = Field(..., ge=0)
    dst_bytes: float = Field(..., ge=0)
    land: float = Field(..., ge=0)
    wrong_fragment: float = Field(..., ge=0)
    urgent: float = Field(..., ge=0)
    hot: float = Field(..., ge=0)
    num_failed_logins: float = Field(..., ge=0)
    logged_in: float = Field(..., ge=0)
    num_compromised: float = Field(..., ge=0)
    root_shell: float = Field(..., ge=0)
    su_attempted: float = Field(..., ge=0)
    num_root: float = Field(..., ge=0)
    num_file_creations: float = Field(..., ge=0)
    num_shells: float = Field(..., ge=0)
    num_access_files: float = Field(..., ge=0)
    num_outbound_cmds: float = Field(..., ge=0)
    is_host_login: float = Field(..., ge=0)
    is_guest_login: float = Field(..., ge=0)
    count: float = Field(..., ge=0)
    srv_count: float = Field(..., ge=0)
    serror_rate: float = Field(..., ge=0)
    srv_serror_rate: float = Field(..., ge=0)
    rerror_rate: float = Field(..., ge=0)
    srv_rerror_rate: float = Field(..., ge=0)
    same_srv_rate: float = Field(..., ge=0)
    diff_srv_rate: float = Field(..., ge=0)
    srv_diff_host_rate: float = Field(..., ge=0)
    dst_host_count: float = Field(..., ge=0)
    dst_host_srv_count: float = Field(..., ge=0)
    dst_host_same_srv_rate: float = Field(..., ge=0)
    dst_host_diff_srv_rate: float = Field(..., ge=0)
    dst_host_same_src_port_rate: float = Field(..., ge=0)
    dst_host_srv_diff_host_rate: float = Field(..., ge=0)
    dst_host_serror_rate: float = Field(..., ge=0)
    dst_host_srv_serror_rate: float = Field(..., ge=0)
    dst_host_rerror_rate: float = Field(..., ge=0)
    dst_host_srv_rerror_rate: float = Field(..., ge=0)
    label_kdd: str

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


class CICSchema(BaseModel):
    Flow_Duration: float = Field(..., ge=0)
    Total_Fwd_Packets: float = Field(..., ge=0)
    Total_Backward_Packets: float = Field(..., ge=0)
    Total_Length_of_Fwd_Packets: Optional[float] = Field(default=None, ge=0)
    Total_Length_of_Bwd_Packets: Optional[float] = Field(default=None, ge=0)
    Flow_Bytes_per_s: Optional[float] = Field(default=None, ge=0, alias="Flow_Bytes/s")
    Flow_Packets_per_s: Optional[float] = Field(default=None, ge=0, alias="Flow_Packets/s")
    Label: str

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        extra = "allow"


class UnionSchema(BaseModel):
    label_binary: int = Field(..., ge=0, le=1)
    label_multiclass: str

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


# ---------------------------------------------------------------------------
# Helper utilities


def normalize_column_name(name: str) -> str:
    """Normalize a column name to snake_case for lookup purposes."""

    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_").lower()


def normalize_headers(columns: Iterable[str]) -> List[str]:
    """Normalize a sequence of headers without altering the original order."""

    return [normalize_column_name(col) for col in columns]


def read_csv_any(path: str | Path) -> pd.DataFrame:
    """Read a CSV file with automatic delimiter detection.

    Parameters
    ----------
    path:
        Path to the CSV file.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        sample = handle.read(4096)

    try:
        dialect = csv.Sniffer().sniff(sample)
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = None

    try:
        has_header = csv.Sniffer().has_header(sample)
    except csv.Error:
        header_keywords = ("label", "flow", "protocol", "duration", "packet")
        lower_sample = sample.lower()
        has_header = any(keyword in lower_sample for keyword in header_keywords)

    sep = delimiter if delimiter not in (None, "") else None
    engine = "c" if sep else "python"

    df = pd.read_csv(
        file_path,
        sep=sep,
        engine=engine,
        header=0 if has_header else None,
        encoding="utf-8",
        on_bad_lines="skip",
    )

    if not has_header:
        df.columns = [f"column_{idx}" for idx in range(df.shape[1])]

    return df


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a dataframe to Parquet format."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(file_path, index=False)
    except Exception as exc:  # pragma: no cover - depends on optional engines
        raise RuntimeError("Parquet support requires `pyarrow` or `fastparquet`.") from exc


def _value_or_none(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_with_model(
    df: pd.DataFrame, model: type[BaseModel], fields: Sequence[str]
) -> None:
    """Validate a dataframe using the provided Pydantic model."""

    if df.empty:
        return

    max_rows = min(len(df), 5)
    for _, row in df.iloc[:max_rows].iterrows():
        payload = {field: _value_or_none(row.get(field)) for field in fields}
        model(**payload)  # type: ignore[arg-type]


def map_protocols(series: pd.Series) -> pd.Series:
    """Map protocol codes to human readable values."""

    def _map(value: Any) -> str:
        if pd.isna(value):
            return "UNKNOWN"
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return "UNKNOWN"
            if stripped.isdigit():
                value_int = int(stripped)
                return PROTOCOL_MAP.get(value_int, str(value_int))
            return stripped.upper()
        if isinstance(value, (int, np.integer)):
            return PROTOCOL_MAP.get(int(value), str(int(value)))
        if isinstance(value, (float, np.floating)) and value.is_integer():
            return PROTOCOL_MAP.get(int(value), str(int(value)))
        return str(value)

    return series.apply(_map)


def derive_packet_counts_from_bytes(
    byte_series: pd.Series, average_size: float = AVERAGE_TCP_PACKET_SIZE
) -> pd.Series:
    """Approximate packet counts from byte counters using a TCP MSS estimate."""

    numeric = pd.to_numeric(byte_series, errors="coerce").fillna(0.0)
    counts = (numeric / max(average_size, 1.0)).round().clip(lower=0)
    return counts.astype(int)


def derive_throughput(bytes_total: pd.Series, duration_ms: pd.Series) -> pd.Series:
    """Compute bytes per second with zero-division protection."""

    duration_sec = duration_ms.astype(float) / 1000.0
    duration_safe = duration_sec.where(duration_sec > 0, other=1e-9)
    return bytes_total.astype(float) / duration_safe


def normalize_labels(labels: pd.Series) -> pd.Series:
    """Normalize textual labels to lowercase strings without trailing spaces."""

    return labels.astype(str).str.strip().str.lower()


def nsl_flag_to_tcp_counts(flag: str) -> Dict[str, int]:
    """Translate NSL-KDD symbolic flags into TCP flag counters."""

    normalized = str(flag).strip().upper()
    mapping = NSL_FLAG_MAPPING.get(normalized, {"syn": 0, "ack": 0, "fin": 0, "rst": 0})
    return dict(mapping)


def map_nsl_flags_to_state(flags: pd.Series) -> pd.Series:
    """Map NSL-KDD flag symbols to harmonized connection state identifiers."""

    def _map(flag: Any) -> int:
        normalized = str(flag).strip().upper()
        return NSL_CONNECTION_STATE_MAP.get(normalized, 6)

    return flags.apply(_map).astype(int)


def ensure_cic_tcp_flag_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure TCP flag counter columns exist for CIC-IDS-2017 data."""

    result = df.copy()
    for column in ("SYN_Flag_Count", "ACK_Flag_Count", "FIN_Flag_Count", "RST_Flag_Count"):
        if column not in result.columns:
            result[column] = 0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).astype(int)
    return result


def derive_cic_connection_state(df: pd.DataFrame) -> pd.Series:
    """Derive harmonized connection state identifiers from CIC TCP flags."""

    syn = pd.to_numeric(df.get("SYN_Flag_Count", 0), errors="coerce").fillna(0).astype(int)
    ack = pd.to_numeric(df.get("ACK_Flag_Count", 0), errors="coerce").fillna(0).astype(int)
    fin = pd.to_numeric(df.get("FIN_Flag_Count", 0), errors="coerce").fillna(0).astype(int)
    rst = pd.to_numeric(df.get("RST_Flag_Count", 0), errors="coerce").fillna(0).astype(int)

    state = pd.Series(0, index=df.index, dtype=int)
    state.loc[(syn > 0) & (ack == 0) & (rst == 0)] = 1
    state.loc[rst > 0] = 2
    state.loc[fin > 0] = 4
    state.loc[(syn == 0) & (ack == 0) & (fin == 0) & (rst == 0)] = 6
    return state


def derive_land_flag(df: pd.DataFrame) -> pd.Series:
    """Determine whether flows originate and terminate at the same endpoint."""

    if "Source_IP" in df.columns and "Destination_IP" in df.columns:
        source = df["Source_IP"].astype(str).str.strip()
        destination = df["Destination_IP"].astype(str).str.strip()
        match = (source != "") & (source == destination)
        return match.astype(int)
    return pd.Series([0] * len(df), index=df.index, dtype=int)


def _prepare_cic_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for column in df.columns:
        normalized = normalize_column_name(column)
        canonical = CIC_CANONICAL_NAME_BY_NORMALIZED.get(normalized)
        if canonical:
            rename_map[column] = canonical
    renamed = df.rename(columns=rename_map)

    for canonical in CIC_CANONICAL_COLUMNS:
        if canonical not in renamed.columns:
            renamed[canonical] = np.nan

    return renamed


def _prepare_nsl_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    renamed.columns = NSL_RAW_COLUMNS[: len(renamed.columns)]
    return renamed


def _assemble_union_dataframe(
    base: pd.DataFrame, mapping: Dict[str, pd.Series | np.ndarray | Any]
) -> pd.DataFrame:
    data: Dict[str, Any] = {}
    for column in UNION_COLUMNS:
        if column in mapping:
            data[column] = mapping[column]
        elif column in STRING_COLUMNS_UNION:
            data[column] = pd.Series(["" for _ in base.index], index=base.index)
        else:
            data[column] = pd.Series([np.nan for _ in base.index], index=base.index)
    return pd.DataFrame(data, index=base.index)


def _compute_summary(df: pd.DataFrame) -> Dict[str, Any]:
    nan_share = df.isna().mean().sort_values(ascending=False)
    top_nan = nan_share.head(20).to_dict()

    numeric = df.select_dtypes(include=["number"])
    numeric_stats = {
        column: {
            "min": None if numeric[column].dropna().empty else float(numeric[column].min()),
            "max": None if numeric[column].dropna().empty else float(numeric[column].max()),
        }
        for column in numeric.columns
    }

    return {
        "rows": int(df.shape[0]),
        "nan_share_top20": top_nan,
        "numeric_min_max": numeric_stats,
        "schema_version": SCHEMA_VERSION,
    }


# ---------------------------------------------------------------------------
# Transformation functions


def to_common_from_nsl(df: pd.DataFrame) -> pd.DataFrame:
    """Transform NSL-KDD dataframe to the common subset schema."""

    base = df.copy()
    duration_sec = pd.to_numeric(base["duration"], errors="coerce").fillna(0.0)
    duration_ms = duration_sec * 1000.0
    protocol = map_protocols(base["protocol_type"])

    fwd_bytes_series = pd.to_numeric(base["src_bytes"], errors="coerce").fillna(0.0)
    bwd_bytes_series = pd.to_numeric(base["dst_bytes"], errors="coerce").fillna(0.0)
    fwd_bytes = fwd_bytes_series.astype(int)
    bwd_bytes = bwd_bytes_series.astype(int)

    fwd_packets = derive_packet_counts_from_bytes(fwd_bytes_series)
    bwd_packets = derive_packet_counts_from_bytes(bwd_bytes_series)

    total_bytes = fwd_bytes_series + bwd_bytes_series
    flow_bytes_per_s = derive_throughput(total_bytes, duration_ms)

    connection_state = map_nsl_flags_to_state(base["flag"])
    urgent_count = pd.to_numeric(base["urgent"], errors="coerce").fillna(0.0)

    duration_safe = duration_sec + 1e-3
    count_series = pd.to_numeric(base["count"], errors="coerce").fillna(0.0)
    srv_count_series = pd.to_numeric(base["srv_count"], errors="coerce").fillna(0.0)
    connection_rate = (count_series / duration_safe).replace([np.inf, -np.inf], 0.0)
    service_rate = (srv_count_series / duration_safe).replace([np.inf, -np.inf], 0.0)

    error_rate = pd.to_numeric(base["serror_rate"], errors="coerce").fillna(0.0)
    land = pd.to_numeric(base["land"], errors="coerce").fillna(0).astype(int)

    label_multiclass = normalize_labels(base["label_kdd"])
    label_binary = (label_multiclass != "normal").astype(np.int8)

    common = pd.DataFrame(
        {
            "duration_ms": duration_ms.astype(float),
            "protocol": protocol.astype(str),
            "fwd_bytes": fwd_bytes.astype(int),
            "bwd_bytes": bwd_bytes.astype(int),
            "fwd_packets": fwd_packets.astype(int),
            "bwd_packets": bwd_packets.astype(int),
            "connection_state": connection_state.astype(int),
            "urgent_count": urgent_count.astype(float),
            "connection_rate": connection_rate.astype(float),
            "service_rate": service_rate.astype(float),
            "error_rate": error_rate.astype(float),
            "land": land.astype(int),
            "flow_bytes_per_s": flow_bytes_per_s.astype(float),
            "label_binary": label_binary,
            "label_multiclass": label_multiclass,
        }
    )

    common = common.loc[:, COMMON_COLUMNS]
    validate_common(common)
    return common


def _select_numeric_column(
    df: pd.DataFrame, candidates: Sequence[str]
) -> Tuple[str, pd.Series]:
    for candidate in candidates:
        if candidate in df.columns:
            series = pd.to_numeric(df[candidate], errors="coerce")
            if series.notna().any():
                return candidate, series
    for candidate in candidates:
        if candidate in df.columns:
            return candidate, pd.to_numeric(df[candidate], errors="coerce")
    raise KeyError(f"None of the candidate columns {candidates} are present")


def to_common_from_cic(df: pd.DataFrame) -> pd.DataFrame:
    """Transform CIC-IDS-2017 dataframe to the common subset schema."""

    base = df.copy()
    base = ensure_cic_tcp_flag_cols(base)

    duration_ms = pd.to_numeric(base["Flow_Duration"], errors="coerce").fillna(0.0)

    _, fwd_series = _select_numeric_column(
        base, ("Total_Fwd_Bytes", "Total_Length_of_Fwd_Packets")
    )
    _, bwd_series = _select_numeric_column(
        base, ("Total_Backward_Bytes", "Total_Length_of_Bwd_Packets")
    )

    fwd_bytes_series = fwd_series.fillna(0.0)
    bwd_bytes_series = bwd_series.fillna(0.0)

    fwd_bytes = fwd_bytes_series.astype(int)
    bwd_bytes = bwd_bytes_series.astype(int)

    fwd_packets = pd.to_numeric(base["Total_Fwd_Packets"], errors="coerce").fillna(0).astype(int)
    bwd_packets = pd.to_numeric(base["Total_Backward_Packets"], errors="coerce").fillna(0).astype(int)

    flow_bytes = base.get("Flow_Bytes/s", np.nan)
    flow_bytes = pd.to_numeric(flow_bytes, errors="coerce")
    derived_flow_bytes = derive_throughput(
        fwd_bytes_series.astype(float) + bwd_bytes_series.astype(float), duration_ms
    )
    flow_bytes_per_s = flow_bytes.fillna(derived_flow_bytes)

    protocol_column = base["Protocol"] if "Protocol" in base.columns else pd.Series(
        ["UNKNOWN"] * len(base), index=base.index
    )
    protocol = map_protocols(protocol_column)

    connection_state = derive_cic_connection_state(base)
    urgent_count = pd.to_numeric(base.get("URG_Flag_Count", 0), errors="coerce").fillna(0.0)
    connection_rate = pd.to_numeric(base.get("Flow_Packets/s", 0), errors="coerce").fillna(0.0)
    service_rate = pd.to_numeric(base.get("Fwd_Packets/s", 0), errors="coerce").fillna(0.0)

    total_packets = (fwd_packets + bwd_packets).astype(float)
    rst_counts = pd.to_numeric(base["RST_Flag_Count"], errors="coerce").fillna(0.0)
    total_packets_safe = total_packets.replace(0, np.nan)
    error_rate = (rst_counts / total_packets_safe).fillna(0.0)

    land = derive_land_flag(base)

    label_series = base["Label"] if "Label" in base.columns else pd.Series(
        ["unknown"] * len(base), index=base.index
    )
    normalized_labels = normalize_labels(label_series)
    label_binary = (~normalized_labels.str.contains("benign", case=False)).astype(np.int8)

    common = pd.DataFrame(
        {
            "duration_ms": duration_ms.astype(float),
            "protocol": protocol.astype(str),
            "fwd_bytes": fwd_bytes.astype(int),
            "bwd_bytes": bwd_bytes.astype(int),
            "fwd_packets": fwd_packets.astype(int),
            "bwd_packets": bwd_packets.astype(int),
            "connection_state": connection_state.astype(int),
            "urgent_count": urgent_count.astype(float),
            "connection_rate": connection_rate.astype(float),
            "service_rate": service_rate.astype(float),
            "error_rate": error_rate.astype(float),
            "land": land.astype(int),
            "flow_bytes_per_s": flow_bytes_per_s.astype(float),
            "label_binary": label_binary,
            "label_multiclass": normalized_labels,
        }
    )

    common = common.loc[:, COMMON_COLUMNS]
    validate_common(common)
    return common


def to_union_from_nsl(df: pd.DataFrame) -> pd.DataFrame:
    """Create the union schema dataframe for NSL-KDD data."""

    base = df.copy()
    mapping: Dict[str, pd.Series] = {column: base[column] for column in NSL_CANONICAL_COLUMNS}
    mapping["label_kdd"] = base["label_kdd"]
    mapping["difficulty"] = base["difficulty"] if "difficulty" in base.columns else ""

    label_binary = (normalize_labels(base["label_kdd"]) != "normal").astype(np.int8)
    label_multiclass = normalize_labels(base["label_kdd"])

    mapping["label_binary"] = label_binary
    mapping["label_multiclass"] = label_multiclass
    mapping["label_cic"] = pd.Series(["" for _ in base.index], index=base.index)
    mapping["Label"] = mapping["label_cic"]

    union_df = _assemble_union_dataframe(base, mapping)
    validate_union(union_df)
    return union_df


def to_union_from_cic(df: pd.DataFrame) -> pd.DataFrame:
    """Create the union schema dataframe for CIC-IDS-2017 data."""

    base = df.copy()
    mapping: Dict[str, pd.Series] = {}
    for column in CIC_CANONICAL_COLUMNS:
        mapping[column] = base[column] if column in base.columns else np.nan

    if "Label" not in mapping:
        raise KeyError("CIC dataset must contain a 'Label' column")

    label_series = base["Label"].astype(str)
    label_binary = (~normalize_labels(label_series).str.contains("benign")).astype(np.int8)
    label_multiclass = normalize_labels(label_series)

    mapping["label_cic"] = label_series
    mapping["label_binary"] = label_binary
    mapping["label_multiclass"] = label_multiclass
    mapping["label_kdd"] = pd.Series(["" for _ in base.index], index=base.index)
    mapping["difficulty"] = pd.Series(["" for _ in base.index], index=base.index)

    union_df = _assemble_union_dataframe(base, mapping)
    validate_union(union_df)
    return union_df


# ---------------------------------------------------------------------------
# Validation helpers


def validate_common(df: pd.DataFrame) -> None:
    missing = [column for column in COMMON_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing common columns: {missing}")

    if list(df.columns) != COMMON_COLUMNS:
        raise ValueError("Common dataframe must preserve the prescribed column order")

    _validate_with_model(df, CommonSchema, COMMON_COLUMNS)


def validate_union(df: pd.DataFrame) -> None:
    missing = [column for column in UNION_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing union columns: {missing}")

    _validate_with_model(df, UnionSchema, ["label_binary", "label_multiclass"])


def validate_nsl(df: pd.DataFrame) -> None:
    fields = NSL_CANONICAL_COLUMNS + ["label_kdd"]
    _validate_with_model(df, NSLSchema, fields)


def validate_cic(df: pd.DataFrame) -> None:
    fields = ["Flow_Duration", "Total_Fwd_Packets", "Total_Backward_Packets", "Label"]
    _validate_with_model(df, CICSchema, fields)


# ---------------------------------------------------------------------------
# Harmonization pipelines


def harmonize_nsl(path_csv: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load and harmonize NSL-KDD data into union and common schemas."""

    raw = read_csv_any(path_csv)
    prepared = _prepare_nsl_columns(raw)
    prepared.rename(columns={"label": "label_kdd"}, inplace=True)
    validate_nsl(prepared)

    union_df = to_union_from_nsl(prepared)
    common_df = to_common_from_nsl(prepared)
    summary = _compute_summary(union_df)
    return union_df, common_df, summary


def harmonize_cic(path_csv: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load and harmonize CIC-IDS-2017 data into union and common schemas."""

    raw = read_csv_any(path_csv)
    prepared = _prepare_cic_columns(raw)
    validate_cic(prepared)

    union_df = to_union_from_cic(prepared)
    common_df = to_common_from_cic(prepared)
    summary = _compute_summary(union_df)
    return union_df, common_df, summary


@dataclass(frozen=True)
class HarmonizationResult:
    """Convenience container for harmonization outputs."""

    union: pd.DataFrame
    common: pd.DataFrame
    summary: Dict[str, Any]

