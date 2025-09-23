"""Unit tests for harmonization utilities."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.harmonization import (
    COMMON_COLUMNS,
    CIC_CANONICAL_COLUMNS,
    NSL_CANONICAL_COLUMNS,
    map_protocols,
    nsl_flag_to_tcp_counts,
    to_common_from_cic,
    to_common_from_nsl,
    to_union_from_cic,
    to_union_from_nsl,
    validate_common,
)


def _make_nsl_sample() -> pd.DataFrame:
    rows: Dict[str, list] = {column: [0.0, 0.0] for column in NSL_CANONICAL_COLUMNS}
    rows.update(
        {
            "duration": [1.0, 2.5],
            "protocol_type": ["tcp", "udp"],
            "service": ["http", "domain"],
            "flag": ["SF", "REJ"],
            "src_bytes": [100, 50],
            "dst_bytes": [25, 10],
        }
    )
    df = pd.DataFrame(rows)
    df["label_kdd"] = pd.Series(["normal", "neptune"])
    df["difficulty"] = pd.Series([10, 20])
    return df


def _make_cic_sample() -> pd.DataFrame:
    data = {
        "Flow_Duration": [1000.0, 2000.0],
        "Total_Fwd_Packets": [10, 5],
        "Total_Backward_Packets": [4, 1],
        "Total_Length_of_Fwd_Packets": [400, 200],
        "Total_Length_of_Bwd_Packets": [120, 30],
        "SYN_Flag_Count": [1, 0],
        "ACK_Flag_Count": [1, 0],
        "FIN_Flag_Count": [0, 0],
        "RST_Flag_Count": [0, 1],
        "Flow_Bytes/s": [520.0, np.nan],
        "Label": ["BENIGN", "DoS Hulk"],
        "Protocol": [6, "17"],
    }
    df = pd.DataFrame(data)
    for column in CIC_CANONICAL_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
    return df


def test_nsl_flag_to_tcp_counts() -> None:
    assert nsl_flag_to_tcp_counts("SF") == {"syn": 1, "ack": 1, "fin": 0, "rst": 0}
    assert nsl_flag_to_tcp_counts("S0")["ack"] == 0
    assert nsl_flag_to_tcp_counts("unknown") == {"syn": 0, "ack": 0, "fin": 0, "rst": 0}


def test_to_common_from_nsl() -> None:
    df = _make_nsl_sample()
    common = to_common_from_nsl(df)
    assert list(common.columns) == COMMON_COLUMNS
    assert common.loc[0, "duration_ms"] == pytest.approx(1000.0)
    assert common.loc[1, "tcp_flag_rst"] == 1
    assert common.loc[0, "label_binary"] == 0
    assert common.loc[1, "label_binary"] == 1
    assert common["fwd_pkts"].isna().all()
    validate_common(common)


def test_to_common_from_cic() -> None:
    cic = _make_cic_sample()
    common = to_common_from_cic(cic)
    assert list(common.columns) == COMMON_COLUMNS
    assert common.loc[0, "protocol"] == "TCP"
    assert common.loc[1, "protocol"] == "UDP"
    assert common.loc[0, "total_pkts"] == 14
    assert common.loc[1, "flow_bytes_per_s"] > 0
    assert set(common["label_binary"]) == {0, 1}


def test_to_union_from_nsl_preserves_labels() -> None:
    df = _make_nsl_sample()
    union = to_union_from_nsl(df)
    assert "label_kdd" in union.columns
    assert "label_cic" in union.columns
    assert union.loc[0, "label_multiclass"] == "normal"
    assert union.loc[1, "label_binary"] == 1


def test_to_union_from_cic_alignment() -> None:
    df = _make_cic_sample()
    union = to_union_from_cic(df)
    assert "Label" in union.columns
    assert union.loc[0, "label_kdd"] == ""
    assert union.loc[1, "label_binary"] == 1
    assert union.loc[0, "label_multiclass"] == "benign"


def test_map_protocols_handles_numeric() -> None:
    series = pd.Series([6, 17, 1, 99, "udp", " "])
    mapped = map_protocols(series)
    assert mapped.iloc[0] == "TCP"
    assert mapped.iloc[1] == "UDP"
    assert mapped.iloc[2] == "ICMP"
    assert mapped.iloc[3] == "99"
    assert mapped.iloc[4] == "UDP"
    assert mapped.iloc[5] == "UNKNOWN"
