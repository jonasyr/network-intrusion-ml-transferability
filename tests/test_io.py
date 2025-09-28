"""Tests for I/O helpers used in harmonization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing.harmonization import (
    normalize_headers,
    read_csv_any,
    write_parquet,
)


def test_normalize_headers_removes_extra_spaces() -> None:
    headers = [" Flow Duration ", "Protocol", "Total  Fwd Packets"]
    normalized = normalize_headers(headers)
    assert normalized == ["flow_duration", "protocol", "total_fwd_packets"]


def test_read_csv_any_detects_header(tmp_path: Path) -> None:
    path = tmp_path / "with_header.csv"
    path.write_text("col_a,col_b\n1,2\n", encoding="utf-8")
    df = read_csv_any(path)
    assert list(df.columns) == ["col_a", "col_b"]
    assert df.shape == (1, 2)


def test_read_csv_any_without_header(tmp_path: Path) -> None:
    path = tmp_path / "no_header.csv"
    path.write_text("1,2\n3,4\n", encoding="utf-8")
    df = read_csv_any(path)
    assert list(df.columns) == ["column_0", "column_1"]
    assert df.iloc[0, 0] == 1


def test_write_parquet_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "sample.parquet"
    write_parquet(df, path)
    loaded = pd.read_parquet(path)
    pd.testing.assert_frame_equal(df, loaded)
