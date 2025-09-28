"""
Input/Output utilities for the ML network anomaly detection project.
"""

import pandas as pd
from pathlib import Path
from typing import Union
import warnings


def read_csv_any(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Read CSV file with robust error handling and encoding detection.

    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments to pass to pd.read_csv

    Returns:
        pd.DataFrame: Loaded dataframe

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If file cannot be read with any encoding
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Common encodings to try
    encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]

    # Default pandas read_csv arguments
    default_kwargs = {"on_bad_lines": "skip", "low_memory": False}

    # Update with user-provided kwargs
    default_kwargs.update(kwargs)

    for encoding in encodings:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = pd.read_csv(file_path, encoding=encoding, **default_kwargs)
                print(
                    f"Successfully read {file_path.name} with {encoding} encoding ({len(df):,} rows)"
                )
                return df
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            if encoding == encodings[-1]:  # Last encoding attempt
                raise Exception(
                    f"Could not read {file_path} with any encoding. Last error: {e}"
                )
            continue

    # Should never reach here, but just in case
    raise Exception(f"Unexpected error reading {file_path}")
