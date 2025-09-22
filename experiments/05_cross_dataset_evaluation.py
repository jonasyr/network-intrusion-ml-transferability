#!/usr/bin/env python3
"""Run the full cross-dataset evaluation pipeline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, Tuple

current_dir = Path(__file__).parent


def run_script(script: Path, description: str) -> bool:
    print("=" * 80)
    print(f"🧪 {description}")
    print("=" * 80)
    try:
        subprocess.run([sys.executable, str(script)], check=True)
        return True
    except subprocess.CalledProcessError as exc:  # pragma: no cover - command failure
        print(f"❌ {description} failed: {exc}")
        return False


def main() -> bool:
    scripts: Iterable[Tuple[Path, str]] = [
        (current_dir / "04_cross_dataset_nsl_to_cic.py", "NSL-KDD → CIC-IDS-2017"),
        (current_dir / "05_cross_dataset_cic_to_nsl.py", "CIC-IDS-2017 → NSL-KDD"),
        (current_dir / "06_bidirectional_analysis.py", "Bidirectional summary"),
    ]

    success = True
    for script_path, description in scripts:
        if not script_path.exists():
            print(f"⚠️ Missing script: {script_path}")
            success = False
            continue
        success = run_script(script_path, description) and success

    if success:
        print("\n🎯 Cross-dataset pipeline complete!")
    else:
        print("\n⚠️ Cross-dataset pipeline finished with errors.")

    return success


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
