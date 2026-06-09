"""
Aggregate FreeSurfer subcortical segmentation volumes from aseg.stats files
into a single DataFrame.

Usage:
    python fetch_aseg_volumes.py

Output:
    aseg_volumes.tsv  — tab-separated file with columns:
    participant_id, session, <StructName_1>, <StructName_2>, ...
"""

import re
import glob
import pandas as pd
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_BASEDIRECTORY = "/neurospin/rlink/PUBLICATION/rlink-mri-anat"
GLOB_PATTERN = "derivatives/freesurfer-reconall-v7.1.1/sub-*/ses-*/stats/aseg.stats"
OUTPUT_FILE = "fs_aseg_volumes.tsv"
# ──────────────────────────────────────────────────────────────────────────────


def parse_aseg_stats(filepath: Path) -> dict:
    """
    Parse a single aseg.stats file and return a dict of {StructName: Volume_mm3}.
    Reads only the data table that follows the ColHeaders line.
    """
    volumes = {}
    in_table = False

    with open(filepath, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")

            # Detect the header line that precedes the data table
            if line.startswith("# ColHeaders"):
                in_table = True
                continue

            if not in_table:
                continue

            # Stop at blank lines or comment lines after the table starts
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                break

            # Data line: Index SegId NVoxels Volume_mm3 StructName ...
            parts = stripped.split()
            if len(parts) < 5:
                continue

            struct_name = parts[4]       # column 5 (0-indexed: 4)
            volume_mm3  = float(parts[3]) # column 4 (0-indexed: 3)
            volumes[struct_name] = volume_mm3

    return volumes


def extract_ids_from_path(filepath: Path) -> tuple[str, str]:
    """
    Extract participant_id and session from a path of the form:
    .../sub-<id>/ses-<session>/stats/aseg.stats
    Returns ("sub-<id>", "<session>")
    """
    parts = filepath.parts
    sub_part = next((p for p in parts if re.match(r"^sub-", p)), None)
    ses_part = next((p for p in parts if re.match(r"^ses-", p)), None)

    participant_id = sub_part if sub_part else "unknown"
    session = ses_part.replace("ses-", "") if ses_part else "unknown"
    return participant_id, session


def main():
    search_path = Path(INPUT_BASEDIRECTORY) / GLOB_PATTERN
    files = sorted(glob.glob(str(search_path)))

    if not files:
        print(f"[WARNING] No aseg.stats files found under:\n  {search_path}")
        return

    print(f"Found {len(files)} aseg.stats file(s). Parsing...")

    rows = []
    for fp in files:
        filepath = Path(fp)
        participant_id, session = extract_ids_from_path(filepath)
        volumes = parse_aseg_stats(filepath)

        if not volumes:
            print(f"  [SKIP] No data parsed from {filepath}")
            continue

        row = {"participant_id": participant_id, "session": session}
        row.update(volumes)
        rows.append(row)
        print(f"  [OK] {participant_id}  ses-{session}  ({len(volumes)} structures)")

    if not rows:
        print("[ERROR] No data collected. Check file paths and format.")
        return

    df = pd.DataFrame(rows)

    # Reorder: participant_id and session first, then structures alphabetically
    id_cols = ["participant_id", "session"]
    struct_cols = sorted([c for c in df.columns if c not in id_cols])
    df = df[id_cols + struct_cols]

    df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"\nSaved {len(df)} rows × {len(df.columns)} columns to: {OUTPUT_FILE}")
    print(df[id_cols + struct_cols[:5]].head())


if __name__ == "__main__":
    main()
