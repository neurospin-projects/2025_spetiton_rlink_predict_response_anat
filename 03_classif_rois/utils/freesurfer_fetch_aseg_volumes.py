"""
Aggregate FreeSurfer subcortical segmentation volumes from aseg.stats files
into a single DataFrame, including global measures from the header.

Output:
    aseg_volumes.tsv  — tab-separated file with columns:
    participant_id, session, <GlobalMeasure_1>, ..., <StructName_1>, ...
"""

import re
import glob
import pandas as pd
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_BASEDIRECTORY = "/neurospin/rlink/PUBLICATION/rlink-mri-anat"
GLOB_PATTERN = "derivatives/freesurfer-reconall-v7.1.1/sub-*/ses-*/stats/aseg.stats"
OUTPUT_FILE = "aseg_volumes.tsv"
# ──────────────────────────────────────────────────────────────────────────────


def parse_aseg_stats(filepath: Path) -> tuple[dict, dict]:
    """
    Parse a single aseg.stats file.

    Returns
    -------
    global_measures : dict  {ShortName: float}
        Parsed from header lines of the form:
        # Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1148821.0, mm^3
    volumes : dict  {StructName: Volume_mm3}
        Parsed from the data table following # ColHeaders.
    """
    global_measures = {}
    volumes = {}
    in_table = False

    # Regex for global measure lines
    # # Measure <ShortName>, <LongName>, <Description>, <Value>, <Unit>
    measure_re = re.compile(
        r"^#\s+Measure\s+(\w+),\s+\w+,\s+[^,]+,\s+([\d.]+),"
    )

    with open(filepath, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")

            # ── Global measures from header ───────────────────────────────
            m = measure_re.match(line)
            if m:
                short_name = m.group(1)   # e.g. BrainSeg, eTIV
                value      = float(m.group(2))
                global_measures[short_name] = value
                continue

            # ── Data table ────────────────────────────────────────────────
            if line.startswith("# ColHeaders"):
                in_table = True
                continue

            if not in_table:
                continue

            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                break

            parts = stripped.split()
            if len(parts) < 5:
                continue

            struct_name = parts[4]        # e.g. Left-Hippocampus
            volume_mm3  = float(parts[3]) # Volume_mm3
            volumes[struct_name] = volume_mm3

    return global_measures, volumes


def extract_ids_from_path(filepath: Path) -> tuple[str, str]:
    """
    Extract participant_id and session from path:
    .../sub-<id>/ses-<session>/stats/aseg.stats
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
        global_measures, volumes = parse_aseg_stats(filepath)

        if not volumes:
            print(f"  [SKIP] No data parsed from {filepath}")
            continue

        row = {"participant_id": participant_id, "session": session}
        row.update(global_measures)   # global measures first
        row.update(volumes)           # then per-structure volumes
        rows.append(row)
        print(f"  [OK] {participant_id}  ses-{session}  "
              f"({len(global_measures)} global measures, {len(volumes)} structures)")

    if not rows:
        print("[ERROR] No data collected. Check file paths and format.")
        return

    df = pd.DataFrame(rows)

    # Reorder: ids | global measures | structures (both alphabetical)
    id_cols      = ["participant_id", "session"]
    global_cols  = sorted([c for c in df.columns
                            if c not in id_cols and not c[0].isupper() or
                            c in global_measures])
    # Safer: separate by whether they were global measure keys
    all_global   = sorted(set().union(*[list(parse_aseg_stats(Path(f))[0].keys())
                                        for f in files]))
    struct_cols  = sorted([c for c in df.columns
                            if c not in id_cols and c not in all_global])
    global_cols  = sorted([c for c in df.columns
                            if c not in id_cols and c not in struct_cols])

    df = df[id_cols + global_cols + struct_cols]

    df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"\nSaved {len(df)} rows × {len(df.columns)} columns → {OUTPUT_FILE}")
    print(f"  Global measures : {len(global_cols)}")
    print(f"  Structures      : {len(struct_cols)}")
    print(df[id_cols + global_cols[:3] + struct_cols[:3]].head())


if __name__ == "__main__":
    main()
