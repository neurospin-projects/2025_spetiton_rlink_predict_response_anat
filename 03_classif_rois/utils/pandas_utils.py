"""
safe_merging_toref: merge a new DataFrame into a reference DataFrame
with full pre-merge diagnostics and a controlled join strategy.

Conventions
-----------
- ref  : reference DataFrame (index = participant_id)
- new  : new DataFrame      (index = participant_id)
- The merged result preserves ref's row order; extra subjects from new
  are appended at the end.
- Missing subjects (in ref but not in new) produce NaN in the new columns.
"""

import pandas as pd


def merge_on_index(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: str = "inner",
    on=None,
    left_on=None,
    right_on=None,
) -> pd.DataFrame:
    """
    Merge two DataFrames on their index plus optional additional columns.

    The index is always included as a join key on both sides.
    `on` / `left_on` / `right_on` add extra column keys (same semantics as
    pd.DataFrame.merge).  The result is indexed by the left DataFrame's index.

    Parameters
    ----------
    left, right : DataFrames to merge
    how         : "inner" (default), "left", "right", or "outer"
    on          : column(s) present in both frames to include as extra keys
    left_on     : column(s) in left  to include as extra keys
    right_on    : column(s) in right to include as extra keys
    """
    def _as_list(x):
        if x is None:
            return []
        return [x] if isinstance(x, str) else list(x)

    left_idx_name = left.index.name or "index"
    right_idx_name = right.index.name or "index"

    left_reset = left.reset_index()
    right_reset = right.reset_index()

    if on is not None:
        extra = _as_list(on)
        left_keys = [left_idx_name] + extra
        right_keys = [right_idx_name] + extra
    else:
        left_keys = [left_idx_name] + _as_list(left_on)
        right_keys = [right_idx_name] + _as_list(right_on)

    merged = left_reset.merge(right_reset, left_on=left_keys, right_on=right_keys, how=how)

    if left_idx_name != right_idx_name and right_idx_name in merged.columns:
        merged = merged.drop(columns=[right_idx_name])

    return merged.set_index(left_idx_name)


def safe_merging_toref(
    ref: pd.DataFrame,
    new: pd.DataFrame,
    how: str = "outer",          # "outer" keeps everyone; change to "left" to drop new-only
    verbose: bool = True,
    add_flags: bool = True,        # Add boolean columns 'in_ref' and 'in_new' to indicate membership
) -> pd.DataFrame:
    """
    Merge `new` into `ref`, preserving ref's row order.
    Extra subjects from `new` are appended at the end.

    Parameters
    ----------
    ref     : reference DataFrame, index = participant_id
    new     : new DataFrame,       index = participant_id
    how     : "outer" (default) keeps all subjects; "left" drops new-only subjects
    verbose : print the diagnostic report
    add_flags : add boolean columns to indicate membership
    Returns
    -------
    merged  : DataFrame with ref's columns first, then new's columns,
              ref subjects in original order, extra subjects at the end.
              A boolean column `in_ref` and `in_new` flags membership.
    """

    ref_ids = set(ref.index)
    new_ids = set(new.index)

    common       = ref_ids & new_ids
    only_in_ref  = ref_ids - new_ids   # missing from new  → NaN in merged new-cols
    only_in_new  = new_ids - ref_ids   # extra in new      → appended at end

    # ── Diagnostic report ────────────────────────────────────────────────────
    if verbose:
        sep = "─" * 60
        print(sep)
        print("PRE-MERGE DIAGNOSTICS")
        print(sep)
        print(f"  ref  subjects          : {len(ref_ids):>6}")
        print(f"  new  subjects          : {len(new_ids):>6}")
        print(f"  Common (will merge)    : {len(common):>6}")
        print()

        if only_in_ref:
            print(f"  ⚠  In ref but NOT in new ({len(only_in_ref)}) "
                  f"→ new columns will be NaN:")
            for i, pid in enumerate(sorted(only_in_ref)):
                idx_pos = ref.index.get_loc(pid)
                print(f"       [{idx_pos:>4}]  {pid}")
        else:
            print("  ✓  No subjects missing from new.")
        print()

        if only_in_new:
            print(f"  ℹ  In new but NOT in ref ({len(only_in_new)}) "
                  f"→ will be appended at the end:")
            for pid in sorted(only_in_new):
                print(f"             {pid}")
        else:
            print("  ✓  No extra subjects in new.")
        print(sep)

    # ── Merge ────────────────────────────────────────────────────────────────
    # Avoid column name collisions in the suffixes (ref wins on overlapping names)
    merged = ref.merge(
        new,
        left_index=True,
        right_index=True,
        how=how,
        suffixes=("", "_new"),   # ref columns keep their names
    )

    # Add membership flags
    merged["in_ref"] = merged.index.isin(ref_ids)
    merged["in_new"] = merged.index.isin(new_ids)

    # ── Reorder rows: ref order first, then new-only appended ────────────────
    ref_order   = [pid for pid in ref.index   if pid in merged.index]
    extra_order = [pid for pid in sorted(only_in_new) if pid in merged.index]
    merged = merged.loc[ref_order + extra_order]

    if verbose:
        print(f"\nMerged DataFrame : {merged.shape[0]} rows × {merged.shape[1]} columns")
        n_missing = merged["in_new"].eq(False).sum()
        n_extra   = merged["in_ref"].eq(False).sum()
        if n_missing:
            print(f"  → {n_missing} row(s) have NaN in new columns (missing from new)")
        if n_extra:
            print(f"  → {n_extra} row(s) appended from new only")
        print()

    if not add_flags:
        merged.drop(columns=["in_ref", "in_new"], inplace=True)

    return merged


# ── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":

    import numpy as np

    # Simulate ref (CAT12, 5 subjects)
    ref = pd.DataFrame(
        {"age": [30, 45, 28, 52, 37], "sex": [1, 0, 1, 0, 1]},
        index=pd.Index(
            ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005"],
            name="participant_id",
        ),
    )

    # Simulate new (FreeSurfer, 4 common + 1 extra + 1 missing)
    new = pd.DataFrame(
        {
            "session":            ["M00"] * 5,
            "Left-Hippocampus":   [3800, 3950, 4100, 3700, 4200],
            "Right-Hippocampus":  [3850, 3900, 4050, 3750, 4150],
        },
        index=pd.Index(
            ["sub-001", "sub-002", "sub-003", "sub-005", "sub-006"],
            name="participant_id",
        ),
    )

    # Filter to M00 (mirrors your pipeline)
    new = new[new.session == "M00"]

    merged = safe_merging_toref(ref, new)
    print(merged.to_string())
