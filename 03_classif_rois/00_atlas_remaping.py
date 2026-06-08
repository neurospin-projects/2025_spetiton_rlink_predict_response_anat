"""
Remap v-4 ROI column names to Neuromorphometrics-aligned v-5 names.

v-4 columns use mixed, inconsistent naming (e.g. 'Right ACgG anterior
cingulate gyrus_GM_Vol').  This script standardises them to the
Neuromorphometrics atlas convention (e.g. 'Right Anterior Cingulate
Gyrus_GM_Vol') and produces three outputs:

  OUTPUT_MAPPING : CSV with the full feature_src → feature_dst mapping table
  OUTPUT_DATA    : v-5 CSV identical to v-4 except feature columns are renamed
  OUTPUT_MD      : Markdown changelog documenting this renaming

Parsing rules (feature_src → feature_dst)
------------------------------------------
1. Non-feature columns (participant_id, age, sex, site, y) are kept as-is.
2. Modality suffix (_GM_Vol, _CSF_Vol, _WM_Vol) is stripped, then
   re-appended unchanged.
3. Leading 'Right'/'Left' is extracted as laterality; absent → 'bilateral'.
4. Prefix rule: if the remainder matches '<ABBREV> <lowercase name>'
   (e.g. 'ACgG anterior cingulate gyrus'), only the descriptive part is
   title-cased ('Anterior Cingulate Gyrus'); the abbreviation is recorded
   but dropped from the destination name.
5. Otherwise the remainder is title-cased directly.
6. 22 manual corrections align ambiguous names with Neuromorphometrics_ROIname
   (e.g. 'Accumbens Area' → 'Accumbens', '3rd Ventricle' → 'Third Ventricle').
7. Neuromorphometrics abbreviations (lAntCinGy / rAntCinGy) are looked up
   from data/atlases_mapping_to_canonical.csv to populate roi_abr_dst and
   feature_abr_dst.

Mapping table columns
---------------------
  feature_src     original column name in v-4
  side            Right | Left | bilateral
  abbrev          atlas abbreviation prefix if present (e.g. ACgG), else ''
  roi_src         raw ROI string extracted from feature_src (no capitalisation)
  roi_dst         Neuromorphometrics_ROIname
  feature_dst     destination column name in v-5
  roi_abr_dst     side-specific Neuromorphometrics abbreviation (e.g. rAntCinGy)
  feature_abr_dst roi_abr_dst + '_' + modality
  modality        GM_Vol | CSF_Vol | WM_Vol
"""
import re
import pandas as pd
from config import config

INPUT_DATA = './data/processed/roi-cat12vbm/study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_v-4.csv'

OUTPUT_DIR = './data/processed/roi-cat12vbm/'
OUTPUT_MAPPING = OUTPUT_DIR + "study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_mapping-v-4-to-v-5.csv"
OUTPUT_DATA = OUTPUT_DIR + 'study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_v-5.csv'
OUTPUT_MD = OUTPUT_DIR + 'study-rlink_mod-cat12vbm_type-roi+age+sex+site_lab-M00_v-5.md'

################################################################################
# %% Load data and atlas
data  = pd.read_csv(config['input_data'])
atlas = pd.read_csv("data/atlases_mapping_to_canonical.csv")

META_COLS = {"participant_id", "age", "sex", "site", "y"}
feat_cols = [c for c in data.columns if c not in META_COLS]

neuro_names = set(atlas["Neuromorphometrics_ROIname"].dropna().unique())

################################################################################
# %% Regex patterns
suffix_re = re.compile(r'_(GM|CSF|WM)_Vol$')
side_re   = re.compile(r'^(Right|Left) ')
prefix_re = re.compile(r'^([A-Z][A-Za-z]+) ([a-z].+)$')

################################################################################
# %% Manual corrections: auto-derived roi_name -> Neuromorphometrics_ROIname
CORRECTIONS = {
    "3Rd Ventricle":                                 "Third Ventricle",
    "4Th Ventricle":                                 "Fourth Ventricle",
    "Accumbens Area":                                "Accumbens",
    "Basal Forebrain":                               "Basal Cerebrum and Forebrain Brain",
    "Brain Stem":                                    "Brainstem",
    "Calcarine Cortex":                              "Calcarine and Cerebrum",
    "Cerebellum Exterior":                           "Exterior Cerebellum",
    "Csf":                                           "CSF",
    "Inf Lat Vent":                                  "Inferior Lateral Ventricle",
    "Medial Frontal Cortex":                         "Medial Frontal Cerebrum",
    "Opercular Part Of The Inferior Frontal Gyrus":  "Inferior Frontal Gyrus",
    "Orbital Part Of The Inferior Frontal Gyrus":    "Inferior Frontal Orbital Gyrus",
    "Parahippocampal Gyrus":                         "Parahippocampus Gyrus",
    "Planum Temporale":                              "Temporal (Planum temporale)",
    "Postcentral Gyrus Medial Segment":              "Medial Postcentral Gyrus",
    "Precentral Gyrus Medial Segment":               "Medial Precentral Gyrus",
    "Superior Frontal Gyrus Medial Segment":         "Superior Medial Frontal Gyrus",
    "Supplementary Motor Cortex":                    "Cerebrum and Motor (supplementary motor)",
    "Transverse Temporal Gyrus":                     "Temporal Transverse Gyrus",
    "Triangular Part Of The Inferior Frontal Gyrus": "Inferior Frontal Angular Gyrus",
    "Ventral Dc":                                    "Ventral Ventricle (Ventral DC)",
    "Cerebellar Vermal Lobules Vi-Vii":              "Cerebellar Vermal Lobules VI-VII",
    "Cerebellar Vermal Lobules Viii-X":              "Cerebellar Vermal Lobules VIII-X",
}

################################################################################
# %% Build mapping
rows = []
for feat in feat_cols:
    m_suf    = suffix_re.search(feat)
    modality = m_suf.group(1) + "_Vol" if m_suf else ""
    base     = suffix_re.sub("", feat)

    m_side       = side_re.match(base)
    side         = m_side.group(1) if m_side else "bilateral"
    roi_src      = side_re.sub("", base)     # raw: "ACgG anterior cingulate gyrus", "Accumbens Area"

    m_prefix = prefix_re.match(roi_src)
    abbrev   = m_prefix.group(1) if m_prefix else ""
    # title-case only the descriptive part so abbreviations are not mangled
    title_key = m_prefix.group(2).title() if m_prefix else roi_src.title()

    roi_dst = CORRECTIONS.get(title_key, title_key)
    feature_dst = (f"{side} " if side != "bilateral" else "") + f"{roi_dst}_{modality}"

    rows.append(dict(feature_src=feat, side=side, abbrev=abbrev,
                     roi_src=roi_src, roi_dst=roi_dst,
                     feature_dst=feature_dst, modality=modality))

mapping_df = pd.DataFrame(rows)

################################################################################
# %% Build roi_abr_dst and feature_abr_dst from Neuromorphometrics_ROIabbr
abbr_lookup = (
    atlas[["Neuromorphometrics_ROIname", "Neuromorphometrics_ROIabbr"]]
    .dropna()
    .drop_duplicates("Neuromorphometrics_ROIname")
    .set_index("Neuromorphometrics_ROIname")["Neuromorphometrics_ROIabbr"]
)

def pick_abbr(roi_dst, side):
    raw = abbr_lookup.get(roi_dst, "")
    if not raw or "/" not in raw:
        return raw
    left_abbr, right_abbr = [s.strip() for s in raw.split(" / ")]
    if side == "Left":
        return left_abbr
    if side == "Right":
        return right_abbr
    # bilateral: strip leading 'l' to get the base abbreviation
    return left_abbr[1:] if left_abbr.startswith("l") else left_abbr

mapping_df["roi_abr_dst"]     = mapping_df.apply(lambda r: pick_abbr(r["roi_dst"], r["side"]), axis=1)
mapping_df["feature_abr_dst"] = mapping_df["roi_abr_dst"] + "_" + mapping_df["modality"]

################################################################################
# %% Check matching
matched   = mapping_df["roi_dst"].isin(neuro_names)
n_unique  = mapping_df["roi_dst"].nunique()
n_matched = mapping_df.loc[matched, "roi_dst"].nunique()

print(mapping_df[["feature_src", "roi_src", "roi_dst", "feature_dst"]].to_string(index=False))
print(f"\n{len(mapping_df)} features → {n_unique} unique ROIs → {n_matched} matched in Neuromorphometrics")

unmatched = mapping_df.loc[~matched, "roi_dst"].unique()
if len(unmatched):
    print("\nSTILL UNMATCHED:")
    for r in sorted(unmatched):
        print(f"  {r!r}")

################################################################################
# %% Save mapping
mapping_df.to_csv(OUTPUT_MAPPING, index=False)
print(f"\nSaved: {OUTPUT_MAPPING}")

################################################################################
# %% Rename feature columns in data and save OUTPUT_DATA
rename_map = mapping_df.set_index("feature_src")["feature_dst"].to_dict()

data_out = pd.read_csv(INPUT_DATA).rename(columns=rename_map)
data_out.to_csv(OUTPUT_DATA, index=False)
print(f"Saved: {OUTPUT_DATA}")

################################################################################
# %% Write markdown changelog
changed = mapping_df[mapping_df["feature_src"] != mapping_df["feature_dst"]]

md_lines = [
    f"# Column renaming: v-4 → v-5",
    f"",
    f"**Input:**  `{INPUT_DATA}`  ",
    f"**Output:** `{OUTPUT_DATA}`  ",
    f"**Mapping:** `{OUTPUT_MAPPING}`",
    f"",
    f"ROI feature columns ({len(mapping_df)} total, {mapping_df['roi_dst'].nunique()} unique ROIs, "
    f"3 modalities: GM\\_Vol, CSF\\_Vol, WM\\_Vol) were renamed to align with the "
    f"[Neuromorphometrics](http://www.neuromorphometrics.com/) atlas convention "
    f"as provided in `data/atlases_mapping_to_canonical.csv`.",
    f"",
    f"## Renaming rules",
    f"",
    f"1. Non-feature columns (`participant_id`, `age`, `sex`, `site`, `y`) are kept as-is.",
    f"2. Modality suffix (`_GM_Vol`, `_CSF_Vol`, `_WM_Vol`) is stripped then re-appended unchanged.",
    f"3. Leading `Right`/`Left` is preserved as-is.",
    f"4. **Prefix rule** — if the ROI part matches `<ABBREV> <lowercase name>` "
    f"(e.g. `ACgG anterior cingulate gyrus`), only the descriptive part is title-cased "
    f"and the abbreviation is dropped (→ `Anterior Cingulate Gyrus`).",
    f"5. Otherwise the ROI part is title-cased directly.",
    f"6. 22 manual corrections align ambiguous names with Neuromorphometrics "
    f"(e.g. `Accumbens Area` → `Accumbens`, `3rd Ventricle` → `Third Ventricle`).",
    f"",
    f"## Changed columns ({len(changed)} of {len(mapping_df)})",
    f"",
    f"| feature_src | feature_dst |",
    f"|---|---|",
]
for _, row in changed.sort_values("feature_src").iterrows():
    md_lines.append(f"| `{row['feature_src']}` | `{row['feature_dst']}` |")

with open(OUTPUT_MD, "w") as fh:
    fh.write("\n".join(md_lines) + "\n")
print(f"Saved: {OUTPUT_MD}")
