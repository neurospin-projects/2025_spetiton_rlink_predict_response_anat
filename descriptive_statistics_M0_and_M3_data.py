import pandas as pd, numpy as np
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.weightstats import ttest_ind as sm_ttest
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.multitest import fdrcorrection

# inputs
ROOT="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat_backup/"
DATA_DIR=ROOT+"data/processed/"
RLINK_DATAFRAME_M00_M03 = DATA_DIR + "df_ROI_age_sex_site_M00_M03_v4labels_280rois_cat12_7.csv" #"df_ROI_age_sex_site_M00_M03_v4labels.csv"
RLINK_DATAFRAME_ALL_M00 = DATA_DIR + "df_ROI_age_sex_site_M00_v4labels_280rois_cat12_7.csv" #"df_ROI_age_sex_site_M00_v4labels.csv"
SELECTED_REGIONS_OF_INTEREST_RLINK = ['lHip_GM_Vol','rHip_GM_Vol','lAmy_GM_Vol', 'rAmy_GM_Vol']
SELECTED_REGIONS_OF_INTEREST_RLINK_JOINED_HEMI = ["Hip_GM_Vol","Amy_GM_Vol"]
ECRF_PATH = "/neurospin/rlink/PUBLICATION/rlink-ecrf/"
    
def get_mean_roi_left_right_hemispheres(df):
    left_cols = [c for c in df.columns if c.startswith("l") and c.endswith("_Vol")]
    cols_not_roi = [c for c in df.columns if not c.endswith("_Vol")]

    new_df = df[cols_not_roi].copy()
    averaged_rois = {}
    for lcol in left_cols:
        base_name = lcol[1:]
        rcol = "r" + base_name
        
        if rcol in df.columns:
            averaged_rois[base_name] = df[[lcol, rcol]].mean(axis=1)
    
    new_df = pd.concat([new_df, pd.DataFrame(averaged_rois, index=df.index)], axis=1)
    return new_df

def cohens_d(a, b):
    """measures effect size between a (GR) and b (PaR/NR): 
    p-values tell us if there is an effect size, Cohen's d tells us how big the difference is
    absolute cohen's d values: 
        0.2: small effect, 0.5: medium effect, 0.8: large effect
    it's basically a weighted average of the two groups' standard deviations
    (weighted by how many values each group has)
    """
    pooled_std = np.sqrt(
        ((len(a)-1)*a.std()**2 + (len(b)-1)*b.std()**2) / (len(a)+len(b)-2)
    )
    return (a.mean() - b.mean()) / pooled_std

def ttest_row(label, t_series, a_series, b_series):
    """
    t_series: total 
    a_series: good responders
    b_series: non- or partial- responders
    To be used for continuous variables like age, illness duration, scores (QIDS, BRMS), BMI ...
    Welch t-test with equal_var = False because variances may not be equal
    """
    t_stat, p = stats.ttest_ind(a_series.dropna(), b_series.dropna(), equal_var=False)
    d = cohens_d(a_series.dropna(), b_series.dropna())
    
    # N for variables with missing data (for summary Table info)
    t_n = t_series.notna().sum()
    a_n = a_series.notna().sum()
    b_n = b_series.notna().sum()
    
    total_str = f"{t_series.mean():.1f} ± {t_series.std():.1f} (N={t_n})"
    a_str     = f"{a_series.mean():.1f} ± {a_series.std():.1f} (N={a_n})"
    b_str     = f"{b_series.mean():.1f} ± {b_series.std():.1f} (N={b_n})"
    p_str     = f"{p:.3f}" if p >= 0.001 else "<0.001"
    return [label, total_str, a_str, b_str, f"{t_stat:.2f}", p_str, f"{d:.2f}"]

def ttest_row_nonull(label, t_series, a_series, b_series):
    """same as ttest_row but doesn't account for NaN values
    only used for age (when unspecified, it's the age at time of RLink study, 
    as used in regression for classification)
    which we have for all subjects"""
    t_stat, p = stats.ttest_ind(a_series, b_series, equal_var=False)
    d = cohens_d(a_series, b_series)
    total_str = f"{t_series.mean():.1f} ± {t_series.std():.1f}"
    a_str     = f"{a_series.mean():.1f} ± {a_series.std():.1f}"
    b_str     = f"{b_series.mean():.1f} ± {b_series.std():.1f}"
    p_str     = f"{p:.3f}" if p >= 0.001 else "<0.001"
    return [label, total_str, a_str, b_str, f"{t_stat:.2f}", p_str, f"{d:.2f}"]

def chi2_row(label, t_series, a_series, b_series, positive_level=None):
    """tests the association between categorical variables, like
    sex, site, episode type, or change in meds
    1. creates a table of observed frequencies with pd.crosstab
    2. compares observed vs. expected frequencies if there was no association
        btw the 2 variables
    3. gives p-values describing whether the two groups are different
    """
    combined = pd.concat([
        a_series.to_frame().assign(group="GR"),
        b_series.to_frame().assign(group="PaR_NR")
    ])
    ct = pd.crosstab(combined.iloc[:, 0], combined["group"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"

    if positive_level:
        # binary: show n (%)
        t_n = (t_series == positive_level).sum()
        a_n = (a_series == positive_level).sum()
        b_n = (b_series == positive_level).sum()
        total_str = f"{t_n} ({100*t_n/t_series.notna().sum():.1f}%)"
        a_str     = f"{a_n} ({100*a_n/a_series.notna().sum():.1f}%)"
        b_str     = f"{b_n} ({100*b_n/b_series.notna().sum():.1f}%)"
    else:
        # multi-category (like first episode type): each level as "label: n (%), ..."
        def fmt_counts(s):
            total_n = s.notna().sum()
            return " | ".join([
                f"{cat}: {n} ({100*n/total_n:.1f}%)"
                for cat, n in s.value_counts().items()
            ])
        total_str = fmt_counts(t_series)
        a_str     = fmt_counts(a_series)
        b_str     = fmt_counts(b_series)

    return [label, total_str, a_str, b_str, f"χ²={chi2:.2f}", p_str, "—"]

ANTIPSYCHOTICS = [
    'olanzapine', 'olanzapin', 'quetiapine', 'quetiapin', 'seroquel', 'quetipin',
    'aripiprazole', 'aripiprazol', 'risperidone', 'haloperidol', 'lurasidone',
    'asenapine', 'levomepromazine', 'levomepromazina', 'chlorpromazine', 
    'clorpromazine', 'clorpomazine', 'amisulpride', 'loxapine', 'cariprazine',
    'brexpiprazole', 'perphenazine', 'promazine', 'chlorprothixene', 'clotiapine',
    'cyamemazine', 'lepticur'
]

ANTIDEPRESSANTS = [
    'citalopram', 'escitalopram', 'fluoxetine', 'fluoxetina', 'sertraline', 'sertralin',
    'venlafaxine', 'venlaflaxin', 'duloxetine', 'bupropion', 'bupoprion', 'bupropione',
    'trazodone', 'trazadone', 'mirtazapine', 'mirtazapin', 'mianserin', 'mansierin',
    'agomelatine', 'vortioxetine', 'brintellix', 'nortriptyline', 'noritren', 
    'clomipramine'
]

MOOD_STABILIZERS = [
    'lamotrigine', 'lamotrigin', 'lamotrigen', 'lamotrigren', 'lamictal', 'lamotrigene',
    'valproic acid', 'valproate', 'natriumvalproat', 'sodium valproate', 'valproat',
    'carbamazepine', 'oxcarbazepine', 'oxcarbazepina', 'topiramate'
]

BENZOS = [
    'delorazepam', 'alprazolam', 'lorazepam', 'diazepam', 'clonazepam', 
    'bromazepam', 'oxazepam', 'lormetazepam', 'triazolam', 'flurazepam',
    'prazepam', 'chlordiazepoxide', 'sobril', 'oxapax'
]

def classify_drug(inn):
    if pd.isna(inn):
        return None
    inn_lower = str(inn).lower().strip()
    
    # Skipping lithium
    if 'lithium' in inn_lower or 'litio' in inn_lower or 'litium' in inn_lower or \
       'quilonum' in inn_lower or 'priadel' in inn_lower or 'teralithe' in inn_lower:
        return 'lithium'
    
    if any(drug in inn_lower for drug in ANTIPSYCHOTICS):
        return 'antipsychotic'
    elif any(drug in inn_lower for drug in ANTIDEPRESSANTS):
        return 'antidepressant'
    elif any(drug in inn_lower for drug in MOOD_STABILIZERS):
        return 'mood_stabilizer'
    elif any(drug in inn_lower for drug in BENZOS):
        return 'benzodiazepine'
    else:
        return 'other'
    
def get_M0M3_delta_info():
    """info for participants with data at both baseline (M0) and 3 months after Li intake (M3) (N=89)"""
    # dataframe with neuroimaging data for participants with both M0 and M3 data
    df = pd.read_csv(RLINK_DATAFRAME_M00_M03)
    df = df.drop_duplicates(subset="participant_id")

    baseline = pd.read_csv(ECRF_PATH+"dataset-clinical_mod-baseline_version-3.tsv", sep="\t")
    # visits file (longitudinal)
    visits = pd.read_csv(ECRF_PATH+"dataset-clinical_mod-visits_form-visit_version-3.tsv", sep="\t")

    # conversion to numeric
    baseline[["QIDSTSC_PRELI", "BRMSTSC_PRELI", "WEIGHT_PRELI", "HEIGHT_PRELI"]] = \
        baseline[["QIDSTSC_PRELI", "BRMSTSC_PRELI", "WEIGHT_PRELI", "HEIGHT_PRELI"]].apply(
            pd.to_numeric, errors="coerce"
        )
    
    visits[["QIDSCW1", "BRMSCW1", "WEIGHT"]] = \
        visits[["QIDSCW1", "BRMSCW1", "WEIGHT"]].apply(
            pd.to_numeric, errors="coerce"
        )
    ###===================== scores delta (depression and mania) + BMI delta ========================================================###
    # filtering to M3 visit only
    m3_data = visits[visits["VISCODE"] == "M3"][["participant_id", "QIDSCW1", "BRMSCW1", "WEIGHT"]]

    clinical = pd.merge(
        baseline[["participant_id", "QIDSTSC_PRELI", "BRMSTSC_PRELI", "WEIGHT_PRELI", "HEIGHT_PRELI"]],
        m3_data,
        on="participant_id",
        how="inner",
        suffixes=("_m0", "_m3")
    )

    # changes
    clinical["delta_QIDS"] = clinical["QIDSCW1"] - clinical["QIDSTSC_PRELI"]
    clinical["delta_BRMS"] = clinical["BRMSCW1"] - clinical["BRMSTSC_PRELI"]
    clinical["BMI_m0"] = clinical["WEIGHT_PRELI"] / ((clinical["HEIGHT_PRELI"] / 100) ** 2)
    clinical["BMI_m3"] = clinical["WEIGHT"] / ((clinical["HEIGHT_PRELI"] / 100) ** 2)
    clinical["delta_BMI"] = clinical["BMI_m3"] - clinical["BMI_m0"]

    # merging with neuroimaging reference dataframe to get response/site
    df_m3_complete = pd.merge(
        df[["participant_id", "response", "site", "age", "sex"]],
        clinical,
        on="participant_id",
        how="inner"
    )

    ###===================== meds type delta ========================================================###
    baseline_meds=baseline.copy()
    # Convert baseline medication columns to binary
    med_cols_baseline = ["ANTIPSY_PLI", "MOOD_PLI", "ANTIDEP_PLI", "BENZOS_PLI"]
    baseline_meds[med_cols_baseline] = baseline_meds[med_cols_baseline].apply(
        pd.to_numeric, errors="coerce"
    )

    # Convert to boolean (assuming 1=yes, 0=no)
    for col in med_cols_baseline:
        baseline_meds[f'{col}_binary'] = baseline_meds[col] == 1

    # get M3 medication status (!=lithium)
    meds = pd.read_csv(ECRF_PATH+"dataset-clinical_mod-visits_form-visit_tab-med_version-3.tsv", sep="\t")
    meds['INN_clean'] = meds['INN'].str.lower().str.strip()
    meds['drug_class'] = meds['INN_clean'].apply(classify_drug)
    # get M3 medications (assuming F_VISIT is M3)!!!
    meds_m3 = meds[meds['FORM_F_VISIT'] == 'F_VISIT_3']

    # initializing medication cols
    df_m3_complete['on_antipsy_m3'] = False
    df_m3_complete['on_mood_m3'] = False
    df_m3_complete['on_antidep_m3'] = False
    df_m3_complete['on_benzos_m3'] = False

    # creating M3 medication indicators
    for participant in df_m3_complete['participant_id'].unique():
        participant_meds_m3 = meds_m3[meds_m3['participant_id'] == participant]
        
        df_m3_complete.loc[df_m3_complete['participant_id'] == participant, 'on_antipsy_m3'] = \
            (participant_meds_m3['drug_class'] == 'antipsychotic').any()
        df_m3_complete.loc[df_m3_complete['participant_id'] == participant, 'on_mood_m3'] = \
            (participant_meds_m3['drug_class'] == 'mood_stabilizer').any()
        df_m3_complete.loc[df_m3_complete['participant_id'] == participant, 'on_antidep_m3'] = \
            (participant_meds_m3['drug_class'] == 'antidepressant').any()
        df_m3_complete.loc[df_m3_complete['participant_id'] == participant, 'on_benzos_m3'] = \
            (participant_meds_m3['drug_class'] == 'benzodiazepine').any()
    
    # merge with baseline info
    df_m3_complete = pd.merge(
    df_m3_complete,
    baseline_meds[['participant_id', 'ANTIPSY_PLI_binary', 'MOOD_PLI_binary', 
                   'ANTIDEP_PLI_binary', 'BENZOS_PLI_binary']],
        on='participant_id',
        how='left'
    )

    # xomputing change for each medication class separately
    df_m3_complete['antipsy_change'] = (
        df_m3_complete['on_antipsy_m3'] != df_m3_complete['ANTIPSY_PLI_binary']
    )

    df_m3_complete['mood_change'] = (
        df_m3_complete['on_mood_m3'] != df_m3_complete['MOOD_PLI_binary']
    )

    df_m3_complete['antidep_change'] = (
        df_m3_complete['on_antidep_m3'] != df_m3_complete['ANTIDEP_PLI_binary']
    )

    df_m3_complete['benzos_change'] = (
        df_m3_complete['on_benzos_m3'] != df_m3_complete['BENZOS_PLI_binary']
    )

    # any change in meds
    df_m3_complete['any_medication_change'] = (
        df_m3_complete['antipsy_change'] | 
        df_m3_complete['mood_change'] | 
        df_m3_complete['antidep_change'] | 
        df_m3_complete['benzos_change']
    )

    print(f"N at baseline: {len(baseline)}")
    print(f"N at M3 (clinical data): {len(clinical)}")
    print(f"N at M3 (with neuroimaging): {len(df_m3_complete)}")

    # Split by response group
    gr_m3 = df_m3_complete[df_m3_complete["response"] == "GR"]
    parnr_m3 = df_m3_complete[df_m3_complete["response"].isin(["NR", "PaR"])]
    total_m3 = df_m3_complete

    N_gr_m3 = len(gr_m3)
    N_parnr_m3 = len(parnr_m3)
    N_total_m3 = len(total_m3)

    print(f"N GR at M3 = {N_gr_m3}")
    print(f"N PaR/NR at M3 = {N_parnr_m3}")
    print(f"Total at M3 = {N_total_m3}")

    rows_m3 = []
    rows_m3.append(["N", str(N_total_m3), str(N_gr_m3), str(N_parnr_m3), "—", "—", "—"])

    rows_m3.append(ttest_row_nonull("Age, years (mean ± SD)",
                        total_m3["age"], gr_m3["age"], parnr_m3["age"]))

    rows_m3.append(chi2_row("Sex, female n (%)",
                        total_m3["sex"], gr_m3["sex"], parnr_m3["sex"],
                        positive_level="female"))

    rows_m3.append(site_row("Site, n", total_m3["site"], gr_m3["site"], parnr_m3["site"]))

    # Clinical outcomes at M3
    rows_m3.append(ttest_row("QIDS at M3 (mean ± SD)",
                        total_m3["QIDSCW1"], gr_m3["QIDSCW1"], parnr_m3["QIDSCW1"]))

    rows_m3.append(ttest_row("BRMS at M3 (mean ± SD)",
                        total_m3["BRMSCW1"], gr_m3["BRMSCW1"], parnr_m3["BRMSCW1"]))

    rows_m3.append(ttest_row("Change in QIDS, M0 to M3 (mean ± SD)",
                        total_m3["delta_QIDS"], gr_m3["delta_QIDS"], parnr_m3["delta_QIDS"]))

    rows_m3.append(ttest_row("Change in BRMS, M0 to M3 (mean ± SD)",
                        total_m3["delta_BRMS"], gr_m3["delta_BRMS"], parnr_m3["delta_BRMS"]))

    rows_m3.append(ttest_row("BMI at M3, kg/m² (mean ± SD)",
                        total_m3["BMI_m3"], gr_m3["BMI_m3"], parnr_m3["BMI_m3"]))

    rows_m3.append(ttest_row("Change in BMI, M0 to M3 (mean ± SD)",
                        total_m3["delta_BMI"], gr_m3["delta_BMI"], parnr_m3["delta_BMI"]))

    rows_m3.append(chi2_row("Any medication change M0 to M3, n (%)",
                        total_m3['any_medication_change'],
                        gr_m3['any_medication_change'],
                        parnr_m3['any_medication_change'],
                        positive_level=True))

    rows_m3.append(chi2_row("Antipsychotic change M0 to M3, n (%)",
                        total_m3['antipsy_change'],
                        gr_m3['antipsy_change'],
                        parnr_m3['antipsy_change'],
                        positive_level=True))

    rows_m3.append(chi2_row("Mood stabilizer change M0 to M3, n (%)",
                        total_m3['mood_change'],
                        gr_m3['mood_change'],
                        parnr_m3['mood_change'],
                        positive_level=True))

    rows_m3.append(chi2_row("Antidepressant change M0 to M3, n (%)",
                        total_m3['antidep_change'],
                        gr_m3['antidep_change'],
                        parnr_m3['antidep_change'],
                        positive_level=True))

    rows_m3.append(chi2_row("Benzodiazepine change M0 to M3, n (%)",
                        total_m3['benzos_change'],
                        gr_m3['benzos_change'],
                        parnr_m3['benzos_change'],
                        positive_level=True))
    df_table_m3 = pd.DataFrame(
        rows_m3,
        columns=["Variable",
                f"Total (N={N_total_m3})",
                f"GR (N={N_gr_m3})",
                f"PaR/NR (N={N_parnr_m3})",
                "Statistic", "p-value", "Cohen's d"]
    )

    col_widths = {
        "Variable"                : 50,
        f"Total (N={N_total_m3})" : 35,
        f"GR (N={N_gr_m3})"       : 35,
        f"PaR/NR (N={N_parnr_m3})": 35,
        "Statistic"               : 12,
        "p-value"                 : 10,
        "Cohen's d"               : 10,
    }

    header = "".join(col.ljust(w) for col, w in col_widths.items())
    separator = "-" * sum(col_widths.values())

    print("\n" + separator)
    print(header)
    print(separator)
    for _, row in df_table_m3.iterrows():
        print("".join(str(val).ljust(w) for val, w in zip(row, col_widths.values())))
    print(separator)

def get_M0_plus(df):
    """returns a dataframe that is df with additional (merged) data taken from the
    dataset-clinical_mod-baseline_version-3.tsv dataframe for the participants described in df,
    which should be participants with data at M0 (N = 116)"""

    dat = pd.read_csv(ECRF_PATH+"dataset-clinical_mod-baseline_version-3.tsv",sep="\t")
    # mood_meds = pd.read_csv(ECRF_PATH+"dataset-clinical_mod-baseline_form-postLi_tab-mood_version-3.tsv",sep="\t",low_memory=False)

    """
    np.sum(dat["AGEMANE2_PLI"].notna() & (dat["AGEMANE2_PLI"] != "ND"))#134 # age first manic episode
    np.sum(dat["AGEHYPOE2_PLI"].notna() & (dat["AGEHYPOE2_PLI"] != "ND")) #112 # age first hypomanic episode
    np.sum(dat["AGEMDE2_PLI"].notna() & (dat["AGEMDE2_PLI"] != "ND")) #133 # age first depressive episode
    np.sum(dat["AGESTBH2_PLI"].notna() & (dat["AGESTBH2_PLI"] != "ND")) #136 # age hospitalization 
    np.sum(dat["AGES2_PLI"].notna() & (dat["AGES2_PLI"] != "ND"))#27 # age first suicide attempt
    """

    list_acronyms = ["AGEMANE2_PLI", "AGEHYPOE2_PLI","AGEMDE2_PLI","AGESTBH2_PLI","AGES2_PLI"]

    dat[list_acronyms] = dat[list_acronyms].apply(
        pd.to_numeric, errors="coerce" # 'coerce': if a value can't be converted, replaces it with nan instead of raising an error.
    )
    dat["AGE_PREMIER_EPISODE"] = dat[["AGEMANE2_PLI", "AGEHYPOE2_PLI","AGEMDE2_PLI"]].min(axis=1)
    # print(np.sum(dat["AGE_PREMIER_EPISODE"].notna() & (dat["AGE_PREMIER_EPISODE"]!="ND"))) # 139
    dat["PREMIER_TYPE"] = np.where(
        dat[["AGEMANE2_PLI", "AGEHYPOE2_PLI","AGEMDE2_PLI"]].notna().any(axis=1),  # at least 1 of three values exist
        dat[["AGEMANE2_PLI", "AGEHYPOE2_PLI","AGEMDE2_PLI"]].idxmin(axis=1).map({
            "AGEMANE2_PLI": "mania",
            "AGEHYPOE2_PLI": "hypomania",
            "AGEMDE2_PLI": "depression"
        }),
        np.nan
    )
    # print(dat["MANE2_PLI"]) # nb manic episodes
    # print(dat["HYPOE2_PLI"]) # nb hypomanic episodes
    # print(dat["MDE2_PLI"]) # nb depressive episodes
    nb_episodes = ["MANE2_PLI","HYPOE2_PLI","MDE2_PLI"]
    dat[nb_episodes+["NBH2_PLI","NBS2_PLI","CGI3_PRELI","WEIGHT_PRELI","HEIGHT_PRELI"]] = \
        dat[nb_episodes+["NBH2_PLI","NBS2_PLI","CGI3_PRELI","WEIGHT_PRELI","HEIGHT_PRELI"]].apply(
            pd.to_numeric, errors="coerce" # 'coerce': if a value can't be converted, replaces it with nan instead of raising an error.
        )

    dfM0_plus = pd.merge(df, dat[["participant_id","AGE_PREMIER_EPISODE","PREMIER_TYPE","NBH2_PLI","NBS2_PLI","WEIGHT_PRELI","HEIGHT_PRELI","CGI3_PRELI"]
                                 +nb_episodes+list_acronyms], on="participant_id",how="inner")

    # change value of age at first episode and hospitalization for subject with age discrepancy
    #     AGEMANE2_PLI  AGEHYPOE2_PLI  AGEMDE2_PLI  AGESTBH2_PLI  AGES2_PLI participant_id  age  illness_duration
    # 26          37.0            NaN          NaN          37.0        NaN      sub-33139   36              -1.0
    
    mask = dfM0_plus["participant_id"] == "sub-33139"
    dfM0_plus.loc[mask, "AGEMANE2_PLI"] = 36.0
    dfM0_plus.loc[mask, "AGESTBH2_PLI"] = 36.0
    dfM0_plus.loc[mask, "AGE_PREMIER_EPISODE"] = 36.0

    dfM0_plus["illness_duration"] = dfM0_plus["age"] - dfM0_plus["AGE_PREMIER_EPISODE"]
    dfM0_plus["episode_density"] = np.where(
        dfM0_plus["illness_duration"] > 0,
        (dfM0_plus["MANE2_PLI"] + dfM0_plus["HYPOE2_PLI"] + dfM0_plus["MDE2_PLI"]) / dfM0_plus["illness_duration"],
        np.nan  # sets to NaN if illness_duration is 0
    )

    dfM0_plus["hospitalization_density"] = np.where(
        dfM0_plus["illness_duration"] > 0,
        dfM0_plus["NBH2_PLI"] / dfM0_plus["illness_duration"],
        np.nan
    )

    dfM0_plus["positive_history_suicide_attempts"] = (
        dfM0_plus["NBS2_PLI"]
        .astype("Float64")  
        .gt(0)              # > 0, preserving <NA> as <NA>
    )
    # print(dfM0_plus["positive_history_suicide_attempts"].value_counts(dropna=False)) # 4 nan values
    # print(dfM0_plus["NBS2_PLI"].value_counts(dropna=False)) # 4 nan values
    # print(dfM0_plus["CGI3_PRELI"]) # mood state at trial entry

    mapping = {
        1: "Depressed",
        2: "Euthymic",
        3: "Dysphoric Mania",
        4: "Euphoric Mania",
        5: "Cycling",
        6: "Hypomanic"
    }

    dfM0_plus["mood_state_at_entry"] = dfM0_plus["CGI3_PRELI"].map(mapping)
    dfM0_plus["BMI_at_entry"] = dfM0_plus["WEIGHT_PRELI"] / ((dfM0_plus["HEIGHT_PRELI"] / 100) ** 2)
    # check that there are no negative values
    assert ((dfM0_plus["illness_duration"] >= 0) | (dfM0_plus["illness_duration"].isna())).all()
    assert ((dfM0_plus["episode_density"] >= 0) | (dfM0_plus["episode_density"].isna())).all()
    assert ((dfM0_plus["hospitalization_density"] >= 0) | (dfM0_plus["hospitalization_density"].isna())).all()

    assert len(dfM0_plus)==dfM0_plus["participant_id"].nunique()," duplicate participant info in dfM0_plus dataframe "

    return dfM0_plus

# site--> chi-square across all levels, displayed as N unique sites
def site_row(label, t_series, a_series, b_series):
    combined = pd.concat([
        a_series.to_frame().assign(group="GR"),
        b_series.to_frame().assign(group="PaR_NR")
    ])
    ct = pd.crosstab(combined.iloc[:, 0], combined["group"])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    p_str = f"{p:.3f}" if p >= 0.001 else "<0.001"
    total_str = f"{t_series.nunique()} sites"
    a_str     = f"{a_series.nunique()} sites"
    b_str     = f"{b_series.nunique()} sites"
    return [label, total_str, a_str, b_str, f"χ²={chi2:.2f}", p_str, "—"]


def basic_descriptive_stats():
    """description of participants data at baseline (M0) (N=116)"""
    dfM0 = pd.read_csv(RLINK_DATAFRAME_ALL_M00)
    fh = pd.read_csv(ECRF_PATH+"dataset-clinical_mod-baseline_form-postLi_tab-famhist_version-3.tsv",sep="\t")

    # convert diag to numeric
    fh["DIAG_FH"] = pd.to_numeric(fh["DIAG_FH"], errors="coerce")
    # Define family history of BD (any first-degree relative with BD I, BD II, or unspecified BD)
    bd_diagnoses = [1, 2, 3]  # BD I, BD II, unspecified BD
    # Get participants with at least one family member with BD
    participants_with_bd_fh = fh[fh["DIAG_FH"].isin(bd_diagnoses)]["participant_id"].unique()

    print("\n" + "="*60)
    print("M0")
    total=dfM0
    N_total = len(total)

    assert N_total==dfM0["participant_id"].nunique()," duplicate participant info in M0 dataframe "
    print("N = ", dfM0["participant_id"].nunique())
    print("age mean = ", round(dfM0["age"].mean(),3), " std = ",round(dfM0["age"].std(),3))
    print("N sites = ", dfM0["site"].nunique())
   
    gr = dfM0[dfM0["response"] == "GR"]
    parnr = dfM0[dfM0["response"].isin(["NR", "PaR"])]
    N_gr = len(gr)
    N_parnr = len(parnr)
    print("N GR = ",N_gr)
    print("N NR/PaR =", N_parnr)
    assert N_gr + N_parnr == dfM0["participant_id"].nunique()
    print("percentage female = ", round(100*(len(dfM0[dfM0["sex"]=="female"])/dfM0["participant_id"].nunique()),3))
    print("percentage male = ", round(100*(len(dfM0[dfM0["sex"]=="male"])/dfM0["participant_id"].nunique()),3))

    dfM0_plus = get_M0_plus(dfM0)
    # add bin variable for fam hist 
    dfM0_plus["family_history_BD"] = dfM0_plus["participant_id"].isin(participants_with_bd_fh)

    gr_plus    = dfM0_plus[dfM0_plus["response"] == "GR"]
    parnr_plus = dfM0_plus[dfM0_plus["response"].isin(["NR", "PaR"])]
    total_plus = dfM0_plus
    # print(gr_plus[gr_plus["sex"]=="female"]["age"].values)
    # print(parnr_plus[parnr_plus["sex"]=="female"]["age"].values)

    # trying grouping mania and hypomania as discussed with Julie
    total_plus["manic_pole_first"] = total_plus["PREMIER_TYPE"].isin(["mania", "hypomania"])
    gr_plus["manic_pole_first"] = gr_plus["PREMIER_TYPE"].isin(["mania", "hypomania"])
    parnr_plus["manic_pole_first"] = parnr_plus["PREMIER_TYPE"].isin(["mania", "hypomania"])

    N_total = len(total_plus)
    rows = []
    rows.append(["N", str(N_total), str(N_gr), str(N_parnr), "—", "—", "—"])

    # Demographics from dfM0
    rows.append(ttest_row_nonull("Age, years (mean ± SD)",
                        total_plus["age"], gr_plus["age"], parnr_plus["age"]))
    rows.append(chi2_row("Sex, female n (%)",
                        total_plus["sex"], gr_plus["sex"], parnr_plus["sex"],
                        positive_level="female"))
    rows.append(site_row("Site, n", total_plus["site"], gr_plus["site"], parnr_plus["site"]))

    # Clinical characteristics from dfM0_plus
    rows.append(ttest_row("Age at first episode, years (mean ± SD)",
                        total_plus["AGE_PREMIER_EPISODE"],
                        gr_plus["AGE_PREMIER_EPISODE"],
                        parnr_plus["AGE_PREMIER_EPISODE"]))

    rows.append(ttest_row("Illness duration, years (mean ± SD)",
                        total_plus["illness_duration"],
                        gr_plus["illness_duration"],
                        parnr_plus["illness_duration"]))

    rows.append(chi2_row("Type of first episode",
                        total_plus["PREMIER_TYPE"],
                        gr_plus["PREMIER_TYPE"],
                        parnr_plus["PREMIER_TYPE"],
                        positive_level=None))
    
    rows.append(chi2_row("Family history of BD, n (%)",
                        total_plus["family_history_BD"],
                        gr_plus["family_history_BD"],
                        parnr_plus["family_history_BD"],
                        positive_level=True))


    # rows.append(chi2_row("Manic-pole first episode, n (%)",
    #                 total_plus["manic_pole_first"],
    #                 gr_plus["manic_pole_first"],
    #                 parnr_plus["manic_pole_first"],
    #                 positive_level=True))

    rows.append(ttest_row("Episode density, episodes/year (mean ± SD)",
                        total_plus["episode_density"],
                        gr_plus["episode_density"],
                        parnr_plus["episode_density"]))

    rows.append(ttest_row("Hospitalization density, hospitalizations/year (mean ± SD)",
                        total_plus["hospitalization_density"],
                        gr_plus["hospitalization_density"],
                        parnr_plus["hospitalization_density"]))

    rows.append(chi2_row("History of suicide attempt(s), n (%)",
                        total_plus["positive_history_suicide_attempts"],
                        gr_plus["positive_history_suicide_attempts"],
                        parnr_plus["positive_history_suicide_attempts"],
                        positive_level=True))

    rows.append(chi2_row("Mood state at trial entry",
                        total_plus["mood_state_at_entry"],
                        gr_plus["mood_state_at_entry"],
                        parnr_plus["mood_state_at_entry"],
                        positive_level=None))

    rows.append(ttest_row("BMI at entry, kg/m² (mean ± SD)",
                        total_plus["BMI_at_entry"],
                        gr_plus["BMI_at_entry"],
                        parnr_plus["BMI_at_entry"]))

    df_table = pd.DataFrame(
        rows,
        columns=["Variable",
                f"Total (N={N_total})",
                f"GR (N={N_gr})",
                f"PaR/NR (N={N_parnr})",
                "Statistic", "p-value", "Cohen's d"]
    )

    col_widths = {
        "Variable"              : 60,  # increased for longer labels
        f"Total (N={N_total})"  : 40,  # increased for multi-category display
        f"GR (N={N_gr})"        : 40,
        f"PaR/NR (N={N_parnr})" : 40,
        "Statistic"             : 12,
        "p-value"               : 10,
        "Cohen's d"             : 10,
    }

    header = "".join(col.ljust(w) for col, w in col_widths.items())
    separator = "-" * sum(col_widths.values())

    print(separator)
    print(header)
    print(separator)
    for _, row in df_table.iterrows():
        print("".join(str(val).ljust(w) for val, w in zip(row, col_widths.values())))
    print(separator)
    
def test_illness_duration_relation_to_GM_atrophy():
    """
    testing whether gray matter atrophy (at baseline = M0) may be related longer illness duration,
    after accounting for age, sex, site, and response
    results : no significant findings between illness duration and GM atrophy in nearly all regions (even if we don't account for response)
    the exceptions are for regions which are likely to be noisy, such as OC (optical chiasm), or they are CSF regions, like rMedPoCGy_CSF_Vol
    that have higher CSF volume associated with longer illness duration
    """
    dfM0 = pd.read_csv(RLINK_DATAFRAME_ALL_M00)

    info = dfM0.copy()
    assert len(info)==len(dfM0["participant_id"].unique())
    # center age
    info["age_c"] = info["age"] - info["age"].mean()
    # merge PaR and NR
    info["response_bin"] = info["response"].replace({"PaR": "PaR_NR", "NR": "PaR_NR"})
    # categorical and GR as reference
    info["response_bin"] = pd.Categorical(info["response_bin"], categories=["PaR_NR", "GR"])
    info = get_M0_plus(info)
    info["illness_duration_c"] = info["illness_duration"] - info["illness_duration"].mean()
    list_roi = [r for r in info.columns if r.endswith("_Vol") and "-" not in r and "+" not in r]

    for roi in list_roi:#SELECTED_REGIONS_OF_INTEREST_RLINK:
        print(f"{roi} evaluation...")
        formula = f"{roi} ~ C(response_bin) + illness_duration_c + C(sex) + age_c + C(site)"
        model = smf.ols(formula, data=info).fit()
        duration_coef = model.params.get("illness_duration_c", np.nan)
        duration_p = model.pvalues.get("illness_duration_c", np.nan)
        duration_ci = model.conf_int().loc["illness_duration_c"]
        
        print(f"Illness duration main effect:")
        print(f"  Coefficient: {duration_coef:.4f}")
        print(f"  p-value: {duration_p:.3f}")
        print(f"  95% CI: [{duration_ci[0]:.4f}, {duration_ci[1]:.4f}]")
        
        if duration_p < 0.05:
            if duration_coef > 0:
                print("  Longer illness duration associated with LARGER volume")
            else:
                print("  Longer illness duration associated with SMALLER volume (atrophy)")
        else:
            print("  No significant association with illness duration")

    print("\n" + "="*60)

def li_effect_male_female_sep(df, list_roi):
    # Separately estimating treatment effect in males and females
    for roi in list_roi:
        print("\nroi ",roi)
        results = []
        for sex_val in ["male", "female"]:
            subset = df[df["sex"] == sex_val]
            
            formula = f"{roi} ~ C(response_bin) + age_c + C(site)"
            model_strat = smf.ols(formula, data=subset).fit()
            
            coef = model_strat.params.get("C(response_bin)[T.GR]", np.nan)
            pval = model_strat.pvalues.get("C(response_bin)[T.GR]", np.nan)
            ci = model_strat.conf_int().loc["C(response_bin)[T.GR]"]
            tval= model_strat.tvalues.get("C(response_bin)[T.GR]", np.nan)

            # Get means for each group
            gr_mean = subset[subset["response_bin"]=="GR"][roi].mean()
            parnr_mean = subset[subset["response_bin"]=="PaR_NR"][roi].mean()
            
            if sex_val=="female" and pval<0.05:print("\n" + "="*120+"SIGNIFICANT for women too!")
            results.append({
                "Sex": sex_val,
                "N": len(subset),
                "N_GR": (subset["response_bin"]=="GR").sum(),
                "N_PaR_NR": (subset["response_bin"]=="PaR_NR").sum(),
                "Mean_GR": gr_mean,
                "Mean_PaR_NR": parnr_mean,
                "Difference": coef,
                "95%_CI_lower": ci[0],
                "95%_CI_upper": ci[1],
                "p_value": pval,
                "t_value": tval
            })

        results_df = pd.DataFrame(results)
        print("\nStratified Analysis by Sex:")
        print(results_df.to_string(index=False))

def testing_interactions_ROI_volumes_and_sex_separate_hemispheres():
    """
    testing whether there are interactions between GM volume at M0 in Hippocampus and Amygdala and sex,
    with separate regions for the 2 hemispheres: regions include 'lAmy_GM_Vol', 'rAmy_GM_Vol', etc.
    tests, for selected regions of interest (bilateral Amygdala and bilateral Hippocampus):
    f"{roi} ~ C(response_bin) * C(sex) + age_c + C(site)"
    """
    print("separately for the 2 hemispheres...")
    info = pd.read_csv(RLINK_DATAFRAME_ALL_M00)
    assert len(info)==len(info["participant_id"].unique())
    # center age
    info["age_c"] = info["age"] - info["age"].mean()

    # merge PaR and NR
    info["response_bin"] = info["response"].replace({"PaR": "PaR_NR", "NR": "PaR_NR"})
    # categorical and GR as reference
    info["response_bin"] = pd.Categorical(info["response_bin"], categories=["PaR_NR", "GR"])

    for roi in SELECTED_REGIONS_OF_INTEREST_RLINK:
        print(f"{roi} evaluation...")
        formula = f"{roi} ~ C(response_bin) * C(sex) + age_c + C(site)"
        model = smf.ols(formula, data=info).fit()
        # print(model.summary())
        interaction_coef = model.params.get("C(response_bin)[T.GR]:C(sex)[T.male]", np.nan)
        interaction_p = model.pvalues.get("C(response_bin)[T.GR]:C(sex)[T.male]", np.nan)
        interaction_ci = model.conf_int().loc["C(response_bin)[T.GR]:C(sex)[T.male]"]
        print(f"Interaction: beta = {interaction_coef:.3f}, p = {interaction_p:.3f}")
        print(f"95% CI: [{interaction_ci[0]:.3f}, {interaction_ci[1]:.3f}]")
        print(f"\nResponse × Sex Interaction:")
        print(f"  Coefficient: {interaction_coef:.3f}")
        print(f"  p-value: {interaction_p:.3f}")

        if interaction_p < 0.05:
            print("  = Treatment response effect DIFFERS significantly between sexes")
        else:
            print("  = No significant sex difference in treatment response effect")

    print("\n" + "="*60)
    # stratified tests for female and males
    li_effect_male_female_sep(info, SELECTED_REGIONS_OF_INTEREST_RLINK)
    # list_roi = [r for r in info.columns if r.endswith("_Vol") and "-" not in r and "+" not in r]
    # li_effect_male_female_sep(info, list_roi)
    # ==> only region statistically significant for both is the right amygdala

def testing_interactions_ROI_volumes_and_sex_merged_hemispheres():
    """
    testing whether there are interactions between GM volume at M0 in Hippocampus and Amygdala and sex,
    with merged regions for the 2 hemispheres: regions include 'Amy_GM_Vol', 'Hip_GM_Vol', etc.,
    which are mean values of the 2 hemisphere GM (respectively, CSF) volumes 
    tests, for selected regions of interest (Amygdala and Hippocampus):
    f"{roi} ~ C(response_bin) * C(sex) + age_c + C(site)"
    """
    print("separately for the 2 hemispheres...")
    info = pd.read_csv(RLINK_DATAFRAME_ALL_M00)
    assert len(info)==len(info["participant_id"].unique())
    # center age
    info["age_c"] = info["age"] - info["age"].mean()

    # merge PaR and NR
    info["response_bin"] = info["response"].replace({"PaR": "PaR_NR", "NR": "PaR_NR"})
    # categorical and GR as reference
    info["response_bin"] = pd.Categorical(info["response_bin"], categories=["PaR_NR", "GR"])

    print("\njointly for the 2 hemispheres (mean of both GM values)...")
    info_joined = get_mean_roi_left_right_hemispheres(info)
    for roi in SELECTED_REGIONS_OF_INTEREST_RLINK_JOINED_HEMI:
        print(f"{roi} evaluation...")
        formula = f"{roi} ~ C(response_bin) * C(sex) + age_c + C(site)"
        model = smf.ols(formula, data=info_joined).fit()
        # print(model.summary())
        interaction_coef = model.params.get("C(response_bin)[T.GR]:C(sex)[T.male]", np.nan)
        interaction_p = model.pvalues.get("C(response_bin)[T.GR]:C(sex)[T.male]", np.nan)
        interaction_ci = model.conf_int().loc["C(response_bin)[T.GR]:C(sex)[T.male]"]
        print(f"Interaction: beta = {interaction_coef:.3f}, p = {interaction_p:.3f}")
        print(f"95% CI: [{interaction_ci[0]:.3f}, {interaction_ci[1]:.3f}]")

        print(f"\nResponse × Sex Interaction:")
        print(f"  Coefficient: {interaction_coef:.3f}")
        print(f"  p-value: {interaction_p:.3f}")
        if interaction_p < 0.05:
            print("   = Treatment response effect DIFFERS significantly between sexes")
        else:
            print("   = No significant sex difference in treatment response effect")
    print("\n" + "="*60)
    # stratified tests for female and males
    li_effect_male_female_sep(info_joined, SELECTED_REGIONS_OF_INTEREST_RLINK_JOINED_HEMI)
    # list_roi = [r for r in info_joined.columns if r.endswith("_Vol") and "-" not in r and "+" not in r and "3" not in r and "4" not in r]
    # li_effect_male_female_sep(info_joined, list_roi)
    # print("\n" + "="*60)

def testing_interactions_ROI_volumes_difference_M3M0_and_sex_separate_hemispheres():
    """
    testing whether there are interactions between {GM volume difference between after lithium (M3) - baseline (M0)} and {sex},
    with separate regions for the 2 hemispheres: regions include 'lAmy_GM_Vol', 'rAmy_GM_Vol', etc.
    tests, for selected regions of interest (bilateral Amygdala and bilateral Hippocampus):
    f"{roi} ~ C(response_bin) * C(sex) + age_c + C(site)"
    """
    print("\nusing the differences between M0 and M3 values...")
    diff_M0M3= pd.read_csv(DATA_DIR+"df_ROI_M03_minus_M00_age_sex_site_v4labels_280rois_cat12_7.csv")
    info = diff_M0M3.copy()
    assert len(info)==len(diff_M0M3["participant_id"].unique())
    # center age
    info["age_c"] = info["age"] - info["age"].mean()

    # merge PaR and NR
    info["response_bin"] = info["response"].replace({"PaR": "PaR_NR", "NR": "PaR_NR"})
    # categorical and GR as reference
    info["response_bin"] = pd.Categorical(info["response_bin"], categories=["PaR_NR", "GR"])

    for roi in SELECTED_REGIONS_OF_INTEREST_RLINK:
        print(f"{roi} evaluation...")
        formula = f"{roi} ~ C(response_bin) * C(sex) + age_c + C(site)"
        model = smf.ols(formula, data=info).fit()
        # print(model.summary())
        interaction_coef = model.params.get("C(response_bin)[T.GR]:C(sex)[T.male]", np.nan)
        interaction_p = model.pvalues.get("C(response_bin)[T.GR]:C(sex)[T.male]", np.nan)
        interaction_ci = model.conf_int().loc["C(response_bin)[T.GR]:C(sex)[T.male]"]
        print(f"Interaction: beta = {interaction_coef:.3f}, p = {interaction_p:.3f}")
        print(f"95% CI: [{interaction_ci[0]:.3f}, {interaction_ci[1]:.3f}]")

        print(f"\nResponse × Sex Interaction:")
        print(f"  Coefficient: {interaction_coef:.3f}")
        print(f"  p-value: {interaction_p:.3f}")

        if interaction_p < 0.05:
            print("  = Treatment response effect DIFFERS significantly between sexes")
        else:
            print("  = No significant sex difference in treatment response effect")

    print("\n" + "="*60)
    list_roi = [r for r in info.columns if r.endswith("_Vol") and "-" not in r and "+" not in r and "3" not in r and "4" not in r]
    # stratified tests for female and males
    # for all regions of interest
    li_effect_male_female_sep(info, list_roi)
    # for selected regions of interest (bilateral amygdala and bilateral hippocampus)
    # li_effect_male_female_sep(info, SELECTED_REGIONS_OF_INTEREST_RLINK)
    print("\n" + "="*60)

def main():

    # get_M0M3_delta_info()
    # basic_descriptive_stats()
    test_illness_duration_relation_to_GM_atrophy()

if __name__=="__main__":
    main()


"""
============================================================
M0
N =  116
age mean =  40.19  std =  13.341
N sites =  16
N GR =  42
N NR/PaR = 74
percentage female =  53.448
percentage male =  46.552

============================================================
M0 and M3
N =  89
age mean =  39.124  std =  13.484
N =  16
N GR =  31
N NR/PaR =  58
percentage female =  55.056
percentage male =  44.944
"""