import numpy as np, pandas as pd
from utils import create_folder_if_not_exists
import os, sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# normative modeling with PCNtoolkit
from PCNtoolkit.pcntoolkit.util.utils import create_bspline_basis
from PCNtoolkit.pcntoolkit.normative import estimate
from PCNtoolkit.pcntoolkit.normative import predict

# for residualization on site
sys.path.append('/neurospin/psy_sbox/temp_sara/')
from pylearn_mulm.mulm.residualizer import Residualizer

# inputs
ROOT="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"
DATA_DIR=ROOT+"data/processed/"
PATH_TO_DATA_OPENBHB = "/neurospin/psy_sbox/analyses/201906_schizconnect-vip-prague-bsnip-biodb-icaar-start_assemble-all/data/cat12vbm/"
DF_FILE = DATA_DIR+"OpenBHB_roi.csv"
FOUR_REGIONS_OF_INTEREST = ["lHip_GM_Vol","rHip_GM_Vol","lAmy_GM_Vol","rAmy_GM_Vol"]
# outputs
DATA_DIR_NM = DATA_DIR+"normative_model/"
NORMATIVE_MODEL_DIR = ROOT + "models/normative_model/"
NORMATIVE_MODEL_RESULTS_DIR = ROOT + "reports/normative_model/"


def residualize_four_regions_df(df_selected_regions, selected_roi_names, \
    formula_res="site", formula_full = "age + sex + site", no_res=False):
    """
    returns a dataframe of as many columns as there are selected_roi_names,
    with roi residualized on site 
    """
    if no_res: 
        features = df_selected_regions[selected_roi_names].to_numpy()
        df_features_residualized = pd.DataFrame(features, columns=selected_roi_names)
        return df_features_residualized
    residualizer = Residualizer(data=df_selected_regions, formula_res=formula_res, formula_full=formula_full)
    Zres = residualizer.get_design_mat(df_selected_regions)
    features = df_selected_regions[selected_roi_names].to_numpy()
    residualizer.fit(features, Zres)
    features_res = residualizer.transform(features, Zres)
    df_features_residualized = pd.DataFrame(features_res, columns=selected_roi_names)
    return df_features_residualized


def hippo_amyg_save_for_normative_model(no_res=True):
    """
    select the four rois we are interested in for normative modeling:
    gray matter volume of bilateral hippocampus and bilateral amygdala

    residualize on site variable with formula y ~ intercept + age + sex + site
    
    """
    assert os.path.exists(DF_FILE),"openBHB roi dataframe file does not exist"
    df_openbhb = pd.read_csv(DF_FILE)
    print("reading ",DF_FILE," ...")
    print("dataframe OpenBHB rois ...\n",df_openbhb)

    # select the regions we're interested in (bilateral hippocampus and amygdala gray matter volumes)
    # rename OpenBHB names for these roi into neuromorphometrics names used for RLink dataframe 
    print("\nHippocampus and Amygdala regions only : \n", df_openbhb)
    atlas_df = pd.read_csv(DATA_DIR+"lobes_Neuromorphometrics_with_dfROI_correspondencies.csv", sep=";")
    atlas_df_four_roi = atlas_df[atlas_df["ROIabbr"].isin(FOUR_REGIONS_OF_INTEREST)]
    dict_change_names = dict(zip(atlas_df_four_roi["ROIabbr"],atlas_df_four_roi["ROI_Neuromorphometrics_labels"]))

    df_openbhb_four_regions  = df_openbhb[FOUR_REGIONS_OF_INTEREST+["age","sex","site"]]
    df_openbhb_four_regions = df_openbhb_four_regions.rename(columns=dict_change_names)
    print("\nSetting names to the same neuromorphometrics atlas names as in the Rlink dataframes:\n",df_openbhb_four_regions)

    # Bin age (into decades) (so that there are no classes with one subject only for age + sex)
    df_openbhb_four_regions['age_bin'] = pd.cut(df_openbhb_four_regions['age'], bins=[0, 20, 30, 40, 50, 60, 70, 100], labels=False)

    # Create stratify label on binned age + sex
    df_openbhb_four_regions['stratify_label'] = df_openbhb_four_regions['age_bin'].astype(str) + "_" + df_openbhb_four_regions['sex'].astype(str)

    # Train-test split stratified on age and sex
    df_openbhb_tr, df_openbhb_te = train_test_split(
        df_openbhb_four_regions,
        test_size=0.2,
        stratify=df_openbhb_four_regions['stratify_label'],
        random_state=42  # to ensure reproducibility
    )

    df_openbhb_tr.drop(columns=['stratify_label', 'age_bin'], inplace=True)
    df_openbhb_te.drop(columns=['stratify_label', 'age_bin'], inplace=True)

    # print(df_openbhb_tr["age"].mean(), df_openbhb_te["age"].mean()) # 33.88 mean age, 33.95 mean age
    # print((df_openbhb_tr['sex'] == 1).mean(),(df_openbhb_te['sex'] == 1).mean()) # 51.77% women, 51.76% women
    four_roi_names = [dict_change_names[k] for k in FOUR_REGIONS_OF_INTEREST]

    # openBHB : residualize roi on site 
    print(df_openbhb_tr,"\n",df_openbhb_te)
    openBHB_features_residualized_tr = residualize_four_regions_df(df_openbhb_tr, four_roi_names,no_res=no_res)
    openBHB_features_residualized_te = residualize_four_regions_df(df_openbhb_te, four_roi_names,no_res=no_res)

    # print("\nOpenBHB tr roi now residualized on site ...\n",openBHB_features_residualized_tr)
    # print("\nOpenBHB te roi now residualized on site ...\n",openBHB_features_residualized_te)

    # openBHB : covariates : age and sex
    openBHB_covariates_tr = df_openbhb_tr[["age","sex"]]
    openBHB_covariates_te = df_openbhb_te[["age","sex"]]
    # print("\nseparate from their covariates...\n",openBHB_covariates_tr)

    print("OpenBHB train features (residualized on site) shape ",np.shape(openBHB_features_residualized_tr), \
          "\n,and covariates ",np.shape(openBHB_covariates_tr))
    print("OpenBHB test features (residualized on site) shape ",np.shape(openBHB_features_residualized_te), \
          "\n,and covariates ",np.shape(openBHB_covariates_te))
    
    # read Rlink dataframe at M0
    df_ROI_age_sex_site_rlink_M0 = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00.csv")
    # print(df_ROI_age_sex_site_rlink_M0["participant_id"])
    df_ROI_age_sex_site_rlink_four_regions_M0 = df_ROI_age_sex_site_rlink_M0[four_roi_names].copy()
    df_ROI_age_sex_site_rlink_four_regions_M0[["age","sex","site"]] = df_ROI_age_sex_site_rlink_M0[["age","sex","site"]]
    print("\nRLink M0 roi with age, sex, site ...\n",df_ROI_age_sex_site_rlink_four_regions_M0)

    # rlink M0: residualize roi on site
    rlink_features_residualized_M0 = residualize_four_regions_df(df_ROI_age_sex_site_rlink_four_regions_M0, four_roi_names,no_res=no_res)
    print("\nRlink M0 roi now residualized on site ...\n",rlink_features_residualized_M0)
    # rlink M0: covariates : age and sex
    rlink_covariates_M0 = df_ROI_age_sex_site_rlink_four_regions_M0[["age","sex"]]
    print("RLink features (residualized on site) shape ",np.shape(rlink_features_residualized_M0), \
          "\n,and covariates ",np.shape(rlink_covariates_M0))
    
    df_ROI_age_sex_site_rlink_M3 = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00_M03.csv")
    df_ROI_age_sex_site_rlink_M3 = df_ROI_age_sex_site_rlink_M3[df_ROI_age_sex_site_rlink_M3["session"]=="M03"]
    # print(df_ROI_age_sex_site_rlink_M3["participant_id"])
    df_ROI_age_sex_site_rlink_four_regions_M3 = df_ROI_age_sex_site_rlink_M3[four_roi_names].copy()
    df_ROI_age_sex_site_rlink_four_regions_M3[["age","sex","site"]] = df_ROI_age_sex_site_rlink_M3[["age","sex","site"]]
    print("\nRLink M3 roi with age, sex, site ...\n",df_ROI_age_sex_site_rlink_four_regions_M3)

    # rlink M3: residualize roi on site
    rlink_features_residualized_M3 = residualize_four_regions_df(df_ROI_age_sex_site_rlink_four_regions_M3, four_roi_names,no_res=no_res)
    print("\nRlink M0 roi now residualized on site ...\n",rlink_features_residualized_M3)
    # rlink M3: covariates : age and sex
    rlink_covariates_M3 = df_ROI_age_sex_site_rlink_four_regions_M3[["age","sex"]]
    print("RLink features (residualized on site) shape ",np.shape(rlink_features_residualized_M3), \
          "\n,and covariates ",np.shape(rlink_covariates_M3))

    no_res_str ="_no_res" if no_res else ""

    openBHB_cov_tr, openBHB_feat_tr = DATA_DIR_NM   + 'cov_openBHB_tr.txt', DATA_DIR_NM + 'resp_openBHB_tr'+no_res_str+'.txt'
    openBHB_cov_te, openBHB_feat_te = DATA_DIR_NM   + 'cov_openBHB_te.txt', DATA_DIR_NM + 'resp_openBHB_te'+no_res_str+'.txt'

    if not os.path.exists(openBHB_cov_tr): openBHB_covariates_tr.to_csv(openBHB_cov_tr, sep = '\t', header=False, index=False)
    if not os.path.exists(openBHB_feat_tr): openBHB_features_residualized_tr.to_csv(openBHB_feat_tr, sep = '\t', header=False, index=False)

    if not os.path.exists(openBHB_cov_te): openBHB_covariates_te.to_csv(openBHB_cov_te, sep = '\t', header=False, index=False)
    if not os.path.exists(openBHB_feat_te): openBHB_features_residualized_te.to_csv(openBHB_feat_te, sep = '\t', header=False, index=False)

    rlinkM0_cov, rlinkM0_feat = DATA_DIR_NM   + 'cov_rlinkM0.txt', DATA_DIR_NM + 'resp_rlinkM0'+no_res_str+'.txt'
    if not os.path.exists(rlinkM0_cov): rlink_covariates_M0.to_csv(rlinkM0_cov, sep = '\t', header=False, index=False)
    if not os.path.exists(rlinkM0_feat): rlink_features_residualized_M0.to_csv(rlinkM0_feat, sep = '\t', header=False, index=False)

    rlinkM3_cov, rlinkM3_feat = DATA_DIR_NM   + 'cov_rlinkM3.txt', DATA_DIR_NM + 'resp_rlinkM3'+no_res_str+'.txt'
    if not os.path.exists(rlinkM3_cov): rlink_covariates_M3.to_csv(rlinkM3_cov, sep = '\t', header=False, index=False)
    if not os.path.exists(rlinkM3_feat): rlink_features_residualized_M3.to_csv(rlinkM3_feat, sep = '\t', header=False, index=False)


def create_bspline_basis_for_covariates(rlink_age_range=True, openBHB_age_range=False):
    """
    create bspline bases for covariates
    openBHB age range is default for bspline bases
    choose either age range of openBHB or rlink for bsplines min and max for age covariate
    
    """
    assert not (rlink_age_range and openBHB_age_range)," both age ranges of rlink and openBHB can't be used at the same time "

    openBHB_cov_tr = DATA_DIR_NM   + 'cov_openBHB_tr.txt'
    openBHB_cov_te = DATA_DIR_NM   + 'cov_openBHB_te.txt'
    rlinkM0_cov = DATA_DIR_NM   + 'cov_rlinkM0.txt'
    rlinkM3_cov = DATA_DIR_NM   + 'cov_rlinkM3.txt'

    # load train covariate data matrices
    openBHB_covariates_tr = np.loadtxt(os.path.join(openBHB_cov_tr))
    openBHB_covariates_te = np.loadtxt(os.path.join(openBHB_cov_te))
    rlink_covariates_M0 = np.loadtxt(os.path.join(rlinkM0_cov))
    rlink_covariates_M3 = np.loadtxt(os.path.join(rlinkM3_cov))

    # add intercept column
    openBHB_covariates_tr = np.concatenate((openBHB_covariates_tr, np.ones((openBHB_covariates_tr.shape[0],1))), axis=1)
    openBHB_covariates_te = np.concatenate((openBHB_covariates_te, np.ones((openBHB_covariates_te.shape[0],1))), axis=1)

    rlink_covariates_M0 = np.concatenate((rlink_covariates_M0, np.ones((rlink_covariates_M0.shape[0],1))), axis=1)
    rlink_covariates_M3 = np.concatenate((rlink_covariates_M3, np.ones((rlink_covariates_M3.shape[0],1))), axis=1)
    
    df_openbhb = pd.read_csv(DF_FILE)

    age_min = round(min(list(df_openbhb["age"].unique())),1)
    age_max = round(max(list(df_openbhb["age"].unique())),1)
    print("OpenBHB min and max age :", age_min, age_max) # 6.5 95.0
    if openBHB_age_range: B = create_bspline_basis(age_min, age_max)

    df_ROI_age_sex_site_rlink_M0 = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00.csv")
    age_min = round(min(list(df_ROI_age_sex_site_rlink_M0["age"].unique())),1)
    age_max = round(max(list(df_ROI_age_sex_site_rlink_M0["age"].unique())),1)
    print("RLink min and max age :", age_min, age_max) # 18 69
    if rlink_age_range: B = create_bspline_basis(age_min, age_max)

    # create the basis expansion for the covariates
    print('Creating basis expansion ...')

    # create Bspline basis set
    Phi_openBHB_tr = np.array([B(i) for i in openBHB_covariates_tr[:,0]]) # age is the first columns
    Phi_openBHB_te = np.array([B(i) for i in openBHB_covariates_te[:,0]])
    Phi_rlink_M0 = np.array([B(i) for i in rlink_covariates_M0[:,0]])
    Phi_rlink_M3 = np.array([B(i) for i in rlink_covariates_M3[:,0]])

    openBHB_covariates_tr = np.concatenate((openBHB_covariates_tr, Phi_openBHB_tr), axis=1)
    openBHB_covariates_te = np.concatenate((openBHB_covariates_te, Phi_openBHB_te), axis=1)
    rlink_covariates_M0 = np.concatenate((rlink_covariates_M0, Phi_rlink_M0), axis=1)
    rlink_covariates_M3 = np.concatenate((rlink_covariates_M3, Phi_rlink_M3), axis=1)

    if openBHB_age_range: str_age_range=""
    if rlink_age_range: str_age_range="_rlink_age_range"

    openBHB_cov_tr = DATA_DIR_NM   + 'cov_openBHB_tr_bspline'+str_age_range+'.txt'
    openBHB_cov_te = DATA_DIR_NM   + 'cov_openBHB_te_bspline'+str_age_range+'.txt'
    rlinkM0_cov = DATA_DIR_NM   + 'cov_rlinkM0_bspline'+str_age_range+'.txt'
    rlinkM3_cov = DATA_DIR_NM   + 'cov_rlinkM3_bspline'+str_age_range+'.txt'

    if not os.path.exists(openBHB_cov_tr): np.savetxt(openBHB_cov_tr, openBHB_covariates_tr)
    if not os.path.exists(openBHB_cov_te): np.savetxt(openBHB_cov_te, openBHB_covariates_te)
    if not os.path.exists(rlinkM0_cov): np.savetxt(rlinkM0_cov, rlink_covariates_M0)
    if not os.path.exists(rlinkM3_cov): np.savetxt(rlinkM3_cov, rlink_covariates_M3)

    print("...done.")
    
def train_normative_model_wblr(rlink_age_range=False, openBHB_age_range=True, no_bspline=False, no_res=True, inscale=None,\
                               scaling_responses=True):
    assert not (rlink_age_range and openBHB_age_range)," both age ranges of rlink and openBHB can't be used at the same time "
    if openBHB_age_range: str_age_range=""
    if rlink_age_range: str_age_range="_rlink_age_range"
    if no_bspline: str_age_range="_no_bspline"
    no_res_str ="_no_res" if no_res else ""

    openBHB_feat_tr = DATA_DIR_NM + 'resp_openBHB_tr'+no_res_str+'.txt'
    openBHB_feat_te = DATA_DIR_NM + 'resp_openBHB_te'+no_res_str+'.txt'

    if no_bspline: openBHB_cov_tr, openBHB_cov_te = DATA_DIR_NM + 'cov_openBHB_tr.txt', DATA_DIR_NM + 'cov_openBHB_te.txt'
    else : 
        openBHB_cov_tr = DATA_DIR_NM   + 'cov_openBHB_tr_bspline'+str_age_range+'.txt'
        openBHB_cov_te = DATA_DIR_NM   + 'cov_openBHB_te_bspline'+str_age_range+'.txt'

    covtr = np.loadtxt(openBHB_cov_tr,dtype=str)
    covte = np.loadtxt(openBHB_cov_te,dtype=str)

    print("covariates :",np.shape(covtr), np.shape(covte))
    print("covariates :",type(covtr), type(covte))

    # load train & test response files
    resptr = np.loadtxt(openBHB_feat_tr,dtype=float)
    respte = np.loadtxt(openBHB_feat_te,dtype=float)
    print("responses :",np.shape(resptr), np.shape(respte), type(resptr), type(respte))

    if scaling_responses: 
        str_scaled_resp = "_scaled_resp"
        openBHB_feat_tr_scaled = DATA_DIR_NM + 'resp_openBHB_tr'+no_res_str+'_scaled.txt'
        openBHB_feat_te_scaled = DATA_DIR_NM + 'resp_openBHB_te'+no_res_str+'_scaled.txt'
        if os.path.exists(openBHB_feat_tr_scaled): openBHB_feat_tr = openBHB_feat_tr_scaled
        if os.path.exists(openBHB_feat_te_scaled): openBHB_feat_te = openBHB_feat_te_scaled

        if not os.path.exists(openBHB_feat_tr_scaled) or not os.path.exists(openBHB_feat_te_scaled):
            print("scaled features have not been saved yet... scaling features ...")
            scaler = StandardScaler()
            scaler.fit(resptr)
            resptr = scaler.transform(resptr)
            respte = scaler.transform(respte)
            print("scaled responses :",np.shape(resptr), np.shape(respte), type(resptr), type(respte))
            np.savetxt(openBHB_feat_tr_scaled, resptr)
            np.savetxt(openBHB_feat_te_scaled, respte)
            openBHB_feat_tr = openBHB_feat_tr_scaled
            openBHB_feat_te = openBHB_feat_te_scaled
            print("saved to ",DATA_DIR_NM)

    else : str_scaled_resp = ""

    if inscale: inscale="standardize"
    str_inscale = "_inscalerTrue" if inscale else ""

    modelname = "wblr_bilateral_hippo_amyg_openBHB_roi"+str_age_range+no_res_str+str_inscale+str_scaled_resp
    path_model = NORMATIVE_MODEL_DIR+modelname
    print(path_model)
    create_folder_if_not_exists(path_model)
    os.chdir(path_model)
    print(openBHB_feat_tr)
    print(openBHB_feat_te)
    estimate(openBHB_cov_tr,
            openBHB_feat_tr,
            testresp=openBHB_feat_te,
            testcov=openBHB_cov_te,
            alg = "blr",
            optimizer = 'powell',
            savemodel=True,
            saveoutput=True,
            standardize=False, warp = "WarpSinArcsinh", inscale=inscale)


def evaluate_rlink_four_rois_using_normative_model(rlink_age_range=False, openBHB_age_range=True, no_bspline=False, no_res=True,\
                                                   inscale=None, scaling_responses=False):
    assert not (rlink_age_range and openBHB_age_range)," both age ranges of rlink and openBHB can't be used at the same time "
    if openBHB_age_range: str_age_range=""
    if rlink_age_range: str_age_range="_rlink_age_range"
    if no_bspline: str_age_range="_no_bspline"
    no_res_str ="_no_res" if no_res else ""
    str_scaled_resp = "_scaled_resp" if scaling_responses else ""
    str_inscale = "_inscalerTrue" if inscale else ""

    rlinkM0_feat, rlinkM3_feat = DATA_DIR_NM + 'resp_rlinkM0'+no_res_str+'.txt', DATA_DIR_NM + 'resp_rlinkM3'+no_res_str+'.txt'

    if no_bspline: rlinkM0_cov, rlinkM3_cov = DATA_DIR_NM + 'cov_rlinkM0.txt', DATA_DIR_NM + 'cov_rlinkM3.txt'
    else: 
        rlinkM0_cov = DATA_DIR_NM + 'cov_rlinkM0_bspline'+str_age_range+'.txt'
        rlinkM3_cov = DATA_DIR_NM + 'cov_rlinkM3_bspline'+str_age_range+'.txt'

    M0cov = np.loadtxt(rlinkM0_cov,dtype=str)
    M3cov = np.loadtxt(rlinkM3_cov,dtype=str)

    M0feat = np.loadtxt(rlinkM0_feat,dtype=str)
    M3feat = np.loadtxt(rlinkM3_feat,dtype=str)

    if scaling_responses: 
        M0feat_scaled = DATA_DIR_NM + 'resp_rlinkM0'+no_res_str+'_scaled.txt' 
        M3feat_scaled = DATA_DIR_NM + 'resp_rlinkM3'+no_res_str+'_scaled.txt'
        if os.path.exists(M0feat_scaled): rlinkM0_feat = M0feat_scaled
        if os.path.exists(M3feat_scaled): rlinkM3_feat = M3feat_scaled

        if not os.path.exists(M0feat_scaled) or not os.path.exists(M3feat_scaled):
            print("Rlink scaled features have not been saved yet... scaling features ...")
            scaler = StandardScaler()
            scaler.fit(M0feat)
            M0feat = scaler.transform(M0feat)
            scaler.fit(M3feat)
            M3feat = scaler.transform(M3feat)
            print("scaled responses :",np.shape(M0feat), np.shape(M3feat), type(M0feat), type(M3feat))
            np.savetxt(M0feat_scaled, M0feat)
            np.savetxt(M3feat_scaled, M3feat)
            rlinkM0_feat, rlinkM3_feat = M0feat_scaled, M3feat_scaled
            print("saved to ",DATA_DIR_NM)

    print("covariates M0 :",np.shape(M0cov), np.shape(M0cov))
    print("covariates M3 :",type(M3cov), type(M3cov))
    print("responses/features M0 :",np.shape(M0feat), np.shape(M0feat))
    print("responses/features M3 :",type(M3feat), type(M3feat))

    modelname = "wblr_bilateral_hippo_amyg_openBHB_roi"+str_age_range+no_res_str+str_inscale+str_scaled_resp
    path_model = NORMATIVE_MODEL_DIR+modelname
    assert os.path.exists(path_model),path_model+ " does not exist"

    path_results_M0 = NORMATIVE_MODEL_RESULTS_DIR+"M0"+str_age_range+no_res_str+str_inscale+str_scaled_resp
    path_results_M3= NORMATIVE_MODEL_RESULTS_DIR+"M3"+str_age_range+no_res_str+str_inscale+str_scaled_resp
    print(path_results_M0)
    print(path_results_M3)
 
    create_folder_if_not_exists(path_results_M0)
    create_folder_if_not_exists(path_results_M3)

    if not os.listdir(path_results_M0): # if folder is empty
        predict(rlinkM0_cov,
                alg='blr',
                respfile=rlinkM0_feat,
                model_path= path_model+"/Models",
                save_path = path_results_M0)
        
    if not os.listdir(path_results_M3): # if folder is empty
        predict(rlinkM3_cov,
            alg='blr',
            respfile=rlinkM3_feat,
            model_path= path_model+"/Models",
            save_path = path_results_M3)


def read_results(label=1, rlink_age_range=False, openBHB_age_range=True, no_bspline=False, no_res=True, inscale=None, \
                 scaling_responses=False):
    
    no_res_str ="_no_res" if no_res else ""
    assert not (rlink_age_range and openBHB_age_range)," both age ranges of rlink and openBHB can't be used at the same time "
    if openBHB_age_range: str_age_range=""
    if rlink_age_range: str_age_range="_rlink_age_range"
    if no_bspline: str_age_range="_no_bspline"
    str_inscale = "_inscalerTrue" if inscale else ""
    str_scaled_resp = "_scaled_resp" if scaling_responses else ""

    zscore_path_M0 = NORMATIVE_MODEL_RESULTS_DIR+"M0"+str_age_range+no_res_str+str_inscale+str_scaled_resp+"/Z_predict.txt"
    zscore_path_M3 = NORMATIVE_MODEL_RESULTS_DIR+"M3"+str_age_range+no_res_str+str_inscale+str_scaled_resp+"/Z_predict.txt"

    z_scores_M0 = np.loadtxt(zscore_path_M0,dtype=str)
    z_scores_M3 = np.loadtxt(zscore_path_M3,dtype=str)
    print(zscore_path_M0)
    print(zscore_path_M3)

    # print(np.shape(z_scores_M0))
    # print(np.shape(z_scores_M3))
    atlas_df = pd.read_csv(DATA_DIR+"lobes_Neuromorphometrics_with_dfROI_correspondencies.csv", sep=";")

    atlas_df_four_roi = atlas_df[atlas_df["ROIabbr"].isin(FOUR_REGIONS_OF_INTEREST)]
    dict_change_names = dict(zip(atlas_df_four_roi["ROIabbr"],atlas_df_four_roi["ROI_Neuromorphometrics_labels"]))
    four_roi_names = [dict_change_names[k] for k in FOUR_REGIONS_OF_INTEREST]

    df_zscores_M0 = pd.DataFrame(z_scores_M0, columns=four_roi_names)
    df_zscores_M3 = pd.DataFrame(z_scores_M3, columns=four_roi_names)

    # print(df_zscores_M0)
    # print(df_zscores_M3)

    print(df_zscores_M0.median(), "\n",df_zscores_M3.median(),"\n")
    
    df_ROI_age_sex_site_rlink_M0 = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00.csv")

    df_ROI_age_sex_site_rlink_M3 = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00_M03.csv")
    df_ROI_age_sex_site_rlink_M3 = df_ROI_age_sex_site_rlink_M3[df_ROI_age_sex_site_rlink_M3["session"]=="M03"]
    df_ROI_age_sex_site_rlink_M3.reset_index(inplace=True)
    df_ROI_age_sex_site_rlink_M3["y"] = df_ROI_age_sex_site_rlink_M3["y"].replace({"GR": 1, "PaR": 0, "NR": 0})
    participant_ids_label = df_ROI_age_sex_site_rlink_M3.loc[df_ROI_age_sex_site_rlink_M3['y'] == label, 'participant_id'].tolist()

    df_zscores_M0["participant_id"]=df_ROI_age_sex_site_rlink_M0["participant_id"].copy()
    df_zscores_M3["participant_id"]=df_ROI_age_sex_site_rlink_M3["participant_id"].copy()

    print(df_zscores_M0)
    df_openbhb = pd.read_csv(DF_FILE)
    print("dataframe OpenBHB rois ...\n",df_openbhb[["participant_id"]+FOUR_REGIONS_OF_INTEREST].mean())

    df_ROI_age_sex_site_rlink_M0 = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00.csv")
    four_rois =['Left Hippocampus_GM_Vol','Right Hippocampus_GM_Vol','Left Amygdala_GM_Vol', 'Right Amygdala_GM_Vol']
    df_ROI_age_sex_site_rlink_M0 = df_ROI_age_sex_site_rlink_M0[df_ROI_age_sex_site_rlink_M0["participant_id"].isin(participant_ids_label)]
    print(df_ROI_age_sex_site_rlink_M0[["participant_id"]+four_rois].mean())

    df_ROI_age_sex_site_rlink_M3 = df_ROI_age_sex_site_rlink_M3[df_ROI_age_sex_site_rlink_M3["participant_id"].isin(participant_ids_label)]
    print(df_ROI_age_sex_site_rlink_M3[["participant_id"]+four_rois].mean())

    quit()

    # add participant_id to zscores dataframes
    merged = pd.merge(df_zscores_M0, df_zscores_M3, on='participant_id', suffixes=('_M0', '_M3'))
    merged = merged[merged["participant_id"].isin(participant_ids_label)]
    merged.reset_index(inplace=True)

    closer_to_zero = pd.DataFrame()
    # print(merged)

    closer_to_zero['participant_id'] = merged['participant_id']

    for col in df_zscores_M0.columns:
        if col != 'participant_id':
            col_M0 = pd.to_numeric(merged[f'{col}_M0'], errors='coerce')
            col_M3 = pd.to_numeric(merged[f'{col}_M3'], errors='coerce')
            
            # Compare absolute values: which is closer to zero?
            closer = np.where(np.abs(col_M0) < np.abs(col_M3), 'M0', 'M3')
            
            closer_to_zero[col] = closer

    counts = closer_to_zero.drop(columns='participant_id').apply(pd.Series.value_counts)
    print(counts.T)  
    n_total = len(closer_to_zero)

    # Count of M3 for each column (i.e., number of times M3 is closer to 0)
    m3_counts = (closer_to_zero.drop(columns='participant_id') == 'M3').sum()

    # Convert to percentage
    m3_percentages = (m3_counts / n_total) * 100

    # Display result
    print(m3_percentages.round(2).to_frame(name='% M3 closer to 0'))

    for col in closer_to_zero.columns:
        if col != 'participant_id':
            if label ==1:
                print(f'\nParticipants where M0 is closer to 0 than M3 for region: {col}')
                ids = closer_to_zero.loc[closer_to_zero[col] == 'M0', 'participant_id']
                print(ids.tolist())
            if label == 0:
                print(f'\nParticipants where M3 is closer to 0 than M0 for region: {col}')
                ids = closer_to_zero.loc[closer_to_zero[col] == 'M3', 'participant_id']
                print(ids.tolist())
                tot = len(df_ROI_age_sex_site_rlink_M0[df_ROI_age_sex_site_rlink_M0["participant_id"].isin(ids.tolist())]["y"])
                n_par = df_ROI_age_sex_site_rlink_M0[
                    df_ROI_age_sex_site_rlink_M0["participant_id"].isin(ids.tolist())
                ]["y"].eq("PaR").sum()
                n_nr = df_ROI_age_sex_site_rlink_M0[
                    df_ROI_age_sex_site_rlink_M0["participant_id"].isin(ids.tolist())
                ]["y"].eq("NR").sum()
                print("percentage of NR ",round(100*(n_nr/tot),2))
                print("percentage of PaR ",round(100*(n_par/tot),2))



def main():
    # hippo_amyg_save_for_normative_model(no_res=False)
    # create_bspline_basis_for_covariates()
    # train_normative_model_wblr(scaling_responses=True, no_res=False)
    # evaluate_rlink_four_rois_using_normative_model(scaling_responses=True, no_res=False)
    read_results(label = 0, no_res=True)
    quit()
    """
    inscaler (standardization of covariates) doesn't change the results
    """

if __name__ == "__main__":
    main()




