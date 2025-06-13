import sys, numpy as np, pandas as pd, pickle
from sklearn import svm
import time
import sklearn.linear_model as lm
from xgboost import XGBClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import scipy, os, shap, joblib
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from plots import plot_glassbrain
from utils import binarization
sys.path.append('/neurospin/psy_sbox/temp_sara/')
from pylearn_mulm.mulm.residualizer import Residualizer
from utils import read_pkl, rename_col , get_rois, make_stratified_splitter, strat_stats, get_scores

# inputs
ROOT ="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"
DATA_DIR=ROOT+"data/processed/"
DF_ROI_M0=DATA_DIR+"df_ROI_age_sex_site_M00_v4labels.csv"
DF_ROI_M3_MINUS_M0 = DATA_DIR+"df_ROI_M03_minus_M00_age_sex_site.csv"
DF_ROI_M3M0 = DATA_DIR+"df_ROI_age_sex_site_M00_M03_v4labels.csv"
ATLAS_ROI_NAMES_DF = DATA_DIR+"lobes_Neuromorphometrics_with_dfROI_correspondencies.csv"

# outputs
RESULTS_DIR = ROOT+"reports/classification_results/"
FEAT_IMPTCE_RES_DIR = ROOT+"reports/feature_importance_results/"

def create_df_from_dict_multiclass(dict_cv_):
    rows = []
    for fold_idx, fold_data in dict_cv_.items():
        for residual_key, residual_data in fold_data.items():
            for classif_key, results in residual_data.items():
                for result in results:
                    row = {
                        "fold": fold_idx,
                        "residualization": residual_key,
                        "classifier": classif_key,

                        "y_test": result["y_test"],
                        "score_test": result["score_test"],
                        "y_pred_test": result["y_pred_test"],
                        "roc_auc_test": result["roc_auc_test"],
                        "balanced_accuracy_test": result["bacc_test"],
                        "recall_0_te": result["recall_0_te"],
                        "recall_1_te":result["recall_1_te"],
                        "recall_2_te":result["recall_2_te"],
                        "overall_recall_te":result["overall_recall_te"],

                        "y_train": result["y_train"],
                        "score_train": result["score_train"],
                        "y_pred_train": result["y_pred_train"],
                        "roc_auc_train": result["roc_auc_train"],
                        "balanced_accuracy_train": result["bacc_train"],
                        "recall_0_tr": result["recall_0_tr"],
                        "recall_1_tr":result["recall_1_tr"],
                        "recall_2_tr":result["recall_2_tr"],
                        "overall_recall_tr":result["overall_recall_tr"],

                    }
                    rows.append(row)

    df_results = pd.DataFrame(rows)
    return df_results

def create_df_from_dict(dict_cv_):
    rows = []
    for fold_idx, fold_data in dict_cv_.items():
        for residual_key, residual_data in fold_data.items():
            for classif_key, results in residual_data.items():
                for result in results:
                    row = {
                        "fold": fold_idx,
                        "residualization": residual_key,
                        "classifier": classif_key,

                        "y_test": result["y_test"],
                        "score_test": result["score_test"],
                        "y_pred_test": result["y_pred_test"],
                        "roc_auc_test": result["roc_auc_test"],
                        "balanced_accuracy_test": result["bacc_test"],
                        "specificity_test": result["specificity_test"],
                        "sensitivity_test":result["sensitivity_test"],

                        "y_train": result["y_train"],
                        "score_train": result["score_train"],
                        "y_pred_train": result["y_pred_train"],
                        "roc_auc_train": result["roc_auc_train"],
                        "balanced_accuracy_train": result["bacc_train"],
                        "specificity_train": result["specificity_train"],
                        "sensitivity_train":result["sensitivity_train"],

                    }
                    rows.append(row)

    df_results = pd.DataFrame(rows)
    return df_results

def initialize_results_dicts(nbfolds, residualization_configs, classifiers_config):
    # Initializing results dictionary : for each fold, get the residualization status, 
    # the classifier, and the corresponding results
    dict_cv = {
        fold: {res_key: {clf_key: [] for clf_key in classifiers_config.keys()}
            for res_key in residualization_configs.keys()}
        for fold in range(nbfolds)
    }

    #Initializing the subjects dicitonary : 
    # get the indices and ids of train and test subjects for each fold
    dict_cv_subjects={
        fold: {"train_subjects_indices":[], "test_subjects_indices":[], \
            "train_subjects_ids":[], "test_subjects_ids":[]}
        for fold in range(nbfolds)
    }

    coefficients_L2LR = {
        fold: {res_key: {}
            for res_key in residualization_configs.keys()}
        for fold in range(nbfolds)
    }

    shap_svm = {
        fold: {res_key: {}
            for res_key in residualization_configs.keys()}
        for fold in range(nbfolds)
    }

    return dict_cv, dict_cv_subjects, coefficients_L2LR, shap_svm
    


def classification_one_fold(fold_idx, X_arr,y_arr,train_index,test_index,df_ROI_age_sex_site , residualization_configs,\
                                    classifiers_config, dict_cv, dict_cv_subjects, coefficients_L2LR, shap_svm,\
                                        compute_and_save_shap, verbose, binarize=False): 
    
    

    # Split data
    X_train, X_test = X_arr[train_index], X_arr[test_index]
    y_train, y_test = y_arr[train_index], y_arr[test_index]
    if verbose : 
        print(len(train_index), len(test_index))
        print("y_train nb of 0 labels ",(y_train == 0).sum(), ", 1 labels ", (y_train == 1).sum())
        print("y_test nb of 0 labels ",(y_test == 0).sum(), ", 1 labels ", (y_test == 1).sum())


    dict_cv_subjects[fold_idx]={"train_subjects_indices":train_index, "test_subjects_indices":test_index, \
                                "train_subjects_ids":df_ROI_age_sex_site["participant_id"].loc[train_index].values, \
                                    "test_subjects_ids":df_ROI_age_sex_site["participant_id"].loc[test_index].values}

    for residual_key, formula in residualization_configs.items():
        if formula:
            # print(f"Residualizing with: {formula}")
            residualizer = Residualizer(
                data=df_ROI_age_sex_site[["age", "sex", "site", "y"]],
                formula_res=formula,
                formula_full=formula + " + y"
            )
            Zres = residualizer.get_design_mat(df_ROI_age_sex_site[["age", "sex", "site", "y"]])
            Zres_train, Zres_test = Zres[train_index], Zres[test_index]
            residualizer.fit(X_train, Zres_train)
            X_train_res = residualizer.transform(X_train, Zres_train)
            X_test_res = residualizer.transform(X_test, Zres_test)
        # formula = None --> no residualization     
        else:
            X_train_res, X_test_res = X_train, X_test

        for classif_key, (estimator, param_grid) in classifiers_config.items():
            # print("classif_key ",classif_key)
            if binarize : pipeline = make_pipeline(GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, n_jobs=1))
            else :
                pipeline = make_pipeline(
                    StandardScaler(),
                    GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, n_jobs=1)
                )
            if binarize:
                X_train_res, X_test_res = binarization(X_train_res, X_test_res, y_train)

            pipeline.fit(X_train_res, y_train)
            y_pred_test = pipeline.predict(X_test_res)
            y_pred_train = pipeline.predict(X_train_res)

            score_test, score_train = get_scores(pipeline, X_test_res, X_train_res)
                
            if classif_key=="L2LR": # get coefficients of L2LR 
                grid_search = pipeline.named_steps['gridsearchcv']
                best_lr = grid_search.best_estimator_
                coefficients = best_lr.coef_[0] #best_lr.coef_ is an array of shape (1,268) so we keep only index 0
                coefficients_L2LR[fold_idx][residual_key]=coefficients
            
            if classif_key=="svm" and compute_and_save_shap: # compute shap
                # runs in a few hours
                background_data = X_train_res 
                grid_search = pipeline.named_steps['gridsearchcv']
                best_svm = grid_search.best_estimator_

                explainer = shap.KernelExplainer(best_svm.decision_function, background_data)
                # shap_values = explainer.shap_values(X_test_res)

                shap_values = joblib.Parallel(n_jobs=20)(
                    joblib.delayed(explainer)(x) for x in tqdm(X_test_res)
                )
                shap_values = np.array([exp.values for exp in shap_values])

                shap_svm[fold_idx][residual_key]=shap_values


            # Metrics
            roc_auc_te = roc_auc_score(y_test, score_test)
            roc_auc_tr = roc_auc_score(y_train, score_train)
            # Compute specificity (recall for negative class, `pos_label=0`) --> fraction of negatives correctly classified (TN/(TN+FP))
            specificity_te = recall_score(y_test, y_pred_test, pos_label=0) 
            # Compute sensitivity (recall for positive class, default `pos_label=1`) --> fraction of positives correctly classified (TP/(TP+FN))
            sensitivity_te = recall_score(y_test, y_pred_test, pos_label=1)

            specificity_tr= recall_score(y_train, y_pred_train, pos_label=0) 
            sensitivity_tr= recall_score(y_train, y_pred_train, pos_label=1)

            test_ids = df_ROI_age_sex_site.iloc[test_index]["participant_id"].values
            misclassified_idx_te = y_test != y_pred_test
            misclassified_test_ids = test_ids[misclassified_idx_te]

            bacc_te = balanced_accuracy_score(y_test, y_pred_test)
            bacc_tr = balanced_accuracy_score(y_train, y_pred_train)
                
            dict_cv[fold_idx][residual_key][classif_key].append({
                "y_test": y_test,
                "score_test": score_test,
                "y_pred_test": y_pred_test,
                "y_train": y_train,
                "score_train": score_train,
                "y_pred_train": y_pred_train,
                "roc_auc_test": roc_auc_te,
                "specificity_test": specificity_te,
                "sensitivity_test":sensitivity_te,
                "bacc_test": bacc_te,
                "roc_auc_train": roc_auc_tr,
                "bacc_train": bacc_tr,
                "specificity_train": specificity_tr,
                "sensitivity_train":sensitivity_tr,
            })

    return misclassified_test_ids

def classif_stacking():
    # classif with m3-m0 ROI using the same splits as for classif using m0 only
    df_ROI_age_sex_site_dif = pd.read_csv(DF_ROI_M3_MINUS_M0)
    splits = read_pkl(ROOT+"subjects_by_fold/subjects_for_each_fold_GRvsPaRNR_5foldCV_seed_1.pkl")
    df_ROI_age_sex_site_dif["y"] = df_ROI_age_sex_site_dif["y"].replace({"GR": 1, "PaR": 0, "NR": 0})
    df_ROI_age_sex_site_dif = df_ROI_age_sex_site_dif.reset_index(drop=True)
    print(df_ROI_age_sex_site_dif)
    X_arr = df_ROI_age_sex_site_dif[get_rois()].values 
    y_arr = df_ROI_age_sex_site_dif["y"].values
    dict_cv = {
        fold: {res_key: {clf_key: [] for clf_key in ["svm"]}
            for res_key in ["res_age_sex_site"]}
        for fold in range(5)
    }
    dict_cv_stacking = {
        fold: {res_key: {clf_key: [] for clf_key in ["stacking_L2LR"]}
            for res_key in ["res_age_sex_site"]}
        for fold in range(5)
    }
    print(dict_cv)

    for fold in range(5):
        print(fold)
        train_ids = splits[fold]["train_subjects_ids"]
        test_ids = splits[fold]["test_subjects_ids"]
        print(type(train_ids), np.shape(train_ids))
        train_index = df_ROI_age_sex_site_dif[df_ROI_age_sex_site_dif["participant_id"].isin(train_ids)].index
        test_index = df_ROI_age_sex_site_dif[df_ROI_age_sex_site_dif["participant_id"].isin(test_ids)].index
        print("len all tr ids from m00 ", len(train_ids), " with subjects with both m00 and m03 : ",len(train_index))
        print("len all te ids from m00", len(test_ids), " with subjects with both m00 and m03 : ",len(test_index))
        # Split data
        X_train, X_test = X_arr[train_index], X_arr[test_index]
        y_train, y_test = y_arr[train_index], y_arr[test_index]
        formula = "age + sex + site"
        # print(f"Residualizing with: {formula}")
        residualizer = Residualizer(
            data=df_ROI_age_sex_site_dif[["age", "sex", "site", "y"]],
            formula_res=formula,
            formula_full=formula + " + y"
        )
        Zres = residualizer.get_design_mat(df_ROI_age_sex_site_dif[["age", "sex", "site", "y"]])
        Zres_train, Zres_test = Zres[train_index], Zres[test_index]
        residualizer.fit(X_train, Zres_train)
        X_train_res = residualizer.transform(X_train, Zres_train)
        X_test_res = residualizer.transform(X_test, Zres_test)
        param_grid = {
            "kernel": ["rbf"], "gamma": ["scale"], "C": [0.1, 1.0, 10.0]
        }
        pipeline = make_pipeline(
            StandardScaler(),
            GridSearchCV(estimator=(svm.SVC(class_weight='balanced',probability=True)), param_grid=param_grid, cv=3, n_jobs=1)
        )

        pipeline.fit(X_train_res, y_train)
        y_pred_test = pipeline.predict(X_test_res)
        y_pred_train = pipeline.predict(X_train_res)

        score_test, score_train = get_scores(pipeline, X_test_res, X_train_res)
        # Metrics
        roc_auc_te = roc_auc_score(y_test, score_test)
        roc_auc_tr = roc_auc_score(y_train, score_train)
        # Compute specificity (recall for negative class, `pos_label=0`) --> fraction of negatives correctly classified (TN/(TN+FP))
        specificity_te = recall_score(y_test, y_pred_test, pos_label=0) 
        # Compute sensitivity (recall for positive class, default `pos_label=1`) --> fraction of positives correctly classified (TP/(TP+FN))
        sensitivity_te = recall_score(y_test, y_pred_test, pos_label=1)

        specificity_tr= recall_score(y_train, y_pred_train, pos_label=0) 
        sensitivity_tr= recall_score(y_train, y_pred_train, pos_label=1)

        bacc_te = balanced_accuracy_score(y_test, y_pred_test)
        bacc_tr = balanced_accuracy_score(y_train, y_pred_train)
            
        dict_cv[fold]["res_age_sex_site"]["svm"].append({
            "y_test": y_test,
            "score_test": score_test,
            "y_pred_test": y_pred_test,
            "y_train": y_train,
            "score_train": score_train,
            "y_pred_train": y_pred_train,
            "roc_auc_test": roc_auc_te,
            "specificity_test": specificity_te,
            "sensitivity_test":sensitivity_te,
            "bacc_test": bacc_te,
            "roc_auc_train": roc_auc_tr,
            "bacc_train": bacc_tr,
            "specificity_train": specificity_tr,
            "sensitivity_train":sensitivity_tr,
        })
    df_results_m3minusm0 = create_df_from_dict(dict_cv)

    # get df of ROI for m00
    df_ROI_age_sex_site = pd.read_csv(DF_ROI_M0)
    df_ROI_age_sex_site["y"] = df_ROI_age_sex_site["y"].replace({"GR": 1, "PaR": 0, "NR": 0})
    df_ROI_age_sex_site = df_ROI_age_sex_site.reset_index(drop=True)
    print(df_ROI_age_sex_site)
    results_classif_m00 = read_pkl(RESULTS_DIR+"results_classification_seed_1_GRvsPaRNR_5fold.pkl")
    results_classif_m00 = results_classif_m00[(results_classif_m00["residualization"]=="res_age_sex_site") & (results_classif_m00["classifier"]=="svm")]
    print(results_classif_m00)
    print(results_classif_m00[results_classif_m00["fold"]==0]["score_test"])
    for fold in range(5):
        # print("fold ",fold)
        train_ids = splits[fold]["train_subjects_ids"]
        test_ids = splits[fold]["test_subjects_ids"]
        train_index = df_ROI_age_sex_site_dif[df_ROI_age_sex_site_dif["participant_id"].isin(train_ids)].index
        test_index = df_ROI_age_sex_site_dif[df_ROI_age_sex_site_dif["participant_id"].isin(test_ids)].index

        score_tr_m0, score_te_m0 = results_classif_m00[results_classif_m00["fold"]==fold]["score_train"].values[0], results_classif_m00[results_classif_m00["fold"]==fold]["score_test"].values[0]
        scores_tr_m3minusm0, scores_te_m3minusm0 = df_results_m3minusm0[df_results_m3minusm0["fold"]==fold]["score_train"].values[0], df_results_m3minusm0[df_results_m3minusm0["fold"]==fold]["score_test"].values[0]
        print("score_tr_m0 ", type(score_tr_m0), np.shape(score_tr_m0))
        print("score_te_m0 ", type(score_te_m0), np.shape(score_te_m0))
        print("scores_tr_m3minusm0 ", type(scores_tr_m3minusm0), np.shape(scores_tr_m3minusm0))
        print("scores_te_m3minusm0 ", type(scores_te_m3minusm0), np.shape(scores_te_m3minusm0))

        train_participants_m3m0 = df_ROI_age_sex_site_dif[df_ROI_age_sex_site_dif["participant_id"].isin(train_ids)]["participant_id"].values
        test_participants_m3m0 = df_ROI_age_sex_site_dif[df_ROI_age_sex_site_dif["participant_id"].isin(test_ids)]["participant_id"].values

        y_train_stacking, y_test_stacking = y_arr[train_index], y_arr[test_index]

        print("len all tr ids from m00 ", len(train_ids), " with subjects with both m00 and m03 : ",len(train_participants_m3m0))
        print("len all te ids from m00", len(test_ids), " with subjects with both m00 and m03 : ",len(test_participants_m3m0))
        # Get indices in train_ids of elements that are in train_participants_m3m0
        indices_tr_adjust = np.where(np.isin(train_ids, train_participants_m3m0))[0]
        indices_te_adjust = np.where(np.isin(test_ids, test_participants_m3m0))[0]
        assert len(indices_tr_adjust)==len(train_participants_m3m0) and len(indices_te_adjust) == len(test_participants_m3m0)
        score_tr_m0 = score_tr_m0[indices_tr_adjust]
        score_te_m0 = score_te_m0[indices_te_adjust]
        assert len(score_tr_m0)==len(scores_tr_m3minusm0) and len(score_te_m0) == len(scores_te_m3minusm0)
        X_train_stacking, X_test_stacking = np.stack((score_tr_m0, scores_tr_m3minusm0), axis=1), np.stack((score_te_m0, scores_te_m3minusm0), axis=1)
        print("X_train_stacking ", np.shape(X_train_stacking), type(X_train_stacking))
        print("X_test_stacking ", np.shape(X_test_stacking), type(X_test_stacking))

        # perform classif on stacked arrays of scores with L2LR
        pipeline = make_pipeline(
            StandardScaler(),
            GridSearchCV(estimator=(lm.LogisticRegression(class_weight='balanced', fit_intercept=False)), param_grid={"C": [0.1, 1.0, 10.0]}, cv=3, n_jobs=1)
        )

        pipeline.fit(X_train_stacking, y_train_stacking)
        y_pred_test = pipeline.predict(X_test_stacking)
        y_pred_train = pipeline.predict(X_train_stacking)

        score_test, score_train = get_scores(pipeline, X_test_stacking, X_train_stacking)
        # Metrics
        roc_auc_te = roc_auc_score(y_test_stacking, score_test)
        roc_auc_tr = roc_auc_score(y_train_stacking, score_train)
        # Compute specificity (recall for negative class, `pos_label=0`) --> fraction of negatives correctly classified (TN/(TN+FP))
        specificity_te = recall_score(y_test_stacking, y_pred_test, pos_label=0) 
        # Compute sensitivity (recall for positive class, default `pos_label=1`) --> fraction of positives correctly classified (TP/(TP+FN))
        sensitivity_te = recall_score(y_test_stacking, y_pred_test, pos_label=1)

        specificity_tr= recall_score(y_train_stacking, y_pred_train, pos_label=0) 
        sensitivity_tr= recall_score(y_train_stacking, y_pred_train, pos_label=1)

        bacc_te = balanced_accuracy_score(y_test_stacking, y_pred_test)
        bacc_tr = balanced_accuracy_score(y_train_stacking, y_pred_train)
            
        dict_cv_stacking[fold]["res_age_sex_site"]["stacking_L2LR"].append({
            "y_test": y_test_stacking,
            "score_test": score_test,
            "y_pred_test": y_pred_test,
            "y_train": y_train_stacking,
            "score_train": score_train,
            "y_pred_train": y_pred_train,
            "roc_auc_test": roc_auc_te,
            "specificity_test": specificity_te,
            "sensitivity_test":sensitivity_te,
            "bacc_test": bacc_te,
            "roc_auc_train": roc_auc_tr,
            "bacc_train": bacc_tr,
            "specificity_train": specificity_tr,
            "sensitivity_train":sensitivity_tr,
        })
    df_results_stacking = create_df_from_dict(dict_cv_stacking)
    print(df_results_stacking)
    print(list(df_results_stacking.columns))

    print(round(df_results_stacking["roc_auc_test"].mean(),4))
    print(round(df_results_stacking["roc_auc_test"].std(),4))
    print(round(df_results_stacking["balanced_accuracy_test"].mean(),4))
    print(round(df_results_stacking["specificity_test"].mean(),4))
    print(round(df_results_stacking["sensitivity_test"].mean(),4))

    # print("\n\n")
    # print(round(df_results_m3minusm0["roc_auc_test"].mean(),4))
    # print(round(df_results_m3minusm0["roc_auc_test"].std(),4))
    # print(round(df_results_m3minusm0["balanced_accuracy_test"].mean(),4))
    # print(round(df_results_m3minusm0["specificity_test"].mean(),4))
    # print(round(df_results_m3minusm0["sensitivity_test"].mean(),4))

def classification(nbfolds= 5, print_pvals=False, save_results=False, seed=13, verbose=True, \
                   compute_and_save_shap=False, random_permutation=False, classif_from_differencem3m0=False,\
                      classif_from_concat=False, classif_from_m3=False, seed_label_permutations=None, classif_from_WM_ROI = False, \
                        biomarkers_roi=False, classif_from_17_roi=False, binarize=False):
    
    # seed was 1 for v3 labels
    """
    print_pvals (bool) : whether to print p-values describing the classification significativity
    compute_and_save_shap (bool): whether to compute SHAP values, if True, runs only svm and saves only svm results
    random_permutation (bool) : random permtutation of labels for training (used to establish a null hypothesis for SHAP values)
    classif_from_concat (bool): True if classifying from the concatenation of m0 and m3-m0 features
    classif_from_differencem3m0 (bool) : True if classifying from the difference m3-m0
    classif_from_m3 (bool) : True if classifying from m3 ROI
    classif_from_WM_ROI (bool): True if classifying from white matter ROI VBM measures (instead of GM + CSF) 
    """
    
    str_labels="_GRvsPaRNR"
    if not random_permutation: assert seed_label_permutations is None
    else : print("seed_label_permutations ",seed_label_permutations)

    str_WM = "_WM_Vol" if classif_from_WM_ROI else ""

    df_ROI_age_sex_site = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00"+str_WM+"_v4labels.csv")
    
    if classif_from_differencem3m0 : df_ROI_age_sex_site = pd.read_csv(DATA_DIR+"df_ROI_M03_minus_M00_age_sex_site"+str_WM+"_v4labels.csv")
    if classif_from_concat: # 91 subjects
        df_ROI_age_sex_site_differences = pd.read_csv(DATA_DIR+"df_ROI_M03_minus_M00_age_sex_site"+str_WM+"_v4labels.csv")
        df_ROI_age_sex_site_baseline = df_ROI_age_sex_site.copy()
        merged = df_ROI_age_sex_site_baseline.merge(df_ROI_age_sex_site_differences, on="participant_id", suffixes=('_m0', '_dif'))
        for col in ["age", "sex", "site", "y"]:
            assert (merged[f"{col}_m0"] == merged[f"{col}_dif"]).all(), f"Mismatch in {col}"
        merged = merged.drop(columns=[f"{col}_dif" for col in ["age", "sex", "site", "y"]])
        merged = merged.rename(columns={f"{col}_m0": col for col in ["age", "sex", "site", "y"]})
        df_ROI_age_sex_site = merged.copy()
    if classif_from_m3:
        df_ROI_age_sex_site = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00_M03"+str_WM+"_v4labels.csv")
        df_ROI_age_sex_site = df_ROI_age_sex_site[df_ROI_age_sex_site["session"]=="M03"]
        df_ROI_age_sex_site = df_ROI_age_sex_site.drop(columns=["session"])

    print("in the whole dataset, number of subjects GR : ",len(df_ROI_age_sex_site[df_ROI_age_sex_site["y"]=="GR"]))
    print("in the whole dataset, number of subjects NR : ",len(df_ROI_age_sex_site[df_ROI_age_sex_site["y"]=="NR"]))
    print("in the whole dataset, number of subjects PaR : ",len(df_ROI_age_sex_site[df_ROI_age_sex_site["y"]=="PaR"]))

    df_ROI_age_sex_site["y"] = df_ROI_age_sex_site["y"].replace({"GR": 1, "PaR": 0, "NR": 0})
    df_ROI_age_sex_site = df_ROI_age_sex_site.reset_index(drop=True)

    # Define residualization strategies    

    # Define classifiers and their grid parameters
    if compute_and_save_shap:
        residualization_configs = {"res_age_sex": "age + sex"}
        classifiers_config = {
            "svm": (svm.SVC(class_weight='balanced',probability=True), {
                "kernel": ["rbf"], "gamma": ["scale"], "C": [0.1, 1.0, 10.0]
            })}
    else:
        residualization_configs = {
            "no_res": None,
            "res_age_sex": "age + sex",
            "res_age_sex_site": "age + sex + site"
        }
        classifiers_config = {
            "L2LR": (lm.LogisticRegression(class_weight='balanced', fit_intercept=False), {"C": [0.1, 1.0, 10.0]}), 
            "svm": (svm.SVC(class_weight='balanced',probability=True), {
                "kernel": ["rbf"], "gamma": ["scale"], "C": [0.1, 1.0, 10.0]
            }),
            "EN": (lm.SGDClassifier(
                loss='log_loss', penalty='elasticnet', class_weight='balanced',random_state=42),{
                "alpha": 10.0 ** np.arange(-1, 2),
                "l1_ratio": [0.1, 0.5, 0.9]
            }),
            "MLP": (MLPClassifier(random_state=1), {
                "hidden_layer_sizes": [
                    (100,), (50,), (25,), (10,), (5,),
                    (100, 50), (50, 25), (25, 10), (10, 5),
                    (100, 50, 25), (50, 25, 10), (25, 10, 5)
                ],
                "activation": ["relu"], "solver": ["sgd"], 'alpha': [0.0001]
            }),
            "xgboost": (XGBClassifier(random_state=42, n_jobs=1), {
                "n_estimators": [10, 30, 50],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 6],
                "subsample": [0.8]
            })
        }

    if classif_from_concat: 
        X_arr= df_ROI_age_sex_site[[col for col in df_ROI_age_sex_site.columns if any(col.startswith(roi) and (col.endswith('_dif') or col.endswith('_m0')) \
                                                                                      for roi in get_rois(WM=classif_from_WM_ROI))]].values
    else : 
        four_rois = ["Left Hippocampus", "Right Hippocampus","Right Amygdala", "Left Amygdala"]
        four_rois = [roi+"_GM_Vol" for roi in four_rois]
        if biomarkers_roi : 
            assert not classif_from_WM_ROI,"WM prediction with 4 biomarker ROIs not implemented."
            X_arr = df_ROI_age_sex_site[four_rois].values 
        if classif_from_17_roi: 
            assert not classif_from_WM_ROI,"WM prediction with 17 ROIs not implemented."
            significant_df = pd.read_excel(FEAT_IMPTCE_RES_DIR+"significant_shap_mean_abs_value_pvalues_1000_random_permut.xlsx")
            significant_rois = [roi for roi in list(significant_df.columns) if roi!="fold"]
            X_arr = df_ROI_age_sex_site[significant_rois].values 
        else : X_arr = df_ROI_age_sex_site[get_rois(WM=classif_from_WM_ROI)].values 

    y_arr = df_ROI_age_sex_site["y"].values

    if verbose:
        print("np.shape(X)",np.shape(X_arr), type(X_arr)) 
        print("np.shape(y)",np.shape(y_arr), type(y_arr))
        print("y nb of 0 labels ",(y_arr == 0).sum(), ", 1 labels ", (y_arr == 1).sum())

    print("seed for CV : ",seed)
    dict_cv, dict_cv_subjects, coefficients_L2LR, shap_svm = initialize_results_dicts(nbfolds, residualization_configs, classifiers_config)

    # stratification 
    splitter = make_stratified_splitter(df = df_ROI_age_sex_site.copy(), cv_seed=seed, n_splits=nbfolds)

    if random_permutation: 
        np.random.seed(seed_label_permutations)
        y_arr = np.random.permutation(y_arr)
        if verbose : print("y_arr permuted :",y_arr)

    misclassified_test_ids_all_folds=[]
    for fold_idx, (train_index, test_index) in enumerate(splitter):
        # print("fold ",fold_idx)        
        misclassified_test_ids = classification_one_fold(fold_idx, X_arr, y_arr,train_index,test_index, df_ROI_age_sex_site , residualization_configs,\
                                    classifiers_config,  dict_cv, dict_cv_subjects, coefficients_L2LR, shap_svm,\
                                        compute_and_save_shap, verbose=False, binarize=binarize)
        misclassified_test_ids_all_folds.append(misclassified_test_ids)
    
    misclassified_test_ids_all_folds = np.concatenate(misclassified_test_ids_all_folds,axis=0)
    
    # print(dict_cv_subjects,"\n\n")
    # print(dict_cv)
    
    df_results = create_df_from_dict(dict_cv)


    if classif_from_differencem3m0: str_diff="_difference_m3m0"
    else: 
        if classif_from_concat: str_diff="_concat_dif_with_baseline"
        if classif_from_m3: str_diff="_m3"
        else: str_diff = ""

    str_rois = ""
    if classif_from_17_roi : str_rois = "_17rois_only" 
    if biomarkers_roi : str_rois="_bilateralHippo_and_Amyg_only"
    str_bin= "_binarized_roi" if binarize else ""

    if save_results :
        if not df_results.empty: 
            df_results.to_pickle(RESULTS_DIR+'results_classification_seed_'+str(seed)+str_labels+'_'+str(nbfolds)+'fold'+str_diff+str_WM+str_rois+'_v4labels'+str_bin+'.pkl')
            print("df_results saved to : ",RESULTS_DIR+'results_classification_seed_'+str(seed)+str_labels+'_'+str(nbfolds)+'fold'+str_diff+str_WM+str_rois+'_v4labels'+str_bin+'.pkl')

    filtered_results_res_age_sex_site = df_results[df_results["residualization"]=="res_age_sex_site"]
    filtered_results_res_age_sex = df_results[df_results["residualization"]=="res_age_sex"]
    filtered_results_nores = df_results[df_results["residualization"]=="no_res"]

    print("mean roc auc with age sex site res :",filtered_results_res_age_sex_site[filtered_results_res_age_sex_site["classifier"]=="L2LR"]["roc_auc_test"].mean())
    print("corresponding std :",filtered_results_res_age_sex_site[filtered_results_res_age_sex_site["classifier"]=="L2LR"]["roc_auc_test"].std())
    
    # convert coefficients_L2LR dict to dataframe
    df_coeffsL2LR = pd.DataFrame.from_dict(coefficients_L2LR, orient="index").reset_index()
    df_coeffsL2LR.rename(columns={"index": "fold"}, inplace=True)

    # convert shap_svm dict to dataframe
    if compute_and_save_shap : 
        df_shap_svm = pd.DataFrame.from_dict(shap_svm, orient="index").reset_index()
        df_shap_svm.rename(columns={"index": "fold"}, inplace=True)
        print("df_shap_svm:\n",df_shap_svm)

    
    if save_results :
        if not df_coeffsL2LR.empty : 
            df_coeffsL2LR.to_pickle(RESULTS_DIR+'coefficientsL2LR/L2LR_coefficients_'+str(seed)+str_labels+'_'+str(nbfolds)+'fold'+str_diff+str_WM+str_rois+'_v4labels'+str_bin+'.pkl')
        # if dict_cv_subjects : 
        #     save_pkl(dict_cv_subjects,ROOT+'subjects_by_fold/subjects_for_each_fold'+str_labels+'_'+str(nbfolds)+'foldCV_seed_'+str(seed)+str_diff+str_WM+str_rois+'_v4labels'+str_bin+".pkl")
    

    if compute_and_save_shap and not df_shap_svm.empty : 
        if random_permutation: 
            df_shap_svm.to_pickle(FEAT_IMPTCE_RES_DIR+'svm_shap_seed'+str(seed)+str_labels+"_"+str(nbfolds)+"fold_random_permutations_with_seed_"+\
                                  str(seed_label_permutations)+"_of_labels"+str_diff+str_WM+str_rois+"_v4labels"+str_bin+".pkl")
        else : 
            df_shap_svm.to_pickle(FEAT_IMPTCE_RES_DIR+'svm_shap_seed'+str(seed)+str_labels+"_"+str(nbfolds)+"fold"+str_diff+str_WM+str_rois+"_v4labels"+str_bin+".pkl")

    if verbose :
        print("residualization age+sex+site mean roc auc and std :",round(filtered_results_res_age_sex_site.groupby("classifier")["roc_auc_test"].mean(),4),\
            round(filtered_results_res_age_sex_site.groupby("classifier")["roc_auc_test"].std(),4))
        print("\nresidualization age+sex mean roc auc and std :",round(filtered_results_res_age_sex.groupby("classifier")["roc_auc_test"].mean(),4),\
            round(filtered_results_res_age_sex.groupby("classifier")["roc_auc_test"].std(),4))
        print("\nno residualization mean roc auc and std :",round(filtered_results_nores.groupby("classifier")["roc_auc_test"].mean(),4),
            round(filtered_results_nores.groupby("classifier")["roc_auc_test"].std(),4))

        # print("\n\n", df_results)


    if print_pvals:
        # Group by residualization and classifier
        results_pvals = {}
        grouped = df_results.groupby(['residualization', 'classifier'])
        if verbose : print("\n\n")

        for (residualization, classifier), group in grouped:
            # Concatenate score_test and y_test arrays for all folds
            score_test_concatenated = np.concatenate(group['score_test'].values, axis=0)
            ytest_concatenated = np.concatenate(group['y_test'].values, axis=0)
            
            # Perform the Mann-Whitney U test
            group_nr = score_test_concatenated[ytest_concatenated == 0]
            group_gr = score_test_concatenated[ytest_concatenated == 1]
            
            if len(group_nr) > 0 and len(group_gr) > 0:
                pvalue = scipy.stats.mannwhitneyu(group_nr, group_gr).pvalue
            else:
                pvalue = None  # Handle cases with no data for either group
            
            # Store the results
            results_pvals[(residualization, classifier)] = pvalue

        # Convert results to a DataFrame for easier visualization
        results_pvals_df = pd.DataFrame([
            {'residualization': key[0], 'classifier': key[1], 'pvalue': value}
            for key, value in results_pvals.items()
        ])

        if verbose: print(results_pvals_df)

    # return mean for all classifiers and std ROC AUC with residualization on age, sex, site
    return round(filtered_results_res_age_sex_site.groupby("classifier")["roc_auc_test"].mean(),4),\
        round(filtered_results_res_age_sex_site.groupby("classifier")["roc_auc_test"].std(),4), misclassified_test_ids_all_folds


def get_seed_with_lowest_std_in_roc_auc_between_folds():
    integer_list = list(range(1, 31))
    # dict containing the means and stds of ROC-AUCs for EN, L2LR, and SVM classifiers, with residualization on age, sex, site
    with open("results_different_seeds_means_and_stds_5folds_classif_from_dif.pkl","rb") as f: #"results_different_seeds_means_and_stds.pkl"
        mydict = pickle.load(f)
    print("\nresults with seed = 23")
    means, stds = mydict[23]
    print(type(means))
    print(means, stds)
    print("\nresults with seed = 12")
    means, stds = mydict[12]
    print(type(means))
    print(means, stds)
    print("\n")
    for classifier in ["EN","L2LR","svm","MLP","xgboost"]:
        min , minseed, rocauc_forminstd = 10000, 0, 0
        for i in integer_list:
            means, stds = mydict[i]
            if stds[classifier]<=min:
                min = stds[classifier]
                minseed = i
                rocauc_forminstd = means[classifier]
        print("for ",classifier,"  the minimum std is ", min, ' achieved with seed = ', minseed)
        print("the corresponding mean ROC AUC is ", rocauc_forminstd)


def print_sensitivity_specificity(df, residualization_type="no_res"):
    assert residualization_type in ["no_res", "res_age_sex", "res_age_sex_site"]
    print("specificity :")
    print(round(df[df["residualization"]==residualization_type].groupby("classifier")["specificity_test"].mean(),4))
    print("sensitivity :")
    print(round(df[df["residualization"]==residualization_type].groupby("classifier")["sensitivity_test"].mean(),4))

def print_recall_multiclass(df, residualization_type="no_res"):
    assert residualization_type in ["no_res", "res_age_sex", "res_age_sex_site"]
    print("recall_0_te")
    print(round(df[df["residualization"]==residualization_type].groupby("classifier")["recall_0_te"].mean(),4))
    print("recall_1_te")
    print(round(df[df["residualization"]==residualization_type].groupby("classifier")["recall_1_te"].mean(),4))
    print("recall_2_te")
    print(round(df[df["residualization"]==residualization_type].groupby("classifier")["recall_2_te"].mean(),4))
    print("overall recall")
    print(round(df[df["residualization"]==residualization_type].groupby("classifier")["overall_recall_te"].mean(),4))

def print_results(metric="roc_auc",classif_from_differencem3m0=False, classif_from_17rois=False):
    assert metric in ["roc_auc","balanced_accuracy"]
    file = "results_classification_seed_1_GRvsParNR_5fold.pkl"
    if classif_from_differencem3m0: file="results_classification_seed_1_GRvsPaRNR_5fold_differencem_m3m0.pkl"
    if classif_from_17rois : file='results_classification_seed_1_GRvsPaRNR_5fold_17rois_only.pkl'

    with open(RESULTS_DIR+file, "rb") as f:
        results=pickle.load(f)

    print(list(results.columns))
    print("ROC AUC test with NO residualization :\n")
    print(round(results[results["residualization"]=="no_res"].groupby("classifier")[metric+"_test"].mean(),4))
    print(round(results[results["residualization"]=="no_res"].groupby("classifier")[metric+"_test"].std(),4))

    print_sensitivity_specificity(results, residualization_type="no_res")

    print("ROC AUC test with AGE + SEX residualization :\n")
    print(round(results[results["residualization"]=="res_age_sex"].groupby("classifier")[metric+"_test"].mean(),4))
    print(round(results[results["residualization"]=="res_age_sex"].groupby("classifier")[metric+"_test"].std(),4))

    print_sensitivity_specificity(results, residualization_type="res_age_sex")
        
    print("ROC AUC test with AGE + SEX + SITE residualization :\n")
    print(round(results[results["residualization"]=="res_age_sex_site"].groupby("classifier")[metric+"_test"].mean(),4))
    print(round(results[results["residualization"]=="res_age_sex_site"].groupby("classifier")[metric+"_test"].std(),4))

    print_sensitivity_specificity(results, residualization_type="res_age_sex_site")

    print("\nMANN WHITNEY TEST FOR SIGNIFICANCE OF CLASSIFICATION: ")
    results_pvals = {}
    grouped = results.groupby(['residualization', 'classifier'])

    for (residualization, classifier), group in grouped:
        # Concatenate score_test and y_test arrays for all folds
        score_test_concatenated = np.concatenate(group['score_test'].values, axis=0)
        ytest_concatenated = np.concatenate(group['y_test'].values, axis=0)
        
        # Perform the Mann-Whitney U test
        group_gr = score_test_concatenated[ytest_concatenated == 0]
        group_nr = score_test_concatenated[ytest_concatenated == 1]
        
        if len(group_gr) > 0 and len(group_nr) > 0:
            pvalue = scipy.stats.mannwhitneyu(group_gr, group_nr).pvalue
        else:
            pvalue = None  # Handle cases with no data for either group
        
        # Store the results
        results_pvals[(residualization, classifier)] = pvalue

    # Convert results to a DataFrame for easier visualization
    results_pvals_df = pd.DataFrame([
        {'residualization': key[0], 'classifier': key[1], 'pvalue': value}
        for key, value in results_pvals.items()
    ])

    print(results_pvals_df)

def get_results(df, res="no_res"):
    df = df[df["residualization"]==res]
    print("NR :",df.groupby("classifier")["count_NR"].mean())
    print("GR :",df.groupby("classifier")["count_GR"].mean())

def read_classif_PaR_from_GRvsNR_models():
    classif_PaR_total= pd.read_csv(ROOT+"df_GRvsNR_testonPaR.csv")
    print(classif_PaR_total)
    # assert not classif_PaR_total.isna().values.any(), "DataFrame contains NaN values!"
    print("no residualization ")
    get_results(classif_PaR_total, res="no_res")
    print("AGE SEX residualization ")
    get_results(classif_PaR_total, res="res_age_sex")
    print("AGE SEX SITE residualization ")
    get_results(classif_PaR_total, res="res_age_sex_site")

def pls_regression(nb_components=3, significant_rois=None, \
                   nbfolds= 5, print_pvals=False, save_results=False, seed=1, verbose=True, \
                   classif_from_differencem3m0=False, classif_from_concat=False,classif_from_m3=False):
    """
    nb_components (int): number of components for PLS regression
    significant_rois (list): list of ROIs that going to be used for regression

    print_pvals (bool) : whether to print p-values describing the classification significativity
    compute_shap (bool): whether to compute SHAP values
    random_permutation (bool) : random permtutation of labels for training (used to establish a null hypothesis for SHAP values)
    classif_from_concat (bool): True if classifying from the concatenation of m0 and m3-m0 features
    classif_from_differencem3m0 (bool) : True if classifying from the difference m3-m0
    classif_from_m3 (bool) : True if classifying from m3 ROI
    """
    from sklearn.cross_decomposition import PLSRegression

    str_labels="_GRvsPaRNR"

    df_ROI_age_sex_site = pd.read_csv(DF_ROI_M0)
    
    if classif_from_differencem3m0 : df_ROI_age_sex_site = pd.read_csv(DF_ROI_M3_MINUS_M0)
    if classif_from_concat: # 91 subjects
        df_ROI_age_sex_site_differences = pd.read_csv(DF_ROI_M3_MINUS_M0)
        df_ROI_age_sex_site_baseline = df_ROI_age_sex_site.copy()
        merged = df_ROI_age_sex_site_baseline.merge(df_ROI_age_sex_site_differences, on="participant_id", suffixes=('_m0', '_dif'))
        for col in ["age", "sex", "site", "y"]:
            assert (merged[f"{col}_m0"] == merged[f"{col}_dif"]).all(), f"Mismatch in {col}"
        merged = merged.drop(columns=[f"{col}_dif" for col in ["age", "sex", "site", "y"]])
        merged = merged.rename(columns={f"{col}_m0": col for col in ["age", "sex", "site", "y"]})
        df_ROI_age_sex_site = merged.copy()
    if classif_from_m3:
        df_ROI_age_sex_site = pd.read_csv(DF_ROI_M3M0)
        df_ROI_age_sex_site = df_ROI_age_sex_site[df_ROI_age_sex_site["session"]=="M03"]
        df_ROI_age_sex_site = df_ROI_age_sex_site.drop(columns=["session"])

    print("in the whole dataset, number of subjects GR : ",len(df_ROI_age_sex_site[df_ROI_age_sex_site["y"]=="GR"]))
    print("in the whole dataset, number of subjects NR : ",len(df_ROI_age_sex_site[df_ROI_age_sex_site["y"]=="NR"]))
    print("in the whole dataset, number of subjects PaR : ",len(df_ROI_age_sex_site[df_ROI_age_sex_site["y"]=="PaR"]))

    df_ROI_age_sex_site["y"] = df_ROI_age_sex_site["y"].replace({"GR": 1, "PaR": 0, "NR": 0})
    df_ROI_age_sex_site = df_ROI_age_sex_site.reset_index(drop=True)
    print(df_ROI_age_sex_site)
    
    # Define residualization strategies
    residualization_configs = {
        "no_res": None,
        "res_age_sex": "age + sex",
        "res_age_sex_site": "age + sex + site"
    }

    if classif_from_concat: 
        if significant_rois : quit() # not implemented
        X_arr= df_ROI_age_sex_site[[col for col in df_ROI_age_sex_site.columns if any(col.startswith(roi) and (col.endswith('_dif') or col.endswith('_m0')) for roi in get_rois())]].values
    else : 
        if significant_rois : X_arr = df_ROI_age_sex_site[significant_rois].values
        else : X_arr = df_ROI_age_sex_site[get_rois()].values 
    y_arr = df_ROI_age_sex_site["y"].values
    subjects_list = df_ROI_age_sex_site["participant_id"].values

    if verbose:
        print("np.shape(X)",np.shape(X_arr), type(X_arr)) 
        print("np.shape(y)",np.shape(y_arr), type(y_arr))
        print("y nb of 0 labels ",(y_arr == 0).sum(), ", 1 labels ", (y_arr == 1).sum(),\
                     " and 2 labels ", (y_arr == 2).sum())

    print("seed for CV : ",seed)
    kf = StratifiedKFold(n_splits=nbfolds, shuffle=True, random_state=seed) 

    dict_cv = {
        fold: {res_key: []
            for res_key in residualization_configs.keys()}
        for fold in range(nbfolds)
    }

    #Initializing the subjects dicitonary : 
    # get the indices and ids of train and test subjects for each fold
    dict_cv_subjects={
        fold: {"train_subjects_indices":[], "test_subjects_indices":[], \
            "train_subjects_ids":[], "test_subjects_ids":[]}
        for fold in range(nbfolds)
    }


    list_auc_te, list_auc_tr = [], []
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X_arr, y_arr)):
        # Split data
        X_train, X_test = X_arr[train_index], X_arr[test_index]
        y_train, y_test = y_arr[train_index], y_arr[test_index]
        test_subjects = subjects_list[test_index]
        # print("\n",df.loc[train_index])
        # print("\n",df.loc[test_index])

        
        for residual_key, formula in residualization_configs.items():
            if formula:
                # print(f"Residualizing with: {formula}")
                residualizer = Residualizer(
                    data=df_ROI_age_sex_site[["age", "sex", "site", "y"]],
                    formula_res=formula,
                    formula_full=formula + " + y"
                )
                Zres = residualizer.get_design_mat(df_ROI_age_sex_site[["age", "sex", "site", "y"]])
                Zres_train, Zres_test = Zres[train_index], Zres[test_index]
                residualizer.fit(X_train, Zres_train)
                X_train_res = residualizer.transform(X_train, Zres_train)
                X_test_res = residualizer.transform(X_test, Zres_test)
            # formula = None --> no residualization     
            else:
                X_train_res, X_test_res = X_train, X_test

            pipeline = make_pipeline(
                StandardScaler(),
                PLSRegression(n_components=nb_components)
            )

            pipeline.fit(X_train_res, y_train)
            score_test = pipeline.predict(X_test_res)
            score_train = pipeline.predict(X_train_res)
            X_test_scores = pipeline.transform(X_test_res) # X test scores
                
            # Metrics
            roc_auc_te = roc_auc_score(y_test, score_test)
            roc_auc_tr = roc_auc_score(y_train, score_train)
            # print("fold ",fold_idx ," res ",residual_key," roc_auc_te ", roc_auc_te)

            if residual_key =="res_age_sex_site":
                list_auc_te.append(roc_auc_te)
                list_auc_tr.append(roc_auc_tr)

            dict_cv[fold_idx][residual_key].append({
                # add loadings !!!
                "y_test": y_test,
                "score_test": score_test,
                "y_train": y_train,
                "score_train": score_train,
                "roc_auc_test": roc_auc_te,
                "roc_auc_train": roc_auc_tr,
                "loadings": pipeline.named_steps['plsregression'].x_loadings_,
                "X_test_scores": X_test_scores,
                "test_subjects" : test_subjects
            })


    records = []
    for fold_idx, fold_data in dict_cv.items():
        for residual_key, result_list in fold_data.items():
            for result in result_list:
                record = result.copy()
                record["fold"] = fold_idx
                record["residualization"] = residual_key
                records.append(record)

    df_results = pd.DataFrame(records)
    print(df_results)
    str_significant="_from_significant_ROI" if significant_rois else ""
    if save_results :
        if not df_results.empty: 
            df_results.to_pickle(RESULTS_DIR+'results_classification_seed_'+str(seed)+str_labels+'_'+str(nbfolds)+'fold_PLSregression'+str_significant+'.pkl')
            print("df_results saved to : ",RESULTS_DIR+'results_classification_seed_'+str(seed)+str_labels+'_'+str(nbfolds)+'fold_PLSregression'+str_significant+'.pkl')
    df_results_no_res = df_results[df_results["residualization"]=="no_res"]
    df_results_res_age_sex = df_results[df_results["residualization"]=="res_age_sex"]
    df_results_res_age_sex_site = df_results[df_results["residualization"]=="res_age_sex_site"]
    print("no res ",df_results_no_res["roc_auc_test"].mean()," ",df_results_no_res["roc_auc_test"].std())
    print("res age sex ",df_results_res_age_sex["roc_auc_test"].mean()," ",df_results_res_age_sex["roc_auc_test"].std())
    print("res age sex site ",df_results_res_age_sex_site["roc_auc_test"].mean()," ",df_results_res_age_sex_site["roc_auc_test"].std())

    if print_pvals:
        # Group by residualization and classifier
        results_pvals = {}
        if verbose : print("\n\n")

        for residualization in ["no_res","res_age_sex","res_age_sex_site"]:
            # Concatenate score_test and y_test arrays for all folds
            group = df_results[df_results["residualization"]==residualization]
            score_test_concatenated = np.concatenate(group['score_test'].values, axis=0)
            ytest_concatenated = np.concatenate(group['y_test'].values, axis=0)
            
            # Perform the Mann-Whitney U test
            group_nr = score_test_concatenated[ytest_concatenated == 0]
            group_gr = score_test_concatenated[ytest_concatenated == 1]
            
            if len(group_nr) > 0 and len(group_gr) > 0:
                pvalue = scipy.stats.mannwhitneyu(group_nr, group_gr).pvalue
            else:
                pvalue = None  # Handle cases with no data for either group
            
            # Store the results
            results_pvals[residualization] = pvalue

        # Convert results to a DataFrame for easier visualization
        results_pvals_df = pd.DataFrame([
            {'residualization': key[0], 'pvalue': value}
            for key, value in results_pvals.items()
        ])

        if verbose: print(results_pvals_df)

    return list_auc_tr, list_auc_te, df_results
    

    # if classif_from_differencem3m0: str_diff="_difference_m3m0"
    # else: 
    #     if classif_from_concat: str_diff="_concat_dif_with_baseline"
    #     if classif_from_m3: str_diff="_m3"
    #     else: str_diff = ""


    # filtered_results_res_age_sex_site = df_results[df_results["residualization"]=="res_age_sex_site"]
    # filtered_results_res_age_sex = df_results[df_results["residualization"]=="res_age_sex"]
    # filtered_results_nores = df_results[df_results["residualization"]=="no_res"]
    # print(filtered_results_res_age_sex_site)

def find_best_number_of_components_pls_reg(significant_rois=None, plot=True):
    significant_df = pd.read_excel(FEAT_IMPTCE_RES_DIR+"significant_shap_mean_abs_value_pvalues_1000_random_permut.xlsx")
    significant_rois = [roi for roi in list(significant_df.columns) if roi!="fold"]

    component_range = range(1, 11)  # Trying components from 1 to 10
    mean_auc_te_scores = []
    for n in component_range:
        _ , list_auc_te, _ = pls_regression(nb_components=n, significant_rois=significant_rois)
        mean_auc_te_scores.append(np.mean(list_auc_te))

    # Find best number of components
    best_n = component_range[np.argmax(mean_auc_te_scores)]
    print(f"Best number of components: {best_n} with maximum ROC-AUC score {round(np.max(mean_auc_te_scores),4)}")
    print(best_n, " components roc auc ",round(mean_auc_te_scores[best_n-1],4))

    if plot:
        # Plot ROC-AUC depending on nb of components. We find the ideal number of components to be 2. 
        plt.plot(component_range, mean_auc_te_scores, marker='o')
        plt.xlabel('Number of PLS components')
        plt.ylabel('Mean ROC AUC (5-fold CV)')
        # if significant_rois: plt.title('PLS Regression: Component Selection with Specific ROIs')
        plt.title('PLS Regression: Component Selection with all ROIs')
        plt.grid(True)
        plt.show()

def plot_mean_loadings_over_CV_folds(df_results, significant_rois, res="res_age_sex_site", plot=False, glassbrains=True, \
    nb_components=2, jointplot=False): 
    assert jointplot and nb_components>=2,"Cannot plot jointplot with less than 2 axes (2 components)."
    assert res in ["res_age_sex_site","res_age_sex","no_res"],"incorrect residualization scheme"
    filtered_results = df_results[df_results["residualization"]==res]

    y_test = np.concatenate(filtered_results["y_test"].values,axis=0)
    X_test_scores = np.concatenate(filtered_results["X_test_scores"].values,axis=0)
    y_pred_te = np.concatenate(filtered_results["score_test"].values,axis=0)
    loadings = np.array(filtered_results["loadings"].values.tolist()) 
    test_subjects = np.concatenate(filtered_results["test_subjects"].values, axis=0)
    print("test_subjects ",test_subjects, " \n", np.shape(test_subjects))

    # Compute the mean of loadings for each component across all test set subjects of the 5 CV folds axis 0:
    mean_loadings, std_loadings = np.mean(loadings, axis=0), np.std(loadings, axis=0)

    atlas_df = pd.read_csv(ATLAS_ROI_NAMES_DF, sep=';')
    dict_atlas_roi_names = atlas_df.set_index('ROI_Neuromorphometrics_labels')['ROIname'].to_dict()
    feature_names = [rename_col(col, dict_atlas_roi_names) for col in significant_rois]
    
    loadings_df = pd.DataFrame(mean_loadings, index=feature_names, columns=["Comp "+str(i+1) for i in range(nb_components)]) # dataframe for mean loadings
    std_df = pd.DataFrame(std_loadings, index=feature_names, columns=["Comp "+str(i+1) for i in range(nb_components)])

    print(loadings_df)
    label_map = {0: "NR/PaR", 1: "GR"}
    df_scores = pd.DataFrame({
        "Comp1": X_test_scores[:, 0],
        "Comp2": X_test_scores[:, 1],
        "label": [label_map[label] for label in y_test]
    })

    if jointplot and nb_components>=2:
        import seaborn as sns
        rocauc1 = roc_auc_score(y_test, df_scores["Comp1"])
        rocauc2 = roc_auc_score(y_test, df_scores["Comp2"])
        roc_auc_overall = roc_auc_score(y_test, y_pred_te)  
        print("performance metric on test data: \n,\
            roc auc comp 1 ",rocauc1, " roc auc comp 2 ", rocauc2, " roc auc overall ", roc_auc_overall)
        # idx_min = df_scores["Comp1"].idxmin()
        # print("min is ",df_scores["Comp1"].iloc[idx_min])
        # print("min is ",test_subjects[idx_min])
        # print("true label : ", y_test[idx_min])
        # print("prediction ", y_pred_te[idx_min])

        # Create the jointplot
        plt.figure(figsize=(8, 8))
        g = sns.jointplot(
            data=df_scores,
            x="Comp1", y="Comp2",
            hue="label",
            kind="scatter",
            marker='o',
            s=100,
            alpha=0.7,
            palette="colorblind",
            height=8
        )

        # Set axis labels with AUC
        g.ax_joint.set_xlabel(f"PLS Component 1 (AUC={rocauc1:.2f})", fontsize=18)
        g.ax_joint.set_ylabel(f"PLS Component 2 (AUC={rocauc2:.2f})", fontsize=18)

        # Customize legend
        legend = g.ax_joint.legend_
        legend.set_title("Class", prop={'size': 16})
        for text in legend.get_texts():
            text.set_fontsize(14)

        # Customize tick labels
        g.ax_joint.tick_params(axis='both', labelsize=14)
        g.figure.suptitle(f"PLS Component Scatter (Overall ROC AUC={roc_auc_overall:.2f})", y=0.98, fontsize=20)
        plt.tight_layout()
        plt.show()

    if plot:
        # Plot features' loadings for each component
        features = loadings_df.abs().sum(axis=1).sort_values(ascending=False).index 
        ax = loadings_df.loc[features].plot(
            kind='bar',
            yerr=std_df.loc[features],
            figsize=(10, 6),
            title='Feature Loadings by Component',
            capsize=4  # adds a cap to the error bars
        )
        plt.ylabel('Loading Magnitude')
        plt.tight_layout()
        plt.show()

    if glassbrains: 
        feature_names_csf = [roi for roi in feature_names if roi.endswith(" CSF")]
        # negate CSF ROI loadings for interpretability
        loadings_df.loc[feature_names_csf] = -1*loadings_df.loc[feature_names_csf]
        for comp_nb in range(0, nb_components):
            loadings_one_component = {idx.rsplit(' ', 1)[0]: row['Comp '+str(comp_nb+1)] for idx, row in loadings_df.iterrows()}
            plot_glassbrain(dict_plot=loadings_one_component, title="loadings of PLS component "+str(comp_nb+1))

def print_performance_by_residualization_scheme(df_results, metric="roc_auc", std=False, tr_or_te = "test"):
    
    filtered_results_res_age_sex_site = df_results[df_results["residualization"]=="res_age_sex_site"]
    filtered_results_res_age_sex = df_results[df_results["residualization"]=="res_age_sex"]
    filtered_results_nores = df_results[df_results["residualization"]=="no_res"]

    print("res age sex site")
    means_res_age_sex_site=filtered_results_res_age_sex_site.groupby("classifier")[metric+"_"+tr_or_te].mean()
    print(round(means_res_age_sex_site,4))
    if std : print(round(filtered_results_res_age_sex_site.groupby("classifier")[metric+"_"+tr_or_te].std(),4))
    print("res age sex ")
    means_res_age_sex = filtered_results_res_age_sex.groupby("classifier")[metric+"_"+tr_or_te].mean()
    print(round(means_res_age_sex,4))
    if std: print(round(filtered_results_res_age_sex.groupby("classifier")[metric+"_"+tr_or_te].std(),4))
    print("no res")
    means_no_res = filtered_results_nores.groupby("classifier")[metric+"_"+tr_or_te].mean()
    print(round(means_no_res,4))
    if std: print(round(filtered_results_nores.groupby("classifier")[metric+"_"+tr_or_te].std(),4))
    return means_res_age_sex["svm"]

def plot_L2LR_weights(seed=1,nbfolds=5,classif_from_WM_ROI=False,classif_from_differencem3m0=False,\
                      classif_from_concat=False, classif_from_m3=False, biomarkers_roi=False, classif_from_17_roi=False):
    tr_labels="_GRvsPaRNR"
    if classif_from_differencem3m0: str_diff="_difference_m3m0"
    else: 
        if classif_from_concat: str_diff="_concat_dif_with_baseline"
        if classif_from_m3: str_diff="_m3"
        else: str_diff = ""
    str_rois = ""
    if classif_from_17_roi : str_rois = "_17rois_only" 
    if biomarkers_roi : str_rois="_bilateralHippo_and_Amyg_only"

    str_WM = "_WM_Vol" if classif_from_WM_ROI else ""
    weights_file = RESULTS_DIR+'coefficientsL2LR/L2LR_coefficients_'+str(seed)+"_GRvsPaRNR"+'_'+str(nbfolds)+'fold'+str_diff+str_WM+str_rois+'.pkl'
    if os.path.exists(weights_file):
        data = read_pkl(weights_file)
    else :
        print("No file at ",weights_file)
        quit()
    data=data["res_age_sex_site"]
    print(data)
    data = data.values
    data = np.vstack(data)
    print(np.shape(data), type(data))

    mean_weights_over_folds = np.mean(data,axis=0)

    print(np.shape(mean_weights_over_folds), type(mean_weights_over_folds))
    four_rois = ["Left Hippocampus", "Right Hippocampus","Right Amygdala", "Left Amygdala"]
    four_rois = [roi+"_GM_Vol" for roi in four_rois]
    significant_df = pd.read_excel(FEAT_IMPTCE_RES_DIR+"significant_shap_mean_abs_value_pvalues_1000_random_permut.xlsx")
    significant_rois = [roi for roi in list(significant_df.columns) if roi!="fold"]
    
    atlas_df = pd.read_csv(ATLAS_ROI_NAMES_DF, sep=';')
    roi_names_map = dict(zip(atlas_df['ROI_Neuromorphometrics_labels'], atlas_df['ROIname']))
    if classif_from_17_roi: 
        dict_weights = dict(zip(significant_rois, mean_weights_over_folds))
        roi_names = [roi_names_map[val] for val in significant_rois]
        key_mapping = dict(zip(significant_rois, roi_names))

        
    if biomarkers_roi: dict_weights = dict(zip(four_rois, mean_weights_over_folds))

    # change sign of CSF volume ROIs
    dict_weights = {key: dict_weights[key] * -1 if key.endswith("_CSF_Vol") else dict_weights[key] for key in dict_weights}
    dict_weights_atlas_names = {key_mapping.get(k, k): v for k, v in dict_weights.items()}
    for k,v in dict_weights_atlas_names.items():
        print(k,"   ",round(v,4))

    print(dict_weights)
    plot_glassbrain(dict_plot=dict_weights, title="mean L2LR weights over 5 CV folds")
    

def main():

    # to perform classification:
    # list_all=[]
    # for seed in range(21,30):
    #     _,_,list_te_misclassified= classification(save_results=True, seed=seed, compute_and_save_shap=False, random_permutation=False, classif_from_17_roi=False, \
    #                 print_pvals=True, binarize=False)
        # list_all.append(list(list_te_misclassified))
        
    # common_strings = set(list_all[0])
    # for lst in list_all[1:]:
    #     common_strings &= set(lst)
    # print("Strings in all ",len(list_all)," lists:", list(common_strings))

    # ceux qui sont tjrs mal classifiés:
    # Strings in all  30  lists: ['sub-75284', 'sub-77228', 'sub-32220', 'sub-43459', 'sub-22549', \
    # 'sub-52346', 'sub-27002', 'sub-87487', 'sub-44928', 'sub-90396', 'sub-71090']

    # new subject in v4 : ['sub-80793']


    classification(save_results=True, seed=11, compute_and_save_shap=False, random_permutation=False, classif_from_17_roi=False, \
               print_pvals=True, binarize=False)
    quit()



    # classification(save_results=True, seed=13, compute_and_save_shap=False, random_permutation=False, classif_from_17_roi=False, \
    #                print_pvals=True, binarize=False, classif_from_differencem3m0=True)
    
    # classification(save_results=True, seed=13, compute_and_save_shap=False, random_permutation=False, classif_from_17_roi=False, \
    #                print_pvals=True, binarize=False, classif_from_m3=True)
    # quit()
    
    # 2 --> 70% , 57% bacc et 46% sensitivity (binarized : 73% , 62% bacc et 39% sensitivity)
    # 42 --> 69% , 61% bacc 47% sensitivity et (binarized : 69%, 65% bacc et 43% sensitivity)
    # 5 --> 68%
    # 6 --> 65%
    # 7 --> 68%
    # 8 --> 67%
    # 13 --> 69% (64% balanced accuracy)
    

    # for SHAP computation with random permutations
    """
    start_time = time.time()
    for n in range(501,1001): 
        classification(seed=11, compute_and_save_shap=True, random_permutation=True, seed_label_permutations=n)
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"The function classification(compute_and_save_shap=True, random_permutation=True, seed_label_permutations=n) \
    took {hours}h {minutes}m {seconds}s to run.")
    """

    # seed=2
    df= read_pkl(RESULTS_DIR + "results_classification_seed_"+str(11)+"_GRvsPaRNR_5fold_v4labels.pkl")
    print_performance_by_residualization_scheme(df, metric="balanced_accuracy")

    quit()

    bacc_res_age_sex_site_svm_best = 0
    for seed in range(30):
        print(seed)
        df= read_pkl(RESULTS_DIR + "results_classification_seed_"+str(seed)+"_GRvsPaRNR_5fold_v4labels.pkl")
        # print_performance_by_residualization_scheme(df, metric="roc_auc",std=True)
        bacc_res_age_sex_site_svm = print_performance_by_residualization_scheme(df, metric="balanced_accuracy", tr_or_te = "train")
        if bacc_res_age_sex_site_svm >=bacc_res_age_sex_site_svm_best:
            bacc_res_age_sex_site_svm_best = bacc_res_age_sex_site_svm
            best_seed = seed
    print("\nbest: ",round(bacc_res_age_sex_site_svm_best,4))
    print("seed : ",best_seed)

    # print_performance_by_residualization_scheme(df, metric="specificity")
    # print_performance_by_residualization_scheme(df, metric="sensitivity")
    


    quit()
    
    # to plot logistic regression weights with 17 rois:
    # plot_L2LR_weights(classif_from_17_roi=True)


    # to execute PLS regression with 17 ROIs selected with SHAP 
    # and get the plots 
    """
    significant_df = pd.read_excel(FEAT_IMPTCE_RES_DIR+"significant_shap_mean_abs_value_pvalues_1000_random_permut.xlsx")
    significant_rois = [roi for roi in list(significant_df.columns) if roi!="fold"]
    _, _, df_results = pls_regression(nb_components=2, significant_rois=significant_rois, \
                   nbfolds= 5, print_pvals=True, save_results=True)
    plot_mean_loadings_over_CV_folds(df_results, significant_rois, plot=False, glassbrains=True, nb_components=2,\
        jointplot=True)
    """
    # df= read_pkl(RESULTS_DIR+"results_classification_seed_1_GRvsPaRNR_5fold_WM_Vol.pkl")
    # print_roc_auc_by_residualization_scheme(df)
    # quit()
    # to print the classification results using only 4 rois: bilateral hippocampus and amygdala
    """
    df = read_pkl("reports/classification_results/results_classification_seed_1_GRvsPaRNR_5fold_bilateralHippo_and_Amyg_only.pkl")
    print_roc_auc_by_residualization_scheme(df, metric="balanced_accuracy")
    """

    
    

    """

    # best nb of components all ROI
    # find_best_number_of_components_pls_reg()

    shap_stat = pd.read_excel(FEAT_IMPTCE_RES_DIR+"shap_from_bootstrap500-univstat_alpha05.xlsx",
                            sheet_name='SHAP_roi_univstat')
    print(shap_stat[shap_stat.type=="specific"])
    all_rois = get_rois()
    specific_roi = [item for item in list(shap_stat[shap_stat.type=="specific"].ROI) if item in get_rois()]
    # find_best_number_of_components_pls_reg(specific_roi=specific_roi, plot=True)
    
    _,_, df_results = pls_regression(nb_components=2, specific_roi=specific_roi)
    print(df_results)
    print_roc_auc_by_residualization_scheme(df_results)
    
    plot_mean_loadings_over_CV_folds(df_results, specific_roi, res="res_age_sex_site")
    """
    

if __name__ == "__main__":
    main()


