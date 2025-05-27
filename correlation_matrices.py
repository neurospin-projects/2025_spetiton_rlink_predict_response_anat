import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from utils import get_scaled_data
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster
import seaborn as sns

# inputs
ROOT ="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"
DATA_DIR=ROOT+"data/processed/"
FEAT_IMPTCE_RES_DIR = ROOT+"reports/feature_importance_results/"

def matrice_corr_biomarkers_rois(significant_17rois=False, display_dendrogram=False, display_correlation_matrix=False, clustering=True):
    df_ROI_age_sex_site = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00.csv")
    df_ROI_age_sex_site["y"] = df_ROI_age_sex_site["y"].replace({"GR": 1, "PaR": 0, "NR": 0})
    df_ROI_age_sex_site = df_ROI_age_sex_site.reset_index(drop=True)
    four_rois = ["Left Hippocampus", "Right Hippocampus","Right Amygdala", "Left Amygdala"]
    four_rois = [roi+"_GM_Vol" for roi in four_rois]
    significant_df = pd.read_excel(FEAT_IMPTCE_RES_DIR+"significant_shap_mean_abs_value_pvalues_1000_random_permut.xlsx")
    significant_rois = [roi for roi in list(significant_df.columns) if roi!="fold"]
    
    df_rois = get_scaled_data(res="res_age_sex_site")
    if significant_17rois: df_rois=df_rois[significant_rois]
    else : df_rois=df_rois[four_rois]

    atlas_df = pd.read_csv(ROOT+"data/processed/lobes_Neuromorphometrics_with_dfROI_correspondencies.csv", sep=';')
    roi_names_map = dict(zip(atlas_df['ROI_Neuromorphometrics_labels'], atlas_df['ROIname']))
    roi_names = [roi_names_map[val] for val in list(df_rois.columns)]
    df_rois.columns = roi_names
    print(df_rois)

    # compute correlation matrix
    corr_matrix = df_rois.corr()
    
    if clustering and significant_17rois: # no clustering on 4 ROIs 
        
        # convert correlation to distance (1 - correlation)
        dist_matrix = 1 - corr_matrix
        dist_condensed = squareform(dist_matrix.values)
        # hierarchical clustering
        linkage_matrix = sch.linkage(dist_condensed, method='ward') 

        plt.figure(figsize=(8, 4))
        max_d = linkage_matrix[-4 + 1, 2]
        dendro = sch.dendrogram(linkage_matrix, labels=corr_matrix.columns, color_threshold=max_d)
        ordered_labels = dendro["ivl"]
        corr_matrix = corr_matrix.loc[ordered_labels, ordered_labels]
        if display_dendrogram:
            plt.title("Hierarchical Clustering Dendrogram")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        
        cluster_labels = fcluster(linkage_matrix, t=4, criterion='maxclust')

        # Optional: assign clusters to variables
        cluster_assignments = dict(zip(corr_matrix.columns, cluster_labels))
        print(cluster_assignments)

    if display_correlation_matrix:
        plt.figure(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Correlation Matrix")
        plt.xticks(rotation=45, ha='right') 
        plt.tight_layout()
        plt.show()

def main():
    # to plot the correlation matrix with selected rois:
    matrice_corr_biomarkers_rois(significant_17rois=True, display_correlation_matrix=True)

if __name__ == "__main__":
    main()


