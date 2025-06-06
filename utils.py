import pickle, os
import pandas as pd, sys
import numpy as np
import nibabel
import xml.etree.ElementTree as ET
from sklearn.preprocessing import StandardScaler

from nilearn import plotting, image
import matplotlib.pyplot as plt
from nilearn.image import resample_to_img
sys.path.append('/neurospin/psy_sbox/temp_sara/')
from pylearn_mulm.mulm.residualizer import Residualizer
# inputs
ROOT ="/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"
DATA_DIR=ROOT+"data/processed/"

FILEM00_ROI = "df_ROI_age_sex_site_fevrier2025_M00_labels_as_strings.csv"
FILEM00_M03_ROI = "df_ROI_age_sex_site_fevrier2025_M00_M03_labels_as_strings.csv"
VOL_FILE_VBM = "/drf/local/spm12/tpm/labels_Neuromorphometrics.nii"
VBMLOOKUP_FILE = "/drf/local/spm12/tpm/labels_Neuromorphometrics.xml"
ONE_SUBJECT_NIFTI = "/neurospin/rlink/REF_DATABASE/derivatives/cat12-vbm-v12.8.2_long/sub-11327/ses-M00/mri/mwp1rusub-11327_ses-M00_acq-3DT1_rec-yBCyGC_run-1_T1w.nii"
BRAIN_MASK_PATH = DATA_DIR+"mni_cerebrum-gm-mask_1.5mm.nii.gz"



def scale_rois_with_tiv(dfROI, all_rois, target_tiv=1500.0):
    """
    aim : scaling the rois of a df of rois with total intracranial volume
    
    dfROI : pandas df of ROIs
    all_rois: list of roi (columns of df) to be scaled using tiv
    target_tiv: target total intracranial volume
    """
    assert "tiv" or "TIV" in list(dfROI.columns), "there should be a column 'tiv' in the dataframe"
    if "tiv" in list(dfROI.columns):
        scaling_factor = target_tiv / dfROI["tiv"]
        dfROI[all_rois+["tiv"]] = dfROI[all_rois+["tiv"]].mul(scaling_factor, axis=0)
    if "TIV" in list(dfROI.columns):
        scaling_factor = target_tiv / dfROI["TIV"]
        dfROI[all_rois+["TIV"]] = dfROI[all_rois+["TIV"]].mul(scaling_factor, axis=0)
    return dfROI

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")

def binarization(X_train, X_test, y_train):
    """
    Binarize features in X_train and X_test based on class-conditional medians from X_train.

    Parameters:
        X_train (ndarray): Training feature matrix (n_samples_train, n_features)
        X_test  (ndarray): Test feature matrix (n_samples_test, n_features)
        y_train (ndarray): Training labels (n_samples_train,), binary (0 or 1)

    Returns:
        X_train_bin (ndarray): Binarized X_train
        X_test_bin  (ndarray): Binarized X_test
    """
    y_train_bool = y_train.astype(bool)

    X_train_bin = np.empty_like(X_train, dtype=float)
    X_test_bin = np.empty_like(X_test, dtype=float)

    for col in range(X_train.shape[1]):
        train_col = X_train[:, col]
        test_col = X_test[:, col]

        median_0 = np.median(train_col[~y_train_bool])
        median_1 = np.median(train_col[y_train_bool])

        threshold = (median_0 + median_1) / 2

        X_train_bin[:, col] = (train_col > threshold).astype(float)
        X_test_bin[:, col] = (test_col > threshold).astype(float)

    return X_train_bin, X_test_bin

def get_scaled_data(res="no_res", dataframe=None, WM_roi=False):
    assert res in ["res_age_sex_site", "res_age_sex", "no_res"],"not the right residualization option for parameter 'res'!"

    # Read data
    if dataframe is None: 
        if WM_roi : df_ROI_age_sex_site = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00_WM_Vol.csv")
        else : df_ROI_age_sex_site = pd.read_csv(DATA_DIR+"df_ROI_age_sex_site_M00.csv")
    else:
        assert isinstance(dataframe, pd.DataFrame), "the 'dataframe' variable provided is not a pandas DataFrame!" 
        df_ROI_age_sex_site = dataframe

    df_ROI_age_sex_site_res = df_ROI_age_sex_site.copy()
    df_ROI_age_sex_site_res["y"] = df_ROI_age_sex_site_res["y"].replace({"GR": 1, "PaR": 0, "NR": 0})
    list_roi = get_rois(WM = WM_roi)
    # if we wish to focus on GM ROI (if we do so, still no regions stand out after correction for multiple tests)
    # but the minimum p-value after correction for multiple tests is 0.104 (ROI with pvalues <=0.11 are L&R Amygdala, L&R Hippocampus, 
    # and Left Cerebellum White Matter
    # with CSF volumes only : lowest p-values after correction are 0.2592
    # list_roi = [ r for r in list_roi if r.endswith("_GM_Vol")] 
    X_arr = df_ROI_age_sex_site_res[list_roi].values

    # residualize and scale ROIs
    # 1. fit residualizer
    if res!="no_res":
        if res=="res_age_sex_site": formula = "age + sex + site"
        elif res=="res_age_sex": formula="age + sex"

        residualizer = Residualizer(
            data=df_ROI_age_sex_site_res,
            formula_res=formula,
            formula_full=formula + " + y"
        )
        Zres = residualizer.get_design_mat(df_ROI_age_sex_site_res[["age", "sex", "site", "y"]])
        residualizer.fit(X_arr, Zres)
        X_arr = residualizer.transform(X_arr, Zres)

    # 2. fit scaler
    scaler_ = StandardScaler()
    X_arr = scaler_.fit_transform(X_arr) # for GM and CSF ROI : (116,268) shape --> 116 subjects, 268 ROI (unless M3-M0, in which case n=91; for WM ROI: 134 ROI)

    df_X = pd.DataFrame(X_arr , columns = list_roi)
    df_X[["age", "sex", "site", "y"]]=df_ROI_age_sex_site[["age", "sex", "site", "y"]]

    return df_X

def get_neuromorphometrics_dict():
    tree = ET.parse(VBMLOOKUP_FILE)
    root = tree.getroot()
    labels_to_index_roi = {}
    index_to_label_roi = {}
    # Find the 'data' section where ROI labels and their indices are stored
    data_section = root.find('data')
    # Iterate through each 'label' element within the 'data' section
    for label in data_section.findall('label'):
        index = label.find('index').text  # Get the text of the 'index' element
        name = label.find('name').text    # Get the text of the 'name' element
        labels_to_index_roi[name] = int(index)         # Add to dictionary
        index_to_label_roi[int(index)]=name

    return index_to_label_roi

def get_rois(WM=False):
    dict_n = get_neuromorphometrics_dict()
    roi_neuromorphometrics = list(dict_n.values())
    roi_neuromorphometrics = [roi for roi in roi_neuromorphometrics if roi not in ["Left vessel","Right vessel"]]
    
    if WM: 
        all_rois = [roi+"_WM_Vol" for roi in roi_neuromorphometrics]
        assert len(all_rois)==134,"wrong number of ROIs" # 134 ROI by hemisphere in Neuromorphometrics

    else: 
        all_rois = [roi+"_GM_Vol" for roi in roi_neuromorphometrics]+[roi+"_CSF_Vol" for roi in roi_neuromorphometrics]
        assert len(all_rois)==134*2,"wrong number of ROIs" # 134 ROI by hemisphere in Neuromorphometrics

    return all_rois

def round_sci(x, sig=2):
    """
        round with scientific notation elements from a df column
        use : df['column_name'].apply(lambda x: round_sci(x, sig=<chosen sigma value>))
    """
    if pd.isna(x) or x == 0:
            return str(x)
    return f"{x:.{sig}e}".replace("e+0", "e").replace("e+","e").replace("e0", "e")

def rename_col(col, correspondence_dict):
    """
    rename columns of a ROI dataframe using correspondence_dict
    
    """
    base_name = correspondence_dict.get(col, col)  # Fallback to original name if not in dict
    if "_GM_Vol" in col:
        return f"{base_name} GM"
    elif "_CSF_Vol" in col:
        return f"{base_name} CSF"
    else:
        return base_name
    
def save_pkl(dict_or_array, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(dict_or_array, file)
    print(f'Item saved to {file_path}')

def read_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def clean_keys_with_check(d):
    cleaned = {}
    for k, v in d.items():
        if k.endswith('_CSF_Vol'):
            base = k[:-8]
        elif k.endswith('_GM_Vol'):
            base = k[:-7]
        else:
            base = k
        if base in cleaned:
            raise ValueError(f"Duplicate base key: {base}")
        cleaned[base] = v
    return cleaned

def info_data():
    df_ROI_age_sex_site = pd.read_csv(ROOT+FILEM00_ROI)
    print("N subjects whole dataset :", len(df_ROI_age_sex_site))
    print("in the whole dataset, number of subjects GR : ",len(df_ROI_age_sex_site[df_ROI_age_sex_site["y"]=="GR"]))
    print("in the whole dataset, number of subjects NR : ",len(df_ROI_age_sex_site[df_ROI_age_sex_site["y"]=="NR"]))
    print("in the whole dataset, number of subjects PaR : ",len(df_ROI_age_sex_site[df_ROI_age_sex_site["y"]=="PaR"]))
    df_ROI_age_sex_site["y"] = df_ROI_age_sex_site["y"].replace({"GR": 1, "PaR": 0, "NR": 0})
    print("in the whole dataset, number of labels 0 : ",len(df_ROI_age_sex_site[df_ROI_age_sex_site["y"]==0]))
    print("in the whole dataset, number of labels 1 : ",len(df_ROI_age_sex_site[df_ROI_age_sex_site["y"]==1]))
    print("in the whole dataset, percentage of women : ",round(100*len(df_ROI_age_sex_site[df_ROI_age_sex_site["sex"]==1])/len(df_ROI_age_sex_site),2))
    print("in the whole dataset, mean age : ",round(np.mean(df_ROI_age_sex_site["age"].values),2))
    print("in the whole dataset, std age : ",round(np.std(df_ROI_age_sex_site["age"].values),2))

    df_ROI_age_sex_site_longitudinal = pd.read_csv(ROOT+FILEM00_M03_ROI)
    check = df_ROI_age_sex_site_longitudinal.groupby('participant_id')[['age', 'sex', 'site', 'y']].nunique()
    assert (check.max() <= 1).all(), "Some participant_ids have different values across age, sex, site, y for session m00 and m03."

    df_unique = df_ROI_age_sex_site_longitudinal[['participant_id', 'age', 'sex', 'site', 'y']].drop_duplicates(subset='participant_id')
    print("\n\nin the longitudinal dataset (subjects with both m00 and m03 data), number of subjects : ",len(df_unique))
    print("in the longitudinal dataset, number of subjects GR : ",len(df_unique[df_unique["y"]=="GR"]))
    print("in the longitudinal dataset, number of subjects NR : ",len(df_unique[df_unique["y"]=="NR"]))
    print("in the longitudinal dataset, number of subjects PaR : ",len(df_unique[df_unique["y"]=="PaR"]))
    df_unique["y"] = df_unique["y"].replace({"GR": 1, "PaR": 0, "NR": 0})
    print("in the longitudinal dataset, number of labels 0 : ",len(df_unique[df_unique["y"]==0]))
    print("in the longitudinal dataset, number of labels 1 : ",len(df_unique[df_unique["y"]==1]))
    print("in the longitudinal dataset, percentage of women : ",round(100*len(df_unique[df_unique["sex"]==1])/len(df_unique),2))
    print("in the longitudinal dataset, mean age : ",round(np.mean(df_unique["age"].values),2))
    print("in the longitudinal dataset, std age : ",round(np.std(df_unique["age"].values),2))

def format_roi(row):
    basename = row['ROIbasename']
    if isinstance(basename, str):
        if basename.startswith('Left/Right:'):
            hemi = 'Left' if row['hemisphere'] == 'l' else 'Right'
            clean_name = basename.split(':', 1)[-1].strip().replace('-', ' ')
            return f"{hemi} {clean_name}"
        else:
            return basename.split(':', 1)[-1].strip().replace('-', ' ')
    else:
        print(f"Non-string ROIbasename type: {type(basename)} at index {row.name}")
        return None  
    


def get_reshaped_4D(array, brain_mask_path_):
    nifti_mask = nibabel.load(brain_mask_path_)
    mask_data = nifti_mask.get_fdata()
    if array.squeeze().shape!= mask_data.shape:
        print("shape different!")
        mask_data = resample_to_img(nifti_mask, nibabel.load(ONE_SUBJECT_NIFTI), interpolation='nearest').get_fdata()
    mask_flat = mask_data.ravel()
    image_shape = mask_data.shape
    nb_subjects = array.shape[0]

    # Create an empty array to fill, with shape (n_subjects, x, y, z)
    reshaped_img = np.zeros((nb_subjects, *image_shape), dtype=array.dtype)

    # Get indices where mask == 1
    mask_indices = np.where(mask_flat == 1)[0]

    for i in range(nb_subjects):
        reshaped_subject = np.zeros(mask_flat.shape, dtype=array.dtype)
        reshaped_subject[mask_indices] = array[i]
        reshaped_img[i] = reshaped_subject.reshape(image_shape)

    return reshaped_img

def get_ROI_info_from_voxels(img=None):
    """
    img: nifti image or 3D array 
    Get the ROIs of importance in the Neuromorphometrics atlas for different classifiers 
        (for forward maps computed from voxel-wise VBM images). 

    ref_arr corresponds to the atlas image, and covariates_arr to the array of covariates from the forward model.
    
    """
    assert img is not None, "the nifti image given to plot_ROI_neuromorphometrics_from_VBMvoxelwise_fwd_maps is None"

    tree = ET.parse(VBMLOOKUP_FILE)
    root = tree.getroot()
    labels_to_index_roi = {}
    index_to_label_roi = {}
    # Find the 'data' section where ROI labels and their indices are stored
    data_section = root.find('data')
    # Iterate through each 'label' element within the 'data' section
    for label in data_section.findall('label'):
        index = label.find('index').text  # Get the text of the 'index' element
        name = label.find('name').text    # Get the text of the 'name' element
        labels_to_index_roi[name] = int(index)         # Add to dictionary
        index_to_label_roi[int(index)]=name
    # print(labels_to_index_roi)
    # neuromorphometrics vol file read
    ref_im = nibabel.load(VOL_FILE_VBM)
    if ref_im.shape!=img.shape and isinstance(img, nibabel.Nifti1Image):
        print("shape different!")
        ref_im = resample_to_img(ref_im, nibabel.load(ONE_SUBJECT_NIFTI), interpolation='nearest')
    else : 
        nifti_mask = nibabel.load(BRAIN_MASK_PATH)

    ref_arr = np.array(ref_im.get_fdata())
    # mask reference image to get fair ratios
    print(np.unique(ref_arr),np.shape(ref_arr))
    ref_arr = ref_im.get_fdata() * nifti_mask.get_fdata()
    print(np.unique(ref_arr),np.shape(ref_arr))
   
    if  isinstance(img, nibabel.Nifti1Image): img_array = np.array(img.get_fdata())
    elif isinstance(img, np.ndarray): 
        assert img.ndim==3,"array is not 3D!"
        img_array=img

    print(type(img_array), np.shape(img_array))
    
    # print("atlas image ",type(ref_im), np.shape(ref_im))    
    labels = list(set(labels_to_index_roi.values()))
    # print(np.shape(labels), type(labels))

    data = {"name": [], "t-statistics": [], "ratio": [], "nb_voxels":[], "percentage_pos": [], "percentage_neg": []}
    for label in labels:
        # Find coordinates of all points in the reference array (MNI space) that match 'label'
        if label in set(np.unique(ref_arr)):
            points_ref = np.asarray(np.where(ref_arr == label)).T
            # moyenne des voxels de l'image nifti sur les points de ref
            gm_data = np.asarray([img_array[loc[0], loc[1], loc[2]] for loc in points_ref])
            ratio = np.count_nonzero(gm_data) * 100 / len(gm_data)
            gm_data[gm_data == 0] = np.nan
            # do not include ROIs that have only 0 as covariates of their composing voxels
            all_nan = np.isnan(gm_data).all()
            if not all_nan: 
                gm = np.nanmean(gm_data)
                percen_pos = 100*np.sum(gm_data[~np.isnan(gm_data)]>0)/(np.sum(gm_data[~np.isnan(gm_data)]>=0)+np.sum(gm_data[~np.isnan(gm_data)]<0))
                percen_neg = 100*np.sum(gm_data[~np.isnan(gm_data)]<0)/(np.sum(gm_data[~np.isnan(gm_data)]>=0)+np.sum(gm_data[~np.isnan(gm_data)]<0))
                data["name"].append(index_to_label_roi[label])
                # if percen_neg>percen_pos  : gm = -gm # for when we compute the mean of the absolute values
                data["t-statistics"].append(gm)
                data["ratio"].append(ratio)
                data["nb_voxels"].append(len(gm_data[~np.isnan(gm_data)]))
                data["percentage_pos"].append(percen_pos)
                data["percentage_neg"].append(percen_neg)
            else : print("all nan! ",index_to_label_roi[label])

        else : print("label ", label ," not in ref image (nii atlas)")

    df = pd.DataFrame.from_dict(data)

    sorted_df = df.sort_values(by='t-statistics')
    ordered_names = sorted_df['name'].to_numpy() # names of ROI from lowest to highest cov values
    sorted_df = sorted_df[sorted_df['nb_voxels'] > 10]
    print(sorted_df)

   
    print("\n")
    print(df[df["name"]=="Left Pallidum"])
    print(df[df["name"]=="Right Pallidum"])
    print(df[df["name"]=="Right Putamen"])
    print(df[df["name"]=="Left Putamen"])
    print("\n")

    # create a dict with the nams of ROI et 
    regions_dict = dict(zip(sorted_df['name'], sorted_df['t-statistics']))
    print(regions_dict)

    texture_arr = np.zeros(ref_arr.shape, dtype=float)
    
    for name, val in regions_dict.items():
        # texture_arr[ref_arr == labels_to_index_roi[name]] = val
        texture_arr[ref_arr == labels_to_index_roi["Left Caudate"]] = 1
        texture_arr[ref_arr == labels_to_index_roi["Right Caudate"]] = 1



    texture_im = nibabel.Nifti1Image(texture_arr, ref_im.affine)
    # plot_surface(texture_im,0,cmap,vmin,vmax,title_)

    display = plotting.plot_glass_brain(
        texture_im,
        display_mode="ortho",
        colorbar=True,
        cmap=plt.cm.coolwarm,
        # vmin =  np.min(texture_arr), 
        # vmax =  np.max(texture_arr),  
        plot_abs=False ,
        alpha = 0.95 ,
        threshold=0,
        title="glass brain")
    plotting.show()
    display.savefig('test.png')

