import nibabel
import pandas as pd, numpy as np, re
import nilearn.plotting as plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# inputs
ROOT = "/neurospin/signatures/2025_spetiton_rlink_predict_response_anat/"
VOL_FILE_VBM = "/drf/local/spm12/tpm/labels_Neuromorphometrics.nii"
DATA_DIR = ROOT+"data/processed/"


def plot_glassbrain(dict_plot=None, title="", list_negative=None, neuromorphometrics_roi_names=True): 
    """
        Aim : plot glassbrain of specfic ROI from SHAP values obtained with an SVM-RBF and VBM ROI features
    """    

    print("dict_plot\n")
    for k,v in dict_plot.items():
        print(k, "  ",v)

    ref_im = nibabel.load(VOL_FILE_VBM)
    ref_arr = ref_im.get_fdata()
    # labels = sorted(set(np.unique(ref_arr).astype(int))- {0}) # 136 labels --> 'Left Inf Lat Vent', 'Right vessel', 'Left vessel' missing in data
    atlas_df = pd.read_csv(DATA_DIR+"lobes_Neuromorphometrics_with_dfROI_correspondencies.csv", sep=';')
    texture_arr = np.zeros(ref_arr.shape, dtype=float)
    
    for name, val in dict_plot.items():
        # each baseid is the number associated to the ROI in the nifti image
        if neuromorphometrics_roi_names: baseids = atlas_df[(atlas_df['ROI_Neuromorphometrics_labels'] == name)]["ROIbaseid"].values
        else : baseids = atlas_df[(atlas_df['ROIname'] == name)]["ROIbaseid"].values

        int_list = list(map(int, re.findall(r'\d+', baseids[0])))
        if "Left" in name: 
            if len(int_list)==2: baseid = int_list[1]
            else : baseid = int_list[0]
        else : baseid = int_list[0]
        if list_negative:
            if name in list_negative: texture_arr[ref_arr == baseid] = -val
            else : texture_arr[ref_arr == baseid] = val
        else : texture_arr[ref_arr == baseid] = val

    print("nb unique vals :",len(np.unique(texture_arr)), " \n",np.unique(texture_arr))
    print(np.shape(texture_arr))

    cmap = plt.cm.coolwarm
    vmin = np.min(texture_arr)
    vmax = np.max(texture_arr)
    print("vmin vmax texture arr", vmin,"     ",vmax)
    texture_im = nibabel.Nifti1Image(texture_arr, ref_im.affine)
    if vmin==0:
        # if all values are positive, the color map should be a gradient from white (0) to red (max value)
        red_from_coolwarm = plt.cm.coolwarm(vmax)
        cmap = LinearSegmentedColormap.from_list("white_to_red", ["white", red_from_coolwarm])

    
    plotting.plot_glass_brain(
        texture_im,
        display_mode="ortho",
        colorbar=True,
        cmap=cmap,
        plot_abs=False ,
        alpha = 0.95 ,
        threshold=0,
        title=title)
    plotting.show() 

