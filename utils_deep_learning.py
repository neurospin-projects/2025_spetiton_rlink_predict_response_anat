import nibabel
import numpy as np
import pickle
from nilearn.image import resample_to_img

ONE_SUBJECT_NIFTI = "/neurospin/rlink/REF_DATABASE/derivatives/cat12-vbm-v12.8.2_long/sub-11327/ses-M00/mri/mwp1rusub-11327_ses-M00_acq-3DT1_rec-yBCyGC_run-1_T1w.nii"

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



def read_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data