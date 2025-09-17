from torch.utils.data.dataset import Dataset
from abc import ABC
import pandas as pd
import numpy as np
import os, torch
from typing import Callable
from sara_utils import read_pkl, get_reshaped_4D
import nibabel, json, gc
from sklearn.model_selection import train_test_split


BRAIN_MASK = "binary_brain_mask.npy"


class RlinkDataset(ABC, Dataset):


    def __init__(self, split: str='train', transforms: Callable[[np.ndarray], np.ndarray]=None):
        
        self.transforms = transforms

        if self.transforms : 
            print("transforms applied")

        self.df= pd.read_pickle("mris_without_brainmask.pkl") 
        self.y = np.array(self.df['y'].tolist())
        self.labels = None
        self.data = None
        self.X_train, self.X_test = None,None
        self.y_train, self.y_test = None, None
        self.brainmask = np.load(BRAIN_MASK)
        assert split in ["train", "test"]
        
        if split=="train":
            ########## things to do when we use train as it is the first time we create the instance RlinkDataset ################
            # loading data from the dataframe file # columns for X, y, age, sex, centernum (number associated to the site for each subject)
            subjects_tr = np.loadtxt("tr_subjects.csv", delimiter=",", dtype=str)
            df_tr = self.df[self.df["participant_id"].isin(subjects_tr)]
            # extracting MRIs and labels from the dataframe
            Xtrain = df_tr['X'].tolist()
            self.y_train = np.array(df_tr['y'].tolist())
            
            # applying the binary brain mask to MRIs
            masked_X = []
            for element in Xtrain:
                masked_X.append(element*self.brainmask)

            self.X_train = np.array(masked_X)
            ##################################################################################################################

            # self.X_train, self.X_test, self.y_train, self.y_test, self.subjects_tr, self.subjects_te = \
            #     train_test_split(X, y, subjects,  test_size=0.25, stratify=y, random_state=42)
            # print("Train  / Test split done. \n\n")
            # print("X_train, X_test, y_train, y_test", self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)
            num_zeros = np.bincount(self.y )[0]
            num_ones = np.bincount(self.y )[1]
            print("Number of non responders in total:", num_zeros)
            print("Number of good responders  in total:", num_ones,"\n")

            num_zeros = np.bincount(self.y_train)[0]
            num_ones = np.bincount(self.y_train)[1]

            print("Number of non responders in training set:", num_zeros)
            print("Number of good responders  in training set:", num_ones,"\n")

            # np.savetxt("tr_subjects.csv", self.subjects_tr, delimiter=",", fmt="%s")
            # np.savetxt("te_subjects.csv", self.subjects_te, delimiter=",", fmt="%s")
            
            self.data = np.reshape(self.X_train, (self.X_train.shape[0], 1, *self.X_train.shape[1:]))
            self.labels = self.y_train
            print("shape data ", np.shape(self.data))

        if split=="test":
            # extracting MRIs and labels from the dataframe
            subjects_te = np.loadtxt("te_subjects.csv", delimiter=",", dtype=str)
            df_te = self.df[self.df["participant_id"].isin(subjects_te)]
            Xtest = df_te['X'].tolist()
            self.y_test = np.array(df_te['y'].tolist())

            # applying the binary brain mask to MRIs
            print("\napplying mask ...")
            masked_X = []
            for element in Xtest:
                masked_X.append(element*self.brainmask)
            print("...done.\n")

            self.X_test = np.array(masked_X)

            num_zeros = np.bincount(self.y_test)[0]
            num_ones = np.bincount(self.y_test)[1]

            print("Number of non responders in testing set:", num_zeros)
            print("Number of good responders  in testing set:", num_ones,"\n")

            self.data = np.reshape(self.X_test, (self.X_test.shape[0], 1, *self.X_test.shape[1:]))
            self.labels = self.y_test
            print("shape data ", np.shape(self.data))
            assert len(self.data)==len(self.labels)
            # np.savetxt("tr_subjectsBIS.csv", self.subjects_tr, delimiter=",", fmt="%s")
            # np.savetxt("te_subjectsBIS.csv", self.subjects_te, delimiter=",", fmt="%s")
        
        self.shape = np.shape(self.data)
        gc.collect()

    def _get_train(self, idx: int):
        assert len(self.y_train)==len(self.X_train)
        assert self.y_train is not None, "labels are missing"
        assert self.X_train is not None, "data is missing"

        return self.X_train, self.y_train, self.subjects_tr
    
    def _get_test(self, idx: int):
        assert len(self.y_test)==len(self.X_test)
        assert self.y_test is not None, "labels are missing"
        assert self.X_test is not None, "data is missing"

        return self.X_test, self.y_test, self.subjects_te
    
    def __getitem__(self, idx: int):
        sample, target = self.data[idx], self.labels[idx]

        if self.transforms:
            sample = self.transforms(sample)

        return sample, target.astype(np.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __get_df__(self):
        return self.df
    
    

    def __str__(self):
        return "%s-%s-%s"%(type(self).__name__)
    
        
    
    
