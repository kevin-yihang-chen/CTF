from __future__ import print_function, division

import glob
import os

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
import torch
from scipy import stats
from torch.utils.data import Dataset
from torchvision import transforms

from utils.utils import generate_split, nth


def save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index=True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(d) for d in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns=['train', 'val', 'test'])
    df.to_csv(filename)


class Generic_WSI_Classification_Dataset(Dataset):
    def __init__(self,
                 csv_path='dataset_csv/example.csv',
                 shuffle=False,
                 seed=7,
                 print_info=True,
                 label_dict=None,
                 filter_dict=None,
                 ignore=None,
                 patient_strat=False,
                 label_col=None,
                 patient_voting='max'):
        self.label_dict = label_dict or {}
        self.num_classes = len(set(self.label_dict.values()))
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids = (None, None, None)
        self.data_dir = None
        self.rad_dir = None
        self.csv_path = csv_path
        self.label_col = label_col or 'label'
        # Subclasses overwrite these.  Defined here so split-generation tools
        # can use the base class without needing the radiology / WSI plumbing.
        self.survival = False
        self.cohort = None

        slide_data = pd.read_csv(csv_path)
        slide_data = self.filter_df(slide_data, filter_dict or {})
        slide_data = self.df_prep(slide_data, self.label_dict, ignore or [], self.label_col)

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        self.slide_data = slide_data
        self.patient_data_prep(patient_voting)
        self.cls_ids_prep()

        if print_info:
            self.summarize()

    def cls_ids_prep(self):
        self.patient_cls_ids = [np.where(self.patient_data['label'] == i)[0]
                                for i in range(self.num_classes)]
        self.slide_cls_ids = [np.where(self.slide_data['label'] == i)[0]
                              for i in range(self.num_classes)]

    def patient_data_prep(self, patient_voting='max'):
        patients = np.unique(np.array(self.slide_data['case_id']))
        patient_labels = []
        for p in patients:
            label = self.slide_data.loc[self.slide_data['case_id'] == p, 'label'].values
            if patient_voting == 'max':
                patient_labels.append(label.max())
            elif patient_voting == 'maj':
                patient_labels.append(stats.mode(label)[0])
            else:
                raise NotImplementedError
        self.patient_data = {'case_id': patients, 'label': np.array(patient_labels)}

    @staticmethod
    def df_prep(data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()
        data = data[~data['label'].isin(ignore)].reset_index(drop=True)
        for i in data.index:
            data.at[i, 'label'] = label_dict[data.loc[i, 'label']]
        return data

    def filter_df(self, df, filter_dict):
        if not filter_dict:
            return df
        mask = np.full(len(df), True, bool)
        for key, val in filter_dict.items():
            mask = np.logical_and(mask, df[key].isin(val))
        return df[mask]

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        return len(self.slide_data)

    def summarize(self):
        print(f'label column: {self.label_col}')
        print(f'label dictionary: {self.label_dict}')
        print(f'number of classes: {self.num_classes}')
        print('slide-level counts:\n', self.slide_data['label'].value_counts(sort=False))

    def get_split_from_df(self, all_splits, split_key='train'):
        split = all_splits[split_key].dropna().reset_index(drop=True)
        if len(split) == 0:
            return None
        mask = self.slide_data['slide_id'].isin(split.tolist())
        df_slice = self.slide_data[mask].reset_index(drop=True)
        return Generic_Split(df_slice, data_dir=self.data_dir, rad_dir=self.rad_dir,
                             num_classes=self.num_classes, survival=self.survival,
                             cohort=self.cohort)

    def return_splits(self, from_id=True, csv_path=None):
        if from_id:
            return [self._split_from_ids(ids) for ids in
                    (self.train_ids, self.val_ids, self.test_ids)]

        assert csv_path is not None
        all_splits = pd.read_csv(csv_path, dtype=self.slide_data['slide_id'].dtype)
        return [self.get_split_from_df(all_splits, key) for key in ('train', 'val', 'test')]

    def _split_from_ids(self, ids):
        if not ids:
            return None
        data = self.slide_data.loc[ids].reset_index(drop=True)
        return Generic_Split(data, data_dir=self.data_dir, rad_dir=self.rad_dir,
                             num_classes=self.num_classes, survival=self.survival,
                             cohort=self.cohort)

    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    # ------------------------------------------------------------------
    # 10-fold split generation (used by ``make_splits.py``).
    # ------------------------------------------------------------------
    def create_splits(self, k=10, val_num=(25, 25), test_num=(40, 40),
                      custom_test_ids=None):
        settings = {
            'n_splits': k,
            'val_num': val_num,
            'test_num': test_num,
            'seed': self.seed,
            'custom_test_ids': custom_test_ids,
        }
        if self.patient_strat:
            settings.update({'cls_ids': self.patient_cls_ids,
                             'samples': len(self.patient_data['case_id'])})
        else:
            settings.update({'cls_ids': self.slide_cls_ids,
                             'samples': len(self.slide_data)})
        self.split_gen = generate_split(**settings)

    def set_splits(self, start_from=None):
        ids = nth(self.split_gen, start_from) if start_from else next(self.split_gen)

        if self.patient_strat:
            slide_ids = [[] for _ in range(len(ids))]
            for s, split in enumerate(ids):
                for idx in split:
                    case_id = self.patient_data['case_id'][idx]
                    slide_ids[s].extend(
                        self.slide_data[self.slide_data['case_id'] == case_id].index.tolist()
                    )
            self.train_ids, self.val_ids, self.test_ids = slide_ids[0], slide_ids[1], slide_ids[2]
        else:
            self.train_ids, self.val_ids, self.test_ids = ids

    def test_split_gen(self, return_descriptor=False):
        df = None
        if return_descriptor:
            index = [list(self.label_dict.keys())[list(self.label_dict.values()).index(i)]
                     for i in range(self.num_classes)]
            columns = ['train', 'val', 'test']
            df = pd.DataFrame(np.zeros((len(index), len(columns)), dtype=np.int32),
                              index=index, columns=columns)

        for split_name, ids in (('train', self.train_ids),
                                ('val', self.val_ids),
                                ('test', self.test_ids)):
            print(f'\nnumber of {split_name} samples: {len(ids)}')
            labels = self.getlabel(ids)
            unique, counts = np.unique(labels, return_counts=True)
            for u, c in zip(unique, counts):
                print(f'  number of samples in cls {u}: {c}')
                if return_descriptor:
                    df.loc[index[u], split_name] = c

        assert len(np.intersect1d(self.train_ids, self.test_ids)) == 0
        assert len(np.intersect1d(self.train_ids, self.val_ids)) == 0
        assert len(np.intersect1d(self.val_ids, self.test_ids)) == 0
        return df


class Generic_MIL_Dataset(Generic_WSI_Classification_Dataset):
    """Loads paired pathology features + radiology images.

    ``cohort`` controls how the radiology image is decoded.  Options:

    * ``'tcga'``     : nifti volume per case, one folder per case under ``rad_dir``.
    * ``'dicom_2d'`` : single-slice DICOM + nifti mask (used for single-slice CT).
    * ``'nifti_3d'`` : 3-D nifti volume + matching ``_seg`` mask (used for MRI).
    """

    def __init__(self, data_dir, rad_dir, survival, cohort='tcga', **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.rad_dir = rad_dir
        self.use_h5 = True
        self.survival = survival
        self.cohort = cohort
        self.rad_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def _load_path_features(self, slide_id):
        h5_path = os.path.join(self.data_dir, 'h5_files', f'{slide_id}.h5')
        with h5py.File(h5_path, 'r') as hdf5:
            features = hdf5['features'][:]
            coords = hdf5['coords'][:]
        return torch.from_numpy(features), coords

    def _load_radiology(self, slide_id, rad_id=None):
        if self.cohort == 'tcga':
            base = os.path.join(self.rad_dir, slide_id[:12])
            radiology_path = glob.glob(os.path.join(base, '*'))[0]
            volume = nib.load(radiology_path).get_fdata().astype(np.float32)
            volume = torch.from_numpy(volume)
            T = volume.shape[2]
            stack = torch.zeros((T, 3, 224, 224))
            for t in range(T):
                slc = volume[:, :, t]
                slc = slc / (slc.max() + 1e-5)
                slc = slc.unsqueeze(0).repeat(3, 1, 1)
                stack[t] = self.rad_transform(slc)
            return stack

        if self.cohort == 'dicom_2d':
            radiology_path = os.path.join(self.rad_dir, f'{rad_id}.dcm')
            mask_path = os.path.join(self.rad_dir, f'{rad_id}_mask.nii')
            img = pydicom.dcmread(radiology_path).pixel_array.astype(np.float32)
            img = img / (img.max() + 1e-5)
            img = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)
            mask = torch.from_numpy(nib.load(mask_path).get_fdata().astype(np.float32))
            img = img * mask.permute(2, 0, 1)
            return self.rad_transform(img)

        if self.cohort == 'nifti_3d':
            radiology_path = os.path.join(self.rad_dir, rad_id)
            mask_path = radiology_path.replace('img', 'seg')
            image = np.squeeze(nib.load(radiology_path).get_fdata(dtype=np.float32))
            mask = np.squeeze(nib.load(mask_path).get_fdata(dtype=np.float32))
            masked = torch.from_numpy(image * mask)
            T = masked.shape[2]
            stack = torch.zeros((T, 3, 224, 224))
            for t in range(T):
                slc = masked[:, :, t]
                slc = slc / (slc.max() + 1e-5)
                slc = slc.unsqueeze(0).repeat(3, 1, 1)
                stack[t] = self.rad_transform(slc)
            return stack

        raise ValueError(f'Unknown cohort type: {self.cohort}')

    def __getitem__(self, idx):
        slide_id = self.slide_data['slide_id'][idx]
        rad_id = self.slide_data['radiology_id'][idx] if 'radiology_id' in self.slide_data.columns else None

        path_features, coords = self._load_path_features(slide_id)
        rad_img = self._load_radiology(slide_id, rad_id)

        if self.survival:
            censor = 0 if self.slide_data['censor'][idx] == 0 else 1
            survival_days = self.slide_data['survival_days'][idx]
            survival_interval = self.slide_data['survival_interval'][idx]
            return (path_features, rad_img, None, censor, survival_days,
                    survival_interval, slide_id, coords)

        label_col = 'label' if 'label' in self.slide_data.columns else 'grade'
        label = self.slide_data[label_col][idx]
        return path_features, rad_img, label, slide_id, coords


class Generic_Split(Generic_MIL_Dataset):
    def __init__(self, slide_data, data_dir=None, rad_dir=None,
                 num_classes=2, survival=True, cohort='tcga'):
        self.use_h5 = False
        self.slide_data = slide_data
        self.data_dir = data_dir
        self.rad_dir = rad_dir
        self.num_classes = num_classes
        self.survival = survival
        self.cohort = cohort
        self.slide_cls_ids = [np.where(self.slide_data['label'] == i)[0]
                              for i in range(self.num_classes)]
        self.rad_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def __len__(self):
        return len(self.slide_data)
