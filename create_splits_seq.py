import pdb
import os
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['RCC', 'BRAIN', 'camelyon16', 'BRCA', 'NSCLC', 'ESCA_typing', 'SV_COAD', 'SV_ESCA', 'SV_READ', 'SV_STAD', 'BRCA_HER2', 'STAD', 'CRC_MSI', 'PRAD', 'BRCA_SV', 'BRCA_SV_IDC', 'NSCLC_SV_LUAD', 'RCC_SV_CCRCC', 'RCC_SV_PRCC', 'STAD_SV', 'BLCA_SV', 'SYSU_SV'], default='SYSU_SV')
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')

args = parser.parse_args()

# import ipdb; ipdb.set_trace()
if args.task == 'NSCLC':
    args.n_classes=2
    label_csv = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-nsclc_label.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-nsclc_label.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'LUAD':0, 'LUSC':1},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'BRCA':
    args.n_classes=2
    label_csv = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_label.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'IDC':0, 'ILC':1},
                            patient_strat=True,
                            ignore=['MDLC', 'PD', 'ACBC', 'IMMC', 'BRCNOS', 'BRCA', 'SPC', 'MBC', 'MPT'])
    
elif args.task == 'BRCA_SV':
    args.n_classes=4
    label_csv = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_surv_label.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_surv_label.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3},
                            patient_strat=False,
                            # label_col='survival_interval',
                            ignore=[])
    
elif args.task == 'BRCA_SV_IDC':
    args.n_classes=4
    label_csv = pd.read_csv('../Dataset/TCGA-BRCA/tcga-brca_surv_IDC_label.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_surv_IDC_label.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3},
                            patient_strat=False,
                            # label_col='survival_interval',
                            ignore=[])
    
elif args.task == 'NSCLC_SV_LUAD':
    args.n_classes=4
    label_csv = pd.read_csv('../Dataset/TCGA-NSCLC/tcga-nsclc_surv_LUAD_label.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-nsclc_surv_LUAD_label.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3},
                            patient_strat=False,
                            # label_col='survival_interval',
                            ignore=[])
    
elif args.task == 'RCC_SV_CCRCC':
    args.n_classes=4
    label_csv = pd.read_csv('../Dataset/TCGA-RCC/tcga-kidney_surv_CCRCC_label.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-RCC/tcga-kidney_surv_CCRCC_label.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3},
                            patient_strat=False,
                            # label_col='survival_interval',
                            ignore=[])
    
elif args.task == 'RCC_SV_PRCC':
    args.n_classes=4
    label_csv = pd.read_csv('../Dataset/TCGA-RCC/tcga-kidney_surv_PRCC_label.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-RCC/tcga-kidney_surv_PRCC_label.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3},
                            patient_strat=False,
                            # label_col='survival_interval',
                            ignore=[])
    
elif args.task == 'STAD_SV':
    args.n_classes=4
    label_csv = pd.read_csv('../Dataset/TCGA-STAD/tcga-stad_surv_label.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-STAD/tcga-stad_surv_label.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3},
                            patient_strat=False,
                            # label_col='survival_interval',
                            ignore=[])
    
elif args.task == 'BLCA_SV':
    args.n_classes=4
    label_csv = pd.read_csv('../Dataset/TCGA-BLCA/tcga-blca_surv_label.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BLCA/tcga-blca_surv_label.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3},
                            patient_strat=False,
                            # label_col='survival_interval',
                            ignore=[])
    
elif args.task == 'BRCA_HER2':
    args.n_classes=2
    label_csv = pd.read_csv('../Dataset/TCGA-BRCA/BRCA_HER2_label.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRCA/BRCA_HER2_label.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Negative':0, 'Positive':1},
                            patient_strat=True,
                            ignore=['Equivocal'])
    
elif args.task == 'STAD':
    args.n_classes=3
    label_csv = pd.read_csv('../Dataset/TCGA-STAD/labels/ebv_msi_others.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-STAD/labels/ebv_msi_others.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'EBV':0, 'MSI':1, 'others':2},
                            patient_strat=True,
                            ignore=[])
    
elif args.task == 'CRC_MSI':
    args.n_classes=3
    label_csv = pd.read_csv('../Dataset/TCGA-CRC/tcga-crc_label_msi.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-CRC/tcga-crc_label_msi.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'MSS':0, 'MSI':1},
                            patient_strat=True,
                            ignore=['Indeterminate'])

elif args.task == 'camelyon16':
    args.n_classes=2
    label_csv = pd.read_csv('../Dataset/Camelyon16/camelyon16_label.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/Camelyon16/camelyon16_label.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal':0, 'tumor':1},
                            patient_strat=True,
                            ignore=[])
elif args.task == 'PRAD':
    args.n_classes=4
    label_csv = pd.read_csv('../Dataset/TCGA-PRAD/tcga-prad_label_gleason.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-PRAD/tcga-prad_label_gleason.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {6:0, 7:1, 8:2, 9:3},
                            patient_strat=True,
                            ignore=[10])

elif args.task == 'BRAIN':
    args.n_classes=2
    label_csv = pd.read_csv('../Dataset/TCGA-BRAIN/tcga-brain_label2.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRAIN/tcga-brain_label2.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'GBM':0, 'LGG':1},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'SV_COAD':
    args.n_classes=4
    label_csv = pd.read_csv('/data1/r10user9/WSI/CL_Dataset/COAD/COAD_survival_label.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/data1/r10user9/WSI/CL_Dataset/COAD/COAD_survival_label.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'SV_ESCA':
    args.n_classes=4
    label_csv = pd.read_csv('/data1/r10user9/WSI/CL_Dataset/ESCA/ESCA_survival_label.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/data1/r10user9/WSI/CL_Dataset/ESCA/ESCA_survival_label.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'SV_READ':
    args.n_classes=4
    label_csv = pd.read_csv('/data1/r10user9/WSI/CL_Dataset/READ/READ_survival_label.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/data1/r10user9/WSI/CL_Dataset/READ/READ_survival_label.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'SV_STAD':
    args.n_classes=4
    label_csv = pd.read_csv('/data1/r10user9/WSI/CL_Dataset/STAD/STAD_survival_label.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/data1/r10user9/WSI/CL_Dataset/STAD/STAD_survival_label.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3},
                            patient_strat=True,
                            ignore=[])
    
elif args.task == 'SYSU_SV':
    args.n_classes=4
    label_csv = pd.read_csv('../Dataset/SYSU/sysu_label_clean.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/SYSU/sysu_label_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3},
                            patient_strat=False,
                            # label_col='survival_interval',
                            ignore=[])

# elif args.task == 'RCC':
#     args.n_classes=2
#     label_csv = pd.read_csv('../Dataset/TCGA-BRAIN/tcga-brain_label2.csv')
#     dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/TCGA-BRAIN/tcga-brain_label2.csv',
#                             shuffle = False, 
#                             seed = args.seed, 
#                             print_info = True,
#                             label_dict = {'GBM':0, 'LGG':1},
#                             patient_strat=True,
#                             ignore=[])

else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = '10fold_splits/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))

            split_csv = pd.read_csv(os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)))
            for j in range(len(split_csv)):
                split_csv.loc[j, 'label'] = label_csv[label_csv['slide_id'] == split_csv.loc[j, 'Unnamed: 0']]['label'].values[0]
            split_csv = split_csv.rename(columns={'Unnamed: 0':'slide_id'})
            split_csv.to_csv(os.path.join(split_dir, 'splits_{}_bool_label.csv'.format(i)), index=None)
