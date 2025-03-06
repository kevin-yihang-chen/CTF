from __future__ import print_function

import argparse
import pdb
import os
import math
import shutil
import warnings

warnings.filterwarnings('ignore')

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset


# pytorch imports
import torch

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--rad_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--st_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['hipt_lgp', 'hipt_n', 'hit', 'clam_sb', 'clam_mb', 'mil', 'dsmil', 'mamba', 'transmil', 'dtfdmil', 's4', 'wikg', 'mambamil', 'patchgcn', 'moe'], default='clam_sb', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['RCC', 'camelyon16', 'BRCA', 'NSCLC', 'ESCA_typing', 'task_1_tumor_vs_normal',  'task_2_tumor_subtyping', 'BRCA_HER2', 'STAD_EBVMSI', 'CRC_MSI', 'ESCA', 'PRAD', 'BRCA_SV', 'BRCA_SV_IDC', 'NSCLC_SV_LUAD', 'RCC_SV_CCRCC', 'RCC_SV_PRCC', 'STAD_SV', 'BLCA_SV', 'SYSU_SV'])
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
parser.add_argument('--test_name', type=str, help='Test name')

## A-ViT
parser.add_argument('--ponder_token_scale', default=0.0005, type=float, help="")
parser.add_argument('--pretrained', action='store_true',
                    help='raise to load pretrained.')
parser.add_argument('--act_mode', default=4, type=int,
                    help='4-token act, make sure this is always 4, other modes are only used for initial method comparison and exploration')
parser.add_argument('--tensorboard', action='store_true',
                    help='raise to load pretrained.')
parser.add_argument('--gate_scale', default=100., type=float, help="constant for token control gate rescale")
parser.add_argument('--gate_center', default= 3., type=float, help="constant for token control gate re-center, negatived when applied")
parser.add_argument('--warmup_epoch', default=0, type=int, help="warm up epochs for act")
parser.add_argument('--distr_prior_alpha', default=0.001, type=float, help="scaling for kl of distributional prior")

parser.add_argument('--survival', action='store_true', default=False, 
                     help='survival prediction problem')
parser.add_argument('--fuse', default=None, type=str, help='fuse modalities')

args = parser.parse_args()



def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

        fold_col = folds[:i+1].tolist()

        if i == folds[-1]:
            fold_col.append('mean')
            all_test_auc.append(np.mean(all_test_auc))
            all_val_auc.append(np.mean(all_val_auc))
            all_test_acc.append(np.mean(all_test_acc))
            all_val_acc.append(np.mean(all_val_acc))

            fold_col.append('std')
            all_test_auc.append(np.std(all_test_auc))
            all_val_auc.append(np.std(all_val_auc))
            all_test_acc.append(np.std(all_test_acc))
            all_val_acc.append(np.std(all_val_acc))

        final_df = pd.DataFrame({'folds': fold_col, 'test_auc': all_test_auc, 
            'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

        if len(folds) != args.k:
            save_name = 'summary_partial_{}_{}.csv'.format(start, end)
        else:
            save_name = 'summary.csv'
        final_df.to_csv(os.path.join(args.results_dir, save_name))



device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Dataset')


if args.task == 'SYSU_SV':
    args.n_classes = 4
    dataset = Generic_MIL_Dataset(csv_path = '/data1/WSI/Pathology_Radiology/Dataset/SYSU/sysu_label_clean.csv',
                            data_dir= args.data_root_dir,
                            rad_dir = args.rad_dir,
                            st_dir = args.st_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3},
                            patient_strat=False,
                            label_col='survival_interval',
                            ignore=[],
                            survival=True)

    # import ipdb;ipdb.set_trace()
else:
    raise NotImplementedError
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

# args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed) + '_' + args.test_name)
args.results_dir = os.path.join(args.results_dir, str(args.exp_code), 's{}'.format(args.seed) + '_' + args.test_name)
# if not os.path.isdir(args.results_dir):
if os.path.exists(args.results_dir):
    shutil.rmtree(args.results_dir)
os.makedirs(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('10fold_splits', args.task + '_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('10fold_splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")


