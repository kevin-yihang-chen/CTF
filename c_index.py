import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.utils import get_split_loader
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
from models.par import PaR
import argparse
import os



parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--rad_dir', type=str, default=None,
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
parser.add_argument('--task', type=str, choices=['SYSU_SV','LGG_SV','GBM_SV'])
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
parser.add_argument('--pretrain', default=None, type=str, help="pretrained model path")
parser.add_argument('--survival', action='store_true', default=False,
                     help='survival prediction problem')
parser.add_argument('--fuse', default=None, type=str, help='fuse modalities')
parser.add_argument('--PT', action='store_true', help='whether to use prompt tuning')
parser.add_argument('--n_ctx', default=8, type=int, help='number of prompt tokens')
parser.add_argument('--cls_hidden_dim', default=128, type=int, help='hidden dimension of the classifier')
parser.add_argument('--cls_layers', default=1, type=int, help='number of layers in the classifier')

args = parser.parse_args()

if args.split_dir is None:
    args.split_dir = os.path.join('10fold_splits', args.task + '_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('10fold_splits', args.split_dir)

if args.task == 'SYSU_SV':
    args.n_classes = 4
    dataset = Generic_MIL_Dataset(csv_path = '/data1/WSI/Pathology_Radiology/Dataset/SYSU/sysu_label_clean.csv',
                            data_dir= args.data_root_dir,
                            rad_dir = args.rad_dir,
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3},
                            patient_strat=False,
                            label_col='survival_interval',
                            ignore=[],
                            survival=True)

    # import ipdb;ipdb.set_trace()
elif args.task == 'LGG_SV':
    args.n_classes = 4
    dataset = Generic_MIL_Dataset(csv_path = '/data1/yhchen/TCGA-LGG/TCGA-LGG_survival_info.csv',
                            data_dir= args.data_root_dir,
                            rad_dir = args.rad_dir,
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3},
                            patient_strat=False,
                            label_col='survival_interval',
                            ignore=[],
                            survival=True)
elif args.task == 'GBM_SV':
    args.n_classes = 4
    dataset = Generic_MIL_Dataset(csv_path = '/data1/yhchen/TCGA-GBM/TCGA-GBM_survival_info.csv',
                            data_dir= args.data_root_dir,
                            rad_dir = args.rad_dir,
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3},
                            patient_strat=False,
                            label_col='survival_interval',
                            ignore=[],
                            survival=True)

all_data, _, _ = dataset.return_splits(from_id=False,
                csv_path='{}/all_samples.csv'.format(args.split_dir))
loader = get_split_loader(all_data)
all_risk_scores = np.zeros((len(loader)))
all_censorships = np.zeros((len(loader)))
all_event_times = np.zeros((len(loader)))
sim_paths = []
sim_rads = []
slide_ids = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PaR(n_classes=args.n_classes,concept_risk_align=False,args=args)
model.load_state_dict(torch.load(args.pretrain))
model = model.to(device)
model.eval()

def calculate_risk(h):
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()

with torch.no_grad():
    for i, (data, data2, data3, censor, survival_months, label, slide_id, coords) in enumerate(loader):
        try:
            data, data2, data3, label = data.to(device), data2.to(device), data3.to(device), label.to(device)
        except:
            data, label = data[0].to(device), label.to(device)
        if args.fuse == 'PIBD':
            h, _, _, _, _ = model(data, data2, data3)
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
            censor, label, survival_months = censor.to(device), label.to(device), survival_months.to(device)
            risk = calculate_risk(h)
            all_risk_scores[i] = risk[0]
            all_censorships[i] = censor
            all_event_times[i] = survival_months
        elif args.fuse == 'MPMoE':
            logits, Y_hat, Y_prob, _ = model(data, data2, data3)
            sim_path, sim_rad = model.get_sim()
            sim_paths.append(sim_path.cpu())
            sim_rads.append(sim_rad.cpu())
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            survival_labels = torch.LongTensor(1)
            survival_labels[0] = label
            survival_labels = survival_labels.cuda()
            censoreds = torch.LongTensor(1)
            censoreds[0] = censor
            censoreds = censoreds.cuda()
            risk = -torch.sum(S, dim=1).cpu().detach().numpy()
            all_risk_scores[i] = risk
            all_censorships[i] = censor
            all_event_times[i] = survival_months
        slide_ids.append(slide_id[0])
# save the results as csv
results_df = pd.DataFrame({
    'slide_id': slide_ids,
    'risk_score': all_risk_scores,
    'censor': all_censorships,
    'event_time': all_event_times
    # 'sim_path': sim_paths,
    # 'sim_rad': sim_rads
})
results_df.to_csv(os.path.join(args.results_dir, f'risk_scores_{args.task}.csv'), index=False)