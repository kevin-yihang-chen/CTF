"""Top-level training entry point for CTF.

Performs a 10-fold cross-validated training run on a single task.  The
``TASKS`` dict below ships a single worked example — ``LGGGBM_Grading``
(TCGA-GBMLGG grading) — to mirror the example concept files under
``concepts/``.  Add new tasks by appending an entry with the same fields:
``csv_path``, ``cohort``, ``survival``, ``n_classes``, ``label_dict``,
``label_col``.  Pre-computed splits are read from
``splits/<TASK>/splits_{i}.csv``.
"""

from __future__ import print_function

import argparse
import os
import shutil
import warnings

import numpy as np
import pandas as pd
import torch
import wandb

warnings.filterwarnings('ignore')

from datasets.dataset_generic import Generic_MIL_Dataset
from utils.core_utils import train
from utils.file_utils import save_pkl


# ---------------------------------------------------------------------------
# Per-task configuration.
# ---------------------------------------------------------------------------
# ``cohort`` controls the radiology decoder used by ``Generic_MIL_Dataset``.
# Adjust the CSV paths to point to your local data CSV (see the README and the
# template under ``data_csv_template/``).
# ---------------------------------------------------------------------------
TASKS = {
    # Worked example: TCGA-GBMLGG grading (3-way WHO grade).
    'LGGGBM_Grading': {
        'csv_path': './data_csv/TCGA_LGGGBM_info.csv',
        'cohort': 'tcga',
        'survival': False,
        'n_classes': 3,
        'label_dict': {'G2': 0, 'G3': 1, 'G4': 2},
        'label_col': 'grade',
    },
}


def build_parser():
    parser = argparse.ArgumentParser(description='CTF training')
    parser.add_argument('--data_root_dir', type=str, required=True,
                        help='Directory containing pre-extracted pathology features (.h5).')
    parser.add_argument('--rad_dir', type=str, required=True,
                        help='Directory containing radiology images.')
    parser.add_argument('--task', type=str, required=True, choices=list(TASKS.keys()))
    parser.add_argument('--exp_code', type=str, required=True,
                        help='Top-level experiment name; results go under ``results/<exp_code>/``.')
    parser.add_argument('--test_name', type=str, required=True,
                        help='Run name used for wandb and the per-fold sub-folder.')

    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--reg', type=float, default=1e-6)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--k_start', type=int, default=-1)
    parser.add_argument('--k_end', type=int, default=-1)
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--split_dir', type=str, default=None,
                        help='Override for the splits directory.  Default: splits/<task>/')

    parser.add_argument('--testing', action='store_true', help='Use a small subset of the data for debugging.')
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--weighted_sample', action='store_true')

    # CTF model parameters.
    parser.add_argument('--n_concept', type=int, default=256, help='Number of pathology concepts (radiology uses 128).')
    parser.add_argument('--n_ctx', type=int, default=12, help='Tunable prompt length.')
    parser.add_argument('--cls_hidden_dim', type=int, default=128)
    parser.add_argument('--cls_layers', type=int, default=2)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='CTF')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Optional wandb entity. Falls back to your default if unset.')
    return parser


def main(args, dataset, settings):
    os.makedirs(args.results_dir, exist_ok=True)

    start = 0 if args.k_start == -1 else args.k_start
    end = args.k if args.k_end == -1 else args.k_end

    all_test_auc, all_val_auc, all_test_acc, all_val_acc = [], [], [], []
    folds = np.arange(start, end)

    for i in folds:
        run_kwargs = dict(
            project=args.wandb_project,
            group=args.test_name,
            config=settings,
            name=f'{args.test_name}_fold_{i}',
            mode='offline',
        )
        if args.wandb_entity:
            run_kwargs['entity'] = args.wandb_entity
        if i != 0:
            run_kwargs['reinit'] = 'finish_previous'
        run = wandb.init(**run_kwargs)

        seed_torch(args.seed)
        train_split, val_split, test_split = dataset.return_splits(
            from_id=False,
            csv_path=os.path.join(args.split_dir, f'splits_{i}.csv'),
        )
        results, test_auc, val_auc, test_acc, val_acc = train(
            (train_split, val_split, test_split), i, args,
        )
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)

        save_pkl(os.path.join(args.results_dir, f'split_{i}_results.pkl'), results)

        fold_col = folds[:i + 1].tolist()
        if i == folds[-1]:
            for stat, store in (
                ('mean', np.mean), ('std', np.std),
            ):
                fold_col.append(stat)
                all_test_auc.append(store(all_test_auc))
                all_val_auc.append(store(all_val_auc))
                all_test_acc.append(store(all_test_acc))
                all_val_acc.append(store(all_val_acc))

        df = pd.DataFrame({
            'folds': fold_col,
            'test_auc': all_test_auc, 'val_auc': all_val_auc,
            'test_acc': all_test_acc, 'val_acc': all_val_acc,
        })
        save_name = ('summary_partial_{}_{}.csv'.format(start, end)
                     if len(folds) != args.k else 'summary.csv')
        df.to_csv(os.path.join(args.results_dir, save_name))
        run.finish()


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = build_parser().parse_args()

    # Implicit constants for this release: only the CTF (model_type=moe,
    # fuse=MPMoE) configuration is supported.  Setting them here keeps
    # ``args`` compatible with the rest of the codebase.
    args.model_type = 'moe'
    args.fuse = 'MPMoE'
    args.PT = True
    args.drop_out = True
    args.bag_loss = 'ce'
    args.percent_update = 0.0

    seed_torch(args.seed)

    cfg = TASKS[args.task]
    args.n_classes = cfg['n_classes']
    args.survival = cfg['survival']

    dataset = Generic_MIL_Dataset(
        csv_path=cfg['csv_path'],
        data_dir=args.data_root_dir,
        rad_dir=args.rad_dir,
        cohort=cfg['cohort'],
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict=cfg['label_dict'],
        patient_strat=False,
        label_col=cfg['label_col'],
        ignore=[],
        survival=cfg['survival'],
    )

    args.results_dir = os.path.join(args.results_dir, args.exp_code, f's{args.seed}_{args.test_name}')
    if os.path.exists(args.results_dir):
        shutil.rmtree(args.results_dir)
    os.makedirs(args.results_dir)

    if args.split_dir is None:
        args.split_dir = os.path.join('splits', args.task)
    else:
        args.split_dir = os.path.join('splits', args.split_dir)
    assert os.path.isdir(args.split_dir), f'Splits dir not found: {args.split_dir}'

    settings = {
        'task': args.task,
        'max_epochs': args.max_epochs,
        'lr': args.lr,
        'reg': args.reg,
        'seed': args.seed,
        'n_concept': args.n_concept,
        'n_ctx': args.n_ctx,
        'cls_hidden_dim': args.cls_hidden_dim,
        'cls_layers': args.cls_layers,
        'split_dir': args.split_dir,
    }

    with open(os.path.join(args.results_dir, f'experiment_{args.exp_code}.txt'), 'w') as f:
        print(settings, file=f)

    print('################# Settings ###################')
    for key, val in settings.items():
        print(f'{key}: {val}')

    main(args, dataset, settings)
    print('finished!')
