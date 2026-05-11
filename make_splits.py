"""Generate the 10-fold cross-validation splits expected by ``train.py``.

Usage::

    python make_splits.py --task LGGGBM_Grading \\
        --label_csv data_csv/TCGA_LGGGBM_info.csv

This produces, under ``splits/<TASK>/``:

* ``splits_<i>.csv``              — long-format split (one column each for train/val/test slide IDs).
* ``splits_<i>_bool.csv``         — wide boolean matrix indexed by slide ID.
* ``splits_<i>_descriptor.csv``   — class-balance summary per fold.
* ``splits_<i>_bool_label.csv``   — slide-level labels merged onto the boolean matrix.

The label CSV must contain at least:

* ``case_id``           — one row per slide; multiple slides may share a case.
* ``slide_id``          — unique slide identifier (used as the split key).
* the column named by ``--label_col`` for the target task.

For survival tasks, also include the ``survival_interval`` column the splits
script records into ``splits_<i>_bool_label.csv``.
"""

import argparse
import os

import numpy as np
import pandas as pd

from datasets.dataset_generic import Generic_WSI_Classification_Dataset, save_splits


# Per-task configuration mirroring ``train.py``'s ``TASKS`` dict, but for the
# splitting stage (no radiology decoder needed).  Add new entries with the same
# fields when extending CTF to other cohorts.
TASKS = {
    # Worked example: TCGA-GBMLGG grading (3-way WHO grade).
    'LGGGBM_Grading': {'survival': False, 'n_classes': 3,
                       'label_dict': {'G2': 0, 'G3': 1, 'G4': 2},
                       'label_col': 'grade'},
}


def build_parser():
    parser = argparse.ArgumentParser(description='Create 10-fold CV splits for CTF.')
    parser.add_argument('--task', required=True, choices=list(TASKS.keys()))
    parser.add_argument('--label_csv', required=True,
                        help='Path to the label CSV (see data_csv_template/ for column layout).')
    parser.add_argument('--out_dir', default='splits',
                        help='Root output directory; per-task subfolders are created beneath it.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--k', type=int, default=10, help='Number of folds.')
    parser.add_argument('--val_frac', type=float, default=0.1)
    parser.add_argument('--test_frac', type=float, default=0.1)
    return parser


def main():
    args = build_parser().parse_args()
    cfg = TASKS[args.task]

    label_df = pd.read_csv(args.label_csv)
    dataset = Generic_WSI_Classification_Dataset(
        csv_path=args.label_csv,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict=cfg['label_dict'],
        patient_strat=False,
        label_col=cfg['label_col'],
        ignore=[],
    )

    counts = np.array([len(c) for c in dataset.patient_cls_ids])
    val_num = np.round(counts * args.val_frac).astype(int)
    test_num = np.round(counts * args.test_frac).astype(int)
    print(f'per-class val / test sizes: {val_num.tolist()} / {test_num.tolist()}')

    split_dir = os.path.join(args.out_dir, args.task)
    os.makedirs(split_dir, exist_ok=True)
    dataset.create_splits(k=args.k, val_num=val_num, test_num=test_num)

    for i in range(args.k):
        dataset.set_splits()
        descriptor_df = dataset.test_split_gen(return_descriptor=True)
        splits = dataset.return_splits(from_id=True)
        save_splits(splits, ['train', 'val', 'test'],
                    os.path.join(split_dir, f'splits_{i}.csv'))
        save_splits(splits, ['train', 'val', 'test'],
                    os.path.join(split_dir, f'splits_{i}_bool.csv'),
                    boolean_style=True)
        descriptor_df.to_csv(os.path.join(split_dir, f'splits_{i}_descriptor.csv'))

        # Merge the survival / grade label onto the boolean split matrix —
        # ``concept_generate/selection.py`` reads this for MI ranking.
        bool_path = os.path.join(split_dir, f'splits_{i}_bool.csv')
        split_csv = pd.read_csv(bool_path)
        slide_col = 'Unnamed: 0'
        for j in range(len(split_csv)):
            slide_id = split_csv.loc[j, slide_col]
            row = label_df[label_df['slide_id'] == slide_id]
            if len(row) == 0:
                continue
            split_csv.loc[j, cfg['label_col']] = row[cfg['label_col']].values[0]
            if cfg['survival'] and 'survival_interval' in label_df.columns and cfg['label_col'] != 'survival_interval':
                split_csv.loc[j, 'survival_interval'] = row['survival_interval'].values[0]
            # ``selection.py`` looks for a generic ``label`` column.
            split_csv.loc[j, 'label'] = cfg['label_dict'][row[cfg['label_col']].values[0]]
        split_csv = split_csv.rename(columns={slide_col: 'slide_id'})
        split_csv.to_csv(os.path.join(split_dir, f'splits_{i}_bool_label.csv'), index=False)

    print(f'wrote {args.k} folds to {split_dir}')


if __name__ == '__main__':
    main()
