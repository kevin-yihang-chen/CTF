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
parser.add_argument('--task', type=str, choices=['SYSU_SV','LGG_SV','GBM_SV','Beijing_Grading',"LGGGBM_Grading", 'SYSU_Grading'], default='LGGGBM_Grading')
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')
parser.add_argument('--shots', type=int, nargs='+', default=None,
                    help='List of shots (number of training samples per class) to generate splits for.')

args = parser.parse_args()


def create_few_shot_splits(label_csv, label_col, n_shots, val_frac, test_frac, seed):

    random_state = np.random.RandomState(seed)

    df = label_csv
    # 2. Get class labels and count directly from the dataframe
    class_labels = sorted(df[label_col].unique())
    n_classes = len(class_labels)
    all_train_indices, all_val_indices, all_test_indices = [], [], []
    print(f"\n--- Creating splits for {n_shots}-shot with seed {seed} ---")
    # 3. Iterate through each actual class label (e.g., 1, 2, 3...)
    for i, label in enumerate(class_labels):
        # Find the original DataFrame indices for all slides in the current class
        class_indices = df.index[df[label_col] == label].tolist()

        num_slides_in_class = len(class_indices)
        class_indices = np.array(class_indices)
        test_num = round(num_slides_in_class * test_frac)
        val_num = round(num_slides_in_class * val_frac)
        train_num = n_shots
        print(
            f"Class '{label}' (idx {i}): Total={num_slides_in_class}, Train={train_num}, Val={val_num}, Test={test_num}")
        if train_num + val_num + test_num > num_slides_in_class:
            raise ValueError(f"Not enough data in class '{label}' for the specified splits. "
                             f"Required: {train_num + val_num + test_num}, "
                             f"Available: {num_slides_in_class}. "
                             "Consider reducing val_frac or test_frac.")
        # Shuffle the indices for this class
        random_state.shuffle(class_indices)
        # Split the *indices* into train, val, and test sets for this class
        class_test_indices = class_indices[:test_num]
        class_val_indices = class_indices[test_num: test_num + val_num]
        class_train_indices = class_indices[test_num + val_num: test_num + val_num + train_num]
        # Add the split indices to the master lists
        all_test_indices.extend(class_test_indices)
        all_val_indices.extend(class_val_indices)
        all_train_indices.extend(class_train_indices)
    # Map the final lists of indices back to the actual slide_id strings
    train_ids = df['slide_id'].iloc[all_train_indices].tolist()
    val_ids = df['slide_id'].iloc[all_val_indices].tolist()
    test_ids = df['slide_id'].iloc[all_test_indices].tolist()
    # Shuffle the final lists of slide_ids for randomness across classes
    random_state.shuffle(train_ids)
    random_state.shuffle(val_ids)
    random_state.shuffle(test_ids)
    return train_ids, val_ids, test_ids

# import ipdb; ipdb.set_trace()

    
if args.task == 'SYSU_SV':
    args.n_classes=4
    label_csv = pd.read_csv('../Dataset/SYSU/sysu_label_clean.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '../Dataset/SYSU/sysu_label_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3},
                            patient_strat=False,
                            label_col='survival_interval',
                            ignore=[])
elif args.task == 'LGG_SV':
    args.n_classes=4
    label_csv = pd.read_csv('/data1/yhchen/TCGA-LGG/TCGA-LGG_survival_info.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/data1/yhchen/TCGA-LGG/TCGA-LGG_survival_info.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3},
                            patient_strat=False,
                            label_col='survival_interval',
                            ignore=[])

elif args.task == 'GBM_SV':
    args.n_classes=4
    label_csv = pd.read_csv('/data1/yhchen/TCGA-GBM/TCGA-GBM_survival_info.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/data1/yhchen/TCGA-GBM/TCGA-GBM_survival_info.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3},
                            patient_strat=False,
                            label_col='survival_interval',
                            ignore=[])

elif args.task == 'Beijing_Grading':
    args.n_classes=5
    label_csv = pd.read_csv('/home/yhchen/Documents/CONCH/BJJST_label.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/home/yhchen/Documents/CONCH/BJJST_label.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {0:0, 1:1, 2:2, 3:3, 4:4},
                            patient_strat=False,
                            label_col='label',
                            ignore=[])

elif args.task == 'LGGGBM_Grading':
    args.n_classes=3
    label_csv = pd.read_csv('/home/yhchen/Documents/CONCH/TCGA_LGGGBM_info.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/home/yhchen/Documents/CONCH/TCGA_LGGGBM_info.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {"G2":0, "G3":1, "G4":2},
                            patient_strat=False,
                            label_col='grade',
                            ignore=[])

elif args.task == 'SYSU_Grading':
    args.n_classes=5
    label_csv = pd.read_csv('/home/yhchen/Documents/CONCH/sysu_label.csv')
    dataset = Generic_WSI_Classification_Dataset(csv_path = '/home/yhchen/Documents/CONCH/sysu_label.csv',
                            shuffle = False,
                            seed = args.seed,
                            print_info = True,
                            label_dict = {1:0, 2:1, 3:2, 4:3, 5:4},
                            patient_strat=False,
                            label_col='T',
                            ignore=[])

else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    if args.shots is None:
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
                    if "SV" in args.task:
                        split_csv.loc[j, 'survival_interval'] = label_csv[label_csv['slide_id'] == split_csv.loc[j, 'Unnamed: 0']]['survival_interval'].values[0]
                    elif "Grading" in args.task:
                        if args.task == "LGGGBM_Grading":
                            split_csv.loc[j, 'label'] = label_csv[label_csv['slide_id'] == split_csv.loc[j, 'Unnamed: 0']]['grade'].values[0]
                        elif args.task == "SYSU_Grading":
                            split_csv.loc[j, 'label'] = label_csv[label_csv['slide_id'] == split_csv.loc[j, 'Unnamed: 0']]['T'].values[0]
                        else:
                            split_csv.loc[j, 'label'] = label_csv[label_csv['slide_id'] == split_csv.loc[j, 'Unnamed: 0']]['label'].values[0]
                split_csv = split_csv.rename(columns={'Unnamed: 0':'slide_id'})
                split_csv.to_csv(os.path.join(split_dir, 'splits_{}_bool_label.csv'.format(i)), index=None)
    else:
        for n_shots in args.shots:

            split_dir = os.path.join('splits', str(args.task), f'{n_shots}_shot')
            os.makedirs(split_dir, exist_ok=True)
            print(f"\n==============================================")
            print(f"Generating splits for {n_shots} shots")
            print(f"Saving to: {split_dir}")
            print(f"==============================================")

            for i in range(args.k):
                trial_seed = args.seed + i

                # Generate the lists of slide IDs
                train_ids, val_ids, test_ids = create_few_shot_splits(label_csv=label_csv,
                                                                      label_col='grade',
                                                                      n_shots=n_shots,
                                                                      val_frac=args.val_frac,
                                                                      test_frac=args.test_frac,
                                                                      seed=trial_seed)
                # --- Direct saving using pandas ---
                # Convert each list of IDs to a pandas Series
                s_train = pd.Series(train_ids, name='train')
                s_val = pd.Series(val_ids, name='val')
                s_test = pd.Series(test_ids, name='test')

                # Concatenate the Series into a DataFrame. They will be aligned side-by-side.
                df = pd.concat([s_train, s_val, s_test], axis=1)

                # Define the output path
                output_path = os.path.join(split_dir, f'splits_{i}.csv')

                # Save to CSV, ensuring not to write the default row index (0, 1, 2...)
                df.to_csv(output_path, index=False)

                print(f"Saved {output_path} (Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)})")
        print("\nAll splits generated successfully.")
